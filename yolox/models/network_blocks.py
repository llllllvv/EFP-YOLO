#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .deform_cov import DeformableConv2d
from .whsa import *

# XBN block
def GroupNorm(num_features, num_groups=64, eps=1e-5, affine=True):
    if num_groups > num_features:
        print('------arrive maxum groub numbers of:', num_features)
        num_groups = num_features
    return nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine)

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu",bn='bn',is_focus=False,
    ):
        super().__init__()
        # same padding
        if is_focus:
            pad = 2
        else:
            pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn == 'bn' else GroupNorm(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

    

class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels,activation):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)  # 这里N=4与原文一致
        self.conv1 = BaseConv(in_channels, inter_channels, 1,1,act=activation)  # 四个1x1卷积用来减小channel为原来的1/N
        self.conv2 = BaseConv(in_channels, inter_channels, 1,1,act=activation)
        self.conv3 = BaseConv(in_channels, inter_channels, 1,1,act=activation)
        self.conv4 = BaseConv(in_channels, inter_channels, 1,1,act=activation)
        self.out = BaseConv(in_channels * 2, out_channels, 1,1,act=activation)  # 最后的1x1卷积缩小为原来的channel

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)  # 自适应的平均池化，目标size分别为1x1,2x2,3x3,6x6
        return avgpool(x)

    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)  # concat 四个池化的结果
        x = self.out(x)
        return x

        
        
class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1, dilation=vision + 1, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = get_activation(name="silu")

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = get_activation(name="silu") if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        
class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, ksize=1, bias=False,stride=1)
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, ksize=1, bias=False,stride=1)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))) #fm 就是激活加卷积，我觉得这平均池化用的贼巧妙
        feat = torch.mul(x, atten) #相乘，得到重要特征
        x = x + feat #再加上
        feat = self.conv(x) #最后一层 1*1 的卷积
        return feat

class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128):
        super().__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc) #FSM 部分
        # self.offset = nn.Conv2d(out_nc*2,
        #           out_nc,
        #           kernel_size=1,
        #           stride=1,
        #           padding=0,
        #           bias=True)
        self.dcpack_L2 = DeformableConv2d(out_nc, out_nc, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # weight_init.c2_xavier_fill(self.offset)


    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]: #如果两者不一样就 upsample
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  #0~1 * feats 经过 FDM 模块
        # offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        # 至于上面 feat_up 为什么 *2 我也不是很清楚

        feat_align = self.relu(self.dcpack_L2(feat_up))  # [feat, offset]
        return feat_align + feat_arm
        
class Conv2d(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu",bn='bn',padding=0,
    ):
        super().__init__()
        # same padding
        pad = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn == 'bn' else GroupNorm(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):#baseConv块
        return self.act(self.bn(self.conv(x)))
        # return self.bn(self.conv(x))


class CSPLayer_ho_res2net_new(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act,bn='xbn')

        self.bottleneck_1=Bottleneck_ho(# Bottleneck Bottleneck_res2net Bottleneck_ho
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
        self.bottleneck_2 = Bottleneck_conv(# Bottleneck_conv Bottleneck_res2net_conv Bottleneck_conv
            hidden_channels, hidden_channels, 1.0, depthwise, act=act
        )
        self.bottleneck_3 = Bottleneck_conv(
            hidden_channels, hidden_channels, 1.0, depthwise, act=act
        )
        self.bottleneck_4 = Bottleneck_conv(
            hidden_channels, hidden_channels, 1.0, depthwise, act=act
        )

        self.activate= get_activation(name="silu",inplace=False)



    def forward(self, x):
        x_1 = self.conv1(x)   #x
        x_2 = self.conv2(x)

        out = self.bottleneck_1(x_1)#k1
        outx = 1 / 6. * out

        out = self.bottleneck_2(0.5 * out) + x_1#k2
        outx += 1 / 3. * out

        out = self.bottleneck_3(0.5 * out) + x_1#k3
        outx += 1 / 3. * out

        out = self.bottleneck_4(out) + x_1#k4
        outx += 1 / 6. * out

        out = outx + x_1
        out = self.activate(out)

        x = torch.cat((out, x_2), dim=1)
        return self.conv3(x)


class Bottleneck_ho(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        #self.ECA = ECA(channel=out_channels)#在原来的基础上加上了eca模块
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class Bottleneck_conv(nn.Module):
    # Standard bottleneck 红色的resunit
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, act=act)#3*3卷积 如果使用xbn的话 就加上bn=‘xbn’
        #self.ECA = ECA(channel=out_channels)#在原来的基础上加上了eca模块 76.03是没有的

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y
    


class CSPLayer_swin(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        act="silu",
        w=40,
        h=40,
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        self.m1 = Bottleneck_Swin(
                hidden_channels, hidden_channels, n=1,
            )
        self.m2 =Bottleneck_Swin(
                hidden_channels, hidden_channels, n=1,
            )
        self.m3 = Bottleneck_Swin(
                hidden_channels, hidden_channels, n=1,
            )
        # self.m = nn.Sequential(*module_list)
        # self.conv_ghost = GhostModule(hidden_channels,hidden_channels,kernel_size=3,stride=1)
        self.conv_dw = DWConv(hidden_channels,hidden_channels,ksize=3,stride=1)
        module_list_channel = [nn.Conv2d(hidden_channels,hidden_channels,kernel_size=1,stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels,hidden_channels,kernel_size=1,stride=1),
                               nn.Sigmoid()]
        module_list_spatial = [nn.Conv2d(hidden_channels,hidden_channels,kernel_size=1,stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels,hidden_channels,kernel_size=1,stride=1),
                               nn.Sigmoid()]
        self.channel_action = nn.Sequential(*module_list_channel)
        self.spatial_action = nn.Sequential(*module_list_spatial)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv_dw(self.conv2(x))
        _,_,H,W=x_2.size()
        apt=nn.AdaptiveAvgPool2d(H)
        x_2=apt(x_2)
        x_2_m = self.channel_action(x_2)
        x_1 = self.m1(x_1,x_2_m)
        x_2 = self.conv_dw(self.spatial_action(x_1) * x_2)
        apt = nn.AdaptiveAvgPool2d(H)
        x_2 = apt(x_2)
        x_2_m = self.channel_action(x_2)
        x_1 = self.m2(x_1,x_2_m)
        x_2 = self.conv_dw(self.spatial_action(x_1) * x_2)
        apt = nn.AdaptiveAvgPool2d(H)
        x_2 = apt(x_2)
        x_2_m = self.channel_action(x_2)
        x_1 = self.m3(x_1, x_2_m)
        x_2 = self.spatial_action(x_1) * x_2
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)





class Bottleneck_Swin(nn.Module):
    # Standard bottleneck 红色的resunit
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=1.0,
        act="silu",
        heads=4,
        resolution=None,
        window_size=7,
        qkv_bias=True,
        attn_drop=0.,
        drop=0.,
        n=1,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        # self.conv2 = MHSA(out_channels, width=int(resolution[0]), height=int(resolution[1]), heads=heads)#3*3卷积 如果使用xbn的话 就加上bn=‘xbn’
        self.window_size = window_size
        self.patch_embed1 = PatchEmbed(
            patch_size=1, in_c=hidden_channels, embed_dim=hidden_channels,
            norm_layer=nn.LayerNorm)
        num_heads = hidden_channels // 32
        self.swin = SwinTransformerBlock(
                dim=hidden_channels,
                num_heads=8,
                window_size=7,
                shift_size=0,
                mlp_ratio=4,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm)

        self.use_add = shortcut and in_channels == out_channels

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / 7)) * 7
        Wp = int(np.ceil(W / 7)) * 7
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=None)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        w_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, 7)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, 7 * 7)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x1,x2):
        y = self.conv1(x1)
        b, c, H, W = y.shape
        y, H, W = self.patch_embed1(y)
        x2,H2,W2 = self.patch_embed1(x2)
        attn_mask = self.create_mask(x=y, H=H, W=W)
        self.swin.H, self.swin.W = H, W
        self.swin.H2, self.swin.W2 = H2, W2
        y = self.swin(y,x2,attn_mask)
        y = y.permute(0, 2, 1).view(b, c, H, W)
        if self.use_add:
            y = y + x1
        return y

class CSPLayer_swin_new(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        self.m1=Bottleneck_swin_new(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
        self.m2 = Bottleneck_swin_new(
            hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
        )
        self.m3 = Bottleneck_swin_new(
            hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
        )
        self.w1=Swin_csp(hidden_channels, hidden_channels, n=1)
        self.w2=Swin_csp(hidden_channels, hidden_channels, n=1)
        self.w3=Swin_csp(hidden_channels, hidden_channels, n=1)
        module_list_channel = [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        self.channel_action = nn.Sequential(*module_list_channel)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1,x_1_m = self.m1(x_1)
        _, _, H, W = x_1_m.size()
        apt = nn.AdaptiveAvgPool2d(H)
        x_1_m = apt(x_1_m)
        x_2 = self.w1(x_2,self.channel_action(x_1_m))
        x_1,x_1_m = self.m2(x_1)
        _, _, H, W = x_1_m.size()
        apt = nn.AdaptiveAvgPool2d(H)
        x_1_m = apt(x_1_m)
        x_2 = self.w2(x_2, self.channel_action(x_1_m))
        x_1,x_1_m = self.m3(x_1)
        _, _, H, W = x_1_m.size()
        apt = nn.AdaptiveAvgPool2d(H)
        x_1_m = apt(x_1_m)
        x_2 = self.w3(x_2, self.channel_action(x_1_m))
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Bottleneck_swin_new(nn.Module):
    # Standard bottleneck 红色的resunit
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        # Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = DWConv(hidden_channels, out_channels, 3, stride=1, act=act)#3*3卷积 如果使用xbn的话 就加上bn=‘xbn’

        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        y1=y
        if self.use_add:
            y = y + x
        return y,y1

class Swin_csp(nn.Module):
    # Standard bottleneck 红色的resunit
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=1.0,
        act="silu",
        heads=4,
        resolution=None,
        window_size=7,
        qkv_bias=True,
        attn_drop=0.,
        drop=0.,
        n=1,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.window_size = window_size
        self.patch_embed1 = PatchEmbed(
            patch_size=1, in_c=hidden_channels, embed_dim=hidden_channels,
            norm_layer=nn.LayerNorm)
        num_heads = hidden_channels // 32
        self.swin = SwinTransformerBlock(
                dim=hidden_channels,
                num_heads=8,
                window_size=7,
                shift_size=0,
                mlp_ratio=4,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm)

        self.use_add = shortcut and in_channels == out_channels

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / 7)) * 7
        Wp = int(np.ceil(W / 7)) * 7
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=None)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        w_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, 7)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, 7 * 7)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x1,x2):
        b, c, H, W = x1.shape
        y, H, W = self.patch_embed1(x1)
        x2,H2,W2 = self.patch_embed1(x2)
        attn_mask = self.create_mask(x=y, H=H, W=W)
        self.swin.H, self.swin.W = H, W
        self.swin.H2, self.swin.W2 = H2, W2
        y = self.swin(y,x2,attn_mask)
        y = y.permute(0, 2, 1).view(b, c, H, W)
        return y


class CSPLayer_swin_transformer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        act="silu",
        w=40,
        h=40,
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        self.m1 = Bottleneck_Swin_transformer(
                hidden_channels, hidden_channels, n=1,
            )
        self.m2 =Bottleneck_Swin_transformer(
                hidden_channels, hidden_channels, n=1,
            )
        self.m3 = Bottleneck_Swin_transformer(
                hidden_channels, hidden_channels, n=1,
            )
        # self.m = nn.Sequential(*module_list)
        # self.conv_ghost = GhostModule(hidden_channels,hidden_channels,kernel_size=3,stride=1)
        self.conv_dw = DWConv(hidden_channels,hidden_channels,ksize=3,stride=1)
        module_list_channel = [nn.Conv2d(hidden_channels,hidden_channels,kernel_size=1,stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels,hidden_channels,kernel_size=1,stride=1),
                               nn.Sigmoid()]
        module_list_spatial = [nn.Conv2d(hidden_channels,hidden_channels,kernel_size=1,stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels,hidden_channels,kernel_size=1,stride=1),
                               nn.Sigmoid()]
        self.channel_action = nn.Sequential(*module_list_channel)
        self.spatial_action = nn.Sequential(*module_list_spatial)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv_dw(self.conv2(x))
        _,_,H,W=x_2.size()
        apt=nn.AdaptiveAvgPool2d(H)
        x_2=apt(x_2)
        x_2_m = self.channel_action(x_2)
        x_1 = self.m1(x_1*x_2_m)
        x_2 = self.conv_dw(self.spatial_action(x_1) * x_2)#将相加改成相乘
        # x_2 = self.conv_dw(self.spatial_action(x_1) + x_2)
        # x_2 = self.conv_dw(x_2)
        apt = nn.AdaptiveAvgPool2d(H)
        x_2 = apt(x_2)
        x_2_m = self.channel_action(x_2)
        x_1 = self.m2(x_1*x_2_m)
        x_2 = self.conv_dw(self.spatial_action(x_1) * x_2)
        # x_2 = self.conv_dw(x_2)
        apt = nn.AdaptiveAvgPool2d(H)
        x_2 = apt(x_2)
        x_2_m = self.channel_action(x_2)
        x_1 = self.m3(x_1*x_2_m)
        x_2 = self.spatial_action(x_1) * x_2
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)



class Bottleneck_Swin_transformer(nn.Module):
    # Standard bottleneck 红色的resunit
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=1.0,
        act="silu",
        heads=4,
        resolution=None,
        window_size=7,
        qkv_bias=True,
        attn_drop=0.,
        drop=0.,
        n=1,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        # self.conv2 = MHSA(out_channels, width=int(resolution[0]), height=int(resolution[1]), heads=heads)#3*3卷积 如果使用xbn的话 就加上bn=‘xbn’
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.patch_embed1 = PatchEmbed(
            patch_size=1, in_c=hidden_channels, embed_dim=hidden_channels,
            norm_layer=nn.LayerNorm)
        num_heads = hidden_channels // 32
        self.swin = nn.ModuleList([
            SwinTransformerBlock(
                dim=hidden_channels,
                num_heads=8,
                window_size=7,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=4,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=0.,
                norm_layer=nn.LayerNorm)
            for i in range(2)])

        self.use_add = shortcut and in_channels == out_channels

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / 7)) * 7
        Wp = int(np.ceil(W / 7)) * 7
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=None)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        w_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, 7)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, 7 * 7)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        y = self.conv1(x)
        b, c, H, W = y.shape
        y, H, W = self.patch_embed1(y)
        attn_mask = self.create_mask(x=y, H=H, W=W).to(y.device)

        for blk in self.swin:
            blk.H, blk.W = H, W
            y = blk(y, attn_mask)


        y = y.permute(0, 2, 1).view(b, c, H, W)
        if self.use_add:
            y = y + x
        return y
        
        
class CSPLayer_swin_resnest(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            # AFFBottleneck(
            #     hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            # )
            # Bottleneck_swin_resnest(   #可以选择dw/gnconv+swin
            #     hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            # )
            Bottleneck_resnest(#这里是dw+gnconv的组合，用gnconv代替swin
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Bottleneck_resnest(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        window_size=7,
        qkv_bias=True,
        attn_drop=0.,
        drop=0.,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1x1_1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv1x1_2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv1x1_3 = BaseConv(hidden_channels*2, hidden_channels, 1, stride=1, act=act)

        self.conv_dw = BaseConv(hidden_channels, hidden_channels, 3, stride=1, act=act)#3*3卷积 如果使用xbn的话 就加上bn=‘xbn’
        self.conv_gn = gnconv(hidden_channels,size=7)
        self.use_add = shortcut and in_channels == out_channels

        module_list_channel = [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        module_list_spatial = [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        self.channel_action = nn.Sequential(*module_list_channel)
        self.spatial_action = nn.Sequential(*module_list_spatial)

    def forward(self, x):
        x_1 = self.conv_dw(self.conv1x1_1(x))
        _, _, H, W = x_1.size()
        apt = nn.AdaptiveAvgPool2d(H)
        x_1_m = apt(x_1)
        x_1_m = self.channel_action(x_1_m)
        # x_1要经过dw卷积
        # x_2 = self.conv1x1_2(x)
        x_2 = self.conv1x1_2(x) * x_1_m
        # x_2要经过gnconv
        x_2 = self.conv_gn(x_2)
        x_2_m = self.spatial_action(x_2)
        x_1 = x_1 * x_2_m
        y = self.conv1x1_3(torch.cat((x_1,x_2),dim=1))
        if self.use_add:
            y = y + x
        return y


class gnconv(nn.Module):  # gnconv模块
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0,size=3):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), size, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        self.scale = s

    def forward(self, x):
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = self.proj_out(x)

        return x


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

class CSPLayer_ResNest(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        w, h,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",

    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Resnest_Bottleneck(
                hidden_channels, hidden_channels,w=w,h=h,
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

class Resnest_Bottleneck(nn.Module):
    # Standard bottleneck 红色的resunit
    def __init__(
        self,

        in_channels,
        out_channels,
        w, h,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",

    ):
        super().__init__()
        # hidden_channels = int(out_channels * expansion)
        self.conv1_1 = BaseConv(in_channels, in_channels//2, 1, stride=1, act=act)
        # self.split_attn= SplAtConv2d_wulianjie(
        #     in_channels=in_channels//2, channels=in_channels//2, groups=2, radix=2, w=w,h=h)
        self.split_attn = SplAtConv2d_lianjie1(
            in_channels=in_channels // 2, channels=in_channels // 2, groups=2, radix=2, w=w, h=h)
        self.use_add = shortcut and in_channels == out_channels
        # self.channel_shuffle=nn.ChannelShuffle(4)
        self.conv1_2 = BaseConv(in_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        y1 = self.conv1_1(x)
        y2 = y1
        y1 = self.split_attn(y1)
        y2 = self.split_attn(y2)
        y = torch.cat((y1,y2),dim=1)
        # y = self.channel_shuffle(y)
        y = self.conv1_2(y)
        if self.use_add:
            y = y + x
        return y
# 有链接+swin
class SplAtConv2d_lianjie1(nn.Module):
    """
    Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, w,h, groups=2,
                 radix=2, reduction_factor=4, norm_layer=nn.BatchNorm2d,
                 window_size=7,
                 qkv_bias=True,
                 attn_drop=0.,
                 drop=0.,
                 ):
        super(SplAtConv2d_lianjie1, self).__init__()
        # #
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix#2
        self.cardinality = groups #2
        self.channels = channels

        # self.radix_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
        #               groups=groups * radix, bias=bias, **kwargs),
        #     norm_layer(channels * radix),
        #     nn.ReLU(inplace=True)
        # )
        self.conv1x1 = BaseConv(in_channels,(channels * radix)//2,ksize=1,stride=1,act="silu")
        self.conv3x3 = DWConv((channels * radix)//2,(channels * radix)//2,ksize=3,stride=1,act="silu")
        self.convgn = gnconv((channels * radix)//2,size=7)
        hidden_channels = (channels * radix)//2
        module_list_channel = [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        module_list_spatial = [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        self.channel_action = nn.Sequential(*module_list_channel)
        self.spatial_action = nn.Sequential(*module_list_spatial)
        # self.channel_shuffle = nn.ChannelShuffle(2)

        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        self.relu = get_activation(name="silu",inplace=True)
        self.rsoftmax = rSoftMax(radix, groups)

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.patch_embed1 = PatchEmbed(
            patch_size=1, in_c=hidden_channels, embed_dim=hidden_channels,
            norm_layer=nn.LayerNorm)
        num_heads = hidden_channels // 32
        self.swin = nn.ModuleList([
            SwinTransformerBlock(
                dim=hidden_channels,
                num_heads=8,
                window_size=7,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=4,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=0.,
                norm_layer=nn.LayerNorm)
            for i in range(2)])

    def channel_shuffle(self,x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / 7)) * 7
        Wp = int(np.ceil(W / 7)) * 7
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=None)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        w_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, 7)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, 7 * 7)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        # -------------------------------
        # 经过radix_conv即组卷积产生multi branch个分支U
        # U等分成radix个组，组求和得到gap通道内的值
        # x = self.radix_conv(x)#这里分开  然后channel shuffle？+concat channels * radix,
        x1 = self.conv1x1(x)
        x1 = self.conv3x3(x1)
        _, _, H, W = x1.size()
        apt = nn.AdaptiveAvgPool2d(H)
        x_1_m = apt(x1)
        x_1_m = self.channel_action(x_1_m)
        x2 = self.conv1x1(x) * x_1_m
        b, c, H, W = x2.shape
        x_2, H, W = self.patch_embed1(x2)
        attn_mask = self.create_mask(x=x_2, H=H, W=W).to(x_2.device)
        for blk in self.swin:
            blk.H, blk.W = H, W
            x_2 = blk(x_2, attn_mask)

        x2 = x_2.permute(0, 2, 1).view(b, c, H, W)
        x_2_m = self.spatial_action(x2)
        x1 = x1 * x_2_m

        x = self.channel_shuffle(torch.cat((x1,x2),dim=1),groups=2)

        batch, rchannel = x.shape[:2]
        splited = torch.split(x, rchannel // self.radix, dim=1)
        gap = sum(splited)
        # -------------------------------
        # gap通道内 avgpool + fc1 + fc2 + softmax
        # 其中softmax是对radix维度进行softmax
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        # -------------------------------
        # 将gap通道计算出的和注意力和原始分出的radix组个branchs相加得到最后结果
        attens = torch.split(atten, rchannel // self.radix, dim=1)
        out = sum([att * split for (att, split) in zip(attens, splited)])
        # -------------------------------
        # 返回一个out的copy, 使用contiguous是保证存储顺序的问题
        return out.contiguous()


# 对radix维度进行softmax
class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)

        x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        # x: [Batchsize, radix, cardinality, h, w]
        x = F.softmax(x, dim=1)  # 对radix维度进行softmax
        x = x.reshape(batch, -1)

        return x
        
        
        
        
    
class CSPLayer_mokuai(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            # AFFBottleneck(
            #     hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            # )
            mokuai(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)
        
    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1,x_2), dim=1)
        return self.conv3(x)

class mokuai(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1x1_1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        self.conv1x1_2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        self.conv1x1_3 = BaseConv(hidden_channels*2, hidden_channels, 1, stride=1, act=act)

        self.conv_dw = BaseConv(hidden_channels, hidden_channels, 3, stride=1, act=act)#3*3卷积 如果使用xbn的话 就加上bn=‘xbn’
        self.conv_gn = gnconv(hidden_channels,size=7)
        self.use_add = shortcut and in_channels == out_channels

        module_list_channel = [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        module_list_spatial = [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        self.channel_action = nn.Sequential(*module_list_channel)
        self.spatial_action = nn.Sequential(*module_list_spatial)


    def forward(self, x):
        x_1 = self.conv_dw(self.conv1x1_1(x))
        _, _, H, W = x_1.size()
        apt = nn.AdaptiveAvgPool2d(H)
        x_1_m = apt(x_1)
        x_1_m = self.channel_action(x_1_m)
        # x_1要经过dw卷积
        # x_2 = self.conv1x1_2(x)
        x_2 = self.conv1x1_2(x) * x_1_m
        # x_2要经过gnconv
        x_2 = self.conv_gn(x_2)
        x_2_m = self.spatial_action(x_2)
        x_1 = x_1 * x_2_m
        
        y = self.conv1x1_3(torch.cat((x_1, x_2),dim=1))
        return y

    
    
class CSPLayer_srt(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
            self,
            in_channels,
            out_channels,
            n=1,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck_srt(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Bottleneck_srt(nn.Module):
    # Standard bottleneck
    def __init__(
            self,
            in_channels,
            out_channels,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1_1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        self.conv1_2 = BaseConv(hidden_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        self.conv_gonv = BaseConv(hidden_channels, hidden_channels, 3, stride=1, act=act)
        self.conv_whsa = gnconv(hidden_channels,size=3)
        self.use_add = shortcut and in_channels == out_channels
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()

    def forward(self, x):
        y1 = self.conv_gonv(self.conv1_1(x))
        if self.use_add:
            y = self.coefficient2(y1) + self.coefficient1(x)
        y2 = self.conv_whsa(self.conv1_2(y))
        if self.use_add:
            y2 = self.coefficient4(y2) + self.coefficient3(y)
        return y2

class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out
    
    
    
    
class CSPLayer_mokuai_xin(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act, bn='xbn')
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act, bn='xbn')
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        self.m1 = mokuai_1(hidden_channels, hidden_channels, expansion=1.0, depthwise=False, act="silu")
        self.m2 = mokuai_2(hidden_channels*2, hidden_channels, expansion=1.0, act="silu")
        self.m3 = mokuai_3(hidden_channels*3, hidden_channels, expansion=1.0, act="silu")
        self.conv1x1=BaseConv(hidden_channels*4, hidden_channels, 1, stride=1, act=act, bn='xbn')


    def forward(self, x):
        x_1 = self.conv1(x)#上面1*1卷积
        a = x_1
        x_2 = self.conv2(x)#底下的卷积
        b = self.m1(x_1)
        x_1 = torch.cat((a,b),dim=1)
        c = self.m2(x_1)
        x_1 = torch.cat((a,b,c),dim=1)
        d = self.m3(x_1)
        x_1 = torch.cat((a,b,c,d),dim=1)
        x_1 = self.conv1x1(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class mokuai_1(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, act=act,bn='xbn')
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y
    
    
class mokuai_2(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=1.0,
        act="silu",
        heads=4,
        resolution=None,
        window_size=7,
        qkv_bias=True,
        attn_drop=0.,
        drop=0.,
        n=1,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, hidden_channels, 3, stride=1, act=act)
        self.conv3 = BaseConv(hidden_channels, hidden_channels, 3, stride=1, act=act)
        self.swin = SwinTransformerBlock(
                dim=hidden_channels,
                num_heads=8,
                window_size=7,
                shift_size=0,
                mlp_ratio=4,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm)
        self.patch_embed = PatchEmbed(
            patch_size=1, in_c=hidden_channels, embed_dim=hidden_channels,
            norm_layer=nn.LayerNorm)
        self.use_add = shortcut and in_channels == out_channels
        self.conv4 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)
        module_list_channel = [nn.AdaptiveAvgPool2d(1),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        module_list_spatial = [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        self.channel_action = nn.Sequential(*module_list_channel)
        self.spatial_action = nn.Sequential(*module_list_spatial)
        
    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / 7)) * 7
        Wp = int(np.ceil(W / 7)) * 7
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=None)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        w_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, 7)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, 7 * 7)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


    def forward(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        
        _, _, H, W = y.size()
        apt = nn.AdaptiveAvgPool2d(H)
        y_m = apt(y)
        y_m = self.channel_action(y_m)        
        
        b, c, H, W = y.shape
        y_swin, H, W = self.patch_embed(y*y_m)
        attn_mask = self.create_mask(x=y_swin, H=H, W=W).to(y_swin.device)

        self.swin.H, self.swin.W = H, W
        y_swin = self.swin(y_swin, attn_mask)
        y_swin= y_swin.permute(0, 2, 1).view(b, c, H, W)
        
        y_swin=self.spatial_action(y_swin)
        
        y = self.conv4(y*y_swin)
        return y
    
    
class mokuai_3(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=1.0,
        act="silu",
        heads=4,
        resolution=None,
        window_size=7,
        qkv_bias=True,
        attn_drop=0.,
        drop=0.,
        n=1,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, hidden_channels, 3, stride=1, act=act)
        self.conv3 = BaseConv(hidden_channels, hidden_channels, 3, stride=1, act=act)
        self.conv4 = BaseConv(hidden_channels, hidden_channels, 3, stride=1, act=act)
        self.swin = SwinTransformerBlock(
                dim=hidden_channels,
                num_heads=8,
                window_size=7,
                shift_size=0,
                mlp_ratio=4,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm)
        self.use_add = shortcut and in_channels == out_channels
        self.patch_embed = PatchEmbed(
            patch_size=1, in_c=hidden_channels, embed_dim=hidden_channels,
            norm_layer=nn.LayerNorm)
        self.conv5 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)
        module_list_channel = [nn.AdaptiveAvgPool2d(1),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        module_list_spatial = [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        self.channel_action = nn.Sequential(*module_list_channel)
        self.spatial_action = nn.Sequential(*module_list_spatial)
        
    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / 7)) * 7
        Wp = int(np.ceil(W / 7)) * 7
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=None)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        w_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, 7)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, 7 * 7)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


    def forward(self, x):
        y = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        
        _, _, H, W = y.size()
        apt = nn.AdaptiveAvgPool2d(H)
        y_m = apt(y)
        y_m = self.channel_action(y_m)    
        
        b, c, H, W = y.shape
        y_swin, H, W = self.patch_embed(y*y_m)
        attn_mask = self.create_mask(x=y_swin, H=H, W=W).to(y_swin.device)

        self.swin.H, self.swin.W = H, W
        y_swin = self.swin(y_swin, attn_mask)
        y_swin = y_swin.permute(0, 2, 1).view(b, c, H, W)
        
        y_swin=self.spatial_action(y_swin)
        y=self.conv5(y*y_swin)
        return y
    
    
    
class CSPLayer_conformer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
            self,
            in_channels,
            out_channels,
            n=1,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act,bn='xbn')
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        self.m=Bottleneck_conformer(
                hidden_channels, hidden_channels, shortcut, 1.0
            )
        self.m2=Bottleneck_conformer(
                hidden_channels, hidden_channels, shortcut, 1.0
            )

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x_1 = self.m2(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Bottleneck_conformer(nn.Module):
    # Standard bottleneck
    def __init__(
            self,
            in_channels,
            out_channels,
            shortcut=True,
            expansion=1.0,
            act="silu",
            heads=4,
            resolution=None,
            window_size=7,
            qkv_bias=True,
            attn_drop=0.,
            drop=0.,
            n=1,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv4 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv7 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        self.conv2 = Conv(hidden_channels, hidden_channels, 3, stride=1, act=act)
        self.conv5 = Conv(hidden_channels, hidden_channels, 3, stride=1, act=act)
        self.conv8 = Conv(hidden_channels, hidden_channels, 3, stride=1, act=act)

        self.conv3 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)
        self.conv6 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)
        self.conv9 = BaseConv(hidden_channels, out_channels, 1, stride=1, act=act)
        self.swin = self.swin = SwinTransformerBlock(
                dim=hidden_channels,
                num_heads=8,
                window_size=7,
                shift_size=0,
                mlp_ratio=4,
                qkv_bias=True,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm)
        self.use_add = shortcut and in_channels == out_channels
        self.patch_embed1 = PatchEmbed(
            patch_size=1, in_c=hidden_channels, embed_dim=hidden_channels,
            norm_layer=nn.LayerNorm)
        module_list_channel = [nn.AdaptiveAvgPool2d(1),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        module_list_spatial = [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.BatchNorm2d(hidden_channels),
                               nn.GELU(),
                               nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1),
                               nn.Sigmoid()]
        self.channel_action = nn.Sequential(*module_list_channel)
        self.spatial_action = nn.Sequential(*module_list_spatial)

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / 7)) * 7
        Wp = int(np.ceil(W / 7)) * 7
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=None)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        w_slices = (slice(0, -7),
                    slice(-7, -3),
                    slice(-3, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, 7)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, 7 * 7)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        y = self.conv2(self.conv1(x))

        _, _, H, W = y.size()
        apt = nn.AdaptiveAvgPool2d(H)
        y_m = apt(y)
        y_m = self.channel_action(y_m)

        b, c, H, W = x.shape
        swin_y, H, W = self.patch_embed1(x*y_m)
        attn_mask = self.create_mask(x=swin_y, H=H, W=W)
        self.swin.H, self.swin.W = H, W
        swin_y = self.swin(swin_y,attn_mask)
        swin_y = swin_y.permute(0, 2, 1).view(b, c, H, W)

        y1 = self.conv3(y)
        if self.use_add:
            y1 = y1 + x
        y2 = self.conv4(y1)
        swin_y_m=self.spatial_action(swin_y)
        y2 = self.conv5(y2*swin_y_m)

        _, _, H, W = y2.size()
        apt = nn.AdaptiveAvgPool2d(H)
        y_m = apt(y2)
        y_m = self.channel_action(y_m)

        y2 = self.conv6(y2)
        if self.use_add:
            y2 = y2 + y1

        b, c, H, W = swin_y.shape
        swin_y, H, W = self.patch_embed1(swin_y * y_m)
        attn_mask = self.create_mask(x=swin_y, H=H, W=W)
        self.swin.H, self.swin.W = H, W
        swin_y = self.swin(swin_y, attn_mask)
        swin_y = swin_y.permute(0, 2, 1).view(b, c, H, W)
        swin_y_m = self.spatial_action(swin_y)

        y3 = self.conv7(y2)
        y3= self.conv9(self.conv8(y3*swin_y_m))
        if self.use_add:
            y3 = y3+y2

        return y3