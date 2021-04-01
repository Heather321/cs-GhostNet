#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Time : 2021/3/16 16:59
@Author : Xiaoqing Hu
@File : GhostSCNet.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(ChannelAttention, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_avg = self.avg_pool(x)
        x_man = self.max_pool(x)
        x_se = x_avg + x_man
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return self.gate_fn(x_se)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# 输入特征图，输出卷积平滑后的特征图，并保持特征图大小不变，个数不变
class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., spa_kernel = 3, save_result = False):
        super(GhostBottleneck, self).__init__()
        self.save_result = save_result
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if se_ratio < 1 :
            self.schannel = ChannelAttention(mid_chs, se_ratio=se_ratio)
        else:
            self.schannel = nn.Sequential()

        # Squeeze-and-excitation
        if spa_kernel > 1:
            self.spatial = SpatialAttention(kernel_size=spa_kernel)
        else:
            self.spatial = nn.Sequential()

            # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)
        if self.save_result:
            save_modulechannels(x.cpu(), modelname = 'ghostSC2', modulename = 'ghost1', cat = 'feature_0')

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        se = self.schannel(x)
        spa = self.spatial(x)

        x = x * se * spa
        if self.save_result:
            save_modulechannels(x.cpu(), modelname='ghostSC2', modulename='sespa', cat='feature_0')
        # 2nd ghost bottleneck
        x = self.ghost2(x)
        if self.save_result:
            save_modulechannels(x.cpu(), modelname='ghostSC2', modulename='ghost2', cat='feature_0')

        x += self.shortcut(residual)
        return x


class GhostSCNet2(nn.Module):
    def __init__(self, class_num, pca, window, save_result=False):
        super(GhostSCNet2, self).__init__()
        self.pca = pca
        self.window = window
        self.conv = nn.Sequential(
            nn.Conv2d(pca, 64, 3),  # 17
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.layer2 = nn.Sequential(
            GhostBottleneck(64, 128, 256, se_ratio=0.5, spa_kernel=3, save_result=save_result),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3),  # 15
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 128, 3),  # 13
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.fc = nn.Sequential(
            nn.Linear(128* (window-6) * (window-6), 1024),
            nn.ReLU(),
            nn.Linear(1024, 225),
            nn.ReLU(),
            nn.Linear(225, class_num),
        )

    def forward(self, x, save_result = False, class_num = 0):
        class_name = 'feature' + str(class_num)
        basePath = f'result/ghostSC2/{class_name}'
        x = self.conv(x)
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x = self.layer4(x)
        
        out = self.fc(x.view(-1, 128* (self.window-6) * (self.window-6)))
        return out
