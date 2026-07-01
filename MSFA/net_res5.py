# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
from network_module2 import Conv2dLayer
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self,channel,ratio= 16):
        super(SELayer,self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.gap = nn.AdaptiveAvgPool2d(1)          # sq 压缩，自适应平均池化
        self.fc = nn.Sequential(                    # ex 激励，2个fc
                nn.Linear(channel, channel //ratio, bias=False),
                nn.ReLU(inplace = True),
                nn.Linear(channel //ratio, channel, bias=False),
                nn.Sigmoid()
        )
        # nn.Linear(in_features，out_features，bias=False)
# n_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
# out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]。

    def forward(self, x):
        b,c,h,w = x.size()
        y = self.gap(x).view(b,c)           # sq 压缩
        y = self.fc(y).view(b,c,1,1)        # ex 激励
        return x * y.expand_as(x)

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

class Attention(nn.Module):
    def __init__(self, channel):
        super(Attention, self).__init__()

        self.ca = SELayer(channel)
        self.sa = SpatialAttention()
        self.main = nn.Sequential(
            Conv2dLayer(channel, channel, kernel_size=3, stride=1, padding=1, activation='relu'),
            Conv2dLayer(channel, channel, kernel_size=1, stride=1, padding=0, activation='relu')
        )

    def forward(self, x):
        y = self.ca(x) * x
        y = self.sa(y) * y
        x_att = x + y
        return self.main(x_att) + x_att


class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.sa = Attention(channel)

    def forward(self, xs, anchor):
        b,c,h,w = anchor.size()
        ans = torch.zeros_like(anchor)

        for i, x in enumerate(xs):
            if x.shape[-1] > w:
                x = F.adaptive_avg_pool2d(x, (h,w))
                x = self.sa(x)
            elif x.shape[-1] < w:
                x = F.interpolate(x, size=(h,w),mode='bilinear', align_corners=True)
                x = self.sa(x)
            elif x.shape[-1] == w:
                x = self.sa(x)

            ans = ans + x

        return ans

class MDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        # self.sa = Attention(channel)

    def forward(self, xs, anchor):
        b,c,h,w = anchor.size()
        ans = torch.zeros_like(anchor)

        for i, x in enumerate(xs):
            if x.shape[-1] > w:
                x = F.adaptive_avg_pool2d(x, (h,w))
                # x = self.sa(x)
            elif x.shape[-1] < w:
                x = F.interpolate(x, size=(h,w),mode='bilinear', align_corners=True)
                # x = self.sa(x)
            elif x.shape[-1] == w:
                x = x
                # x = self.sa(x)

            ans = ans + x

        return ans

class MDC(nn.Module):
    def __init__(self, channel):
        super().__init__()

        # self.sa = Attention(channel)

    def forward(self, xs, anchor):
        b,c,h,w = anchor.size()
        ans = anchor

        for i, x in enumerate(xs):
            if x.shape[-1] > w:
                x = F.adaptive_avg_pool2d(x, (h, w))
                # x = self.sa(x)
            elif x.shape[-1] < w:
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
                # x = self.sa(x)
            elif x.shape[-1] == w:
                x = x
                # x = self.sa(x)

            ans = torch.cat((ans, x), dim=1)

        return ans

class AJ(nn.Module):
    def __init__(self, channel):
        super(AJ, self).__init__()

        self.ca = SELayer(channel)
        self.sa = SpatialAttention()
        self.conv1 = Conv2dLayer(channel, channel, kernel_size=3, stride=1, padding=1, activation='relu')
        self.conv2 = Conv2dLayer(channel, channel, kernel_size=3, stride=1, padding=1, activation='relu')
        self.conv3 = nn.Conv2d(channel*2, channel, kernel_size=1, stride=1)
        self.conv4 = Conv2dLayer(channel, channel, kernel_size=3, stride=1, padding=1, activation='relu')
        self.conv5 = Conv2dLayer(channel, channel, kernel_size=3, stride=1, padding=1, activation='relu')
        self.conv6 = Conv2dLayer(channel*2, channel, kernel_size=3, stride=1, padding=1, activation='relu')
    def forward(self, low, mid, high):
        b, c, h, w = low.size()
        x_mid = F.interpolate(mid, size=(h, w), mode='bilinear', align_corners=True)
        x = torch.cat((self.conv1(low), self.conv2(x_mid)), dim=1)
        x = self.conv3(x)
        x_high = F.interpolate(high, size=(h, w), mode='bilinear', align_corners=True)
        x_fu = torch.cat((x, x_high), dim=1)
        x = self.conv4(x)
        x1 = self.sa(x) + x
        x_high2 = self.conv5(high)
        x3 = self.ca(x_high2) + x_high2
        x2 = self.conv6(x_fu)
        x_high3 = F.interpolate(self.ca(x_high2), size=(h, w), mode='bilinear', align_corners=True)
        x2 = self.sa(x) * x2 * x_high3
        x3 = F.interpolate(x3, size=(h, w), mode='bilinear', align_corners=True)
        out = x1 + x2 + x3

        return out

class SCM(nn.Module):
    def __init__(self,channel):
        super(SCM,self).__init__()
        self.main = nn.Sequential(
            Conv2dLayer(channel,channel//4,kernel_size=3,stride=1,padding=1,activation='relu'),
            Conv2dLayer(channel//4, channel // 2, kernel_size=1, stride=1, padding=0, activation='relu'),
            Conv2dLayer(channel//2, channel // 2, kernel_size=3, stride=1, padding=1, activation='relu'),
            Conv2dLayer(channel//2, channel, kernel_size=1, stride=1, padding=0, activation='relu')
        )

        self.conv = Conv2dLayer(channel * 2,channel,kernel_size=1,stride=1,padding=0,activation='relu')

    def forward(self,x):
        x = torch.cat([x,self.main(x)],1)
        return self.conv(x)


class R2CAB(nn.Module):
    def __init__(self, channel, stride=1, scale=5, basewidth=256):
        super(R2CAB, self).__init__()
        width = int(math.floor(basewidth / scale))
        self.conv1 = nn.Conv2d(channel, width * scale, kernel_size=1, stride=stride, bias=True)     # conv 1*1
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 2    #3
        convs = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=True))
        self.convs = nn.ModuleList(convs)
        self.conv3 = nn.Conv2d(width * scale, channel, kernel_size=1, stride=stride, bias=True)      # conv 1*1

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.width = width

        self.conv_fft = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, kernel_size=1, stride=stride, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, kernel_size=1, stride=stride, bias=True)
        )


    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        # [N, width * scale, H, W]
        out = self.relu(self.conv1(x))   # conv 1*1

        # scale * [N, width , H, W]  ,分组
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)      # 三个conv3*3的合并
        out = torch.cat((out, spx[self.nums]), 1)      # 三个conv3*3和直接下来的合并
        sf = torch.fft.rfft2(spx[-1])   #傅里叶变换
        sf_im = sf.imag  #虚部
        sf_re = sf.real  #实部
        sf_f = torch.cat([sf_re, sf_im], 1)  #合并起来  把complex转为float类型
        sf = self.conv_fft(sf_f)  #卷积
        sf_re, sf_im = torch.chunk(sf, 2, 1)  #
        sf = torch.complex(sf_re, sf_im)
        sf = torch.fft.irfft2(sf,s=(H,W))
        out = torch.cat((out, sf), 1)   #三个conv3*3和直接下来的还有经过FFT的合并
        out = self.conv3(out)        # conv 1*1

        out += residual
        return out

# class R2CAB(nn.Modul):
#     def __init__(self, channel, stride=1, scale=4, basewidth=256):
#         super(R2CAB, self).__init__()
#         width = int(math.floor(basewidth / scale))
#         self.conv1 = nn.Conv2d(channel, width * scale, kernel_size=1, stride=stride, bias=True)     # conv 1*1
#         if scale == 1:
#             self.nums = 1
#         else:
#             self.nums = scale - 1    #3
#         convs = []
#         for i in range(self.nums):
#             convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=True))
#         self.convs = nn.ModuleList(convs)
#         self.conv3 = nn.Conv2d(width * scale, channel, kernel_size=1, stride=stride, bias=True)      # conv 1*1
#
#         self.relu = nn.ReLU(inplace=True)
#         self.scale = scale
#         self.width = width
#
#     def forward(self, x):
#         residual = x
#         # [N, width * scale, H, W]
#         out = self.relu(self.conv1(x))   # conv 1*1
#
#         # scale * [N, width , H, W]  ,分组
#         spx = torch.split(out, self.width, 1)
#         for i in range(self.nums):
#             if i == 0:
#                 sp = spx[i]
#             else:
#                 sp = sp + spx[i]
#             sp = self.convs[i](sp)
#             sp = self.relu(sp)
#             if i == 0:
#                 out = sp
#             else:
#                 out = torch.cat((out, sp), 1)      # 三个conv3*3的合并
#         out = torch.cat((out, spx[self.nums]), 1)      # 三个conv3*3和直接下来的合并
#         out = self.conv3(out)        # conv 1*1
#         out += residual
#         return out


# spatial-spectral domain attention learning(SDL)
class SPA_attention(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(SPA_attention, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.sigmoid = nn.Sigmoid()
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.con = nn.Conv2d(self.inplanes,self.inplanes,kernel_size=5,padding=2,stride=1,bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):


        spa_x = self.conv_v_left(x)
        spa_x = torch.mean(spa_x,dim=1)
        spa_x = torch.unsqueeze(spa_x,dim=1)
        mask_sp = self.sigmoid(spa_x)
        x = self.relu(self.con(x))

        out = x * mask_sp

        return out
