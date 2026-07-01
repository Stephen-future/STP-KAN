# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:23:35 2022

@author: Administrator
"""
import torch
import torch.nn as nn
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return  y.expand_as(x)

class DF_fusion_block(nn.Module):
    def __init__(self, base_chan):
        super().__init__()
        self.fusion1=nn.Conv2d(15*base_chan, 4*base_chan, kernel_size=1)
        self.fusion_att1 = nn.Sequential(nn.Conv2d(4*base_chan, 4*base_chan, kernel_size = 1),
                                 nn.BatchNorm2d(4*base_chan),
                                 nn.Sigmoid())
        self.fusion_out=nn.Conv2d(4*base_chan, 4*base_chan, kernel_size = 1)
        self.atrous1=nn.Conv2d(8*base_chan, base_chan, kernel_size=3,dilation=1, padding=1)    #############1、7、13、19是什么意思？
        self.atrous2=nn.Conv2d(4*base_chan, base_chan, kernel_size=3,dilation=7, padding=7)
        self.atrous3=nn.Conv2d(2*base_chan, base_chan, kernel_size=3,dilation=13,padding=13)
        self.atrous4=nn.Conv2d(  base_chan, base_chan, kernel_size=3,dilation=19,padding=19)
        self.se1=SE_Block(base_chan)
        self.se2=SE_Block(base_chan)
        self.se3=SE_Block(base_chan)
        self.se4=SE_Block(base_chan)
    def forward(self, x1,x2,x3,x4):   
        w_f1=self.atrous1(x1)
        w_f1=self.se1(w_f1)
#        f1=x1*w_f1
        
        w_f2=self.atrous2(x2)
        w_f2=self.se2(w_f2) 
#        f2=x2*w_f2
        
        w_f3=self.atrous3(x3)
        w_f3=self.se3(w_f3) 
#        f3=x3*w_f3
        
        w_f4=self.atrous4(x4)
        w_f4=self.se4(w_f4) 
#        f4=x4*w_f4
        DF_fusion_x=torch.cat([x1,x2,x3,x4], dim=1)  #15 base_chan
        DF_fusion_x_out=self.fusion1(DF_fusion_x) #4 base_chan
        DF_fusion_att=torch.cat([w_f1, w_f2, w_f3, w_f4], dim=1) #4 base_chan
        DF_fusion_att_out = self.fusion_att1(nn.ReLU()(DF_fusion_att))         #activate gate
        DF_fusion=DF_fusion_x_out*DF_fusion_att_out
        DF_fusion=self.fusion_out(DF_fusion)
        return DF_fusion
    
class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei



#import torch
#import torch.nn as nn
#class SE_Block(nn.Module):
#    def __init__(self, ch_in, reduction=4):
#        super(SE_Block, self).__init__()
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
#        self.fc = nn.Sequential(
#            nn.Linear(ch_in, ch_in // reduction, bias=False),
#            nn.ReLU(inplace=True),
#            nn.Linear(ch_in // reduction, ch_in, bias=False),
#            nn.Sigmoid()
#        )
#
#    def forward(self, x):
#        b, c, _, _ = x.size()
#        y = self.avg_pool(x).view(b, c)
#        y = self.fc(y).view(b, c, 1, 1)
#        return  y.expand_as(x)
#
#class DF_fusion_block(nn.Module):
#    def __init__(self, base_chan):
#        super().__init__()
#        self.atrous1=nn.Conv2d(8*base_chan, 8*base_chan, kernel_size=3,dilation=1, padding=1)
#        self.atrous2=nn.Conv2d(4*base_chan, 4*base_chan, kernel_size=3,dilation=7, padding=7)
#        self.atrous3=nn.Conv2d(2*base_chan, 2*base_chan, kernel_size=3,dilation=13,padding=13)
#        self.atrous4=nn.Conv2d(  base_chan,   base_chan, kernel_size=3,dilation=19,padding=19)
#        self.se1=SE_Block(8*base_chan)
#        self.se2=SE_Block(4*base_chan)
#        self.se3=SE_Block(2*base_chan)
#        self.se4=SE_Block(base_chan)
#    def forward(self, x1,x2,x3,x4):   
#        w_f1=self.atrous1(x1)
#        w_f1=self.se1(w_f1)
#        f1=x1*w_f1
#        
#        w_f2=self.atrous2(x2)
##        print(w_f2.shape)
#        w_f2=self.se2(w_f2) 
##        print(w_f2.shape)
#        f2=x2*w_f2
#        
#        w_f3=self.atrous3(x3)
#        w_f3=self.se3(w_f3) 
#        f3=x3*w_f3
#        
#        w_f4=self.atrous4(x4)
#        w_f4=self.se4(w_f4) 
#        f4=x4*w_f4
#        
#        DF_fusion=torch.cat([f1,f2,f3,f4], dim=1)
#        return DF_fusion