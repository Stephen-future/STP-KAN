import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
# from timm.models.layers import trunc_normal_
# from mish import Mish

# ----------------------------------------
#               Conv2d Block
# ----------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'reflect', activation = 'relu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        # elif activation == 'mish':      # 我加的
        #     self.activation = Mish()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    
    def forward(self, x):
        x = self.pad(x)          # 不支持对2D进行填充
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class CALayer(nn.Module):
    def __init__(self,channel,ratio= 32):
        super(CALayer,self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.gap = nn.AdaptiveAvgPool2d(1)          # sq 压缩，自适应平均池化
        self.fc = nn.Sequential(                    # ex 激励，2个fc
                nn.Linear(channel, channel //ratio, bias=False),
                nn.ReLU(inplace = True),
                nn.Linear(channel //ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b,c,h,w = x.size()
        y = self.gap(x).view(b,c)           # sq 压缩
        y = self.fc(y).view(b,c,1,1)        # ex 激励
        return x * y.expand_as(x)


class R2CAB(nn.Module):
    def __init__(self, channel, stride=1, scale=4, basewidth=256):
        super(R2CAB, self).__init__()
        width = int(math.floor(16 * (basewidth / 64.0)))     # 64,4个子集
        self.conv1 = nn.Conv2d(channel, width * scale, kernel_size=1, stride=stride, bias=True)     # conv 1*1
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        convs = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=True))
        self.convs = nn.ModuleList(convs)
        # nn.ModuleList 是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。
        self.conv3 = nn.Conv2d(width * scale, channel, kernel_size=1, stride=stride, bias=True)      # conv 1*1

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.width = width
        # SELayer
        self.ca = CALayer(channel)

    def forward(self, x):
        residual = x
        # [N, width * scale, H, W]
        out = self.relu(self.conv1(x))

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
                out = torch.cat((out, sp), 1)      # 合并

        out = torch.cat((out, spx[self.nums]), 1)      # 合并
        out = self.conv3(out)
        out = self.ca(out)          # 压缩和激励

        out += residual
        return out




