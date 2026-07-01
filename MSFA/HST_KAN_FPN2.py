import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
from net_res5 import R2CAB,Attention,MDI
from fastkanconv import FastKANConvLayer
from katransformer import KAN
from fusion import AFF, iAFF
import matplotlib.pyplot as plt


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def _no_grad_trunc_normal_(tensor, mean, std, a, b):            # [1,1,64,64]，μ（均值），σ（标准差）
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.          # x在标准正态分布下的累计概率

    if (mean < a - 2 * std) or (mean > b + 2 * std):            # mean超出范围[a-2*std,b+2*std]警告
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)                          # 计算上下界a、b在标准正态分布中对应的的累计概率
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)                   # 在区间(2*l-1, 2*u-1)上对张量进行均匀分布初始化
        tensor.erfinv_()                                        # 对张量中的每个元素应用反误差函数，将其从[0,1]的范围映射回标准正态分布的范围
        tensor.mul_(std * math.sqrt(2.))                        # 将标准正态分布转换回原始的正态分布
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)                             # 保证所有值在[a,b]范围内
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):        # 截断正态分布的初始化
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        自定义 LayerNorm 层
        Args:
            normalized_shape: 输入张量的最后一个维度大小
            eps: 防止除零的小值
            elementwise_affine: 是否使用可学习的仿射变换参数
        """
        super(CustomLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 [*, normalized_shape]
        Returns:
            正则化后的张量
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized
class PreNorm(nn.Module):                                       # LN层
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = CustomLayerNorm(dim)                           # 对最后一维（dim个通道）进行归一化

    def forward(self, x, *args, **kwargs):                      # 接受任意数量的参数(元组，字典)
        x = self.norm(x)                                        # 先归一化
        return self.fn(x, *args, **kwargs)                      # 然后输入网络


class GELU(nn.Module):                                          # 高斯误差激活函数
    def forward(self, x):
        return F.gelu(x)


class HS_MSA(nn.Module):
    def __init__(
            self,
            dim,                                                # 32,64;128
            window_size=(8, 8),
            dim_head=28,                                        # 32,64;128
            heads=8,                                            # 1,2;4
            only_local_branch=False                             # True,False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.only_local_branch = only_local_branch

        # position embedding
        if only_local_branch:
            seq_l = window_size[0] * window_size[1]                                    # 计算窗口中的元素总数
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))          # 创建一个可训练的4维张量[1,1,64,64]
            trunc_normal_(self.pos_emb)                                                # 进行截断正态分布的初始化
        else:
            seq_l1 = window_size[0] * window_size[1]
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l1, seq_l1)) # 嵌入位置信息的可学习的参数[1,1,1,64,64]
            h, w = 240//self.heads, 240//self.heads                                      # h=120,w=120
            seq_l2 = h*w//seq_l1                                                       # 225
            self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l2, seq_l2)) # 嵌入位置信息的可学习的参数[1,1,1,225,225]
            trunc_normal_(self.pos_emb1)
            trunc_normal_(self.pos_emb2)

        inner_dim = dim_head * heads                                                   # 32，64
        self.to_q = nn.Linear(dim, inner_dim, bias=False)                              # 定义线性变换层，将输入转换为Q
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)                         # 定义线性变换层，将输入转换为K、V
        self.to_out = nn.Linear(inner_dim, dim)                                        # 线性映射输出为原始维度

    def forward(self, x, h_2, w_2, context=None):
        """
        x: [b,h,w,c]                                                                    #[b,32,240,240],[b,64,120,120]
        return out: [b,h,w,c]
        """
        context = default(context, x)

        b, h, w, c = x.shape
        w_size = self.window_size

        seq_l1 = w_size[0] * w_size[1]
        h2, w2 = h_2 // self.heads, w_2 // self.heads  # h=120,w=120
        seq_l2 = h2 * w2 // seq_l1  # 225
        pos_emb2 = nn.Parameter(torch.Tensor(1, 1, self.heads // 2, seq_l2, seq_l2).to('cuda'))  # 嵌入位置信息的可学习的参数[1,1,1,225,225]
        trunc_normal_(pos_emb2)

        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size' # 检查输入的高度和宽度是否可以被窗口大小整除
        if self.only_local_branch:
            x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1]) # 变成3维[1280*b,64,28]
            q = self.to_q(x_inp)                                                        # 将输入转换为Q
            k, v = self.to_kv(x_inp).chunk(2, dim=-1)                                   # 将输入转换为KV,并把通道维度均分为两部分，赋值给K、V
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v)) # 将q、k、v转换为[b*1280,1,64,28]形式
            q *= self.scale
            sim = einsum('b h i d, b h j d -> b h i j', q, k)                           # 在q、k的最后两个维度做矩阵乘法（k取转置）[b*1280,1,64,64]，在空间维度计算窗口内的注意力得分
            sim = sim + self.pos_emb
            attn = sim.softmax(dim=-1)                                                  # 将最后一个维度（注意力得分）的值归一化且和为1
            out = einsum('b h i j, b h j d -> b h i d', attn, v)                        # 在最后两个维度做矩阵乘法，根据注意力得分来加权求和（V）值[b*1280,1,64,28]
            out = rearrange(out, 'b h n d -> b n (h d)')                                # 把h和d维度合并[b*1280,64,28]
            out = self.to_out(out)                                                      # 转换成初始通道数[b*1280,64,28]
            out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])                                               # 恢复原始维度[b,256,320,28]
        else:
            q = self.to_q(x)
            k, v = self.to_kv(context).chunk(2, dim=-1)
            q1, q2 = q[:,:,:,:c//2], q[:,:,:,c//2:]                                     # 把q、k、v按通道维度均分成两个
            k1, k2 = k[:,:,:,:c//2], k[:,:,:,c//2:]
            v1, v2 = v[:,:,:,:c//2], v[:,:,:,c//2:]

            # local branch
            q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                              b0=w_size[0], b1=w_size[1]), (q1, k1, v1))    # 将q2、k2、v2转换为[b,h*w/(b0*b1),b0*b1,c]形式[b,320,64,28]
            q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q1, k1, v1)) # 把h、d分开变成5维,[b,320,1,64,28]
            q1 *= self.scale
            sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)                  # 在空间维度计算窗口内的注意力得分
            sim1 = sim1 + self.pos_emb1
            attn1 = sim1.softmax(dim=-1)
            out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

            # non-local branch
            q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q2, k2, v2)) # 将q2、k2、v2转换为[b,h*w/(b0*b1),b0*b1,c]形式[b,320,64,28]
            q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone())) # 改变维度顺序,[b,64,320,28]
            q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q2, k2, v2)) # 将维度c拆分为h、d,[b,64,1,320,28]
            q2 *= self.scale
            sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)                  # 在q、k的最后两个维度做矩阵乘法（k取转置），计算每个窗口的注意力得分,[b,64,1,320,320]
            sim2 = sim2 + pos_emb2
            attn2 = sim2.softmax(dim=-1)                                                # 将最后一个维度（注意力得分）的值归一化且和为1
            out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)               # 在最后两个维度做矩阵乘法，根据注意力得分来加权求和（V2）值,[b,64,1,320,28]
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')                        # 把h和d维度合并,[b,64,320,28]
            out2 = out2.permute(0, 2, 1, 3)                                             # 把维度顺序改回初始顺序,[b,320,64,28]

            out = torch.cat([out1,out2],dim=-1).contiguous()                            # 将局部和非局部分支的输出合并，并确保新的张量在内存中是连续的,[b,320,64,56]
            out = self.to_out(out)
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])                                               # 恢复原始维度,[b,128,160,56]
        return out

class HSAB(nn.Module):
    def __init__(
            self,
            dim,                                                                        # 32,64;128
            window_size=(12, 12),
            dim_head=64,                                                                # 32,64;128
            heads=8,                                                                    # 1,2;4
            num_blocks=2,                                                               # 1
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, HS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, only_local_branch=(heads==1))),
                PreNorm(dim, KAN(
            in_features=dim,
            hidden_features=int(dim * 4.0),
            act_layer=nn.GELU,
            drop=float(0.),
            act_init='gelu',
        ))                                      # 先归一化，再输入FFN
            ]))

    def forward(self, x, h2, w2):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b0, c0, h0, w0 = x.shape
        x = x.permute(0, 2, 3, 1)                                                       # 改变维度顺序，[b,h,w,c]（方便后面归一化）
        for (attn, kan) in self.blocks:
            x = attn(x, h_2=h2, w_2=w2) + x                                                             # 网络输出和原输出相加
            x1 = rearrange(x, 'b h w c -> b (h w) c')
            x2 = kan(x1)
            x2 = rearrange(x2, 'b (h w) c -> b h w c', h=h0, w=w0)
            x = x2 + x
        out = x.permute(0, 3, 1, 2)                                                     # 改回原顺序，[b,c,h,w]
        return out

class HSAB_cross(nn.Module):
    def __init__(
            self,
            dim,                                                                        # 128
            window_size=(12, 12),
            dim_head=64,                                                                # 128
            heads=8,                                                                    # 4
            num_blocks=2,                                                               # 1
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, HS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, only_local_branch=(heads==1))),
                PreNorm(dim, KAN(
                    in_features=dim,
                    hidden_features=int(dim * 4.0),
                    act_layer=nn.GELU,
                    drop=float(0.),
                    act_init='gelu',
                ))  # 先归一化，再输入FFN
            ]))

    def forward(self, x, y, h2, w2):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b0, c0, h0, w0 = x.shape
        x = x.permute(0, 2, 3, 1)                                                       # 改变维度顺序，[b,h,w,c]（方便后面归一化）
        y = y.permute(0, 2, 3, 1)
        for (attn, kan) in self.blocks:
            x = attn(x, context=y, h_2=h2, w_2=w2) + x                                                             # 网络输出和原输出相加
            x1 = rearrange(x, 'b h w c -> b (h w) c')
            x2 = kan(x1)
            x2 = rearrange(x2, 'b (h w) c -> b h w c', h=h0, w=w0)
            x = x2 + x
        for (attn, kan) in self.blocks:
            y = attn(y, context=x, h_2=h2, w_2=w2) + y                                                             # 网络输出和原输出相加
            y1 = rearrange(y, 'b h w c -> b (h w) c')
            y2 = kan(y1)
            y2 = rearrange(y2, 'b (h w) c -> b h w c', h=h0, w=w0)
            y = y2 + y
        out = x.permute(0, 3, 1, 2)                                                     # 改回原顺序，[b,c,h,w]
        out_2 = y.permute(0, 3, 1, 2)
        return out, out_2

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),                               # 1×1的卷积层（把通道数扩大mult倍）
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),  # 3×3的卷积层
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),                               # 1×1的卷积层（把通道缩小mult倍）
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))                                           # 改变维度顺序输入网络,[b,c,h,w](便于卷积操作)
        return out.permute(0, 2, 3, 1)                                                  # 改回原顺序,[b,h,w,c]


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.double_conv = nn.Sequential(
            FastKANConvLayer(self.in_channels, self.out_channels // 2, padding=1, kernel_size=3, stride=1,
                             kan_type='RBF'),
            nn.BatchNorm2d(self.out_channels // 2),
            nn.ReLU(inplace=True),
            FastKANConvLayer(self.out_channels // 2, self.out_channels, padding=1, kernel_size=3, stride=1,
                             kan_type='RBF'),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x1 = self.conv2(x1)
        x1 = x1 + x
        x2 = self.relu(self.conv3(x1))
        x2 = self.conv4(x2)
        x2 = x2 + x1
        return x2

class DecoderBranch(nn.Module):
    def __init__(self, output_channels):
        super(DecoderBranch, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, output_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x1 = self.conv2(x1)
        x1 = x1 + x
        x2 = self.relu(self.conv5(x1))
        x2 = self.conv3(x2)
        x2 = x2 + self.conv4(x1)
        return x2

class HST(nn.Module):
    def __init__(self, opt, dim=64, num_blocks=[1, 1, 1]):
        super(HST, self).__init__()
        self.dim = dim
        self.scales = len(num_blocks)                                                  # 3
        # Input projection
        self.embedding_9 = nn.Conv2d(9, self.dim, 3, 1, 1, bias=False)              # 3×3的卷积层（把通道数从变为32）
        self.embedding_16 = nn.Conv2d(16, self.dim, 3, 1, 1, bias=False)

        # Fusion
        self.activ = nn.GELU()

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        self.encoder_layers2 = nn.ModuleList([])
        self.encoder_layers3 = nn.ModuleList([])
        self.encoder_layers4 = nn.ModuleList([])
        dim_scale = dim                                                                # 64
        for i in range(self.scales-1):
            self.encoder_layers.append(nn.ModuleList([
                HSAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim), # (64,1,64,1),(128,1,64,2)
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),              # 4×4的下采样层，步长为2，通道数翻倍，输出[b,128,120,120](第一次)、[b,256,60,60](第二次)
            ]))
            dim_scale *= 2                                                             # 第二次循环值翻倍,跳出循环时翻了4倍

        dim_scale = dim  # 64
        for i in range(self.scales - 1):
            self.encoder_layers2.append(nn.ModuleList([
                HSAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                # (64,1,64,1),(128,1,64,2)
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
                # 4×4的下采样层，步长为2，通道数翻倍，输出[b,128,120,120](第一次)、[b,256,60,60](第二次)
            ]))
            dim_scale *= 2

        dim_scale = dim  # 64
        for i in range(self.scales - 1):
            self.encoder_layers3.append(nn.ModuleList([
                HSAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                # (64,1,64,1),(128,1,64,2)
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
                # 4×4的下采样层，步长为2，通道数翻倍，输出[b,128,120,120](第一次)、[b,256,60,60](第二次)
            ]))
            dim_scale *= 2

        dim_scale = dim  # 64
        for i in range(self.scales - 1):
            self.encoder_layers4.append(nn.ModuleList([
                HSAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                # (64,1,64,1),(128,1,64,2)
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
                # 4×4的下采样层，步长为2，通道数翻倍，输出[b,128,120,120](第一次)、[b,256,60,60](第二次)
            ]))
            dim_scale *= 2

        # Bottleneck
        self.bottleneck = HSAB_cross(dim=dim_scale, dim_head=dim, heads=dim_scale // dim, num_blocks=num_blocks[-1]) # (256,64,4,1)
        # self.bottleneck2 = HSAB(dim=dim_scale, dim_head=dim, heads=dim_scale // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_scale*2, dim_scale, stride=2, kernel_size=2, padding=0, output_padding=0), # 2×2的上采样层（反卷积层），步长为2（通道数减半），输出[b,256,120,120],[b,128,240,240]
                nn.Conv2d(dim_scale*2, dim_scale, 1, 1, bias=False),  # 1×1的卷积层（通道数减半）,输出[b,128,120,120],[b,64,240,240]
                iAFF(channels=dim_scale),
                HSAB(dim=dim_scale, num_blocks=num_blocks[self.scales - 2 - i], dim_head=dim*2,
                     heads=(dim_scale) // (dim*2)),                     # (256,1,128,2),(128,1,128,1)
            ]))
            dim_scale //= 2                                              # 第二次循环值减半,跳出循环时缩小了4倍

        # Output projection
        # self.mapping_9 = nn.Conv2d(128, 9, kernel_size=3, padding=1)
        # self.mapping_16 = nn.Conv2d(128, 16, kernel_size=3, padding=1)

        self.conv_x1 = nn.Conv2d(768, 64, kernel_size=3, padding=1)
        self.conv_x2 = nn.Conv2d(384, 64, kernel_size=3, padding=1)
        self.conv_y1 = nn.Conv2d(768, 64, kernel_size=3, padding=1)
        self.conv_y2 = nn.Conv2d(384, 64, kernel_size=3, padding=1)

        self.scm1 = Attention(opt.start_channels * (2 ** 2))
        self.scm2 = Attention(opt.start_channels * (2 ** 3))
        self.scm3 = Attention(opt.start_channels * (2 ** 4))

        # self.scm4 = Attention(opt.start_channels * (2 ** 3))
        # self.scm5 = Attention(opt.start_channels * (2 ** 4))
        # self.scm6 = Attention(opt.start_channels * (2 ** 3))
        # self.scm7 = Attention(opt.start_channels * (2 ** 4))

        self.mdi2 = MDI(opt.start_channels * (2 ** 1))
        self.mdi3 = MDI(opt.start_channels * (2 ** 1))
        self.conv1 = nn.Conv2d(dim_scale * 2, self.dim, 3, 1, 1, bias=False)
        # self.resblock = ResBlock()
        # self.resblock2 = ResBlock()
        self.resblock3 = ResBlock()
        self.resblock4 = ResBlock()
        self.resblock5 = ResBlock()
        self.resblock6 = ResBlock()
        self.resblock7 = ResBlock()
        self.resblock8 = ResBlock()
        self.resblock9 = ResBlock()
        self.resblock10 = ResBlock()
        self.resblock11 = ResBlock()
        self.resblock12 = ResBlock()
        self.conv_out9 = DecoderBranch(output_channels=9)
        self.conv_out16 = DecoderBranch(output_channels=16)

        #### activation function
        self.apply(self._init_weights)                                   # 初始化模型的权重（将权重初始化递归地运用在所有子模块上）

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):                                     # 对线性层权重进行截断正态分布初始化，偏置初始化为0
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, CustomLayerNorm):                                # 对LN层偏置初始化为0，权重初始化为1
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, channel, y, channel2):
        """
        x: [b,c,h,w]                                                        # [b,29,256,310]
        return out:[b,c,h,w]
        """
        # x_t = x[3, 1, :, :].cpu().numpy()
        # y_t = y[3, 1, :, :].cpu().numpy()
        # plt.figure(figsize=(12,6))
        # plt.subplot(1,2,1)
        # plt.imshow(x_t,cmap='gray')
        # plt.colorbar()
        # plt.subplot(1, 2, 2)
        # plt.imshow(y_t, cmap='gray')
        # plt.colorbar()
        # plt.show()
        b, c, h_inp, w_inp = x.shape
        hb, wb = 48, 48
        pad_h = (hb - h_inp % hb) % hb                                      # 0
        pad_w = (wb - w_inp % wb) % wb                                      # 10
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')                  # 对x的最后两维度（h,w）扩充（右侧pad_w和下侧pad_h）,保证输入数据能被（16，16）窗口整除
        y = F.pad(y, [0, pad_w, 0, pad_h], mode='reflect')

        b2, c2, h2, w2 = x.shape
        # Embedding
        fea = self.embedding_16(x)                                             # 3×3的卷积层（把通道数变为64）,x0
        fea_2 = self.embedding_16(y)
        x = x[:, :64, :, :]                                                    # [b,32,240,240]
        y = y[:, :64, :, :]

        # Encoder
        fea_encoder = []
        fea_encoder_2 = []

        for (HSAB, FeaDownSample) in self.encoder_layers:
            fea = HSAB(fea, h2, w2)  # [b,64,256,320],[b,128,128,160]
            fea_encoder.append(fea)  # 将两次HSAB的输出放入fea_encoder，方便后面的concat
            fea = FeaDownSample(fea)  # 输出[b,128,128,160](第一次)、[b,256,64,80](第二次)

        for (HSAB, FeaDownSample) in self.encoder_layers2:
            fea_2 = HSAB(fea_2, h2, w2)
            fea_encoder_2.append(fea_2)
            fea_2 = FeaDownSample(fea_2)

        # Bottleneck
        fea, fea_2 = self.bottleneck(fea, fea_2, h2, w2)
        # fea = self.bottleneck(fea, h2, w2)
        # fea_2 = self.bottleneck2(fea_2, h2, w2)
        fea_total = torch.cat([fea, fea_2], dim=1)
        fea_total = self.scm3(fea_total) + fea_total       # 512
        fea_totals = fea_total
        # Decoder
        n = 0
        FeaUpSample = self.decoder_layers[n][0]
        Fution = self.decoder_layers[n][1]
        fuse2 = self.decoder_layers[n][2]
        HSAB = self.decoder_layers[n][3]
        fea_total = FeaUpSample(fea_total)  # 512->256
        fea_decoder = torch.cat([fea_encoder[self.scales - 2 - n], fea_encoder_2[self.scales - 2 - n]], dim=1)
        fea_total = fuse2(fea_total, self.scm2(fea_decoder))  # 256
        fea_total = HSAB(fea_total, h2, w2)
        fea_totals2 = fea_total

        n = 1
        FeaUpSample = self.decoder_layers[n][0]
        Fution = self.decoder_layers[n][1]
        fuse3 = self.decoder_layers[n][2]
        HSAB = self.decoder_layers[n][3]
        fea_total = FeaUpSample(fea_total)  # 256->128
        fea_decoder = torch.cat([fea_encoder[self.scales - 2 - n], fea_encoder_2[self.scales - 2 - n]], dim=1)
        fea_total = fuse3(fea_total, self.scm1(fea_decoder))  # 128
        fea_total = HSAB(fea_total, h2, w2)
        fea_total = self.conv1(fea_total)                     # 64
        # fea_total = self.resblock2(self.resblock(fea_total))
        # fea_total = self.resblock4(self.resblock3(fea_total_x)) + fea_total
        # Mapping

        fea_total_1 = self.resblock6(self.resblock5(fea_total))
        fea_total1 = self.resblock8(self.resblock7(fea_total_1)) + fea_total
        fea_total_2 = self.resblock10(self.resblock9(fea_total))
        fea_total2 = self.resblock12(self.resblock11(fea_total_2)) + fea_total

        fea_1 = fea_total1
        fea_2 = fea_total2
        fea_encoder_3 = []
        fea_encoder_4 = []

        for (HSAB, FeaDownSample) in self.encoder_layers3:
            fea_1 = HSAB(fea_1, h2, w2)  # [b,64,256,320],[b,128,128,160]
            fea_1 = FeaDownSample(fea_1)  # 输出[b,128,128,160](第一次)、[b,256,64,80](第二次)
            fea_encoder_3.append(fea_1)

        for (HSAB, FeaDownSample) in self.encoder_layers4:
            fea_2 = HSAB(fea_2, h2, w2)
            fea_2 = FeaDownSample(fea_2)
            fea_encoder_4.append(fea_2)

        fea_decoder_x1 = torch.cat([fea_totals, fea_encoder_3[1]], dim=1)  # 512+256->768
        fea_decoder_x2 = torch.cat([fea_totals2, fea_encoder_3[0]], dim=1)  # 256+128->384
        fea_decoder_y1 = torch.cat([fea_totals, fea_encoder_4[1]], dim=1)
        fea_decoder_y2 = torch.cat([fea_totals2, fea_encoder_4[0]], dim=1)

        fea_total_x1 = self.conv_x1(fea_decoder_x1)
        fea_total_x2 = self.conv_x2(fea_decoder_x2)
        fea_total_y1 = self.conv_y1(fea_decoder_y1)
        fea_total_y2 = self.conv_y2(fea_decoder_y2)

        fea_total1 = self.mdi2([fea_total_x1, fea_total_x2, fea_total1], fea_total1)
        fea_total2 = self.mdi3([fea_total_y1, fea_total_y2, fea_total2], fea_total2)

        out = self.conv_out16(fea_total1) + x                                   # 将输出和输入x相加[b,28,256,320]
        out_2 = self.conv_out16(fea_total2) + y

        return out[:, :, :h_inp, :w_inp], out_2[:, :, :h_inp, :w_inp]