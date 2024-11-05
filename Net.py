import torch
from torch import nn
from einops import rearrange
import numbers
import math
import torch.nn.functional as F

class Bn_Cov2d_Relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bn_Cov2d_Relu, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.BN = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.seq(x)
        x = self.BN(x)
        x = self.ReLU(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.Bn_Cov2d_Relu1 = Bn_Cov2d_Relu(2, out_channels=62)
        self.Bn_Cov2d_Relu2 = Bn_Cov2d_Relu(64, out_channels=128)
        self.Bn_Cov2d_Relu3 = Bn_Cov2d_Relu(192, out_channels=128)
        self.Bn_Cov2d_Relu4 = Bn_Cov2d_Relu(320, out_channels=128)
    def forward(self, x1,x2):
        x1 = x1
        x2 = x2
        x = torch.cat([x1, x2], dim=1)
        x1 = x
        x2 = self.Bn_Cov2d_Relu1(x)
        y1 = torch.cat([x1, x2], dim=1)
        x3 = self.Bn_Cov2d_Relu2(y1)
        y2 = torch.cat([x1,x2,x3], dim=1)
        x4 = self.Bn_Cov2d_Relu3(y2)
        y3 = torch.cat([x1,x2,x3,x4], dim=1)
        out = self.Bn_Cov2d_Relu4(y3)
        return out

class OutlookAttention(nn.Module):
    """
    Implementation of outlook attention
    --dim: hidden dim
    --num_heads: number of heads
    --kernel_size: kernel size in each window for outlook attention
    return: token features after outlook attention
    """

    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim**-0.5

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size**4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)
        return x

class BaseFeature(nn.Module):
    def __init__(self):
        super(BaseFeature, self).__init__()
        kernel_size = 7
        self.Conv1 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.Bn_Cov2d_Relu1 = Bn_Cov2d_Relu(64, 64)
        self.sigmoid = nn.Sigmoid()
        self.Outlook_Attention = OutlookAttention(dim=64,num_heads=8)
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Bn_Cov2d_Relu1(x)
        x = x.view(x.size(0), x.size(3), x.size(2),x.size(1))
        x = self.Outlook_Attention(x)
        x = x.view(x.size(0), x.size(3), x.size(2), x.size(1))
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)
class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2
class DetailFeature(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeature, self).__init__()
        self.Conv1 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.Bn_Cov2d_Relu1 = Bn_Cov2d_Relu(64, 64)
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Bn_Cov2d_Relu1(x)
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

# Normalisation Mean = 0 Variances = 1
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    # 64 2 False
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        #  64 256 1 False
        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 256 256 3 1 1 256 False
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        # 128 64 1 False
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# Multi_Attention(Q,K,V) 。
class Attention(nn.Module):
    # 64 8 Flase
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        # 8行一列 全1 温度参数来调节注意力的尖峰度
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 64 64*3 1 Flase
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 64*3 64*3 3 1 1 64*3 False
        # groups=64*3 每个输入通道连接对应的输出通道不会连接到其他的输出通道
        # 用来捕获特征的局部相关性
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 64 64 1 False
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # (8 64 128 128)
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        # 64 64 64
        q, k, v = qkv.chunk(3, dim=1)
        # (8 (8 192) 128 128)->(8 8 192 (128 128))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
class TransformerBlock(nn.Module):
    # 64 8 2 False WithBias
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        # 每个样本的特征进行均值为 0、方差为 1 的标准化处理
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # 64 8 False 输出是经过自注意力加权后的特征张量，具有与输入相同的维度和形状。64 8 False
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # 64 2 False 前馈神经网络其作用是对输入特征进行非线性变换
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # Attention(Q,K,V) 输出是经过自注意力加权后的特征张量，具有与输入相同的维度和形状。
        x = x + self.attn(self.norm1(x))
        #  前馈神经网络 其作用是对输入特征进行非线性变换
        x = x + self.ffn(self.norm2(x))
        return x

class Restormer_Decoder(nn.Module):
    def __init__(self,
                 # 输入通道数
                 inp_channels=1,
                 out_channels=1,
                 dim=128,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.reduce_channel1 = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid = nn.Sigmoid()
        self.r1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.r2 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
    def forward(self, inp_img, share_feature, ir_global, ir_local, vi_global, vi_local):
        if (ir_local is None) and (ir_global is None):
            all_feature = torch.cat((share_feature,vi_global,vi_local), dim=1)
            all_feature = self.r1(all_feature)
            all_feature = self.encoder_level2(all_feature)
            out = self.output(all_feature)+inp_img
        elif (vi_global is None) and (vi_local is None):
            all_feature = torch.cat((share_feature, ir_global, ir_local), dim=1)
            all_feature = self.r2(all_feature)
            all_feature = self.encoder_level2(all_feature)
            out = self.output(all_feature)+inp_img
        else:
            all_feature = torch.cat((share_feature, ir_global,vi_global, ir_local, vi_local), dim=1)
            all_feature = self.reduce_channel1(all_feature)
            all_feature = self.encoder_level2(all_feature)
            out = self.output(all_feature)+inp_img
        return self.sigmoid(out)

if __name__ == '__main__':
    # input = torch.randn(1, 1, 128, 128)
    # input2 = torch.randn(1, 1, 128, 128)
    # encoder1 = DenseBlock()
    # output = encoder1(input, input2)
    # print(output.size())
    # input3 = torch.randn(1, 1, 128, 128)
    # encoder2 = BaseFeature()
    # output = encoder2(input3)
    # print(output.size())
    input4 = torch.randn(1, 1, 128, 128)
    encoder3 = DetailFeature()
    output = encoder3(input4)
    print(output.size())
    # input5 = torch.randn(1, 64, 128, 128)
    # input4 = torch.randn(1, 64, 128, 128)
    # input3 = torch.randn(1, 64, 128, 128)
    # input2 = torch.randn(1, 64, 128, 128)
    # input1 = torch.randn(1, 128, 128, 128)
    # decoder = Restormer_Decoder()
    # output = decoder(input1, input2, input3, input4, input5)
    # print(output.shape)


