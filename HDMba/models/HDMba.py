import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import einops
from einops import rearrange
# import tqdm
# 系统相关的库
import math
import os
import urllib.request
from zipfile import ZipFile
# from transformers import *
from timm.models.layers import DropPath, to_2tuple

torch.autograd.set_detect_anomaly(True)

# 配置标识和超参数
USE_MAMBA = 1
DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM = 0
# 设定所用设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 人为定义的超参数
batch_size = 4    # 批次大小
last_batch_size = 81  # 最后一个批次大小
current_batch_size = batch_size
different_batch_size = False
h_new = None
temp_buffer = None


# 定义S6模块
class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(S6, self).__init__()
        # 一系列线性变换
        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)
        # 设定一些超参数
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size
        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        # 参数初始化
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)

        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)

        # 定义内部参数h和y
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)

    # 离散化函数
    def discretization(self):
        # 离散化函数定义介绍在Mamba论文中的28页
        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)
        # dA = torch.matrix_exp(A * delta)  # matrix_exp() only supports square matrix
        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))
        return self.dA, self.dB

    # 前行传播
    def forward(self, x):
        # 参考Mamba论文中算法2
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))
        # 离散化
        self.discretization()

        if DIFFERENT_H_STATES_RECURRENT_UPDATE_MECHANISM:
            # 如果不使用'h_new'，将触发本地允许错误
            global current_batch_size
            current_batch_size = x.shape[0]

            if self.h.shape[0] != current_batch_size:
                different_batch_size = True
                # 缩放h的维度匹配当前的批次
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h[:current_batch_size, ...]) + rearrange(x, "b l d -> b l d 1") * self.dB
            else:
                different_batch_size = False
                h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB

            # 改变y的维度
            self.y = torch.einsum('bln,bldn->bld', self.C, h_new)

            # 基于h_new更新h的信息
            global temp_buffer
            temp_buffer = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()

            return self.y
        else:
            # 将会触发错误
            # 设置h的维度
            h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
            y = torch.zeros_like(x)

            h = torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

            #设置y的维度
            y = torch.einsum('bln,bldn->bld', self.C, h)
            return y


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: str = 'cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


# 定义MambaBlock模块
class MambaBlock(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(MambaBlock, self).__init__()
        self.inp_proj = nn.Linear(d_model, 2 * d_model, device=device)
        self.out_proj = nn.Linear(2 * d_model, d_model, device=device)
        # 残差连接
        self.D = nn.Linear(d_model, 2 * d_model, device=device)
        # 设置偏差属性
        self.out_proj.bias._no_weight_decay = True
        # 初始化偏差
        nn.init.constant_(self.out_proj.bias, 1.0)
        # 初始化S6模块
        self.S6 = S6(seq_len, 2 * d_model, state_size, device)
        # 添加1D卷积
        self.conv = nn.Conv1d(seq_len, seq_len, kernel_size=3, padding=1, groups=seq_len, device=device)
        # 添加线性层
        self.conv_linear = nn.Linear(2 * d_model, 2 * d_model, device=device)
        # 正则化
        self.norm = RMSNorm(d_model, device=device)

    def forward(self, x):
        # 参考Mamba论文中的图3
        x = self.norm(x)
        x_proj = self.inp_proj(x)
        # 1D卷积操作
        x_conv = self.conv(x_proj)
        x_conv_act = F.silu(x_conv)  # Swish激活
        # 线性操作
        x_conv_out = self.conv_linear(x_conv_act)
        # S6模块操作
        x_ssm = self.S6(x_conv_out)
        x_act = F.silu(x_ssm)  # Swish激活
        # 残差连接
        x_residual = F.silu(self.D(x))
        x_combined = x_act * x_residual
        x_out = self.out_proj(x_combined)
        return x_out


# 输入：序列长度 模型维度 状态大小
class Mamba(nn.Module):
    def __init__(self, seq_len, d_model, state_size, device):
        super(Mamba, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size
        self.mamba_block1 = MambaBlock(self.seq_len, self.d_model, self.state_size, device)
        self.mamba_block2 = MambaBlock(self.seq_len, self.d_model, self.state_size, device)
        self.mamba_block3 = MambaBlock(self.seq_len, self.d_model, self.state_size, device)

    def forward(self, x):
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SSMamba(nn.Module):
    def __init__(self, dim, window_size, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.mamba = Mamba(seq_len=self.window_size ** 2, d_model=dim, state_size=dim, device=device)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # Change shape to (B, L, C)
        x = self.norm1(x)

        x = x.view(B, H, W, C)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Apply Mamba
        attn_windows = self.mamba(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.norm2(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # Change shape back to (B, C, H, W)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)).view(B, H, W, C).permute(0, 3, 1, 2).contiguous())

        return x


class SSMaBlock(nn.Module):
    def __init__(self,
                 dim=32,
                 window_size=7,
                 depth=4,
                 mlp_ratio=2,
                 drop_path=0.0):
        super(SSMaBlock, self).__init__()
        self.ssmablock = nn.Sequential(*[
            SSMamba(dim=dim, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        out = self.ssmablock(x)
        out = self.conv(out) + x
        return out


class HDMba(nn.Module):
    def __init__(self,
                 inp_channels=305,
                 dim=32,
                 window_size=7,
                 depths=[4, 4, 4, 4],
                 mlp_ratio=2,
                 bias=False,
                 drop_path=0.0
                 ):
        super(HDMba, self).__init__()

        self.conv_first = nn.Conv2d(inp_channels, dim, 3, 1, 1)  # shallow feature extraction
        self.num_layers = depths
        self.layers = nn.ModuleList()

        for i_layer in range(len(self.num_layers)):
            layer = SSMaBlock(dim=dim,
                              window_size=window_size,
                              depth=depths[i_layer],
                              mlp_ratio=mlp_ratio,
                              drop_path=drop_path)
            self.layers.append(layer)

        self.output = nn.Conv2d(int(dim), dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_delasta = nn.Conv2d(dim, inp_channels, 3, 1, 1)  # reconstruction from features

    def forward(self, inp_img):
        f1 = self.conv_first(inp_img)
        x = f1
        for layer in self.layers:
            x = layer(x)

        x = self.output(x + f1)
        x = self.conv_delasta(x) + inp_img
        return x

from torchsummary import summary

if __name__ == '__main__':
    net = HDMba().to(device)
    # x_train = torch.randn(2, 305, 64, 64).to(device)
    # out_train = net(x_train)
    # print("Training output shape:", out_train.shape)  # Expected output shape should be (2, 32, 64, 64)
    # summary(net.cuda(), (305, 64, 64))







############################################################################################
# class SSMamba(nn.Module):
#     def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=0,
#                  mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#
#         if min(self.input_resolution) <= self.window_size:
#             # if window size is larger than input resolution, we don't partition windows
#             self.shift_size = 0
#             self.window_size = min(self.input_resolution)
#
#         self.norm1 = norm_layer(dim)
#         self.mamba = Mamba(seq_len=self.window_size ** 2, d_model=dim, state_size=dim, device=device)
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#     def forward(self, x):
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#
#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)
#         # Pad feature maps to multiples of window size
#         pad_r = (self.window_size - W % self.window_size) % self.window_size
#         pad_b = (self.window_size - H % self.window_size) % self.window_size
#         x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
#         _, Hp, Wp, _ = x.shape
#
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#         else:
#             shifted_x = x
#
#         x_windows = window_partition(shifted_x, self.window_size)
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
#
#         # Apply Mamba
#         attn_windows = self.mamba(x_windows)
#
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#         shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
#
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x
#
#         if pad_r > 0 or pad_b > 0:
#             x = x[:, :H, :W, :].contiguous()
#
#         x = x.view(B, H * W, C)
#
#         # FFN
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#
#         return x
#
#
# # if __name__ == '__main__':
# #     net = SSMamba(dim=30, input_resolution=[32, 32], num_heads=3).to(device)
# #     x = torch.randn(4, 32*32, 30).to(device)
# #     out = net(x)
# #     print(out.shape)
#
#
# class SSMaBlock(nn.Module):
#     def __init__(self,
#                  dim=32,
#                  num_head=3,
#                  input_resolution=[32,32],
#                  window_size=7,
#                  depth=3,
#                  mlp_ratio=2,
#                  drop_path=0.0):
#         super(SSMaBlock, self).__init__()
#         self.ssmablock = nn.Sequential(*[SSMamba(dim=dim, input_resolution=input_resolution, num_heads=num_head, window_size=window_size,
#                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                  mlp_ratio=mlp_ratio,
#                                  drop_path = drop_path,
#                                 # drop_path=drop_path[i],
#                                  )
#             for i in range(depth)])
#         self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
#
#     def forward(self, x):
#         out = self.ssmablock(x)
#         # out = self.conv(out) + x
#         return out
#
#
# if __name__ == '__main__':
#     net = SSMaBlock(dim=30, input_resolution=[32, 32]).to(device)
#     x = torch.randn(4, 32*32, 30).to(device)
#     out = net(x)
#     print(out.shape)