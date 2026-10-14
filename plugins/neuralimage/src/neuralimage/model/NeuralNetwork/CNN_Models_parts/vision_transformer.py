"""CNN_Models: vision transformer."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class SwiGLUMLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_proj, gate = self.fc1(x).chunk(2, dim=-1)
        x = x_proj * F.silu(gate)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)

class HybridOverlapPatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, patch_stride=8, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.kernel = max(3, patch_size // 2)
        self.stride = max(1, patch_stride // 2)
        self.padding = max(1, self.kernel // 4)

        stem_mid = max(32, embed_dim // 8)
        stem_high = max(64, embed_dim // 4)
        self.stem1 = ConvBNAct(in_channels, stem_mid)
        self.stem2 = ConvBNAct(stem_mid, stem_high, stride=2)
        self.proj = nn.Conv2d(stem_high, embed_dim, kernel_size=self.kernel, stride=self.stride, padding=self.padding)

    def output_hw(self, h, w):
        h2 = math.floor((h + 2 - 3) / 2 + 1)
        w2 = math.floor((w + 2 - 3) / 2 + 1)
        out_h = math.floor((h2 + 2 * self.padding - self.kernel) / self.stride + 1)
        out_w = math.floor((w2 + 2 * self.padding - self.kernel) / self.stride + 1)
        return out_h, out_w

    def forward(self, x):
        stem1 = self.stem1(x)
        stem2 = self.stem2(stem1)
        tok = self.proj(stem2)
        h, w = tok.shape[-2:]
        tokens = tok.flatten(2).transpose(1, 2)
        return tokens, (h, w), {"stem1": stem1, "stem2": stem2}

class Learnable2DPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, base_h, base_w):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, embed_dim, base_h, base_w))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x, h, w):
        pos = self.pos
        if pos.shape[-2:] != (h, w):
            pos = F.interpolate(pos, size=(h, w), mode='bilinear', align_corners=False)
        pos = pos.flatten(2).transpose(1, 2)
        return x + pos

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = SwiGLUMLP(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=1, img_size=256, patch_size=16,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        patch_stride = max(2, patch_size // 2)
        self.patch_embed = HybridOverlapPatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        base_h, base_w = self.patch_embed.output_hw(img_size, img_size)
        self.pos_encoding = Learnable2DPositionalEncoding(embed_dim, base_h, base_w)
        self.dropout = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim, num_heads, mlp_ratio, dropout, drop_path=dpr[i]
            ) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        tokens, (h, w), features = self.patch_embed(x)
        x = self.pos_encoding(tokens, h, w)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(x.shape[0], self.embed_dim, h, w)
        return x, features
