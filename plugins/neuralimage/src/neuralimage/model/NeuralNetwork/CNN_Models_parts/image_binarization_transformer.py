"""CNN_Models: image binarization transformer."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

from neuralimage.model.NeuralNetwork.CNN_Models_parts.vision_transformer import (
    ConvBNAct,
    VisionTransformer,
)

@register_model('Transformer', model_type=ModelType.experimental)
class ImageBinarizationTransformer(nn.Module):
    def __init__(self, in_channels=1, img_size=256, patch_size=16,
                 embed_dim=756, depth=6, num_heads=12, mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        stem_mid = max(32, embed_dim // 8)
        stem_high = max(64, embed_dim // 4)
        self.token_to_high = nn.Sequential(
            nn.Conv2d(embed_dim, stem_high, kernel_size=1, bias=False),
            nn.BatchNorm2d(stem_high),
            nn.GELU(),
        )
        self.fuse_high = nn.Sequential(
            ConvBNAct(stem_high + stem_high, stem_high),
            ConvBNAct(stem_high, stem_high),
        )
        self.fuse_low = nn.Sequential(
            ConvBNAct(stem_high + stem_mid, stem_mid),
            ConvBNAct(stem_mid, stem_mid),
        )
        self.out_head = nn.Sequential(
            nn.Conv2d(stem_mid, stem_mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_mid, 1, kernel_size=1),
        )
        self.confidence_head = nn.Conv2d(stem_mid, 1, kernel_size=1)

    def forward(self, x):
        original_hw = x.shape[-2:]
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        token_map, features = self.vit(x)
        stem1 = features["stem1"]
        stem2 = features["stem2"]

        token_map = self.token_to_high(token_map)
        token_map = F.interpolate(token_map, size=stem2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse_high(torch.cat([token_map, stem2], dim=1))

        x = F.interpolate(x, size=stem1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse_low(torch.cat([x, stem1], dim=1))
        decoded = self.out_head[:-1](x)
        mask_logits = self.out_head[-1](decoded)
        confidence = self.confidence_head(decoded)
        if mask_logits.shape[-2:] != original_hw:
            mask_logits = F.interpolate(mask_logits, size=original_hw, mode='bilinear', align_corners=False)
        if confidence.shape[-2:] != original_hw:
            confidence = F.interpolate(confidence, size=original_hw, mode='bilinear', align_corners=False)
        return {'mask': mask_logits, 'confidence': confidence}
