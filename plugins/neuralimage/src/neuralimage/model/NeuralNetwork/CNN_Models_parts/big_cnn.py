"""CNN_Models: big cnn."""
from __future__ import annotations
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

# @register_model(name='Big 5M')
class BigCnn(nn.Module):
    """
    Autoencoder with significantly higher capacity than MediumOptimised
    (roughly 4x parameters by widening channels).
    """
    def __init__(self, input_channels: int = 1):
        super(BigCnn, self).__init__()

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            # 1 -> 128
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 128 -> 192
            nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 192 -> 256
            nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 256 -> 320
            nn.Conv2d(256, 320, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),

            # 320 -> 384
            nn.Conv2d(320, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            # 384 -> 320
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(384, 320, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.LeakyReLU(0.2, inplace=True),

            # 320 -> 256
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(320, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 256 -> 192
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),

            # 192 -> 128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(192, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 -> 128 (РїРѕСЃР»РµРґРЅРёР№ В«РїРµСЂРµС…РѕРґРЅС‹Р№В» СЃР»РѕР№)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Р’С‹С…РѕРґРЅРѕР№ СЃР»РѕР№ -> 1 РєР°РЅР°Р»
            nn.Conv2d(128, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
