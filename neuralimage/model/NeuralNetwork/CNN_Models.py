import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.cpu.amp import GradScaler,autocast

from lib import System
from .registrator import *
from .blocks import *

@register_model(name='S 660k')
class SmallFCNN(nn.Module):
    def __init__(self, input_channels = 1):
        super(SmallFCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

@register_model(name='M 720k')
class MediumFCNN(nn.Module):
    def __init__(self, input_channels = 1):
        super(MediumFCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# @register_model(name='MT 1M')
class MediumFCNNTranspose(nn.Module):
    def __init__(self, input_channels = 1):
        super(MediumFCNNTranspose, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(96, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# @register_model(name='MS 720k')
class MediumFCNNSELU(nn.Module):
    def __init__(self, input_channels = 1):
        super(MediumFCNNSELU, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2, stride=2, padding=0),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.SELU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 96, kernel_size=3, padding=1),
            nn.SELU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.SELU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.SELU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SELU(),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# @register_model(name='MLP 720k')
class MediumFCNNLessPool(nn.Module):
    def __init__(self, input_channels = 1):
        super(MediumFCNNLessPool, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# @register_model(name='MO 1.3M')
class MediumOptimised(nn.Module):
    def __init__(self, input_channels = 1):
        super(MediumOptimised, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(128, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(160, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(192, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(160, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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

@register_model('Unet 21.6M')
class Unet(nn.Module):
    def __init__(self,in_channels: int = 1):
        super().__init__()

        self.e1 = ConvBlock(in_channels, 64, norm='bn', act='leaky', pooling=True)
        self.e2 = ConvBlock(64,  128, norm='bn', act='leaky', pooling=True)
        self.e3 = ConvBlock(128, 256, norm='bn', act='leaky', pooling=True)
        self.e4 = ConvBlock(256, 512, norm='bn', act='leaky', pooling=True)

        self.b1 = Bottleneck(512)

        self.d1 = ConvTransposeEncoder(1024, 512, 256)
        self.d2 = ConvTransposeEncoder(256, 256, 128)
        self.d3 = ConvTransposeEncoder(128, 128, 64)
        self.d4 = ConvTransposeEncoder(64, 64, 64)

        self.out_conv = ConvBlock(64, 1, norm='bn', act='leaky', pooling=False)


    def forward(self, x):

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        b = self.b1(e4)

        d1 = self.d1(b, e4)
        d2 = self.d2(d1, e3)
        d3 = self.d3(d2, e2)
        d4 = self.d4(d3, e1)

        o = self.out_conv(d4)

        return o

@register_model('Wellnet 86.5M')
class Wellnet(nn.Module):
    def __init__(self,in_channels: int = 1):
        super().__init__()

        self.start_conv = nn.Sequential(
            ConvBlock(in_channels,64, norm='bn', act='leaky', pooling=False),
            ConvBlock(64, 128, norm='bn', act='leaky', pooling=True),
        )

        self.e1 = ConvBlock(128, 256, norm='bn', act='leaky', pooling=True)
        self.e2 = ConvBlock(256, 512, norm='bn', act='leaky', pooling=True)
        self.e3 = ConvBlock(512, 1024, norm='bn', act='leaky', pooling=True)

        self.b1 = Bottleneck(1024)

        self.d1 = ConvTransposeEncoder(2048, 1024, 512)
        self.d2 = ConvTransposeEncoder(512, 512, 256)
        self.d3 = ConvTransposeEncoder(256, 256, 128)

        self.finish_upsample = nn.Sequential(
            ConvTransposeEncoder(128, 0, 64),
            ConvBlock(64, 1, norm='bn', act='leaky', droput=0.1, pooling=False)
        )


    def forward(self, x):

        x = self.start_conv(x)

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        b = self.b1(e3)

        d1 = self.d1(b, e3)
        d2 = self.d2(d1, e2)
        d3 = self.d3(d2, e1)

        o = self.finish_upsample(d3)


        return o

@register_model('Wellnet2')
class Wellnet2(nn.Module):
    def __init__(self,in_channels: int = 1):
        super().__init__()

        self.start_conv = nn.Sequential(
            ConvBlock(in_channels, 64, norm='bn', act='leaky', droput=0.1, pooling=False),
            ConvBlock(64, 64, norm='bn', act='leaky', droput=0.1, pooling=False),
            ConvBlock(64, 64, norm='bn', act='leaky', droput=0.1, pooling=True)
        )

        self.e1 = nn.Sequential(
            ConvBlock(64, 128, norm='bn', act='leaky', droput=0.2, pooling=False),
            ConvBlock(128, 128, norm='bn', act='leaky', droput=0.2, pooling=False),
            ConvBlock(128, 128, norm='bn', act='leaky', droput=0.2, pooling=True)
        )

        self.e2 = nn.Sequential(
            ConvBlock(128, 256, norm='bn', act='leaky', droput=0.3, pooling=False),
            ConvBlock(256, 256, norm='bn', act='leaky', droput=0.3, pooling=False),
            ConvBlock(256, 256, norm='bn', act='leaky', droput=0.3, pooling=True)
        )
        self.e3 = nn.Sequential(
            ConvBlock(256, 512, norm='bn', act='leaky', droput=0.3, pooling=False),
            ConvBlock(512, 512, norm='bn', act='leaky', droput=0.3, pooling=False),
            ConvBlock(512, 512, norm='bn', act='leaky', droput=0.3, pooling=True)
        )

        self.b1 = Bottleneck(512)

        self.d1 = ConvTransposeEncoder(1024, 512, 512, norm='bn', act='leaky', droput=0.3)
        self.d2 = ConvTransposeEncoder(512, 256, 256, norm='bn', act='leaky', droput=0.2)
        self.d3 = ConvTransposeEncoder(256, 128, 128, norm='bn', act='leaky', droput=0.1)

        self.finish_upsample = nn.Sequential(
            ConvTransposeEncoder(128, 0, 64, norm='bn', act='leaky', droput=0.1),
            ConvBlock(64, 1, norm='bn', act='leaky', droput=0.1, pooling=False)
        )


    def forward(self, x):

        x = self.start_conv(x)

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        b = self.b1(e3)

        d1 = self.d1(b, e3)
        d2 = self.d2(d1, e2)
        d3 = self.d3(d2, e1)

        o = self.finish_upsample(d3)


        return o

@register_model('Wellnet2 mini')
class Wellnet2Mini(nn.Module):
    def __init__(self,in_channels: int = 1):
        super().__init__()
        base_channels = 32

        self.start_conv = nn.Sequential(
            ConvBlock(in_channels, base_channels, norm='bn', act='leaky', droput=0.1, pooling=False),
            ConvBlock(base_channels, base_channels, norm='bn', act='leaky', droput=0.1, pooling=False),
            ConvBlock(base_channels, base_channels, norm='bn', act='leaky', droput=0.1, pooling=True)
        )

        self.e1 = nn.Sequential(
            ConvBlock(base_channels, base_channels*2, norm='bn', act='leaky', droput=0.2, pooling=False),
            ConvBlock(base_channels*2, base_channels*2, norm='bn', act='leaky', droput=0.2, pooling=False),
            ConvBlock(base_channels*2, base_channels*2, norm='bn', act='leaky', droput=0.2, pooling=True)
        )

        self.e2 = nn.Sequential(
            ConvBlock(base_channels*2, base_channels*4, norm='bn', act='leaky', droput=0.3, pooling=False),
            ConvBlock(base_channels*4, base_channels*4, norm='bn', act='leaky', droput=0.3, pooling=False),
            ConvBlock(base_channels*4, base_channels*4, norm='bn', act='leaky', droput=0.3, pooling=True)
        )
        self.e3 = nn.Sequential(
            ConvBlock(base_channels*4, base_channels*8, norm='bn', act='leaky', droput=0.3, pooling=False),
            ConvBlock(base_channels*8, base_channels*8, norm='bn', act='leaky', droput=0.3, pooling=False),
            ConvBlock(base_channels*8, base_channels*8, norm='bn', act='leaky', droput=0.3, pooling=True)
        )

        self.b1 = Bottleneck(base_channels*8)

        self.d1 = ConvTransposeEncoder(base_channels*16, base_channels*8, base_channels*8, norm='bn', act='leaky', droput=0.3)
        self.d2 = ConvTransposeEncoder(base_channels*8, base_channels*4, base_channels*4, norm='bn', act='leaky', droput=0.2)
        self.d3 = ConvTransposeEncoder(base_channels*4, base_channels*2, base_channels*2, norm='bn', act='leaky', droput=0.1)

        self.finish_upsample = nn.Sequential(
            ConvTransposeEncoder(base_channels*2, 0, base_channels, norm='bn', act='leaky', droput=0.1),
            ConvBlock(base_channels, 1, norm='bn', act='leaky', droput=0.1, pooling=False)
        )


    def forward(self, x):

        x = self.start_conv(x)

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        b = self.b1(e3)

        d1 = self.d1(b, e3)
        d2 = self.d2(d1, e2)
        d3 = self.d3(d2, e1)

        o = self.finish_upsample(d3)


        return o

@register_model('EfficientUNet')
class EfficientUNet(nn.Module):
    """
    Lightweight U-Net-like model.
    num_classes=1 -> binary logits (use BCEWithLogitsLoss + sigmoid at inference)
    num_classes>1 -> multi-class logits (use CrossEntropyLoss + softmax at inference)
    """
    def __init__(self, in_ch=1):
        super().__init__()
        gn_groups = 8
        base_ch = 32
        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8
        c5 = base_ch * 16

        self.stem = ResDSBlock(in_ch, c1, gn_groups=gn_groups)
        self.down1 = Down(c1, c2, gn_groups=gn_groups)
        self.down2 = Down(c2, c3, gn_groups=gn_groups)
        self.down3 = Down(c3, c4, gn_groups=gn_groups)
        self.down4 = Down(c4, c5, gn_groups=gn_groups)

        self.up4 = Up(c5, c4, c4, gn_groups=gn_groups)
        self.up3 = Up(c4, c3, c3, gn_groups=gn_groups)
        self.up2 = Up(c3, c2, c2, gn_groups=gn_groups)
        self.up1 = Up(c2, c1, c1, gn_groups=gn_groups)

        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.stem(x)      # 1x
        s2 = self.down1(s1)    # 1/2
        s3 = self.down2(s2)    # 1/4
        s4 = self.down3(s3)    # 1/8
        b  = self.down4(s4)    # 1/16 (bottleneck)

        x = self.up4(b, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.head(x)



# @register_model('Wellnet Ultra')
class WellnetUltra(nn.Module):
    def __init__(self,in_channels: int = 1):
        super().__init__()

        self.start_conv = nn.Sequential(
            ConvBlock(in_channels,64,norm='bn', act='leaky'),
            ConvBlock(64, 128, norm='bn', act='leaky'),
        )

        self.e1 = ConvBlock(128, 256)
        self.e2 = ConvBlock(256, 512)
        self.e3 = ConvBlock(512, 1024)
        self.e4 = ConvBlock(1024, 2048)

        self.b1 = Bottleneck(2048)

        self.d1 = ConvTransposeEncoder(4096, 2048, 1024)
        self.d2 = ConvTransposeEncoder(1024, 1024, 512)
        self.d3 = ConvTransposeEncoder(512, 512, 256)
        self.d4 = ConvTransposeEncoder(256, 256, 128)

        self.finish_upsample = nn.Sequential(
            ConvTransposeEncoder(128, 0, 64),
            ConvTransposeEncoder(64, 0, 64),
            ConvTransposeEncoder(64, 0,1)
        )


    def forward(self, x):

        x = self.start_conv(x)

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        b = self.b1(e4)

        d1 = self.d1(b, e4)
        d2 = self.d2(d1, e3)
        d3 = self.d3(d2, e2)
        d4 = self.d4(d3, e1)

        o = self.finish_upsample(d4)


        return o

class MultiLayerFCNN(nn.Module):
    def __init__(self, input_channels, layers, start_filter, step_filter):
        super(MultiLayerFCNN, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current_filter = start_filter

        # Encoder layers
        for i in range(layers + 1):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(current_filter if i > 0 else input_channels, current_filter, kernel_size=3, padding=1),
                    nn.BatchNorm2d(current_filter),
                    nn.LeakyReLU(0.2)
                )
            )
            if i != layers:
                self.encoders.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            current_filter += step_filter

        # Bottleneck layer (without pooling)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_filter - step_filter, current_filter, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_filter),
            nn.LeakyReLU(0.2)
        )

        # Decoder layers
        for i in range(layers + 1):
            current_filter -= step_filter
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(current_filter + step_filter, current_filter, kernel_size=3, padding=1),
                    nn.BatchNorm2d(current_filter),
                    nn.LeakyReLU(0.2)
                )
            )
            if i != layers - 1:
                self.decoders.append(nn.Upsample(scale_factor=2, mode='nearest'))

        # Output layer
        self.output_layer = nn.Conv2d(current_filter, 1, kernel_size=3, padding=1)

    def forward(self, x):
        skip_connections = []

        # Encoding path
        for layer in self.encoders:
            x = layer(x)
            if isinstance(layer, nn.Sequential):
                skip_connections.append(x)

        x = self.bottleneck(x)

        # Decoding path
        for layer in self.decoders:
            x = layer(x)

        return x

class DepthwiseSeparableConv(nn.Module):
    """ EfficientNet-Style Depthwise Separable Convolution """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DenseResidualBlock(nn.Module):
    """ Dense Residual Block with Multi-Scale Feature Fusion """

    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, growth_rate)
        self.conv2 = DepthwiseSeparableConv(in_channels + growth_rate, growth_rate)
        self.conv3 = DepthwiseSeparableConv(in_channels + 2 * growth_rate, in_channels)

    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(torch.cat([x, out1], dim=1)))
        out3 = self.conv3(torch.cat([x, out1, out2], dim=1))
        return x + out3  # Residual connection


class UpsampleBlock(nn.Module):
    """ Learnable Deconvolution for High-Quality Upsampling """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return F.gelu(self.conv(x))


# @register_model(name='CNN Uber')
class UberModel(nn.Module):
    """ Fully Convolutional Network with Maximum Optimization """

    def __init__(self, input_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            DepthwiseSeparableConv(input_channels, 64),
            DenseResidualBlock(64),
            DepthwiseSeparableConv(64, 128),
            DenseResidualBlock(128),
            DepthwiseSeparableConv(128, 256),
            SEBlock(256),
            DenseResidualBlock(256),
            nn.AdaptiveAvgPool2d(1)
        )

        self.decoder = nn.Sequential(
            UpsampleBlock(256, 128),  # 1 -> 2
            DenseResidualBlock(128),
            UpsampleBlock(128, 64),  # 2 -> 4
            DenseResidualBlock(64),
            UpsampleBlock(64, 32),  # 4 -> 8
            DenseResidualBlock(32),
            UpsampleBlock(32, 16),  # 8 -> 16
            DenseResidualBlock(16),
            UpsampleBlock(16, 8),  # 16 -> 32
            DenseResidualBlock(8),
            UpsampleBlock(8, 4),  # 32 -> 64
            DenseResidualBlock(4),
            UpsampleBlock(4, 2),  # 64 -> 128
            DenseResidualBlock(2),
            UpsampleBlock(2, 1),  # 128 -> 256
            DepthwiseSeparableConv(1, input_channels)  # СѓР¶Рµ 256x256
            # UpsampleBlock(256, 128),
            # DenseResidualBlock(128),
            # UpsampleBlock(128, 64),
            # DenseResidualBlock(64),
            # UpsampleBlock(64, 32),
            # DepthwiseSeparableConv(32, input_channels),
            # nn.Tanh()
        )

    def forward(self, x, tta=False):
        x = self.encoder(x)
        x = self.decoder(x)

        if tta:
            return (x + self.forward(x.flip(2)).flip(2) + self.forward(x.flip(3)).flip(3)) / 3

        return x


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


@register_model('Transformer')
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

    def forward(self, x):
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
        x = self.out_head(x)
        return x

CHANNELS = 1
if __name__ == '__main__':
    model_names = get_registered_models()
    gpus = System.check_gpu_availability()
    devices_list = [torch.device(f"cuda:{gpu}") for gpu in range(gpus)]
    scaler = GradScaler()

    # Set environment variable to avoid fragmentation
    import os

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    for model in model_names:
        # try:
        model_test = create_model(model, CHANNELS)
        model_test = model_test.to(devices_list[0])

        trainable = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
        print(f"Trainable parameters for {model}: {trainable:,}")

        # Reduce batch size to prevent memory issues
        dummy = torch.randn(1, CHANNELS, 32, 32)  # batch=1, single-channel image
        dummy = dummy.to(devices_list[0])

        with autocast():
            out = model_test(dummy)

        out = out.to('cpu')
        torch.cuda.empty_cache()
        print('output shape:', out.shape)

        # except torch.cuda.OutOfMemoryError as e:
        #     print(f"OutOfMemoryError with model {model}: {e}")
        #     torch.cuda.empty_cache()  # Clear cache and continue with next model
        #     continue
        # except Exception as e:
        #     print(f"Error with model {model}: {e}")
        #     torch.cuda.empty_cache()
        #     continue  # expected shape: (4, 1, 256, 256)
    # model_test = create_model('Transformer', 1, 256, 16)
    # model_test = model_test.to(devices_list[0])
    # trainable = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
    # rainable = model_test.parameters()
    # print(f"Trainable parameters for {'Transformer'}: {trainable:,}")
    # dummy = torch.randn(4, 1, 512, 512)  # batch=4, single-channel image
    # dummy = dummy.to(devices_list[0])
    # out = model_test(dummy)
    # out = out.to('cpu')
    # print('output shape:', out.shape)  # expected shape: (4, 1, 256, 256)

    # model = BigCnnV2(inputs=1, base_ch=64, latent_dim=256,
    #                  use_se=True, use_res=True,
    #                  norm='gn', act='gelu')

