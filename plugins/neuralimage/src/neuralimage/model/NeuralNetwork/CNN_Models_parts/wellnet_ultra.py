"""CNN_Models: wellnet ultra."""
from __future__ import annotations
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

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
