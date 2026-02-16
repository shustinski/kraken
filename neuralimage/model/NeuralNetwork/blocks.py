import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch,k=3, s=1,p=1,gn_groups=8):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        g = min(gn_groups,out_ch)
        self.norm = nn.GroupNorm(g, out_ch)

        self.act =  nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        return self.act(x)

class ResDSBlock(nn.Module):
    """Two DSConv with residual (when shape matches)."""
    def __init__(self, in_ch, out_ch, gn_groups=8):
        super().__init__()
        self.conv1 = DSConv(in_ch, out_ch, gn_groups=gn_groups)
        self.conv2 = DSConv(out_ch, out_ch, gn_groups=gn_groups)
        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.skip is not None:
            identity = self.skip(identity)
        return x + identity


class Down(nn.Module):
    """Downsample by stride-2 depthwise-separable conv."""
    def __init__(self, in_ch, out_ch, gn_groups=8):
        super().__init__()
        self.down = DSConv(in_ch, out_ch, s=2, gn_groups=gn_groups)   # stride=2
        self.block = ResDSBlock(out_ch, out_ch, gn_groups=gn_groups)

    def forward(self, x):
        x = self.down(x)
        return self.block(x)


class Up(nn.Module):
    """Upsample + concat skip + refine."""
    def __init__(self, in_ch, skip_ch, out_ch, gn_groups=8):
        super().__init__()
        self.refine = ResDSBlock(in_ch + skip_ch, out_ch, gn_groups=gn_groups)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # spatial safety (in case H/W not divisible by 16)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch,
                 norm='bn',
                 act='gelu',
                 droput:float=0,
                 stride=1,
                 pooling:bool=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        else:
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)

        self.act =  nn.GELU() if act == 'gelu' else (
                    nn.SiLU()) if act == 'silu' else nn.LeakyReLU(0.2, inplace=True)

        self.use_dropout = bool(droput)
        self.dropout = nn.Dropout2d(p=droput)

        self.pooling = bool(pooling)
        if pooling:
            self.pooling_layer = nn.MaxPool2d(2, stride=2, padding=0)
        else:
            self.pooling_layer = None


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.use_dropout:
            x = self.dropout(x)
        if self.pooling and self.pooling_layer is not None:
            x = self.pooling_layer(x)
        return x

class ConvTransposeEncoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels,
                 norm='bn',
                 act='gelu',
                 droput: float = 0,
                 stride=1
                 ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                                                 stride=2, padding=1, output_padding=1)
        self.conv1 = ConvBlock(out_channels + skip_channels,out_channels, norm=norm, act=act, droput=droput, pooling=False)
        self.conv2 = ConvBlock(out_channels, out_channels, norm=norm, act=act, droput=droput, pooling=False)

    def forward(self, x, skip=None):
        # Transpose convolution
        x = self.conv_transpose(x)
        if skip is not None:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class ConvUpsampleEncoder(nn.Module):
    def __init__(self, skip_channels, out_channels,
                 norm='bn',
                 act='gelu',
                 droput: float = 0,
                 stride=1
                 ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = ConvBlock(out_channels + skip_channels,out_channels, norm=norm, act=act, droput=droput, pooling=False)
        self.conv2 = ConvBlock(out_channels, out_channels, norm=norm, act=act, droput=droput, pooling=False)

    def forward(self, x, skip=None):
        # Transpose convolution
        x = self.upsample(x)
        if skip is not None:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class Bottleneck(nn.Module):

    def __init__(self, in_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_ch, in_ch*2, norm='bn', act='leaky', droput=0.3, pooling=False),
            ConvBlock(in_ch*2, in_ch*2, norm='bn', act='leaky', droput=0.3, pooling=False),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.extinstion = nn.Sequential(
            nn.Linear(ch, ch//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch//reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs,c,_,_ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.extinstion(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)



class ResBlock(nn.Module):
    def __init__(self, ch, norm='gn', act='gelu', use_se=False):
        super().__init__()
        self.conv1 = ConvBlock(ch, ch, norm=norm, act=act)
        self.conv2 = ConvBlock(ch, ch, norm=norm, act=act)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(ch)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        return x + residual



class UpBlock(nn.Module):
    """
    Upsample (ConvTranspose2d) → concat со skip‑тензором →
    ConvBlock (внутри учитывает реальное количество каналов).
    """
    def __init__(self,
                 in_ch,          # каналы входа в ConvTranspose2d
                 out_ch,         # каналы после up‑sampling
                 skip_ch=None,   # каналы у skip‑тензора (по умолчанию = out_ch)
                 use_se=False,
                 norm='gn',
                 act='gelu'):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch,
                                     kernel_size=2, stride=2)

        if skip_ch is None:
            skip_ch = 0
        conv_in = out_ch + skip_ch          # после конкатенации
        self.conv = ConvBlock(conv_in, out_ch,
                              norm=norm, act=act, pooling=False)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_ch)

    def forward(self, x):
        x = self.up(x)                                            # ↑2
        x = self.conv(x)
        if self.use_se:
            x = self.se(x)
        return x
