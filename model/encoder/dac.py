import math
import argbind
from torch import nn
from model.nn.dac.layers import Snake1d
from model.nn.dac.layers import WNConv1d
from model.utils.abs_class import AbsEncoder


class _ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y
    
    
class _EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            _ResidualUnit(dim // 2, dilation=1),
            _ResidualUnit(dim // 2, dilation=3),
            _ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)
    
    
class Encoder(AbsEncoder):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list = [2, 4, 8, 8],
        latent_dim: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, encoder_dim, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in encoder_rates:
            encoder_dim *= 2
            self.block += [_EncoderBlock(encoder_dim, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(encoder_dim),
            WNConv1d(encoder_dim, latent_dim, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.o_dim = latent_dim
        
    def forward(self, x):
        return self.block(x)