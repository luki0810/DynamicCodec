from torch import nn
import math

from model.nn.dac.layers import Snake1d
from model.nn.dac.layers import WNConv1d
from model.nn.dac.layers import WNConvTranspose1d
from model.utils.abs_class import AbsDecoder

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


class _DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            _ResidualUnit(output_dim, dilation=1),
            _ResidualUnit(output_dim, dilation=3),
            _ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(AbsDecoder):
    def __init__(
        self,
        latent_dim: int = 64,
        decoder_dim: int = 1536,
        decoder_rates: list = [8, 8, 4, 2],
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(latent_dim, decoder_dim, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(decoder_rates):
            input_dim = decoder_dim // 2**i
            output_dim = decoder_dim // 2 ** (i + 1)
            layers += [_DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)