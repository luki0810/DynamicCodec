import torch
from torch import nn
import math


from model.utils.abs_class import AbsEncoder
from model.nn.dac.layers import Snake1d
from model.nn.dac.layers import WNConv1d


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
        n_mels: int = 100, # d_in
        encoder_dim: int = 64,
        encoder_rates: list = [2, 2, 2],
        latent_dim: int = 64,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.latent_dim = latent_dim

        blocks = []
        blocks.append(WNConv1d(n_mels, encoder_dim, kernel_size=7, padding=3))

        cur_dim = encoder_dim
        for stride in encoder_rates:
            next_dim = cur_dim * 2
            blocks.append(_EncoderBlock(next_dim, stride=stride)) 
            cur_dim = next_dim

        self.final_dim = cur_dim 

        # cur_dim -> (Snake + Conv1d) -> latent_dim
        blocks += [
            Snake1d(cur_dim),
            WNConv1d(cur_dim, latent_dim, kernel_size=3, padding=1),
        ]

        self.block = nn.Sequential(*blocks)
        self.o_dim = latent_dim

        ds = 1
        for s in encoder_rates:
            ds *= s
        self.downsample_factor = ds

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, n_mels, T)
        return z: (B, latent_dim, T')
        """
        return self.block(mel)
