import torch
from torch import nn
import math
from torch.nn import functional as F


from model.nn.dac.layers import Snake1d
from model.nn.dac.layers import WNConv1d, WNConvTranspose1d
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
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.upsample = WNConvTranspose1d(
            dim,
            dim // 2,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=stride % 2,
        )

        self.res_blocks = nn.Sequential(
            _ResidualUnit(dim // 2, dilation=1),
            _ResidualUnit(dim // 2, dilation=3),
            _ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.res_blocks(x)
        return x



 
class Decoder(AbsDecoder):
    def __init__(
        self,
        n_mels: int = 100, # d_out
        decoder_dim: int = 64,
        decoder_rates: list = [2, 2, 2],
        latent_dim: int = 64,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.latent_dim = latent_dim

        cur_dim = decoder_dim
        for stride in decoder_rates:
            cur_dim *= 2
        self.encoder_final_dim = cur_dim  # MelEncoder.final_dim

        blocks = []

        blocks.append(WNConv1d(latent_dim, self.encoder_final_dim, kernel_size=3, padding=1))
        blocks.append(Snake1d(self.encoder_final_dim))

        dec_dim = self.encoder_final_dim
        for stride in reversed(decoder_rates):
            blocks.append(_DecoderBlock(dec_dim, stride=stride))
            dec_dim = dec_dim // 2  

        blocks += [
            WNConv1d(dec_dim, n_mels, kernel_size=7, padding=3),
        ]

        self.block = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor, target_len: int = None) -> torch.Tensor:
        mel_hat = self.block(z)

        if target_len is not None:
            cur_len = mel_hat.size(-1)
            if cur_len > target_len:
                mel_hat = mel_hat[..., :target_len]
            elif cur_len < target_len:
                pad = target_len - cur_len
                mel_hat = F.pad(mel_hat, (0, pad))

        return mel_hat
    