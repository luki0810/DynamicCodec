import torch
import torch.nn as nn
import math

from model.utils.abs_class import AbsEncoder

from model.nn.cosmos.block import ResnetBlock, AttnBlock
from model.nn.cosmos.utils import Normalize, nonlinearity
from model.nn.cosmos.updown_sample import Downsample
from model.nn.cosmos.patching import Patcher


class Encoder(AbsEncoder):
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 128,
        channels_mult: list[int] = [2, 4, 4],
        num_res_blocks: int = 2,
        attn_resolutions: list[int] = [6, 12],
        dropout: float = 0.0,
        resolution: int = 96,
        z_channels: int = 256,
        spatial_compression: int = 8,
        embedding_dim: int = 1024,
        p_size: int = 2,
        p_method: str = "haar",
        
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # Patcher.
        self.patcher = Patcher(
            p_size, p_method
        )
        in_channels = in_channels * p_size * p_size

        # calculate the number of downsample operations
        self.num_downsamples = int(math.log2(spatial_compression)) - int(
            math.log2(p_size)
        )
        assert (
            self.num_downsamples <= self.num_resolutions
        ), f"we can only downsample {self.num_resolutions} times at most"

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution // p_size
        in_ch_mult = (1,) + tuple(channels_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_downsamples:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=dropout
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )

        # quant-conv
        self.quant_conv = nn.Conv2d(z_channels, embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
                x = x.unsqueeze(1)  # [B,1,n_mels,T]
        assert x.dim() == 4, f"Encoder expects 4D after unsqueeze, got {x.shape}"

        x = self.patcher(x)
        assert x.dim() == 4, f"Patcher must output 4D, got {x.shape}"


        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level < self.num_downsamples:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        return self.quant_conv(h).mean(dim=2)