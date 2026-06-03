from torch import nn
import torch
import math

from model.nn.cosmos.patching import UnPatcher
from model.nn.cosmos.block import ResnetBlock, AttnBlock
from model.nn.cosmos.utils import Normalize, nonlinearity
from model.nn.cosmos.updown_sample import Upsample

from model.utils.abs_class import AbsDecoder


class Decoder(AbsDecoder):
    def __init__(
        self,
        out_channels: int = 1,
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
        patch_method: str = "haar",
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # Mirror the encoder's shape contract (DynamicCodec uses these to
        # decide pad-on-entry / crop-on-exit for the n_mels axis).
        self.resolution = resolution
        self.spatial_compression = spatial_compression
        self.p_size = p_size

        # 与 Encoder: num_downsamples = log2(spatial_compression) - log2(patch_size)
        self.num_upsamples = int(math.log2(spatial_compression)) - int(math.log2(p_size))
        assert self.num_upsamples <= self.num_resolutions, \
            f"we can only upsample {self.num_resolutions} times at most"

        # Encoder 在最后对 H_latent 做 mean(dim=2)，所以 Decoder 需要把它补回去
        # H_latent = (resolution // p_size) // 2**num_upsamples
        assert resolution % p_size == 0, f"resolution={resolution} must be divisible by p_size={p_size}"
        h_after_patch = resolution // p_size
        self.h_latent = h_after_patch // (2 ** self.num_upsamples)
        assert self.h_latent >= 1, f"h_latent computed as {self.h_latent}, check resolution/spatial_compression/p_size"

        # post_quant_conv: (B, embedding_dim, H_latent, W_latent) -> (B, z_channels, H_latent, W_latent)
        self.post_quant_conv = nn.Conv2d(embedding_dim, z_channels, 1)

        # UnPatcher: out_ch = out_channels * p_size^2
        self.unpatcher = UnPatcher(p_size, patch_method)
        out_ch = out_channels * p_size * p_size
        self.out_channels = out_channels

        # z to block_in
        block_in = channels * channels_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling: 这里按“从低分辨率 -> 高分辨率”的顺序构建并顺序遍历
        self.up = nn.ModuleList()
        curr_res_h = self.h_latent  # 只用 H 来做 attn_resolutions 的判定，与 Encoder 的 curr_res 语义一致

        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_out = channels * channels_mult[i_level]

            # 保证 attn 与 block 索引严格对齐：没 attn 的位置填 None
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out

                if curr_res_h in attn_resolutions:
                    attn.append(AttnBlock(block_in))
                else:
                    attn.append(None)

            up = nn.Module()
            up.block = block
            up.attn = attn

            # 需要 upsample 的层数 = num_upsamples（与 Encoder 的 downsample 次数对应）
            # 这里判断：越靠近“瓶颈”的层越先 upsample
            if i_level >= (self.num_resolutions - self.num_upsamples):
                up.upsample = Upsample(block_in)
                curr_res_h *= 2

            self.up.append(up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        期望输入:
          - 与 Encoder 对齐的 3D latent: [B, embedding_dim, W_latent]
        输出:
          - 4D: [B, out_channels, n_mels, T] 或
          - 若 out_channels==1，可选择 squeeze -> [B, n_mels, T]（按你需求）
        """
        # 1) 将 3D latent 补回 H_latent 维度: [B, C, W] -> [B, C, H_latent, W]
        if z.dim() == 3:
            z = z.unsqueeze(2)  # [B, C, 1, W]
            # 简单且对齐：沿 H 复制（相当于 Encoder mean 的“逆”用常数展开）
            z = z.repeat(1, 1, self.h_latent, 1)  # [B, C, H_latent, W]
        assert z.dim() == 4, f"Decoder expects 4D after expand, got {z.shape}"

        # 2) post_quant + conv_in
        z_hat = self.post_quant_conv(z)
        h = self.conv_in(z_hat)

        # 3) middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # 4) upsampling（self.up 已按从低->高顺序 append）
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                attn_layer = self.up[i_level].attn[i_block]
                if attn_layer is not None:
                    h = attn_layer(h)

            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # 5) end + unpatch
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.unpatcher(h)

        # 如果你希望和原始 mel 输入一样输出 3D：[B, n_mels, T]
        if self.out_channels == 1:
            h = h.squeeze(1)

        return h