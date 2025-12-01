from model.nn.dac.layers import Snake1d
from model.nn.dac.layers import WNConv1d, WNConvTranspose1d
from model.utils.abs_class import AbsEncoder, AbsDecoder

import torch
from torch import nn
import math

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




class MelEncoder(AbsEncoder):
    """
    DAC 风格的 Mel 编码器：
    输入 mel: (B, n_mels, T)
    输出 latent: (B, latent_dim, T')
    其中 T' = T / prod(encoder_rates)（四舍五入）
    """
    def __init__(
        self,
        n_mels: int = 100,
        encoder_dim: int = 64,
        encoder_rates: list = [2, 4, 8, 8],
        latent_dim: int = 64,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.latent_dim = latent_dim

        blocks = []
        # 第一层：把 n_mels 提升到 encoder_dim
        blocks.append(WNConv1d(n_mels, encoder_dim, kernel_size=7, padding=3))

        cur_dim = encoder_dim
        # 与 DAC 相同：每个 stride 前先把通道数 *2，然后进入 _EncoderBlock
        for stride in encoder_rates:
            next_dim = cur_dim * 2
            blocks.append(_EncoderBlock(next_dim, stride=stride))  # 注意：内部用 dim//2
            cur_dim = next_dim

        self.final_dim = cur_dim  # encoder 最后 conv 之前的通道数

        # 最后一层：Snake + Conv1d -> latent_dim
        blocks += [
            Snake1d(cur_dim),
            WNConv1d(cur_dim, latent_dim, kernel_size=3, padding=1),
        ]

        self.block = nn.Sequential(*blocks)
        self.o_dim = latent_dim

        # 下采样因子：方便你外面算 T'
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
    
    
    
    
class _DecoderBlock(nn.Module):
    """
    与 _EncoderBlock 镜像：
    - 先上采样 (dim -> dim//2, stride)
    - 再做多尺度 residual
    """
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        # 上采样：ConvTranspose1d
        self.upsample = WNConvTranspose1d(
            dim,
            dim // 2,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=stride % 2,  # 一般可缓解 off-by-one
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
    
    



class MelDecoder(AbsDecoder):
    """
    DAC 风格的 Mel 解码器：
    输入 latent: (B, latent_dim, T')
    输出 mel_hat: (B, n_mels, T_recon)
    """
    def __init__(
        self,
        n_mels: int = 100,
        encoder_dim: int = 64,
        encoder_rates: list = [2, 2, 2],
        latent_dim: int = 64,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.latent_dim = latent_dim

        # 需要知道 encoder 最后一个 block 的通道数（跟 MelEncoder 保持一致）
        cur_dim = encoder_dim
        for stride in encoder_rates:
            cur_dim *= 2
        self.encoder_final_dim = cur_dim  # 对应 MelEncoder.final_dim

        blocks = []

        # 首先把 latent_dim -> encoder_final_dim
        blocks.append(WNConv1d(latent_dim, self.encoder_final_dim, kernel_size=3, padding=1))
        blocks.append(Snake1d(self.encoder_final_dim))

        # 反向遍历 encoder 的 strides，依次上采样
        dec_dim = self.encoder_final_dim
        for stride in reversed(encoder_rates):
            blocks.append(_DecoderBlock(dec_dim, stride=stride))
            dec_dim = dec_dim // 2  # 每个 block 把通道减半

        # 最后一层：把 decoder 最终通道数 -> n_mels
        blocks += [
            WNConv1d(dec_dim, n_mels, kernel_size=7, padding=3),
        ]

        self.block = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor, target_len: int = None) -> torch.Tensor:
        """
        z: (B, latent_dim, T')
        target_len: 如果提供，则在时间维上裁剪/补齐到该长度
        return mel_hat: (B, n_mels, T_recon)
        """
        mel_hat = self.block(z)

        if target_len is not None:
            cur_len = mel_hat.size(-1)
            if cur_len > target_len:
                mel_hat = mel_hat[..., :target_len]
            elif cur_len < target_len:
                pad = target_len - cur_len
                mel_hat = F.pad(mel_hat, (0, pad))

        return mel_hat
    
    
    
    
from model.quantizer.quantize import ResidualVectorQuantize
class MelAE(nn.Module):
    """
    整体的 Mel Auto-Encoder：
    - encoder: MelEncoder (DAC 风格)
    - decoder: MelDecoder (镜像结构)
    """
    def __init__(
        self,
        n_mels: int = 100,
        encoder_dim: int = 64,
        encoder_rates: list = [2, 2, 2],
        latent_dim: int = 64,
    ):
        super().__init__()
        self.encoder = MelEncoder(
            n_mels=n_mels,
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=latent_dim,
        )
        self.quantizer = ResidualVectorQuantize(
            latent_dim=latent_dim,
            n_codebooks=8,
            codebook_size=1024,
            codebook_dim=8,
            quantizer_dropout=0.0,
        )
        self.decoder = MelDecoder(
            n_mels=n_mels,
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=latent_dim,
        )

    @property
    def downsample_factor(self):
        return self.encoder.downsample_factor

    def forward(self, mel: torch.Tensor):
        """
        mel: (B, n_mels, T)
        return:
            z: (B, latent_dim, T')
            mel_hat: (B, n_mels, T)
        """
        z = self.encoder(mel)
        z_q, _, _, _, _ = self.quantizer(z)
        mel_hat = self.decoder(z_q, target_len=mel.size(-1))
        return z, mel_hat
    
    
    
   
from torch.nn import functional as F
from model.utils.melspec import MelSpectrogramFeatures
if __name__ == "__main__":
    # 1. 提取 mel
    mel_extractor = MelSpectrogramFeatures(
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
    )
    dummy_audio = torch.randn(1, 24000)  # 2 个 1 秒音频
    mel = mel_extractor(dummy_audio)     # (2, 100, T)
    print("mel:", mel.shape)

    # 2. MelAE（DAC 风格）
    melae = MelAE(
        n_mels=100,
        encoder_dim=64,
        encoder_rates=[2, 2, 2],
        latent_dim=64,
    )

    z, mel_hat = melae(mel)
    print("z:", z.shape)
    print("mel_hat:", mel_hat.shape)

    # 3. 损失
    loss = F.l1_loss(mel_hat, mel)
    loss.backward()
    print("loss:", loss.item())




