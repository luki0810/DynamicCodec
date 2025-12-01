import torch
import torch.nn as nn
import torch.nn.functional as F
from model.quantizer.quantize import VectorQuantize


class MelEncoder(nn.Module):
    """
    Mel 编码器：输入 Mel 频谱 (B, n_mels, T) -> 潜在表示 (B, latent_dim, T')
    T' 约为 T / (2 ** num_downsamples)
    """
    def __init__(
        self,
        n_mels: int = 100,
        hidden_channels=(256, 512, 512),
        latent_dim: int = 1024,
        leak: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_ch = n_mels

        # 多层 Conv1d 做下采样：kernel=4, stride=2, padding=1 -> 时间长度 /2
        for h in hidden_channels:
            layers.append(
                nn.Conv1d(
                    in_ch,
                    h,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            layers.append(nn.LeakyReLU(leak, inplace=True))
            in_ch = h

        self.conv = nn.Sequential(*layers)
        # 最终投影到 latent_dim，stride=1 不改变 T'
        self.to_latent = nn.Conv1d(in_ch, latent_dim, kernel_size=3, padding=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, n_mels, T)
        return: (B, latent_dim, T')
        """
        x = self.conv(mel)
        z = self.to_latent(x)
        return z


class MelDecoder(nn.Module):
    """
    Mel 解码器：输入潜在表示 (B, latent_dim, T') -> 预测 Mel (B, n_mels, T_recon)
    使用 ConvTranspose1d 做时间维上采样。
    """
    def __init__(
        self,
        n_mels: int = 100,
        hidden_channels=(512, 512, 256),
        latent_dim: int = 128,
        leak: float = 0.2,
    ):
        super().__init__()

        in_ch = latent_dim
        layers = []

        # 多层 ConvTranspose1d：kernel=4, stride=2, padding=1 -> 时间长度 *2
        for h in hidden_channels:
            layers.append(
                nn.ConvTranspose1d(
                    in_ch,
                    h,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                )
            )
            layers.append(nn.LeakyReLU(leak, inplace=True))
            in_ch = h

        self.deconv = nn.Sequential(*layers)
        # 输出到 n_mels 通道
        self.to_mel = nn.Conv1d(in_ch, n_mels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, target_len: int = None) -> torch.Tensor:
        """
        z: (B, latent_dim, T')
        target_len: 如果提供，则在时间维上进行裁剪或 padding 对齐到该长度
        return: (B, n_mels, T_recon)
        """
        x = self.deconv(z)
        mel_hat = self.to_mel(x)

        if target_len is not None:
            cur_len = mel_hat.size(-1)
            if cur_len > target_len:
                mel_hat = mel_hat[..., :target_len]
            elif cur_len < target_len:
                # 简单 padding 到 target_len（常用于对齐）
                pad = target_len - cur_len
                mel_hat = F.pad(mel_hat, (0, pad))
        return mel_hat


class MelAutoEncoder(nn.Module):
    """
    一个简单的封装，方便直接调用 encoder + decoder。
    """
    def __init__(
        self,
        n_mels: int = 100,
        encoder_hidden=(256, 512, 512),
        decoder_hidden=(512, 512, 256),
        latent_dim: int = 128,
    ):
        super().__init__()
        self.encoder = MelEncoder(
            n_mels=n_mels,
            hidden_channels=encoder_hidden,
            latent_dim=latent_dim,
        )
        self.quantizer = VectorQuantize(
            latent_dim=latent_dim,
            codebook_size=1024,
            codebook_dim=latent_dim,
        )
        self.decoder = MelDecoder(
            n_mels=n_mels,
            hidden_channels=decoder_hidden,
            latent_dim=latent_dim,
        )

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
    dummy_audio = torch.randn(2, 24000)  # 2 个 1 秒 24k 的音频
    mel = mel_extractor(dummy_audio)     # (2, 100, T)
    print("mel shape:", mel.shape)

    # 2. 送入编码器-解码器
    ae = MelAutoEncoder(n_mels=100, latent_dim=128)
    z, mel_hat = ae(mel)
    print("latent shape:", z.shape)       # (2, 128, T')
    print("recon mel shape:", mel_hat.shape)  # (2, 100, T_recon≈T)
