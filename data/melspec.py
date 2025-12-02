import torch
import torchaudio
import argbind

from data.base import FeatureExtractor

# from vocos
class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def safe_log(self, x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
        """
        Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

        Args:
            x (Tensor): Input tensor.
            clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

        Returns:
            Tensor: Element-wise logarithm of the input tensor with clipping applied.
        """
        return torch.log(torch.clip(x, min=clip_val))

    def forward(self, audio, **kwargs):
        if audio.dim() == 3 and audio.size(1) == 1:
            audio = audio.squeeze(1)
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        features = self.safe_log(mel)
        return features


@argbind.bind()
def mel_model(
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 100,
    padding: str = "center"
):
    mel_extractor = MelSpectrogramFeatures(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        padding=padding
    )
    return mel_extractor

