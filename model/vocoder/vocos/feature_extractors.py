from typing import List

import torch
import torchaudio
from torch import nn


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


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
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        features = self.safe_log(mel)
        return features


if __name__ == "__main__":
    # Test MelSpectrogramFeatures
    mel_extractor = MelSpectrogramFeatures(
        sample_rate = 24000,
        n_fft = 1024,
        hop_length = 256,
        n_mels = 100,
        padding = "center")
    dummy_audio = torch.randn(2, 16000)  # Batch of 2 audio samples, each 1 second at 16kHz
    mel_features = mel_extractor(dummy_audio)
    print("Mel Spectrogram Features shape:", mel_features.shape)