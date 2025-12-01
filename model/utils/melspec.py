import torch
import pyworld as pw
import numpy as np
import soundfile as sf
import os
import torchaudio
from torchaudio.functional import pitch_shift
import librosa
from librosa.filters import mel as librosa_mel_fn
import torch.nn as nn
import torch.nn.functional as F
import tqdm


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


# from tadicodec
class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        center=False,
    ):
        super(MelSpectrogram, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.sampling_rate = sampling_rate
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center

        mel_basis = {}
        hann_window = {}

        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis = torch.from_numpy(mel).float()
        hann_window = torch.hann_window(win_size)

        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("hann_window", hann_window)

    def forward(self, y):
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(self.mel_basis, spec)
        spec = spectral_normalize_torch(spec)

        return spec


# from vocos
class MelSpectrogramFeatures(nn.Module):
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
    mode = "vocos"
    
    
    if mode == "tadicodec":
        # 初始化梅尔频谱提取器
        mel_fn = MelSpectrogram(
            n_fft=1024,
            num_mels=80,
            sampling_rate=44100,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=8000
        )

        # 示例1：处理单个音频
        audio, sr = sf.read("/mnt/speech/luyongkang/DynamicCodec/wav_file/input_wav/p226_002.wav")  # 读取音频
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # 添加batch维度
        mel_spec = mel_fn(audio_tensor)  # 提取梅尔频谱
        print(f"输入形状: {audio_tensor.shape}")  # torch.Size([1, 音频长度])
        print(f"输出形状: {mel_spec.shape}")     # torch.Size([1, 80, 时间帧数])

        # 示例2：处理批量音频
        batch_audio = torch.randn(4, 44100)  # 4个1秒音频(44.10kHz)
        batch_mel = mel_fn(batch_audio)
        print(f"批量输出形状: {batch_mel.shape}") 
        
        
    elif mode == "vocos":
        # Test MelSpectrogramFeatures
        mel_extractor = MelSpectrogramFeatures(
            sample_rate = 24000,
            n_fft = 1024,
            hop_length = 256,
            n_mels = 100,
            padding = "center")
        dummy_audio = torch.randn(1, 16000)  # Batch of 2 audio samples, each 1 second at 16kHz
        mel_features = mel_extractor(dummy_audio)
        print("Mel Spectrogram Features shape:", mel_features.shape)