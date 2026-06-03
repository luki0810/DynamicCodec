# basic import
import argbind
import math
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from model.utils.mel_pad import right_pad_to_multiple, match_length_lastdim
from model.all_choices import *
from model.utils.abs_class import AbsConvCodec
from model.utils.abs_class import AbsDiscriminator
from model.all_choices import encoder_choices, decoder_choices, quantizer_choices


from model.utils.class_choice.get_default_kwargs import get_default_kwargs
from model.utils.class_choice.nested_dict_action import NestedDictAction

from model.utils.logger import logger


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0)



@argbind.bind(without_prefix=False)
class DynamicDiscriminator(AbsDiscriminator):
    def __init__(
        self,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = 44100,
        bands: list = [],
    ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run Mat, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        super().__init__()
        discs = []
        from model.discriminator.discriminator import MPD, MSD, MRD
        discs += [MPD(p) for p in periods]
        discs += [MSD(r, sample_rate=sample_rate) for r in rates]
        discs += [MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes]
        self.discriminators = nn.ModuleList(discs)

        self.rates = rates
        self.periods = periods
        self.fft_sizes = fft_sizes
        self.sample_rate = sample_rate
        self.bands = bands

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps


@argbind.bind(without_prefix=True)
class DynamicDiscriminatorTask:
    @classmethod
    def build_disc(cls):
        return DynamicDiscriminator()

    @classmethod
    def load_from_folder(cls, **kwargs):
        return DynamicDiscriminator.load_from_folder(**kwargs)
      
@argbind.bind(without_prefix=True)
class DynamicCodec(AbsConvCodec):
    def __init__(
        self,
        sample_rate: int = 66666,
        feature_extractor: Optional[nn.Module] = None,
        encoder: Optional[nn.Module] = None,
        quantizer: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        vocoder: Optional[nn.Module] = None,

        # init weights
        init_weights_fn: Optional[callable] = None,
    ):
        super().__init__()

        # --- important parameters ---
        self.sample_rate = sample_rate
        

        # --- module check ---
        if encoder is None or quantizer is None or decoder is None:
            raise ValueError("encoder/quantizer/decoder is None, please check it")
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.vocoder = vocoder
        self.feature_extractor = feature_extractor
        

        # --- init weights ---
        if init_weights_fn is not None:
            self.apply(init_weights_fn)
        else:
            if "init_weights" in globals():
                self.apply(globals()["init_weights"])

        # --- delay  ---
        self.delay = self.get_delay()

    def preprocess(self, audio_data: torch.Tensor, sample_rate: Optional[int]):
        # sr alignment
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        # Normalize shape: allow (B,T) or (B,1,T) or (B,C,T)
        if audio_data.dim() == 2:
            audio_data = audio_data.unsqueeze(1)

        raw_length = audio_data.shape[-1]  # 原始 wav 长度（samples）


        from data.melspec import MelSpectrogramFeatures
        # input: wav (no feature extractor)
        if self.feature_extractor is None:
            length = raw_length
            hop_length = int(np.prod(self.encoder.encoder_rates))

            # down-sample & up-sample | guaranteed effective recovery (codec hop)
            right_pad = math.ceil(raw_length / hop_length) * hop_length - raw_length
            if right_pad > 0:
                audio_data = F.pad(audio_data, (0, right_pad))

            pad_info = {
                "domain": "wav",
                "raw_length": raw_length,
                "padded_raw_length": audio_data.shape[-1],
                "feat_length": None,
                "padded_feat_length": None,
            }
            return audio_data, length, pad_info

        # mel input (wav -> mel -> pad in mel domain)
        elif isinstance(self.feature_extractor, MelSpectrogramFeatures):
            mel = self.feature_extractor(audio_data)  # (B, n_mels, L)
            n_mels_orig = mel.shape[-2]

            # 2D encoders (e.g. cosmos) treat (n_mels, T) as an image and need
            # n_mels padded up to `encoder.resolution` (a multiple of
            # `spatial_compression`). 1D mel encoders don't expose `resolution`
            # and stay on the no-pad path.
            target_n_mels = getattr(self.encoder, "resolution", None)
            if target_n_mels is not None and target_n_mels > n_mels_orig:
                pad_h = target_n_mels - n_mels_orig
                # F.pad on (B, n_mels, T): (left_w, right_w, left_h, right_h)
                mel = F.pad(mel, (0, 0, 0, pad_h))
            elif target_n_mels is not None and target_n_mels < n_mels_orig:
                raise ValueError(
                    f"encoder.resolution={target_n_mels} < n_mels={n_mels_orig}; "
                    "raise resolution in conf/model/encoder/<name>.yaml or lower n_mels."
                )

            # Pad to a hop multiple along time. When a vocoder is attached we use
            # its hop_length (so its inverse STFT lines up); otherwise fall back
            # to the mel extractor's own hop_length (mel-domain reconstruction).
            hop = (
                self.vocoder.hop_length
                if self.vocoder is not None
                else self.feature_extractor.hop_length
            )
            # 2D encoders also compress the time axis by `spatial_compression`.
            # When that doesn't already divide `hop`, expand to lcm so both
            # constraints are satisfied with one padding pass.
            sc = getattr(self.encoder, "spatial_compression", 1)
            if sc > 1 and hop % sc != 0:
                from math import gcd
                hop = hop * sc // gcd(hop, sc)
            mel, feat_length = right_pad_to_multiple(mel, hop, dim=-1)

            # 这里 length 对于"无 vocoder，仅重建 mel"时，裁剪回原始 mel 帧长
            # 但如果你有 vocoder，并且最终输出 wav，我们仍然要裁 raw_length
            length = feat_length

            pad_info = {
                "domain": "mel",
                "raw_length": raw_length,
                "padded_raw_length": None,
                "feat_length": feat_length,
                "padded_feat_length": mel.shape[-1], # pad 后的 mel 帧长
                "n_mels_orig": n_mels_orig,
                "n_mels_padded": mel.shape[-2] if target_n_mels is not None else None,
            }
            return mel, length, pad_info

        # ssl input
        else:
            # 调用 SSL feature extractor 提取特征: (B, 1, T) -> (B, C, L)
            ssl_feat = self.feature_extractor(audio_data)
            if ssl_feat.dim() == 2:
                ssl_feat = ssl_feat.unsqueeze(0)  # (C, L) -> (1, C, L)
            feat_length = ssl_feat.shape[-1]
            length = feat_length
            pad_info = {
                "domain": "ssl",
                "raw_length": raw_length,
                "padded_raw_length": None,
                "feat_length": feat_length,
                "padded_feat_length": None,
            }
            return ssl_feat, length, pad_info

    def encode(self, audio_data: torch.Tensor):
        z = self.encoder(audio_data)
        z_q, codes, latents, loss_dict, other = self.quantizer(
            z
        )
        return z_q, codes, latents, loss_dict, other

    def decode(self, z: torch.Tensor):
        # z.shape (B, C, T)
        # NOTE: this public path does NOT crop the n_mels axis; cosmos+mel
        # callers should go through forward() which uses pad_info to crop.
        z_hat = self.decoder(z)
        if self.vocoder is not None:
            z_hat = self.vocoder.decode(z_hat)
            if z_hat.dim() == 2:
                z_hat = z_hat.unsqueeze(1)
        return z_hat

    def forward(self, audio_data: torch.Tensor, sample_rate: Optional[int] = None):
        # 保留原始输入长度（samples 或 mel frames 取决于你喂的是什么）
        # 但我们最终以 preprocess 的 pad_info 决定裁剪逻辑
        x_pad, length, pad_info = self.preprocess(audio_data, sample_rate)

        z, codes, latents, loss_dict, other = self.encode(x_pad)

        # ---- decoder + optional vocoder ----
        # We split the .decode() call so we can crop the n_mels axis between
        # decoder and vocoder when a 2D mel encoder (cosmos) padded it.
        feat_hat = self.decoder(z)

        # Crop n_mels back to the original. 2D-image decoders (cosmos) emit
        # (B, resolution, T) which is wider than the canonical n_mels; the
        # vocoder / mel-domain loss expects the canonical width.
        n_mels_orig = pad_info.get("n_mels_orig") if pad_info.get("domain") == "mel" else None
        n_mels_padded = pad_info.get("n_mels_padded") if pad_info.get("domain") == "mel" else None
        if n_mels_padded is not None and feat_hat.dim() == 3 and feat_hat.shape[-2] > n_mels_orig:
            feat_hat = feat_hat[..., :n_mels_orig, :]

        if self.vocoder is not None:
            x_hat = self.vocoder.decode(feat_hat)
            if x_hat.dim() == 2:
                x_hat = x_hat.unsqueeze(1)
        else:
            x_hat = feat_hat

        # ---------- length alignment ----------
        if self.vocoder is not None:
            # vocoder 输出 wav: (B, 1, T) 或 (B, T) -> 统一到 (B, 1, T)
            if x_hat.dim() == 2:
                x_hat = x_hat.unsqueeze(1)

            target_raw_len = int(pad_info["raw_length"])

            # 可选：若你希望先对齐到“padded raw len”，可以尝试推一个目标长度
            # 1) input domain wav
            if pad_info["domain"] == "wav":
                padded_raw_len = int(pad_info["padded_raw_length"])
                x_hat = match_length_lastdim(x_hat, padded_raw_len)

            # 2) input domain mel，且你能拿到 vocoder hop_length，则可以先对齐到 padded_feat_len * hop
            elif pad_info["domain"] == "mel":
                hop = None
                if hasattr(self.vocoder, "hop_length"):
                    hop = int(self.vocoder.hop_length)
                elif hasattr(self.vocoder, "head") and hasattr(self.vocoder.head, "hop_length"):
                    hop = int(self.vocoder.head.hop_length)

                if hop is not None:
                    padded_raw_len = int(pad_info["padded_feat_length"]) * hop
                    x_hat = match_length_lastdim(x_hat, padded_raw_len)
            else: # ssl input
                pass

            # 最终严格裁回原始 wav 长度
            x_hat = x_hat[..., :target_raw_len]

            out_audio = x_hat

        else:
            # no vocoder: output is feature domain (mel/features)
            out_audio = x_hat[..., :length]

        return {
            "audio": out_audio,
            "z": z,
            "codes": codes,
            "latents": latents,
            "loss": loss_dict,
            "other": other,
        }
        
@argbind.bind(without_prefix=True)
class DynamicTask:
    @classmethod
    @argbind.bind(without_prefix=True)
    def build_model(
        cls,
        args,
        input_format: str = "wav",
        encoder: str = "error",
        quantizer: str = "error",
        decoder: str = "error",
        vocoder: str = None,
    ) -> nn.Module:
        logger.info(f"Building model with encoder={encoder}, quantizer={quantizer}, decoder={decoder}, vocoder={vocoder}, input_format={input_format}")
        # 1) encoder
        enc_cls = encoder_choices.get_class(encoder)
        enc_cls = argbind.bind(enc_cls, without_prefix=True)
        enc = enc_cls()

        # 2) quantizer
        q_cls = quantizer_choices.get_class(quantizer)
        q_cls = argbind.bind(q_cls, without_prefix=True)
        qtz = q_cls()

        # 3) decoder
        dec_cls = decoder_choices.get_class(decoder)
        dec_cls = argbind.bind(dec_cls, without_prefix=True)
        dec = dec_cls()
        
        # 4) vocoder
        vo = None
        if vocoder is not None:
            logger.info(f"Building vocoder: {vocoder}")         
            # from pretrained
            # from model.vocoder.voco_istft import Vocoder as Vocos
            # vo = Vocos.from_pretrained("ckpt/models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21")
            
            # from choice
            v_cls = vocoder_choices.get_class(vocoder)
            v_cls = argbind.bind(v_cls, without_prefix=False)
            vo = v_cls()
            
        # 5) feature_extractor
        fem = None
        if input_format == "melspec":
            logger.info("Building mel spectrogram feature extractor")
            with argbind.scope(args):
                from data.melspec import mel_model
                fem = mel_model()
        elif input_format == "repr":
            logger.info("Building SSL feature extractor")
            with argbind.scope(args):
                from data.repr import ssl_model
                fem = ssl_model()

        # 6) combination
        Dyc = argbind.bind(DynamicCodec, without_prefix=True)
        with argbind.scope(args):
            model = Dyc(
                feature_extractor=fem,
                encoder=enc,
                quantizer=qtz,
                decoder=dec,
                vocoder=vo
            )
        logger.info("Dynamic model built successfully")
        return model
    
    @classmethod
    def load_from_folder(cls, **kwargs):
        return DynamicCodec.load_from_folder(**kwargs)
    

if __name__ == "__main__":
    print(get_default_kwargs(DynamicCodec))