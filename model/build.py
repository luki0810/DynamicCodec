# basic import
import argbind
import math
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vocoder.vocos import Vocos
from model.all_choices import *
from model.utils.abs_class import AbsConvCodec
from model.utils.abs_class import AbsDiscriminator
from model.all_choices import encoder_choices, decoder_choices, quantizer_choices


from model.utils.class_choice.get_default_kwargs import get_default_kwargs
from model.utils.class_choice.nested_dict_action import NestedDictAction



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
        sample_rate: int = 44100,
        encoder_rates: List[int] = [2, 4, 8, 8],
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
        self.hop_length = int(np.prod(encoder_rates))

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
            
        # down-sample & up-sample | guaranteed effective recovery
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        if right_pad > 0:
            audio_data = F.pad(audio_data, (0, right_pad))

        if self.feature_extractor is not None:
            audio_data = self.feature_extractor(audio_data)          

        return audio_data

    def encode(self, audio_data: torch.Tensor):
        # encoder: B x 1 x T -> B x D x T'
        z = self.encoder(audio_data)
        z_q, codes, latents, loss_dict, other = self.quantizer(
            z
        )
        return z_q, codes, latents, loss_dict, other

    def decode(self, z: torch.Tensor):
        z_hat = self.decoder(z)
        if self.vocoder is not None:
            z_hat = self.vocoder.decode(z_hat)
        return z_hat

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: Optional[int] = None,
    ):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)

        z, codes, latents, loss_dict, other = self.encode(
            audio_data
        )
        x = self.decode(z)

        return {
            "audio": x[..., :length], # recons waveform
            "z": z,
            "codes": codes,
            "latents": latents,
            "loss": loss_dict,
            "other": other
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
        vocoder_model = None
        if vocoder is not None:
            vocoder_model = Vocos.from_pretrained("charactr/vocos-mel-24khz")

        # 5) feature_extractor
        feature_extractor_model = None
        if input_format == "melspec":
            with argbind.scope(args):
                from data.melspec import mel_model
                feature_extractor_model = mel_model()
        elif input_format == "repr":
            with argbind.scope(args):
                from data.repr import ssl_model
                feature_extractor_model = ssl_model()

        # 6) combination
        model = DynamicCodec(
            feature_extractor=feature_extractor_model,
            encoder=enc,
            quantizer=qtz,
            decoder=dec,
            vocoder=vocoder_model
        )
        return model
    
    @classmethod
    def load_from_folder(cls, **kwargs):
        return DynamicCodec.load_from_folder(**kwargs)
    


if __name__ == "__main__":
    print(get_default_kwargs(DynamicCodec))