# basic import
import argbind
import math
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from model.all_choices import *
from model.utils.abs_class import AbsConvCodec
from model.all_choices import encoder_choices, decoder_choices, quantizer_choices


from model.utils.class_choice.get_default_kwargs import get_default_kwargs
from model.utils.class_choice.nested_dict_action import NestedDictAction
from model.utils.class_choice.types import float_or_none, int_or_none, str2bool, str_or_none



def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0)
        
      
@argbind.bind(without_prefix=True)
class DynamicCodec(AbsConvCodec):
    def __init__(
        self,
        sample_rate: int = 44100,
        encoder_rates: List[int] = [2, 4, 8, 8],
        encoder: Optional[nn.Module] = None,
        quantizer: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,

        # init weights
        init_weights_fn: Optional[callable] = None,
    ):
        super().__init__()

        # --- important parameters ---
        self.sample_rate = sample_rate
        self.hop_length = int(np.prod(encoder_rates))

        # --- module check ---
        if encoder is None or quantizer is None or decoder is None:
            raise ValueError("encoder/quantizer/decoder === None, please check it")
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

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
        return audio_data

    def encode(self, audio_data: torch.Tensor):
        # encoder: B x 1 x T -> B x D x T'
        z = self.encoder(audio_data)
        z_q, codes, latents, loss_dict, other = self.quantizer(
            z
        )
        return z_q, codes, latents, loss_dict, other

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

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
        encoder: str = "error",
        quantizer: str = "error",
        decoder: str = "error",
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

        # 4) combination
        model = DynamicCodec(
            encoder=enc,
            quantizer=qtz,
            decoder=dec,
        )
        print("=========== build model successfully ===========")
        print("encoder: ", encoder)
        print("quantizer: ",quantizer)
        print("decoder: ", decoder)
        print("================================================")
        return model


if __name__ == "__main__":
    print(get_default_kwargs(DynamicCodec))