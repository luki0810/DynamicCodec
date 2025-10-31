import torch
from torch import nn
from typing import Optional
from audiotools.ml import BaseModel
from model.utils.codec_mixin import CodecMixin

class AbsEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class AbsDecoder(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor: ...

class AbsQuantizer(nn.Module):
    def forward(self, z: torch.Tensor, n_quantizers: Optional[int] = None):
        pass
    
class AbsConvCodec(BaseModel, CodecMixin):
    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: Optional[int] = None,
        n_quantizers: Optional[int] = None,
    ):
        
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)

        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers
        )
        x = self.decode(z)

        return {
            "audio": x[..., :length], # recons waveform
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }
