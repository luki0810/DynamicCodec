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
    pass