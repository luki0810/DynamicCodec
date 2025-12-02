
import torch
from torch import nn
from typing import Any, Optional, Dict, Union, Tuple
import yaml

from model.utils.abs_class import AbsVocoder
from model.vocoder.vocos.heads import ISTFTHead
from model.vocoder.vocos.models import VocosBackbone


def instantiate_class(args: Union[Any, Tuple[Any, ...]], init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


class Vocoder(AbsVocoder):
    """
    A lightweight vocoder wrapper that only keeps the Vocos backbone and head.
    It expects pre-computed features of shape (B, C, L) and outputs waveform (B, T).
    """
    def __init__(self, 
        # VocosBackbone args
        input_channels: int = 100,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
        
        # ISTFT head args
        # dim: int, the same as above
        n_fft: int = 1024, 
        hop_length: int = 256, 
        padding: str = "same"
        ):
        super().__init__()
        self.backbone = VocosBackbone(
            input_channels=input_channels,
            dim=dim,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            layer_scale_init_value=layer_scale_init_value,
            adanorm_num_embeddings=adanorm_num_embeddings,
        )
        self.head = ISTFTHead(
            dim=dim,
            n_fft=n_fft,
            hop_length=hop_length,
            padding=padding,
        )


    @classmethod
    def from_hparams(cls, config_path: str) -> "Vocoder":
        """
        Create a VocosForward instance from a YAML config that contains
        'backbone' and 'head' entries in the Lightning-style format:

        backbone:
          class_path: vocos.models.VocosBackbone
          init_args: { ... }

        head:
          class_path: vocos.heads.ISTFTHead
          init_args: { ... }
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        backbone = instantiate_class(args=(), init=config["backbone"])
        head = instantiate_class(args=(), init=config["head"])
        model = cls(backbone=backbone, head=head)
        return model

    def forward(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Args:
            features: (B, C, L) feature tensor.

        Returns:
            (B, T) waveform tensor.
        """
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)
        return audio_output
    
    
    def decode(self, features_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output