
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
    class_module = "model.vocoder." + class_module
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


class Vocoder(AbsVocoder):
    """
    A lightweight vocoder wrapper that only keeps the Vocos backbone and head.
    It expects pre-computed features of shape (B, C, L) and outputs waveform (B, T).
    """
    def __init__(
        self,
        # 原先这些默认参数保留（用于“从参数构建”）
        input_channels: int = 100,
        dim: int = 512,
        intermediate_dim: int = 1536,
        num_layers: int = 8,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
        n_fft: int = 1024,
        hop_length: int = 256,
        padding: str = "same",

        # 新增：允许直接注入（用于从 config 构建）
        backbone: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ):
        super().__init__()

        if backbone is None:
            backbone = VocosBackbone(
                input_channels=input_channels,
                dim=dim,
                intermediate_dim=intermediate_dim,
                num_layers=num_layers,
                layer_scale_init_value=layer_scale_init_value,
                adanorm_num_embeddings=adanorm_num_embeddings,
            )
        if head is None:
            head = ISTFTHead(
                dim=dim,
                n_fft=n_fft,
                hop_length=hop_length,
                padding=padding,
            )

        self.backbone = backbone
        self.head = head

        # 兼容：如果 head 自己带 hop_length，就以 head 为准
        self.hop_length = getattr(head, "hop_length", hop_length)


    @classmethod
    def from_hparams(cls, config_path: str) -> "Vocoder":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 兼容原版 Vocos：config 里可能含 feature_extractor / backbone / head
        if "backbone" not in config or "head" not in config:
            raise KeyError(
                f"Invalid config: expected keys 'backbone' and 'head' in {config_path}, got {list(config.keys())}"
            )

        backbone = instantiate_class(args=(), init=config["backbone"])
        head = instantiate_class(args=(), init=config["head"])

        # 用“注入模块”的方式创建新 Vocoder
        model = cls(backbone=backbone, head=head)
        return model

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> "Vocoder":
        import os
        from huggingface_hub import hf_hub_download

        # 1) 解析 config / model 文件路径：支持本地目录 or HF repo
        if os.path.isdir(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.yaml")
            model_path = os.path.join(repo_id_or_path, "pytorch_model.bin")
        else:
            config_path = hf_hub_download(
                repo_id=repo_id_or_path,
                filename="config.yaml",
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            model_path = hf_hub_download(
                repo_id=repo_id_or_path,
                filename="pytorch_model.bin",
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        # 2) 用 config 构建新模型（只会实例化 backbone/head）
        model = cls.from_hparams(config_path)

        # 3) 加载权重，并过滤掉 feature_extractor/encodec 等无关参数
        state_dict = torch.load(model_path, map_location="cpu")

        filtered = {}
        for k, v in state_dict.items():
            # 只保留新模型需要的两部分
            if k.startswith("backbone.") or k.startswith("head."):
                filtered[k] = v

        missing, unexpected = model.load_state_dict(filtered, strict=False)

        # 可选：打印一次，方便你确认过滤是否符合预期
        # print("[Vocoder.from_pretrained] missing keys:", missing)
        # print("[Vocoder.from_pretrained] unexpected keys:", unexpected)

        model.eval()
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