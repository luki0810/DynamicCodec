import logging
import torch
import torch.nn.functional as F
import fairseq
from typing import Optional, List
from fairseq import tasks
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from omegaconf import OmegaConf



from data.base import FeatureExtractor
from data.ssl.utils.data2vec_audio import Data2VecAudioModel
from whisper.audio import log_mel_spectrogram
from data.ssl.utils.whisper_feature_reader import load_model as load_whisper_model


logger = logging.getLogger("ssl_feature_extractor")

class HubertFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        ckpt_path: str,
        layer: int,
        device: str = "cuda",
        max_chunk: int = 1600000,
    ):
        super().__init__()

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path]
        )
        self.model = models[0].eval().to(device)
        self.task = task
        self.layer = layer
        self.device = torch.device(device)
        self.max_chunk = max_chunk
        self.sample_rate = task.cfg.sample_rate
        self.normalize = bool(getattr(task.cfg, "normalize", False))

        logger.info(f"[HuBERT] device = {self.device}")
        logger.info(f"[HuBERT] TASK CONFIG:\n{self.task.cfg}")
        logger.info(f"[HuBERT] max_chunk = {self.max_chunk}")

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.mean(dim=1)  # (B, T)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        assert audio.dim() == 2, f"Expected (B, C, T) or (B, T) or (T,), got {audio.shape}"

        B, T = audio.shape
        if lengths is None:
            lengths = audio.new_full((B,), T, dtype=torch.long)

        feats_list: List[torch.Tensor] = []
        feat_lengths: List[int] = []

        for b in range(B):
            wav = audio[b, : lengths[b]].float().to(self.device)  # (T_b,)

            if self.normalize:
                wav = F.layer_norm(wav, wav.shape)

            x = wav.view(1, -1)  # (1, T_b)

            chunks = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )  # (1, L_chunk, C)
                chunks.append(feat_chunk)

            feat = torch.cat(chunks, dim=1).squeeze(0)  # (L_b, C)
            feat = feat.transpose(0, 1).contiguous()
            feats_list.append(feat)
            feat_lengths.append(feat.shape[1])
        max_L = max(feat_lengths)
        C = feats_list[0].shape[0]
        feats = audio.new_zeros(
            (B, C, max_L),
            dtype=feats_list[0].dtype,
            device=feats_list[0].device,
        )
        for b, fb in enumerate(feats_list):
            Lb = fb.shape[1]
            feats[b, :, :Lb] = fb
        return feats


class Data2vecFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        ckpt_path: str,
        layer: int,
        device: str = "cuda",
        max_chunk: int = 1600000,
    ):
        super().__init__()

        state = load_checkpoint_to_cpu(ckpt_path)
        cfg = state["cfg"]

        # load task
        task = tasks.setup_task(cfg.task, from_checkpoint=True)
        task.load_state_dict(state["task_state"])

        # load model config
        if "layer_type" not in cfg.model:
            model_config = {k: v for k, v in cfg.model.items()}
            model_config["layer_type"] = "transformer"
            model_config = OmegaConf.create(model_config)
        else:
            model_config = cfg.model

        # fix param name in the state
        state["model"]["final_proj.weight"] = state["model"].pop(
            "final_proj.0.weight"
        )
        state["model"]["final_proj.bias"] = state["model"].pop("final_proj.0.bias")
        if "_ema" in state["model"]:
            del state["model"]["_ema"]

        model = Data2VecAudioModel.build_model(model_config)
        model.load_state_dict(state["model"], strict=True, model_cfg=model_config)

        self.model = model.eval().to(device)
        self.task = task
        self.device = torch.device(device)
        self.max_chunk = max_chunk
        self.sample_rate = task.cfg.sample_rate
        self.normalize = bool(getattr(task.cfg, "normalize", False))
        self.layer = layer - 1

        logger.info(f"[Data2Vec] device = {self.device}")
        logger.info(f"[Data2Vec] TASK CONFIG:\n{self.task.cfg}")
        logger.info(f"[Data2Vec] max_chunk = {self.max_chunk}")

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.mean(dim=1)  # (B, T)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        assert audio.dim() == 2, f"Expected (B, C, T) or (B, T) or (T,), got {audio.shape}"
        B, T = audio.shape
        if lengths is None:
            lengths = audio.new_full((B,), T, dtype=torch.long)
        feats_list: List[torch.Tensor] = []
        feat_lengths: List[int] = []
        for b in range(B):
            wav = audio[b, : lengths[b]].float().to(self.device)  # (T_b,)

            if self.normalize:
                wav = F.layer_norm(wav, wav.shape)
            x = wav.view(1, -1)  # (1, T_b)
            chunks = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                res = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    layer=self.layer,
                )
                feat_chunk = res["x"]  # (1, L_chunk, C)
                chunks.append(feat_chunk)
            feat = torch.cat(chunks, dim=1).squeeze(0)  # (L_b, C)
            feat = feat.transpose(0, 1).contiguous()  # (C, L_b)
            feats_list.append(feat)
            feat_lengths.append(feat.shape[1])

        max_L = max(feat_lengths)
        C = feats_list[0].shape[0]
        feats = audio.new_zeros(
            (B, C, max_L),
            dtype=feats_list[0].dtype,
            device=feats_list[0].device,
        )

        for b, fb in enumerate(feats_list):
            Lb = fb.shape[1]
            feats[b, :, :Lb] = fb
        return feats


class WhisperFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        root: Optional[str],
        ckpt: str,
        layer: int,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = torch.device(device)
        logger.info(f"[Whisper] device = {self.device}")

        self.model = load_whisper_model(
            name=ckpt,
            device=self.device,
            download_root=root,
        ).eval()

        if hasattr(self.model, "decoder"):
            self.model.decoder = None

        self.layer = layer        
        self.sample_rate = 16000

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.mean(dim=1)  # (B, T)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        assert audio.dim() == 2, f"Expected (B, C, T) or (B, T) or (T,), got {audio.shape}"

        B, T = audio.shape
        if lengths is None:
            lengths = audio.new_full((B,), T, dtype=torch.long)

        feats_list: List[torch.Tensor] = []
        feat_lengths: List[int] = []

        for b in range(B):
            wav = audio[b, : lengths[b]].float().to(self.device)  # (T_b,)

            # Whisper pipeline: waveform -> log-mel -> encoder hidden
            mel = log_mel_spectrogram(wav)            # (n_mels, T_mel)
            hidden = self.model.extract_features(
                mel.unsqueeze(0), target_layer=self.layer
            )                                        # (1, L_enc, C)

            audio_length = wav.shape[0]
            feature_length = audio_length // 320
            hidden = hidden[0, :feature_length]       # (L_b, C)

            feat = hidden.transpose(0, 1).contiguous()  # (C, L_b)
            feats_list.append(feat)
            feat_lengths.append(feat.shape[1])

        max_L = max(feat_lengths)
        C = feats_list[0].shape[0]
        feats = audio.new_zeros(
            (B, C, max_L),
            dtype=feats_list[0].dtype,
            device=feats_list[0].device,
        )

        for b, fb in enumerate(feats_list):
            Lb = fb.shape[1]
            feats[b, :, :Lb] = fb

        return feats