import logging
import os
import sys
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from fairseq.data.audio.audio_utils import get_features_or_waveform
from data.melspec import FeatureExtractor

class SSLFeatureExtractor(nn.Module):

    def __init__(self, reader):
        super().__init__()
        self.reader = reader

    @torch.no_grad()
    def _extract_single(self, wav_1d: torch.Tensor) -> torch.Tensor:
        assert wav_1d.dim() == 1, f"Expected 1D tensor, got {wav_1d.shape}"
        x = wav_1d.float()
        if getattr(self.reader.task.cfg, "normalize", False):
            x = F.layer_norm(x, x.shape)
        x = x.view(1, -1)
        feat_chunks = []
        T_total = x.size(1)
        max_chunk = self.reader.max_chunk

        for start in range(0, T_total, max_chunk):
            x_chunk = x[:, start : start + max_chunk]  # (1, T_chunk)
            feat_chunk, _ = self.reader.model.extract_features(
                source=x_chunk,
                padding_mask=None,
                mask=False,
                output_layer=self.reader.layer,
            )
            feat_chunks.append(feat_chunk)

        feat = torch.cat(feat_chunks, dim=1)  # (1, L, C)
        feat = feat.squeeze(0)                # (L, C)
        feat = feat.transpose(0, 1)           # (C, L)

        return feat 

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            assert audio.size(1) == 1
            audio = audio.squeeze(1)

        assert audio.dim() == 2
        B, T = audio.shape

        feats_list = []
        lengths = []

        for b in range(B):
            feat_b = self._extract_single(audio[b])  # (C, L_b)
            feats_list.append(feat_b)
            lengths.append(feat_b.size(-1))

        C = feats_list[0].size(0)
        L_max = max(lengths)

        feats = audio.new_zeros(B, C, L_max)

        for b, feat_b in enumerate(feats_list):
            L_b = feat_b.size(-1)
            feats[b, :, :L_b] = feat_b.to(feats.device)

        return feats 


        

import argbind
@argbind.bind()
def ssl_model(
    model_type: str = 'hubert',
    ckpt_path: str = 'ckpt/to/file',
    device: str = "cuda",
    layer: int = 18,
    max_chunk: int = 1600000,
    use_cpu: bool = False,
    whisper_root: str = None,
    whisper_name: str = None
):
    device = "cpu" if use_cpu else device
    reader = None
    if model_type == "hubert":
        from data.ssl.utils.hubert_feature_reader import HubertFeatureReader
        reader = HubertFeatureReader(ckpt_path, layer, device=device, max_chunk=max_chunk)
    elif model_type == "data2vec":
        from data.ssl.utils.data2vec_feature_reader import Data2vecFeatureReader
        reader = Data2vecFeatureReader(ckpt_path, layer, device=device, max_chunk=max_chunk)
    elif model_type == "whisper":
        from data.ssl.utils.whisper_feature_reader import WhisperFeatureReader
        reader = WhisperFeatureReader(whisper_root, whisper_name, layer, device=device)
    return SSLFeatureExtractor(reader)




def dump(reader, fname: str = "path/to/wav"):
    frames = soundfile.info(fname).frames
    feat = reader.get_feats(fname, frames)
    return feat


    
