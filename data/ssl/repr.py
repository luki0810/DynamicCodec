import logging
import os
import sys
import soundfile
# from model.ssl.utils.feature_utils import get_path_iterator, dump_feature

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
    return reader

def dump(reader, fname: str = "path/to/wav"):
    frames = soundfile.info(fname).frames
    feat = reader.get_feats(fname, frames)
    return feat
