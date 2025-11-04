import logging
import os
import sys
import soundfile
from model.ssl.utils.hubert_feature_reader import HubertFeatureReader
from model.ssl.utils.feature_utils import get_path_iterator, dump_feature
from model.ssl.utils.data2vec_feature_reader import Data2vecFeatureReader
from model.ssl.utils.whisper_feature_reader import WhisperFeatureReader
import argbind

@argbind.bind()
def ssl_model(
    model_type: str = 'hubert',
    ckpt_path: str = 'ckpt/to/file',
    device: str = "cuda",
    layer: int = 18,
    max_chunk: int = 1600000,
    use_cpu: bool = False
):
    device = "cpu" if use_cpu else device
    reader = None
    if model_type == "hubert":
        reader = HubertFeatureReader(ckpt_path, layer, device=device, max_chunk=max_chunk)
        
    return reader

def dump(reader, fname: str = "path/to/wav"):
    frames = soundfile.info(fname).frames
    feat = reader.get_feats(fname, frames)
    return feat
