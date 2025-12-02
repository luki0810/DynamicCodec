import soundfile
from data.ssl.feature_extractor import HubertFeatureExtractor, Data2vecFeatureExtractor, WhisperFeatureExtractor



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
    model_type = model_type.lower()
    if model_type == "hubert":
        return HubertFeatureExtractor(
            ckpt_path=ckpt_path,
            layer=layer,
            device=device,
            max_chunk=max_chunk,
        )
    elif model_type == "data2vec":
        return Data2vecFeatureExtractor(
            ckpt_path=ckpt_path,
            layer=layer,
            device=device,
            max_chunk=max_chunk,
        )
    elif model_type == "whisper":
        return WhisperFeatureExtractor(
            root=whisper_root,
            ckpt=whisper_name or ckpt_path,
            layer=layer,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        




def dump(reader, fname: str = "path/to/wav"):
    frames = soundfile.info(fname).frames
    feat = reader.get_feats(fname, frames)
    return feat

