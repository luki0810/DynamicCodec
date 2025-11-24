import argbind
import subprocess

@argbind.bind()
def manifest(
    dataset_path: str = "path/to/audio_folder",
    manifest_path: str = "path/to/save",
    ext: str = "wav",
    valid_percent: float = 0
):
    cmd = [
        "python", "model/ssl/utils/wav2vec_manifest.py",
        dataset_path,
        "--dest", manifest_path,
        "--ext", ext,
        "--valid-percent", str(valid_percent)
    ]

    print("🚀 Running command (generate manifest tsv file):")
    print(" ".join(cmd))

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    
@argbind.bind()
def dump_feature(
    model_type: str = "hubert",
    tsv_path: str = "path/to/train.tsv",
    ckpt_path: str = "path/to/model",
    layer: int = 18,
    feat_dir: str = "wav_file/ssl_repr",
    device: str = "cuda:0"
):
    cmd = [
        "python3", "model/ssl/utils/dump_feature.py",
        "--model_type", model_type,
        "--tsv_path", tsv_path,
        "--ckpt_path", ckpt_path,
        "--layer", str(layer),
        "--feat_dir", feat_dir,
        "--device", device
    ]

    print("🚀 Running command (dump feature via ssl)")
    print(" ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("------ subprocess error! ------")
        print(f"(stderr):\n{e.stderr}")
    print(result.stdout)
    return feat_dir
    
def dump():
    manifest()
    dump_feature()
    
    
if __name__ == "__main__":
    dump()