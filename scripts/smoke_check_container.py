"""Smoke check that the container's environment matches what main.py / train.py
expect. Invoked by scripts/setup_container.sh; safe to run on its own:

    sudo docker exec dyc_luki python /app/scripts/smoke_check_container.py
"""

from __future__ import annotations

import inspect
import sys


def check_argbind_patch() -> None:
    import argbind

    sig = inspect.signature(argbind.parse_args)
    if "argv" not in sig.parameters:
        sys.exit(f"argbind patch missing: {sig}")
    print(f"  argbind.parse_args  : {sig}")


def check_audiotools_patch() -> None:
    import audiotools.ml.layers.base as alb

    src = inspect.getsource(alb)
    if "weights_only=False" not in src:
        sys.exit("audiotools torch.load patch missing (weights_only=False not found)")
    print("  audiotools torch.load: weights_only=False applied")


def check_manifest_loadable() -> None:
    from audiotools.data.datasets import AudioLoader

    loader = AudioLoader(sources=["data/manifests/vctk/val.csv"], shuffle=False)
    n = sum(len(s) for s in loader.audio_lists)
    if n == 0:
        sys.exit("val.csv loaded but contains no entries")
    sample = loader.audio_lists[0][0]["path"]
    print(f"  AudioLoader val.csv : {n} files, head={sample}")


def main() -> None:
    check_argbind_patch()
    check_audiotools_patch()
    check_manifest_loadable()
    print("  ✓ smoke check passed")


if __name__ == "__main__":
    main()
