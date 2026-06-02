"""Patch audiotools.ml.layers.base to load checkpoints with weights_only=False.

PyTorch 2.6 flipped the default of `torch.load(..., weights_only=)` from False to
True. audiotools 0.7.2 still calls `torch.load` without specifying it, so any
checkpoint that pickles audiotools/argbind objects (which is every checkpoint
this project produces) refuses to load.

The DAC checkpoint at runs/dac-result/best/ was produced by us, so disabling
weights_only here is safe.

Idempotent: skips files that already contain the patched fragment.

The audiotools module location is discovered dynamically, so this script works
across base images (miniforge in the Tencent image, conda env in the public
PyTorch image, system Python, virtualenvs, etc).
"""

from pathlib import Path
import sys


def locate_target() -> Path:
    try:
        import audiotools.ml.layers.base as alb
    except ImportError:
        sys.exit("audiotools.ml.layers.base is not importable in this Python environment")
    target = Path(alb.__file__)
    if not target.is_file():
        sys.exit(f"audiotools.ml.layers.base.__file__ points at {target} which is not a file")
    return target


TARGET = locate_target()

REPLACEMENTS = [
    # primary weights load (line ~172)
    (
        'model_dict = torch.load(location, "cpu")',
        'model_dict = torch.load(location, "cpu", weights_only=False)',
    ),
    # extra-data load: optimizer.pth, scheduler.pth, tracker.pth, metadata.pth (~line 326)
    (
        "extra_data[f.name] = torch.load(f, **kwargs)",
        "extra_data[f.name] = torch.load(f, weights_only=False, **kwargs)",
    ),
]


def main() -> None:
    text = TARGET.read_text()
    changed = False
    for old, new in REPLACEMENTS:
        if new in text:
            print(f"[skip] already patched: {old!r}")
            continue
        if old not in text:
            sys.exit(f"expected fragment not found: {old!r}")
        text = text.replace(old, new, 1)
        changed = True
        print(f"[ok] patched: {old!r}")
    if changed:
        TARGET.write_text(text)
        print(f"[ok] wrote {TARGET}")
    else:
        print("[noop] nothing to do")


if __name__ == "__main__":
    main()
