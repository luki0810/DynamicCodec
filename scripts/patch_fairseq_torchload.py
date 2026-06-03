"""Patch fairseq.checkpoint_utils so torch.load runs with weights_only=False.

PyTorch 2.6 flipped the default of `torch.load(..., weights_only=)` from False
to True. fairseq 0.12.2 still calls `torch.load` without specifying it, so SSL
checkpoints (HuBERT etc.) that pickle fairseq config classes refuse to load
with: `UnpicklingError: Weights only load failed`.

The SSL checkpoints in ckpt/{hubert,data2vec}/ are released by their authors
and pulled into this repo by hand, so disabling weights_only here is the same
trust trade-off the audiotools patch makes.

Idempotent: skips files that already contain the patched fragment.
"""

from pathlib import Path
import sys


def locate_target() -> Path:
    try:
        import fairseq.checkpoint_utils as fc
    except ImportError:
        sys.exit("fairseq.checkpoint_utils is not importable in this Python environment")
    target = Path(fc.__file__)
    if not target.is_file():
        sys.exit(f"fairseq.checkpoint_utils.__file__ points at {target} which is not a file")
    return target


TARGET = locate_target()

REPLACEMENTS = [
    # load_checkpoint_to_cpu: single-line call (fairseq 0.12.2 ~line 315)
    (
        'state = torch.load(f, map_location=torch.device("cpu"))',
        'state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)',
    ),
    # multi-line call in _torch_persistent_save / load_ema variant (~line 878)
    (
        '        new_state = torch.load(\n'
        '            f,\n'
        '            map_location=(\n'
        '                lambda s, _: torch.serialization.default_restore_location(s, "cpu")\n'
        '            ),\n'
        '        )',
        '        new_state = torch.load(\n'
        '            f,\n'
        '            map_location=(\n'
        '                lambda s, _: torch.serialization.default_restore_location(s, "cpu")\n'
        '            ),\n'
        '            weights_only=False,\n'
        '        )',
    ),
]


def main() -> None:
    text = TARGET.read_text()
    changed = False
    for old, new in REPLACEMENTS:
        if new in text:
            print(f"[skip] already patched: {old.splitlines()[0]!r}")
            continue
        if old not in text:
            sys.exit(f"expected fragment not found: {old.splitlines()[0]!r}")
        text = text.replace(old, new, 1)
        changed = True
        print(f"[ok] patched: {old.splitlines()[0]!r}")
    if changed:
        TARGET.write_text(text)
        print(f"[ok] wrote {TARGET}")
    else:
        print("[noop] nothing to do")


if __name__ == "__main__":
    main()
