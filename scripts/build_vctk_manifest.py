"""Generate VCTK manifest CSVs for DynamicCodec training.

Speaker-Dependent split: every speaker contributes to all three sets.
Within each speaker, files are sorted by filename and split by index modulo 50:
    idx % 50 == 0 -> val   (~2 %)
    idx % 50 == 1 -> test  (~2 %)
    otherwise      -> train (~96 %)

This is fully deterministic — no random seed needed, no shuffling.

Output (relative to repo root):
    data/manifests/vctk/train.csv
    data/manifests/vctk/val.csv
    data/manifests/vctk/test.csv

Each CSV has a single column ``path`` containing absolute wav paths,
which is the format audiotools.AudioLoader expects.

The VCTK source directory is read-only — this script never writes anything
under it. Configure the source location via the VCTK_ROOT env var, or edit
the default below.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

VCTK_ROOT = Path(os.environ.get("VCTK_ROOT", "/path/to/vctk/wav48"))
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "manifests" / "vctk"

VAL_MOD = 0
TEST_MOD = 1
MOD = 50


def collect_speaker_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p"))


def collect_speaker_wavs(spk_dir: Path) -> list[Path]:
    return sorted(spk_dir.glob("*.wav"))


def split_by_index(wavs: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    train, val, test = [], [], []
    for idx, wav in enumerate(wavs):
        bucket = idx % MOD
        if bucket == VAL_MOD:
            val.append(wav)
        elif bucket == TEST_MOD:
            test.append(wav)
        else:
            train.append(wav)
    return train, val, test


def write_csv(path: Path, wavs: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path"])
        writer.writeheader()
        for w in wavs:
            writer.writerow({"path": str(w)})


def main() -> None:
    if not VCTK_ROOT.is_dir():
        raise SystemExit(f"VCTK root not found: {VCTK_ROOT}")

    spk_dirs = collect_speaker_dirs(VCTK_ROOT)
    if not spk_dirs:
        raise SystemExit(f"No speaker dirs (p*) under {VCTK_ROOT}")

    all_train: list[Path] = []
    all_val: list[Path] = []
    all_test: list[Path] = []

    per_spk_summary: list[tuple[str, int, int, int]] = []

    for spk_dir in spk_dirs:
        wavs = collect_speaker_wavs(spk_dir)
        if not wavs:
            print(f"[warn] {spk_dir.name}: no wavs, skipping")
            continue
        train, val, test = split_by_index(wavs)
        all_train.extend(train)
        all_val.extend(val)
        all_test.extend(test)
        per_spk_summary.append((spk_dir.name, len(train), len(val), len(test)))

    write_csv(OUT_DIR / "train.csv", all_train)
    write_csv(OUT_DIR / "val.csv", all_val)
    write_csv(OUT_DIR / "test.csv", all_test)

    total = len(all_train) + len(all_val) + len(all_test)
    print(f"speakers : {len(per_spk_summary)}")
    print(f"total wav: {total}")
    print(
        f"train    : {len(all_train)} ({len(all_train) / total * 100:.2f} %) -> {OUT_DIR / 'train.csv'}"
    )
    print(
        f"val      : {len(all_val)} ({len(all_val) / total * 100:.2f} %) -> {OUT_DIR / 'val.csv'}"
    )
    print(
        f"test     : {len(all_test)} ({len(all_test) / total * 100:.2f} %) -> {OUT_DIR / 'test.csv'}"
    )
    print()
    print("per-speaker (first 5 / last 2):")
    for name, t, v, te in (*per_spk_summary[:5], *per_spk_summary[-2:]):
        print(f"  {name}: train={t:4d}  val={v:3d}  test={te:3d}")


if __name__ == "__main__":
    main()
