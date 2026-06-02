"""Probe whether DynamicTask.build_model + a single forward pass succeeds for
various (input_format, encoder, quantizer, decoder, vocoder) combinations.

This is a "does it instantiate and run a forward" check — NOT a full training
validation. A pass here means: dimensions are compatible, code paths exist,
forward returns a tensor without crashing. It does NOT mean training will
converge or that pretrained weights load successfully.

Each combo runs in a fresh subprocess so argbind/import state can't bleed
between probes.

Run from repo root inside the dyc_dev container:
    python scripts/probe_components.py
"""
from __future__ import annotations

import subprocess
import sys


COMBOS = [
    # baseline
    ("wav", "dac", "rvq", "dac", None),
    # other quantizers on dac
    ("wav", "dac", "vq", "dac", None),
    ("wav", "dac", "bsq", "dac", None),
    ("wav", "dac", "fsq", "dac", None),
    # other wav encoders/decoders
    ("wav", "encodec", "rvq", "encodec", None),
    # mel path (CPU is slow, give it more time)
    ("melspec", "mel", "rvq", "mel", None),
    ("melspec", "mel", "rvq", "mel", "vocos"),
    # ssl repr path — requires SSL checkpoint at ckpt/{data2vec,hubert,whisper}/...
    # Probe will report whether it can at least import + resolve config; the
    # actual model load fails at probe time without those checkpoints.
    ("repr", "repcodec", "rvq", "dac", None),
    # cosmos is intentionally NOT registered (image-only, see model/all_choices.py)
    # and therefore not probed here.
]


WORKER = r"""
import sys, os, traceback
sys.path.insert(0, "/app")
os.chdir("/app")
import torch
import argbind
from model.build import DynamicTask
from model.utils.dynamic_argbind_loader import load_config_for_argbind

input_format, encoder, quantizer, decoder, vocoder = sys.argv[1:6]
vocoder = None if vocoder == "None" else vocoder

args = {}
yamls = [
    f"conf/input/{input_format}.yaml",
    f"conf/model/encoder/{encoder}.yaml",
    f"conf/model/decoder/{decoder}.yaml",
    f"conf/model/quantizer/{quantizer}.yaml",
]
if vocoder is not None:
    yamls.append(f"conf/model/vocoder/{vocoder}.yaml")
for y in yamls:
    args.update(load_config_for_argbind(main_yaml=y))
args["input_format"] = input_format
args["encoder"] = encoder
args["quantizer"] = quantizer
args["decoder"] = decoder
args["vocoder"] = vocoder
args.setdefault("sample_rate", 48000)
# probe runs on CPU; force any SSL feature extractor to CPU too so its
# weights live on the same device as our random input tensor.
args["ssl_model.device"] = "cpu"
args["device"] = "cpu"
# probe runs on CPU; force any SSL feature extractor to CPU too so its
# weights live on the same device as our random input tensor.
args["ssl_model.device"] = "cpu"
args["device"] = "cpu"

try:
    with argbind.scope(args):
        model = DynamicTask.build_model(args)
    sr = args.get("sample_rate", 48000)
    n = max(2048, int(sr * 0.05))
    x = torch.randn(1, 1, n)
    with torch.no_grad():
        out = model(x, sr)
    if isinstance(out, dict) and "audio" in out:
        print(f"__RESULT__OK audio {tuple(out['audio'].shape)}")
    else:
        print(f"__RESULT__OK output_type={type(out).__name__}")
except Exception as e:
    head = str(e).splitlines()[0][:200]
    print(f"__RESULT__FAIL {type(e).__name__}: {head}")
"""


def probe(combo):
    if_, e, q, d, v = combo
    label = f"{if_:>7} | {e:>8} + {q:>4} + {d:>8} + voc={v}"
    cmd = ["python", "-c", WORKER, if_, e, q, d, "None" if v is None else v]
    # mel-based combos do a lot of conv work on CPU; give them more headroom
    timeout_s = 300 if e == "mel" else 120
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        for line in r.stdout.splitlines():
            if line.startswith("__RESULT__"):
                tag = line.replace("__RESULT__", "")
                return f"{tag.split()[0]:<4} {label} -> {' '.join(tag.split()[1:])}"
        # no result token — print last stderr line
        err = (r.stderr or r.stdout).strip().splitlines()
        return f"FAIL {label} -> no result, last: {err[-1][:160] if err else '(empty)'}"
    except subprocess.TimeoutExpired:
        return f"FAIL {label} -> timeout ({timeout_s}s)"


def main():
    print(f"Probing {len(COMBOS)} component combinations on CPU "
          f"(random init, no ckpt; each in fresh subprocess)\n")
    for combo in COMBOS:
        print(probe(combo), flush=True)


if __name__ == "__main__":
    main()
