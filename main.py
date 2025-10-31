import torch
import numpy as np
import argbind
import sys
from pathlib import Path
import os

from model.utils.util import to_scalar, pretty_shape
from model.build import DynamicTask
from model.utils.dynamic_argbind_loader import load_config_for_argbind




def _dump_args(args, save_path):
    if save_path.exists():
        try:
            os.remove(save_path)
            print(f"[INFO] Removed existing file: {save_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {save_path}: {e}")      
    argbind.dump_args(args, save_path)

@argbind.bind(without_prefix=True)
def main(load_path: str = "conf/base.yaml", save_path: str = "runs/test/args.yaml"):
    # dynamic load with ${encoder}, ${decoder}, ${quantizer}
    # 这里的dynamic load相当于全部载入，不会检查argbind.unknown
    
    cfg = load_config_for_argbind(main_yaml=load_path)
    args = argbind.parse_args(argv=sys.argv)
    args.update(cfg)
    _dump_args(args=args, save_path=Path(save_path))

    
    # seed
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
    # show all args
    print("--- Loaded Arguments ---")
    print("Loaded arguments:")
    for k, v in args.items():
        print(f"{k}: {v}")
    print("------------------------")

    # dynamice bind operation
    with argbind.scope(args):
        model = DynamicTask.build_model()

    # to device
    device = torch.device("cuda" if (args['device'] == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device).eval()

    # dummy data
    sr = args['sample_rate']
    T = int(sr * args['duration_sec'])
    B = args['batch_size']
    dummy = torch.randn(B, 1, T, device=device)


    # inference
    with torch.no_grad():
        out = model(dummy, sample_rate=sr)


    # output result
    print("\n=== DynamicCodec Test Run ===")
    print(f"device           : {device}")
    print(f"sample_rate      : {sr}")
    if hasattr(model, 'hop_length'):
        print(f"hop_length       : {model.hop_length}")
    if hasattr(model, 'latent_dim'):
        print(f"latent_dim       : {model.latent_dim}")

    print(pretty_shape("input audio", dummy))
    print(pretty_shape("recon audio", out.get("audio")))
    print(pretty_shape("z", out.get("z")))
    print(pretty_shape("codes", out.get("codes")))
    print(pretty_shape("latents", out.get("latents")))
    
    loss_dict = out["loss"]
    for name, value in loss_dict.items():
        print(pretty_shape(name, value))
    print("===========================\n")


if __name__ == "__main__":
    main()
