import torch
import numpy as np
import argbind
import sys
from pathlib import Path
import os
from audiotools import AudioSignal
import soundfile


from model.utils.util import pretty_shape
from model.build import DynamicTask
from model.utils.dynamic_argbind_loader import load_config_for_argbind

import logging
logger = logging.getLogger("main")


def out_print(model, out, input):
    # output result
    print("\n=== DynamicCodec Test Run ===")
    print(f"device           : {model.device}")
    print(f"sample_rate      : {model.sample_rate}")
    if hasattr(model, 'hop_length'):
        print(f"hop_length       : {model.hop_length}")
    if hasattr(model, 'latent_dim'):
        print(f"latent_dim       : {model.latent_dim}")

    print(pretty_shape("input audio", input))
    print(pretty_shape("recon audio", out.get("audio")))
    print(pretty_shape("z", out.get("z")))
    print(pretty_shape("codes", out.get("codes")))
    print(pretty_shape("latents", out.get("latents")))
    
    loss_dict = out["loss"]
    for name, value in loss_dict.items():
        print(pretty_shape(name, value))
    print("===========================\n")

def _dump_args(args, save_path):
    if save_path.exists():
        try:
            os.remove(save_path)
            logger.info(f"[INFO] Removed existing file: {save_path}")
        except Exception as e:
            logger.info(f"[WARN] Could not remove {save_path}: {e}")      
    argbind.dump_args(args, save_path)

@argbind.bind(without_prefix=True)
def main(load_path: str = None, save_path: str = None):
    # dynamic load with ${encoder}, ${decoder}, ${quantizer}
    # 这里的dynamic load相当于全部载入，不会检查argbind.unknown
    cli = argbind.parse_args(argv=sys.argv)
    load_path = cli.get("load_path", load_path)
    save_path = cli.get("save_path", save_path)
    cfg = load_config_for_argbind(main_yaml=load_path)
    args = argbind.parse_args(argv=sys.argv)
    args.update(cfg)
    argpath = Path(save_path)/ "args.yaml"
    _dump_args(args=args, save_path=argpath) # 可以到save_path查看当前使用的args.yaml
  
    # seed
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # args
    sample_rate = args["sample_rate"]
    device = args["device"]
        
    # resume load
    if args.get('resume', False):
        exp_name = args.get('exp_name', None)
        pr_path = os.path.dirname(save_path)
        tag = args.get('tag', 'best')
        kwargs = {
            "folder": f"{pr_path}/{exp_name}/{tag}",
            "map_location": "cpu",
            "package": False
            # package === load full training state
        }
        if (Path(kwargs["folder"]) / "dynamiccodec").exists():
            model, model_extra = DynamicTask.load_from_folder(**kwargs)    
        logger.info(f"[INFO] Loaded model from {kwargs['folder']}")
    else:
        # dynamice build
        logger.info("[WARN] No resume load specified, using randomly initialized model.")
        with argbind.scope(args):
            model = DynamicTask.build_model(args=args)
    model.to(device)
        
        
    #input file
    fname = 'wav_file/input_wav/p226_002.wav'
    signal = AudioSignal(fname)
    signal = signal.to_mono()
    signal.to(model.device)
    model.eval()
    with torch.no_grad():
        out = model(signal.audio_data, signal.sample_rate)


        
    # output    
    out_print(model, out, signal.audio_data)
    
    if args['save'] is True:
        # save output audio
        recon_audio = out["audio"].squeeze().cpu().numpy()
        soundfile.write(Path(save_path)/ "recon.wav", recon_audio, sample_rate)
        logger.info(f"Saved reconstructed audio to {Path(save_path)/ 'recon.wav'}")
    


if __name__ == "__main__":
    main()
