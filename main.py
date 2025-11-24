import torch
import numpy as np
import argbind
import sys
from pathlib import Path
import os
from audiotools import AudioSignal
import soundfile

from data.ssl.repr import ssl_model
from model.utils.util import to_scalar, pretty_shape
from model.build import DynamicTask
from model.utils.dynamic_argbind_loader import load_config_for_argbind

def out_print(model, out):
    # output result
    print("\n=== DynamicCodec Test Run ===")
    print(f"device           : {model.device}")
    print(f"sample_rate      : {model.sample_rate}")
    if hasattr(model, 'hop_length'):
        print(f"hop_length       : {model.hop_length}")
    if hasattr(model, 'latent_dim'):
        print(f"latent_dim       : {model.latent_dim}")


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
            print(f"[INFO] Removed existing file: {save_path}")
        except Exception as e:
            print(f"[WARN] Could not remove {save_path}: {e}")      
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
    _dump_args(args=args, save_path=argpath)

    
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
        print(f"[INFO] Loaded model from {kwargs['folder']}")
    else:
        # dynamice build
        print("[WARN] No resume load specified, using randomly initialized model.")
        with argbind.scope(args):
            model = DynamicTask.build_model()
    model.to(device)
        
        
    #input file
    fname = 'wav_file/input_wav/p226_002.wav'
    
    
    # input_format
    input_format = args['input_format']
    if input_format == 'repr':
        # get_ssl_model
        with argbind.scope(args):
            reader = ssl_model()
        # inference ssl
        frames = soundfile.info(fname).frames
        feat = reader.get_feats(fname, frames)
        feat = feat.unsqueeze(0)
        feat = feat.transpose(1, 2).to(device)
        # inference other
        model.eval()
        with torch.no_grad():
            out = model(feat)
        
    elif input_format == 'wav':
        model.to(device)
        # input
        signal = AudioSignal(fname)
        signal = signal.to_mono() # to single
        signal.to(model.device)
        model.eval()
        with torch.no_grad():
            out = model(signal.audio_data, signal.sample_rate)
            
            
    elif input_format == 'melspec':
        pass
        # TODO: melspectrogram to code to waveform

        
    # output    
    out_print(model, out)
    # save output audio
    recon_audio = out["audio"].squeeze().cpu().numpy()
    soundfile.write(Path(save_path)/ "recon.wav", recon_audio, sample_rate)
    print(f"[INFO] Saved reconstructed audio to {Path(save_path)/ 'recon.wav'}")
    
    


if __name__ == "__main__":
    main()
