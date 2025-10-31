import torch

def to_scalar(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if torch.is_tensor(x):
        if x.ndim == 0:
            return x.item()
        if x.ndim == 1 and x.numel() == 1:
            return x.squeeze().item()
    return x

def pretty_shape(name, x): 
    if x is None: 
        return f"{name}: None" 
    if isinstance(x, (int, float)): 
        return f"{name}: {x}" 
    try: 
        return f"{name}: {tuple(x.shape)}" 
    except Exception: 
        return f"{name}: (unshaped)"
