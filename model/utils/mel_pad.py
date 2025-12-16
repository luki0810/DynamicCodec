import math
import torch
import torch.nn.functional as F
from typing import Tuple

def right_pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = -1) -> Tuple[torch.Tensor, int]:
    """
    Right-pad tensor along `dim` so that length is divisible by `multiple`.
    Return (x_pad, orig_len).
    """
    orig_len = x.size(dim)
    if multiple <= 1:
        return x, orig_len

    pad_len = (math.ceil(orig_len / multiple) * multiple) - orig_len
    if pad_len <= 0:
        return x, orig_len

    # F.pad uses (left, right) for last dim; for dim != -1 we permute
    if dim == -1 or dim == x.dim() - 1:
        return F.pad(x, (0, pad_len)), orig_len

    # generic path: move target dim to last
    perm = list(range(x.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x2 = x.permute(*perm)
    x2 = F.pad(x2, (0, pad_len))
    # invert perm
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    x2 = x2.permute(*inv)
    return x2, orig_len


def crop_to_length(x: torch.Tensor, length: int, dim: int = -1) -> torch.Tensor:
    if x.size(dim) == length:
        return x
    if dim == -1 or dim == x.dim() - 1:
        return x[..., :length]

    slicer = [slice(None)] * x.dim()
    slicer[dim] = slice(0, length)
    return x[tuple(slicer)]



def match_length_lastdim(x: torch.Tensor, target_len: int) -> torch.Tensor:
    cur = x.size(-1)
    if cur > target_len:
        return x[..., :target_len]
    if cur < target_len:
        return F.pad(x, (0, target_len - cur))
    return x