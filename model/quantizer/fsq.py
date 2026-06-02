"""Finite Scalar Quantization (FSQ).

Reference: "Finite Scalar Quantization: VQ-VAE Made Simple" — Mentzer et al. 2023
https://arxiv.org/abs/2309.15505

FSQ replaces the learned codebook of VQ-VAE with a fixed grid of points
defined by per-dimension level counts. Compared to RVQ/VQ:

- No learned codebook → no auxiliary commitment / codebook losses needed
- Codebook size = product(levels) (e.g. levels=[8,5,5,5] -> 1000 entries)
- Each spatial position gets a single integer index in [0, prod(levels))

The interface matches the rest of model/quantizer/* so it can be a drop-in
replacement: forward(z) -> (z_q, indices, z_e, loss_dict, other).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from model.utils.abs_class import AbsQuantizer
from model.nn.dac.layers import WNConv1d


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Differentiable rounding via straight-through estimator."""
    return x + (x.round() - x).detach()


class FiniteScalarQuantize(AbsQuantizer):
    """
    Parameters
    ----------
    latent_dim : int
        Channel dim of the encoder output (input to the quantizer).
    levels : list[int]
        Per-dimension level counts. The code dim equals ``len(levels)`` and the
        total codebook size is ``prod(levels)``. Default ``[8, 5, 5, 5]`` gives
        a 1000-entry codebook (~10 bits / step), matching the FSQ paper's
        recommended setting for speech-scale problems.
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        levels: List[int] = [8, 5, 5, 5],
    ):
        super().__init__()
        if len(levels) == 0:
            raise ValueError("FSQ requires at least one level")
        if any(l < 2 for l in levels):
            raise ValueError(f"Each level must be >=2, got {levels}")
        self.latent_dim = latent_dim
        self.levels = list(levels)
        self.codebook_dim = len(self.levels)
        self.codebook_size = int(torch.tensor(self.levels).prod().item())

        # Per-dim half-range for tanh-based bounding: bounded ∈ [-(L-1)/2, (L-1)/2]
        self.register_buffer(
            "_half_l", torch.tensor([(l - 1) / 2 for l in self.levels])
        )
        # Stride for combining per-dim indices into a single integer index
        # (mixed-radix encoding so the result fits in a single int64 even when
        # different dims have different level counts).
        strides = [1]
        for l in self.levels[:-1]:
            strides.append(strides[-1] * l)
        self.register_buffer("_strides", torch.tensor(strides, dtype=torch.long))

        # Project encoder latent to / from the FSQ code space.
        self.in_proj = WNConv1d(latent_dim, self.codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(self.codebook_dim, latent_dim, kernel_size=1)

    # ---- core quantization ----
    def _bound(self, z_code: torch.Tensor) -> torch.Tensor:
        """Map any-real-valued z_code into [-half_l, +half_l] per dim via tanh."""
        # z_code: (B, C, T); _half_l: (C,)
        half = self._half_l.view(1, -1, 1)
        return half * torch.tanh(z_code)

    def _quantize(self, z_bounded: torch.Tensor) -> torch.Tensor:
        """Round to nearest integer on the FSQ grid with straight-through grads."""
        return _round_ste(z_bounded)

    def _codes_to_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        """Map quantized values (B, C, T) to a single integer index (B, T)."""
        # Shift each dim to [0, L-1], then mixed-radix encode.
        half = self._half_l.view(1, -1, 1)
        shifted = (z_q + half).round().long()  # (B, C, T) ∈ [0, L-1]
        strides = self._strides.view(1, -1, 1)
        return (shifted * strides).sum(dim=1)  # (B, T)

    # ---- public api ----
    def forward(self, z: torch.Tensor):
        """
        Parameters
        ----------
        z : Tensor[B, latent_dim, T]
        Returns
        -------
        z_q : Tensor[B, latent_dim, T]      quantized (after out_proj)
        indices : Tensor[B, T]              integer codebook index per step
        z_e : Tensor[B, codebook_dim, T]    pre-quantization code-space latent
        loss_dict : dict                    empty for FSQ (no aux losses)
        other : dict
        """
        z_e = self.in_proj(z)              # (B, C, T)
        z_bounded = self._bound(z_e)       # (B, C, T) in [-half_l, half_l]
        z_q_code = self._quantize(z_bounded)  # rounded with STE
        indices = self._codes_to_indices(z_q_code)
        z_q = self.out_proj(z_q_code)       # (B, latent_dim, T)

        loss_dict: dict = {}  # FSQ has no commitment / codebook loss
        other = {"levels": self.levels, "codebook_size": self.codebook_size}
        return z_q, indices, z_e, loss_dict, other

    # ---- index <-> code conversions, mirroring other quantizers ----
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of _codes_to_indices: (B, T) int -> (B, codebook_dim, T) values."""
        # indices: (B, T)
        b, t = indices.shape
        out = torch.zeros(b, self.codebook_dim, t, device=indices.device, dtype=torch.float32)
        rem = indices.clone()
        for i, l in enumerate(self.levels):
            digit = (rem % l).float() - (l - 1) / 2.0  # back to centered range
            out[:, i, :] = digit
            rem = torch.div(rem, l, rounding_mode="floor")
        return out

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Convenience: indices (B, T) -> decoded latent (B, latent_dim, T)."""
        z_q_code = self.indices_to_codes(indices)
        return self.out_proj(z_q_code)


# Public alias used by model/all_choices.py
FSQ = FiniteScalarQuantize


if __name__ == "__main__":
    fsq = FiniteScalarQuantize(latent_dim=64, levels=[8, 5, 5, 5])
    z = torch.randn(2, 64, 20)
    z_q, idx, z_e, loss, other = fsq(z)
    print("z_q:", z_q.shape, "indices:", idx.shape, "range:", idx.min().item(), idx.max().item())
    print("codebook_size:", other["codebook_size"])
    rec = fsq.get_codebook_entry(idx)
    print("get_codebook_entry:", rec.shape)
