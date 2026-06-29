"""einops_norms.py

Vector-norm reductions for tensors, expressed with einops patterns.

These helpers let you compute Lp norms over named axes using the familiar
einops ``reduce`` pattern syntax, while delegating the actual computation to
``torch.linalg.vector_norm`` -- which already handles numerical stability and
the inf / -inf / 0 edge cases for you.

The axes that *disappear* between the left and right side of the pattern are
the ones reduced over. When several axes are reduced at once, the norm is taken
over all of them jointly (as if those axes were flattened together).

Example:
-------
>>> import torch
>>> from einops_norms import l1, l2, linf, lp
>>> x = torch.randn(8, 64, 32, 32)          # b c h w
>>> l2(x, 'b c h w -> b h w').shape         # L2 over channels
torch.Size([8, 32, 32])
>>> linf(x, 'b c h w -> b').shape           # L-inf over c, h, w jointly
torch.Size([8])
>>> lp(x, 'b c h w -> b c', p=3).shape      # L3 over the spatial axes
torch.Size([8, 64])
"""

from __future__ import annotations

from typing import Union

import torch
from einops import reduce

__all__ = ["lp", "l1", "l2", "linf"]

Number = Union[int, float]


def _vector_norm_reduction(ord: Number):
    """Build an einops reduction callable that applies ``vector_norm`` with ``ord``.

    einops invokes the returned callable as ``f(tensor, reduced_axes)``, where
    ``reduced_axes`` is the collection of axis indices to collapse. We forward
    those straight to ``torch.linalg.vector_norm`` as ``dim``.
    """

    def reduction(tensor: torch.Tensor, axes) -> torch.Tensor:
        return torch.linalg.vector_norm(tensor, ord=ord, dim=tuple(axes))

    return reduction


def lp(tensor: torch.Tensor, pattern: str, p: Number, **axes_lengths) -> torch.Tensor:
    """Reduce ``tensor`` along the axes dropped by ``pattern`` using the Lp norm.

    Parameters
    ----------
    tensor:
        Input tensor.
    pattern:
        einops reduction pattern, e.g. ``'b c h w -> b h w'``. Axes present on
        the left but absent on the right are reduced over.
    p:
        Order of the norm. Accepts any value ``torch.linalg.vector_norm`` does,
        including ``float('inf')``, ``float('-inf')`` and ``0``.
    **axes_lengths:
        Optional explicit axis sizes, forwarded to ``einops.reduce``.

    Returns:
    -------
    torch.Tensor
        Tensor with the reduced axes removed.
    """
    return reduce(tensor, pattern, _vector_norm_reduction(p), **axes_lengths)


def l1(tensor: torch.Tensor, pattern: str, **axes_lengths) -> torch.Tensor:
    """Sum of absolute values along the reduced axes (Manhattan / taxicab norm)."""
    return lp(tensor, pattern, 1, **axes_lengths)


def l2(tensor: torch.Tensor, pattern: str, **axes_lengths) -> torch.Tensor:
    """Euclidean norm along the reduced axes."""
    return lp(tensor, pattern, 2, **axes_lengths)


def linf(tensor: torch.Tensor, pattern: str, **axes_lengths) -> torch.Tensor:
    """Maximum absolute value along the reduced axes (Chebyshev / sup norm)."""
    return lp(tensor, pattern, float("inf"), **axes_lengths)


# if __name__ == "__main__":
#     # Smoke test: verify the einops-based helpers agree with a direct
#     # torch.linalg.vector_norm call over the equivalent dim.
#     torch.manual_seed(0)
#     x = torch.randn(8, 64, 32, 32)  # b c h w

#     checks = {
#         "l1  (over c)": (l1(x, "b c h w -> b h w"),
#                          torch.linalg.vector_norm(x, ord=1, dim=1)),
#         "l2  (over c)": (l2(x, "b c h w -> b h w"),
#                          torch.linalg.vector_norm(x, ord=2, dim=1)),
#         "linf(over c)": (linf(x, "b c h w -> b h w"),
#                          torch.linalg.vector_norm(x, ord=float("inf"), dim=1)),
#         "lp=3 (h, w)":  (lp(x, "b c h w -> b c", p=3),
#                          torch.linalg.vector_norm(x, ord=3, dim=(2, 3))),
#         "l2 (c, h, w)": (l2(x, "b c h w -> b"),
#                          torch.linalg.vector_norm(x, ord=2, dim=(1, 2, 3))),
#     }

#     for name, (got, want) in checks.items():
#         ok = torch.allclose(got, want, atol=1e-5)
#         print(f"{name:>14}  shape={tuple(got.shape)}  matches_torch={ok}")
