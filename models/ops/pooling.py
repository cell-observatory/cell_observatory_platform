"""Dtype-preserving spatial pooling helpers for binary/label volumes."""

from typing import Sequence, Union

import torch


def _as_triple(kernel_size: Union[int, Sequence[int]]) -> tuple:
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size, kernel_size)
    k = tuple(int(v) for v in kernel_size)
    if len(k) != 3:
        raise ValueError(f"kernel_size must be an int or a 3-sequence, got {kernel_size!r}")
    return k


def any_pool3d(x: torch.Tensor, kernel_size: Union[int, Sequence[int]]) -> torch.Tensor:
    """``F.max_pool3d(x.float(), k, stride=k) > 0`` without the fp32 copy.

    "Is any voxel in the non-overlapping ``k`` cell foreground (``> 0``)?" --
    the reduction used to bring a full-res GT mask down to the stride-``k``
    grid the low-res mask stream is supervised on.

    The ``.float()`` in the max-pool spelling materializes a full fp32 copy of
    the input, which at training volumes (N=96, 25 M voxels) is a ~9 GiB
    transient replacing a 2.25 GiB bool tensor. Reshaping into cells and
    OR-reducing never leaves the input dtype and allocates only the (much
    smaller) reduction outputs.

    Args:
        x: ``(..., Z, Y, X)`` tensor, at least 3-D, any dtype. Leading dims are
            preserved untouched.
        kernel_size: int or 3-sequence. Also used as the stride, matching
            ``F.max_pool3d(..., kernel_size=k, stride=k)``.

    Returns:
        ``(..., Z // kz, Y // ky, X // kx)`` **bool** tensor, ``True`` where any
        voxel of the corresponding cell is ``> 0``. Callers that need the input
        dtype back should ``.to(x.dtype)`` (identical to the old
        ``(pooled > 0).to(dtype)``).

    Floor semantics: like ``F.max_pool3d`` with ``kernel_size == stride`` and no
    padding, trailing voxels that do not fill a whole cell are DROPPED. Each
    spatial dim is truncated to ``k * (dim // k)`` before the reshape so the
    cell grouping is byte-identical to the max-pool for non-divisible shapes
    too. (One deliberate difference: where a spatial dim is smaller than its
    kernel, ``F.max_pool3d`` raises and this returns an empty output dim.)

    Equivalence to the max-pool form is exact for any input without NaNs
    (``max(cell) > 0`` iff some element of the cell is ``> 0``). NaNs are the
    only divergence: ``max_pool3d`` may propagate a NaN and compare False,
    whereas here a positive sibling voxel still reports True. GT masks are
    bool/0-1 valued, so this cannot arise at the call sites.
    """
    if x.dim() < 3:
        raise ValueError(f"any_pool3d expects at least 3 dims, got shape {tuple(x.shape)}")
    kz, ky, kx = _as_triple(kernel_size)
    if kz < 1 or ky < 1 or kx < 1:
        raise ValueError(f"kernel_size must be positive, got {kernel_size!r}")

    z, y, xd = x.shape[-3:]
    nz, ny, nx = z // kz, y // ky, xd // kx
    if (nz * kz, ny * ky, nx * kx) != (z, y, xd):
        # reproduce max_pool3d's floor: drop the remainder voxels
        x = x[..., : nz * kz, : ny * ky, : nx * kx]

    # bool input is already exactly the ">0" predicate; skip the extra copy
    v = x if x.dtype == torch.bool else (x > 0)

    lead = v.shape[:-3]
    # (..., nz, kz, ny, ky, nx, kx) -> OR over the three kernel axes
    cells = v.reshape(*lead, nz, kz, ny, ky, nx, kx)
    return cells.any(-1).any(-2).any(-3)
