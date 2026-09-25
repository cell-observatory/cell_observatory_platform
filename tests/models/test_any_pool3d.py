"""`any_pool3d` must be an exact, allocation-cheaper stand-in for the max-pool form.

The low-res correction loop replaced `F.max_pool3d(gt.float(), k, stride=k) > 0` with `any_pool3d(gt, k)`
at two sites (`models/meta_arch/sam.py` prompt prep and `training/losses.py`
IoU target) to avoid a ~9 GiB fp32 transient at training volumes. That swap is
only safe if the two spellings agree bit-for-bit, INCLUDING the floor semantics
for spatial dims that are not a multiple of the kernel, which is where a naive
reshape silently mis-groups cells.

CPU-only, tiny volumes.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cell_observatory_platform.models.ops.pooling import any_pool3d


def _reference(x: torch.Tensor, k) -> torch.Tensor:
    """The pre-change spelling these call sites used."""
    kk = (k, k, k) if isinstance(k, int) else tuple(k)
    return F.max_pool3d(x.float(), kernel_size=kk, stride=kk) > 0


@pytest.mark.parametrize("shape", [(1, 1, 8, 16, 16), (2, 1, 8, 8, 12), (3, 4, 4, 4)])
@pytest.mark.parametrize("k", [2, 4])
def test_matches_max_pool_on_divisible_shapes(shape, k):
    torch.manual_seed(0)
    x = torch.rand(shape) > 0.7
    if any(d % k for d in shape[-3:]):
        pytest.skip("covered by the non-divisible test")
    got, want = any_pool3d(x, k), _reference(x, k)
    assert got.dtype is torch.bool
    assert got.shape == want.shape
    assert torch.equal(got, want)


@pytest.mark.parametrize("shape", [(1, 1, 9, 17, 18), (2, 1, 6, 10, 14), (1, 1, 5, 5, 5)])
def test_matches_max_pool_floor_semantics_on_non_divisible_shapes(shape):
    """Trailing voxels that do not fill a whole cell are DROPPED, as max_pool3d does."""
    torch.manual_seed(1)
    k = 4
    x = torch.rand(shape) > 0.5
    got, want = any_pool3d(x, k), _reference(x, k)
    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    assert torch.equal(got, want)
    # and the output really is the floor grid
    assert tuple(got.shape[-3:]) == tuple(d // k for d in shape[-3:])


def test_a_lone_foreground_voxel_is_detected_in_every_cell_position():
    """Guards the cell grouping: a single True must light up exactly its own cell."""
    k = 2
    for pos in range(8):
        x = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)
        dz, dy, dx = (pos // 4) % 2, (pos // 2) % 2, pos % 2
        x[0, 0, 2 + dz, 2 + dy, 2 + dx] = True
        got = any_pool3d(x, k)
        assert torch.equal(got, _reference(x, k))
        assert got[0, 0, 1, 1, 1].item() is True
        assert got.sum().item() == 1


def test_non_bool_input_uses_the_gt_zero_predicate():
    torch.manual_seed(2)
    x = torch.randn(2, 1, 8, 8, 8)
    assert torch.equal(any_pool3d(x, 4), _reference(x, 4))


def test_anisotropic_kernel_and_leading_dims_preserved():
    torch.manual_seed(3)
    x = torch.rand(2, 3, 8, 8, 16) > 0.6
    k = (4, 2, 8)
    got = any_pool3d(x, k)
    assert torch.equal(got, _reference(x, k))
    assert got.shape == (2, 3, 2, 4, 2)


def test_all_false_and_all_true_volumes():
    x = torch.zeros(1, 1, 8, 8, 8, dtype=torch.bool)
    assert not any_pool3d(x, 4).any()
    assert any_pool3d(~x, 4).all()


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        any_pool3d(torch.zeros(4, 4), 2)          # < 3 dims
    with pytest.raises(ValueError):
        any_pool3d(torch.zeros(1, 4, 4, 4), 0)    # non-positive kernel
    with pytest.raises(ValueError):
        any_pool3d(torch.zeros(1, 4, 4, 4), (2, 2))  # not a 3-sequence
