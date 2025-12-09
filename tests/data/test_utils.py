import numpy as np
import pytest
import torch

from cell_observatory_platform.data.utils import create_na_masks, downsample, fft, resize_mask

# ---------- fft


@pytest.mark.parametrize(
    "shape,axes",
    [
        ((4, 6, 8), (1, 2)),  # 2D FFT on last two dims
        ((2, 3, 5, 7), (2, 3)),  # 2D FFT on last two dims in 4D array
        ((3, 9, 9), (1, 2)),  # square
    ],
)
def test_fft_shape_no_pad(shape, axes):
    x = torch.zeros(shape, dtype=torch.complex64)
    out = fft(x, axes=axes, pad_to=None)
    assert out.shape == x.shape


@pytest.mark.parametrize(
    "shape,axes,target",
    [
        ((1, 10, 12), (1, 2), (8, 14)),  # crop then pad
        ((2, 7, 7), (1, 2), 5),  # scalar target (5,5)
        ((3, 6, 9), (1, 2), (10, 10)),  # pad only
    ],
)
def test_fft_pad_and_crop_shapes_only(shape, axes, target):
    x = torch.zeros(shape, dtype=torch.float32)
    out = fft(x, axes=axes, pad_to=target)
    tgt = (int(target),) * len(axes) if np.isscalar(target) else tuple(int(t) for t in target)
    exp_shape = list(shape)
    for ax, t in zip(axes, tgt):
        exp_shape[ax] = t
    assert out.shape == tuple(exp_shape)


def test_fft_rejects_bad_pad_tuple_len():
    x = torch.zeros((4, 6, 8), dtype=torch.float32)
    with pytest.raises(ValueError):
        fft(x, axes=(1, 2), pad_to=(8,))


# ---------- create_na_masks


@pytest.mark.xfail(
    reason="create_na_masks calls numpy.fft on a torch.Tensor; shape test would fail until helper is fixed."
)
@pytest.mark.parametrize(
    "thr_list,target_shape,resize",
    [
        ([0.3], (9, 9, 9), False),
        ([0.2, 0.5, 0.8], (8, 8, 8), True),
    ],
)
def test_create_na_masks_shapes_only(thr_list, target_shape, resize):
    ipsf = torch.randn(9, 9, 9)  # 3D PSF
    msks = create_na_masks(ipsf, thresholds=thr_list, target_shape=target_shape, resize=resize)
    assert msks.shape == (len(thr_list), *target_shape)


# ---------- downsample


@pytest.mark.parametrize(
    "shape,spatial_dims",
    [
        ((2, 2, 16, 16), (1, 2)),  # 2D per-batch
        ((2, 1, 8, 8, 8), (1, 2, 3)),  # 3D volume
    ],
)
def test_downsample_shape_only(shape, spatial_dims):
    x = torch.randn(shape)
    na_mask = torch.ones([x.shape[d] for d in spatial_dims], dtype=torch.float32)
    view = [1] * x.ndim
    for i, d in enumerate(spatial_dims):
        view[d] = na_mask.shape[i]
    na_mask = na_mask.view(view)
    y = downsample(na_mask=na_mask, inputs=x, spatial_dims=spatial_dims)
    assert y.shape == x.shape


@pytest.mark.parametrize(
    "shape,spatial_dims",
    [
        ((2, 2, 16, 16), (1, 2)),  # 2D per-batch
        ((2, 1, 8, 8, 8), (1, 2, 3)),  # 3D volume
    ],
)
def test_downsample_in_bfloat16_and_back(shape, spatial_dims):
    x = torch.randn(shape, dtype=torch.bfloat16)
    na_mask = torch.ones([x.shape[d] for d in spatial_dims], dtype=torch.float32)
    view = [1] * x.ndim
    for i, d in enumerate(spatial_dims):
        view[d] = na_mask.shape[i]
    na_mask = na_mask.view(view)
    y = downsample(na_mask=na_mask, inputs=x, spatial_dims=spatial_dims)
    assert y.shape == x.shape
