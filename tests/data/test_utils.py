"""`data/utils.py`: centred FFT with pad/crop, NA-mask construction, spectral
band-limiting (`downsample`) and mask broadcasting (`resize_mask`).

Inputs are chosen so the exact output is known in closed form (unit impulses,
flat kernels, all-pass / DC-only masks)."""

import numpy as np
import pytest
import torch

from cell_observatory_platform.data.utils import create_na_masks, downsample, fft, resize_mask


def _delta(shape, axes):
    """Unit impulse at n//2 on `axes` (ifftshift maps n//2 -> 0, so its spectrum is flat)."""
    x = torch.zeros(shape, dtype=torch.float32)
    idx = [slice(None)] * len(shape)
    for ax in axes:
        idx[ax] = shape[ax] // 2
    x[tuple(idx)] = 1.0
    return x


def _view_mask(mask, x, spatial_dims):
    """Broadcast a spatial mask to `x.ndim` with singleton non-spatial dims."""
    view = [1] * x.ndim
    for i, d in enumerate(spatial_dims):
        view[d] = mask.shape[i]
    return mask.view(view)


# ---------- fft


@pytest.mark.parametrize("shape,axes", [((4, 6, 8), (1, 2)), ((2, 3, 5, 7), (2, 3)), ((3, 9, 9), (1, 2))])
def test_fft_of_centered_delta_is_flat(shape, axes):
    """fft is centred on both sides: an impulse at n//2 transforms to a flat
    all-ones spectrum."""
    out = fft(_delta(shape, axes), axes=axes, pad_to=None)
    assert out.shape == tuple(shape)
    torch.testing.assert_close(out, torch.ones(shape, dtype=out.dtype))


@pytest.mark.parametrize("shape,axes,target", [
    ((1, 10, 12), (1, 2), (8, 14)),   # crop Y, pad X
    ((2, 7, 7), (1, 2), 5),           # scalar target, crop both
    ((3, 6, 9), (1, 2), (10, 11)),    # pad both (even and odd)
])
def test_fft_pad_and_crop_keep_the_delta_centered(shape, axes, target):
    """Centre-crop and zero-pad both keep the n//2 sample at the new n//2, so the
    impulse's spectrum stays flat at the target size."""
    out = fft(_delta(shape, axes), axes=axes, pad_to=target)
    tgt = (int(target),) * len(axes) if np.isscalar(target) else tuple(target)
    exp = list(shape)
    for ax, t in zip(axes, tgt):
        exp[ax] = t
    assert out.shape == tuple(exp)
    torch.testing.assert_close(out, torch.ones(exp, dtype=out.dtype))


def test_fft_rejects_bad_pad_tuple_len():
    with pytest.raises(ValueError):
        fft(torch.zeros((4, 6, 8)), axes=(1, 2), pad_to=(8,))


# ---------- create_na_masks


def test_create_na_masks_delta_psf_passes_every_frequency():
    """A delta PSF has |OTF| == 1 everywhere, so every threshold in [0, 1] keeps
    every frequency, with and without resizing."""
    ipsf = torch.zeros(9, 9, 9)
    ipsf[4, 4, 4] = 1.0
    msks = create_na_masks(ipsf, thresholds=[0.3, 1.0], target_shape=None, resize=False)
    assert msks.shape == (2, 9, 9, 9) and msks.dtype == torch.float32 and (msks == 1).all()
    resized = create_na_masks(ipsf, thresholds=[0.5], target_shape=(4, 4, 4), resize=True)
    assert resized.shape == (1, 4, 4, 4) and (resized == 1).all()


def test_create_na_masks_flat_psf_keeps_only_dc():
    """A flat PSF concentrates all energy at DC; the mask keeps exactly that one
    (centred, fftshifted) voxel per threshold."""
    msks = create_na_masks(torch.ones(5, 5, 5), thresholds=[0.5, 0.01], target_shape=None, resize=False)
    assert msks.sum().item() == 2
    assert msks[0, 2, 2, 2].item() == 1 and msks[1, 2, 2, 2].item() == 1   # fftshift puts DC at n//2


@pytest.mark.parametrize("thr", [-0.1, 1.5])
def test_create_na_masks_rejects_threshold_outside_unit_interval(thr):
    with pytest.raises(ValueError, match="outside"):
        create_na_masks(torch.ones(3, 3, 3), thresholds=[thr], target_shape=None, resize=False)


# ---------- downsample


@pytest.mark.parametrize("shape,spatial_dims", [((2, 2, 16, 16), (2, 3)), ((2, 1, 8, 8, 8), (2, 3, 4))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_downsample_all_pass_mask_is_identity(shape, spatial_dims, dtype):
    """An all-ones mask is the identity; bf16 inputs go through float32 and come
    back in bf16."""
    x = torch.randn(shape).to(dtype)
    mask = _view_mask(torch.ones([shape[d] for d in spatial_dims]), x, spatial_dims)
    y = downsample(na_mask=mask, inputs=x, spatial_dims=spatial_dims)
    assert y.dtype == dtype
    torch.testing.assert_close(y, x)


def test_downsample_dc_only_mask_returns_spatial_mean():
    """Keeping only the (centred) DC bin returns each sample's spatial mean everywhere."""
    x = torch.randn(2, 1, 8, 8)
    mask = torch.zeros(8, 8)
    mask[4, 4] = 1.0                                   # DC bin after fftshift
    y = downsample(na_mask=_view_mask(mask, x, (2, 3)), inputs=x, spatial_dims=(2, 3))
    torch.testing.assert_close(y, x.mean(dim=(2, 3), keepdim=True).expand_as(x))


def test_downsample_per_sample_matches_batched():
    """The memory-saving per-sample loop produces the batched result."""
    x = torch.randn(3, 1, 8, 8, 8)
    m = _view_mask((torch.rand(8, 8, 8) > 0.5).float(), x, (2, 3, 4))
    torch.testing.assert_close(downsample(m, x, (2, 3, 4), per_sample_computation=True),
                               downsample(m, x, (2, 3, 4)))


# ---------- resize_mask


def test_resize_mask_broadcasts_over_layout_axes():
    """A mask already at the spatial shape is broadcast (not resampled) over the
    C and T axes of the requested layout."""
    mask = (torch.rand(4, 8, 8) > 0.5).float()
    out = resize_mask(mask, "ZYXC", channels=2, axial_shape=4, lateral_shape=(8, 8))
    assert out.shape == (4, 8, 8, 2)
    assert torch.equal(out[..., 0], mask) and torch.equal(out[..., 1], mask)
    out_t = resize_mask(mask, "TZYXC", channels=3, timepoints=5, axial_shape=4, lateral_shape=(8, 8))
    assert out_t.shape == (5, 4, 8, 8, 3) and torch.equal(out_t[2, ..., 1], mask)


def test_resize_mask_resamples_when_shape_differs():
    """A mask whose shape differs from the target is resampled (and re-binarised)
    before broadcasting; the requested dtype is applied."""
    out = resize_mask(torch.ones(4, 8, 8), "ZYXC", channels=1, axial_shape=2, lateral_shape=(4, 4), dtype=torch.bool)
    assert out.shape == (2, 4, 4, 1) and out.dtype == torch.bool and out.all()


def test_resize_mask_broadcasts_zyx_over_time_and_channels():
    """A (Z, Y, X) mask is tiled unchanged across every timepoint and channel of a TZYXC tensor."""
    Z, Y, X, T, C = 4, 6, 8, 3, 2
    spatial = torch.zeros(Z, Y, X)
    spatial[1, 2, 3] = 1.0
    mask = resize_mask(
        spatial, input_format="TZYXC", channels=C, timepoints=T,
        axial_shape=Z, lateral_shape=(Y, X), dtype=torch.float32, device=torch.device("cpu"),
    )
    assert mask.shape == (T, Z, Y, X, C)
    for t in range(T):
        for c in range(C):
            assert torch.equal(mask[t, ..., c], spatial)
    assert mask.sum().item() == T * C
