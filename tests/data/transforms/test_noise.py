from unittest.mock import patch

import pytest
import torch

from cell_observatory_platform.data.transforms.noise import MixedPoissonGaussianNoise


def test_mixed_poisson_gaussian_noise_reproducible_with_seed():
    inputs = torch.full((2, 1, 4, 4), 12.0)
    kwargs = dict(
        quantum_efficiency=0.62,
        electrons_per_count=2.5,
        sigma_background_noise=0.2,
        mean_background_offset=4.0,
        seed=123,
    )
    noise_a = MixedPoissonGaussianNoise(**kwargs)
    noise_b = MixedPoissonGaussianNoise(**kwargs)
    in_a = inputs.clone()
    in_b = inputs.clone()
    out_a = noise_a(in_a)
    out_b = noise_b(in_b)

    assert torch.allclose(out_a, out_b), "Noise is not reproducible with seed"
    assert out_a.shape == inputs.shape, "Output shape is not the same as input shape"
    assert torch.any(out_a != inputs), "Output is not different from input"
    assert torch.all((out_a >= 0) & (out_a <= 65535)), "Output is not in valid range [0, 65535]"

def test_mixed_poisson_gaussian_noise_tuple_parameters():
    inputs = torch.full((2, 1, 4, 4), 12.0)
    kwargs = dict(
        quantum_efficiency=(0.62, 0.82),
        electrons_per_count=(2.5, 3.5),
        sigma_background_noise=(0.2, 0.4),
        mean_background_offset=(4.0, 6.0),
        seed=123,
    )
    noise_a = MixedPoissonGaussianNoise(**kwargs)
    noise_b = MixedPoissonGaussianNoise(**kwargs)
    in_a = inputs.clone()
    in_b = inputs.clone()
    out_a = noise_a(in_a)
    out_b = noise_b(in_b)

    assert torch.any(out_a[0] != out_a[1]), "Noise is not different for different batch elements"
    assert torch.allclose(out_a, out_b), "Noise is not reproducible with tuple parameters"
    assert out_a.shape == inputs.shape, "Output shape is not the same as input shape"
    assert torch.any(out_a != inputs), "Noise not added; output is not different from input"
    assert torch.all((out_a >= 0) & (out_a <= 65535)), "Output is not in valid range [0, 65535]"

def test_mixed_poisson_gaussian_noise_returns_background_for_zero_signal():
    inputs = torch.zeros((1, 1, 2, 2))
    noise = MixedPoissonGaussianNoise(
        quantum_efficiency=1.0,
        electrons_per_count=1.0,
        sigma_background_noise=0.0,
        mean_background_offset=7.0,
        seed=0,
    )

    out = noise(inputs)

    assert torch.allclose(out, torch.full_like(inputs, 7.0)), "Output is not the same as expected background"


def test_mixed_poisson_gaussian_noise_updates_data_tensor_in_dict_only():
    inputs = torch.zeros((1, 1, 3, 3))
    targets = torch.ones((1, 1, 3, 3))
    noise = MixedPoissonGaussianNoise(
        quantum_efficiency=1.0,
        electrons_per_count=1.0,
        sigma_background_noise=0.0,
        mean_background_offset=2.0,
        seed=0,
    )
    batch = {"data_tensor": inputs.clone(), "metainfo": {"targets": {"denoising": targets.clone()}}}

    out = noise(batch)

    assert out["data_tensor"].shape == inputs.shape, "Output data tensor shape is not the same as input shape"
    assert torch.allclose(out["metainfo"]["targets"]["denoising"], targets), "Output targets are not the same as input targets"
    assert torch.all(out["data_tensor"] == 2.0), "Output data tensor is not the same as expected background"


def test_noise_does_not_alias_the_float32_input():
    """A float32 input is copied before the in-place sensor pipeline, so the
    caller's tensor (a clean target or a reused buffer slot) is never mutated."""
    x = torch.full((1, 1, 3, 3), 50.0, dtype=torch.float32)
    before = x.clone()
    out = MixedPoissonGaussianNoise(quantum_efficiency=1.0, electrons_per_count=1.0,
                                    sigma_background_noise=0.0, mean_background_offset=5.0, seed=0)(x)
    assert out is not x and torch.equal(x, before)
    assert out.dtype == torch.float32


def test_list_ranges_sample_like_tuple_ranges():
    """Parameter ranges given as lists (as Hydra delivers them) are sampled
    exactly like the equivalent tuples."""
    x = torch.full((4, 1, 4, 4), 30.0)
    kw = dict(electrons_per_count=1.0, sigma_background_noise=0.0, mean_background_offset=0.0, seed=7)
    as_tuple = MixedPoissonGaussianNoise(quantum_efficiency=(0.5, 0.9), **kw)(x.clone())
    as_list = MixedPoissonGaussianNoise(quantum_efficiency=[0.5, 0.9], **kw)(x.clone())
    torch.testing.assert_close(as_list, as_tuple)


def _rank_seeded_noise():
    return MixedPoissonGaussianNoise(
        quantum_efficiency=0.8, electrons_per_count=0.2,
        sigma_background_noise=2.0, mean_background_offset=100.0, seed=1234,
    )


def test_same_rank_replays_same_stream():
    """Two instances built with the same seed on the same rank draw the same noise."""
    x = torch.full((1, 4, 4, 4, 1), 500.0)
    a = _rank_seeded_noise()._add_noise(x.clone())
    b = _rank_seeded_noise()._add_noise(x.clone())
    torch.testing.assert_close(a, b)


def test_ranks_draw_decorrelated_streams():
    """The process rank is folded into the seed, so DDP replicas sharing one
    configured seed still draw different noise."""
    x = torch.full((1, 4, 4, 4, 1), 500.0)
    with patch(
        "cell_observatory_platform.utils.context.process_rank", return_value=0
    ):
        a = _rank_seeded_noise()._add_noise(x.clone())
    with patch(
        "cell_observatory_platform.utils.context.process_rank", return_value=1
    ):
        b = _rank_seeded_noise()._add_noise(x.clone())
    assert not torch.equal(a, b), "rank must decorrelate the noise stream"


def test_mixed_poisson_gaussian_noise_rejects_non_numeric_quantum_efficiency():
    """quantum_efficiency must be a float or a (lo, hi) tuple; anything else fails at construction."""
    with pytest.raises(ValueError, match="quantum_efficiency must be a float or tuple of two floats"):
        MixedPoissonGaussianNoise(
            quantum_efficiency="invalid", electrons_per_count=0.22,
            sigma_background_noise=40.0, mean_background_offset=100.0, seed=42,
        )
