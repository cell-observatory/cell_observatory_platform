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
    print(out_a[0])
    print(out_b[0])
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
    batch = {"data_tensor": inputs.clone(), "metainfo": {"targets": [targets.clone()]}}

    out = noise(batch)

    assert out["data_tensor"].shape == inputs.shape, "Output data tensor shape is not the same as input shape"
    assert torch.allclose(out["metainfo"]["targets"][0], targets), "Output targets are not the same as input targets"
    assert torch.all(out["data_tensor"] == 2.0), "Output data tensor is not the same as expected background"

