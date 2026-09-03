"""`Normalize`: per-sample (per-channel) z-scoring, padding-mask-aware statistics,
Form-D target roles sharing the input's affine frame."""

import pytest
import torch

from cell_observatory_platform.data.data_shapes import MULTICHANNEL_HYPERCUBE
from cell_observatory_platform.data.transforms.normalize import Normalize


def _sample(data: torch.Tensor, has_time: bool = False, **meta) -> dict:
    return {
        "data_tensor": data,
        "metainfo": {"data_types": {"data_tensor": {"has_time": has_time}}, **meta},
    }


def _layout(value: str = "ZYXC") -> MULTICHANNEL_HYPERCUBE:
    return MULTICHANNEL_HYPERCUBE(value)


# ---------------------------------------------------------------------------
# Plain z-scoring
# ---------------------------------------------------------------------------


def test_channel_first_layout_normalizes_each_channel():
    """CZYX: every (sample, channel) slab is z-scored independently, so a channel
    on a wildly different scale ends up with zero mean / unit std like the others."""
    x = torch.rand(2, 3, 4, 4, 4)                          # (B, C, Z, Y, X)
    x[:, 1] = x[:, 1] * 1000 + 7                            # channel 1 on another scale
    out = Normalize(input_layout=_layout("CZYX"))(x)
    torch.testing.assert_close(out.mean(dim=(2, 3, 4)), torch.zeros(2, 3), atol=1e-4, rtol=0)
    torch.testing.assert_close(out.std(dim=(2, 3, 4)), torch.ones(2, 3), atol=1e-3, rtol=0)


def test_unbatched_tensor_is_treated_as_one_sample():
    """An unbatched (Z, Y, X, C) tensor is promoted to batch size one and z-scored
    over all of its voxels."""
    x = torch.rand(4, 4, 4, 1) * 10 + 3
    out = Normalize(input_layout=_layout())(x)
    assert out.shape == (1, 4, 4, 4, 1)
    torch.testing.assert_close(out, ((x - x.mean()) / x.std().clamp_min(1e-4)).unsqueeze(0))


def test_constant_input_normalizes_to_zero_not_nan():
    """A constant sample has std 0; the eps clamp keeps the output finite (all
    zeros) instead of NaN."""
    out = Normalize(input_layout=_layout())(torch.full((1, 4, 4, 4, 1), 9.0))
    assert torch.equal(out, torch.zeros_like(out))          # std clamped to eps, numerator exactly 0


# ---------------------------------------------------------------------------
# Padding-mask-aware statistics
# ---------------------------------------------------------------------------


class TestNormalizePaddingMask:
    def test_masked_stats_match_hand_masked_reference(self):
        """Padded voxels (True in padding_mask) are excluded from the per-sample
        moments; samples without padding get the plain z-score."""
        torch.manual_seed(0)
        B, Z, Y, X, C = 2, 4, 6, 6, 1
        x = torch.rand(B, Z, Y, X, C) * 100
        pm = torch.zeros(B, Z, Y, X, dtype=torch.bool)
        pm[1, :, 3:, :] = True                       # sample 1: half padded
        x[1][pm[1]] = 0.0                            # padded voxels are zeros

        out = Normalize(_layout())(
            {"data_tensor": x.clone(), "metainfo": {"padding_mask": pm}}
        )["data_tensor"]

        # hand-masked reference for sample 1 (valid region only, unbiased std)
        valid_vals = x[1][~pm[1]]
        mean_ref, std_ref = valid_vals.mean(), valid_vals.std()
        torch.testing.assert_close(
            out[1][~pm[1]], (x[1][~pm[1]] - mean_ref) / std_ref.clamp_min(1e-4),
            rtol=1e-5, atol=1e-5,
        )
        # sample 0 (no padding) must equal the plain per-sample z-score
        mean0, std0 = x[0].mean(), x[0].std()
        torch.testing.assert_close(
            out[0], (x[0] - mean0) / std0.clamp_min(1e-4), rtol=1e-5, atol=1e-5
        )

    def test_no_mask_bit_parity_with_plain_path(self):
        """A metainfo without padding_mask takes exactly the same path as a bare
        dict sample."""
        torch.manual_seed(1)
        x = torch.rand(2, 4, 4, 4, 1)
        with_meta = Normalize(_layout())({"data_tensor": x.clone(), "metainfo": {}})
        plain = Normalize(_layout())({"data_tensor": x.clone()})
        torch.testing.assert_close(with_meta["data_tensor"], plain["data_tensor"])

    def test_all_false_mask_equals_unmasked(self):
        """An all-False padding mask (nothing padded) reproduces the unmasked result."""
        torch.manual_seed(2)
        x = torch.rand(2, 4, 4, 4, 1)
        pm = torch.zeros(2, 4, 4, 4, dtype=torch.bool)
        masked = Normalize(_layout())(
            {"data_tensor": x.clone(), "metainfo": {"padding_mask": pm}}
        )["data_tensor"]
        plain = Normalize(_layout())({"data_tensor": x.clone()})["data_tensor"]
        torch.testing.assert_close(masked, plain)


# ---------------------------------------------------------------------------
# Form-D target roles normalized with the input's statistics
# ---------------------------------------------------------------------------


class TestNormalizeTargetRoles:
    def test_input_is_z_scored_per_sample(self):
        """Each sample is z-scored with its OWN statistics (pooled stats would leave
        sample 0 far from zero mean when sample 1 is on another scale); the named
        target keeps the input's shape."""
        g = torch.Generator().manual_seed(0)
        clean = torch.rand((2, 4, 4, 4, 1), generator=g) * 100
        noisy = clean.clone()
        noisy[1] = noisy[1] * 100 + 50                     # sample 1 on a very different scale
        out = Normalize(input_layout=_layout(), normalize_target_roles=["denoising"])(
            _sample(noisy.clone(), targets={"denoising": clean.clone()}))
        x = out["data_tensor"]
        torch.testing.assert_close(x.mean(dim=(1, 2, 3)), torch.zeros(2, 1), atol=1e-5, rtol=0)
        torch.testing.assert_close(x.std(dim=(1, 2, 3)), torch.ones(2, 1), atol=1e-4, rtol=0)
        assert out["metainfo"]["targets"]["denoising"].shape == clean.shape

    def test_input_and_target_share_one_affine_frame(self):
        """The denoising contract: normalize(clean) - normalize(noisy) is the scaled
        residual (clean - noisy) / std, with no mean or scale gap."""
        g = torch.Generator().manual_seed(1)
        clean = torch.rand((1, 4, 4, 4, 1), generator=g) * 1000
        noise = torch.randn((1, 4, 4, 4, 1), generator=g)
        noisy = clean + noise
        n = Normalize(input_layout=_layout(), normalize_target_roles=["denoising"])
        out = n(_sample(noisy.clone(), targets={"denoising": clean.clone()}))
        std = noisy.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
        torch.testing.assert_close(
            out["metainfo"]["targets"]["denoising"] - out["data_tensor"],
            (clean - noisy) / std,
        )

    def test_form_s_targets_raise(self):
        """Form-S (per-sample List[Dict]) GT must never be z-scored."""
        n = Normalize(input_layout=_layout(), normalize_target_roles=["denoising"])
        s = _sample(torch.rand(1, 4, 4, 4, 1), targets=[{"boxes": torch.zeros(0, 6)}])
        with pytest.raises(ValueError, match="never be z-scored"):
            n(s)

    def test_missing_targets_raise(self):
        """normalize_target_roles without any targets points at the transform order."""
        n = Normalize(input_layout=_layout(), normalize_target_roles=["denoising"])
        with pytest.raises(ValueError, match="DeepCopyInputsAsTargets BEFORE"):
            n(_sample(torch.rand(1, 4, 4, 4, 1)))

    def test_unknown_role_raises(self):
        """A role name absent from the targets is a KeyError naming the role."""
        n = Normalize(input_layout=_layout(), normalize_target_roles=["typo"])
        s = _sample(torch.rand(1, 4, 4, 4, 1),
                    targets={"denoising": torch.rand(1, 4, 4, 4, 1)})
        with pytest.raises(KeyError, match="typo"):
            n(s)

    def test_unnamed_roles_pass_through_untouched(self):
        """Only the roles NAMED in normalize_target_roles are normalized."""
        clean = torch.rand(1, 4, 4, 4, 1)
        other = torch.rand(1, 4, 4, 4, 1)
        n = Normalize(input_layout=_layout(), normalize_target_roles=["denoising"])
        out = n(_sample(torch.rand(1, 4, 4, 4, 1),
                        targets={"denoising": clean.clone(), "other": other}))
        assert out["metainfo"]["targets"]["other"] is other

    def test_shape_mismatch_raises(self):
        """A named role whose shape differs from the input cannot share its
        per-sample stats and is rejected."""
        n = Normalize(input_layout=_layout(), normalize_target_roles=["denoising"])
        s = _sample(torch.rand(1, 4, 4, 4, 1),
                    targets={"denoising": torch.rand(1, 2, 4, 4, 1)})
        with pytest.raises(ValueError, match="per-sample stats"):
            n(s)

    def test_option_off_leaves_targets_untouched(self):
        """Without normalize_target_roles the targets object is passed through by identity."""
        clean = torch.rand(1, 4, 4, 4, 1)
        n = Normalize(input_layout=_layout())
        out = n(_sample(torch.rand(1, 4, 4, 4, 1), targets={"denoising": clean}))
        assert out["metainfo"]["targets"]["denoising"] is clean
