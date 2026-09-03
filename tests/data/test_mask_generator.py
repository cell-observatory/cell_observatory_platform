import math
from typing import Optional

import pytest
import torch

from cell_observatory_platform.data.data_shapes import MULTICHANNEL_HYPERCUBE
from cell_observatory_platform.data.masking.mask_generator import MaskGenerator, MaskModes


_FORMAT_MAP = {"CTZYX": "TZYXC", "CTYX": "TYXC", "CZYX": "ZYXC"}

@pytest.fixture
def make_mask_generator():
    def _make(
        maskmode,
        layout,
        time_length: int,
        pattern: Optional[list] = None,
        input_channels: int = 2,
        spatial_shape: tuple = (16, 16, 16),
        temporal_patch_size: Optional[int] = 1,
        lateral_patch_size: int = 16,
        axial_patch_size: int = 16,
        random_masking_ratio: float = 0.5,
        lateral_mask_scale: float = 0.1,
        axial_mask_scale: float = 0.1,
        temporal_mask_scale: float = 0.1,
        aspect_ratio_scale_hw=0.1,
        num_blocks: int = 1,
        mask_unit_size: Optional[tuple] = None,
        q_stride: Optional[tuple] = None,
        q_pool: Optional[int] = None,
    ):
        fmt = layout.value
        has_T = "T" in fmt
        has_Z = "Z" in fmt

        if has_Z:
            assert len(spatial_shape) == 3, "3D layout expects spatial_shape=(Z,Y,X)"
            Z, Y, X = spatial_shape
        else:
            assert len(spatial_shape) == 2, "2D layout expects spatial_shape=(Y,X)"
            Z = 1
            Y, X = spatial_shape

        T = time_length if has_T else 1
        C = input_channels

        axis_size = {"C": C, "T": T, "Z": Z, "Y": Y, "X": X}
        input_shape = tuple(axis_size[a] for a in fmt)

        axis_patch = {
            "C": 1,
            "T": (temporal_patch_size if has_T and temporal_patch_size is not None else 1),
            "Z": (axial_patch_size if has_Z else 1),
            "Y": lateral_patch_size,
            "X": lateral_patch_size,
        }
        patch_shape = tuple(axis_patch[a] for a in fmt)
        input_format = _FORMAT_MAP.get(fmt, fmt)
        if fmt in ("CTZYX", "CTYX", "CZYX"):
            patch_shape = patch_shape[1:]

        def _to_range(v):
            return (v, v) if isinstance(v, (int, float)) else tuple(v)

        lateral_mask_scale_ = _to_range(lateral_mask_scale)
        axial_mask_scale_ = _to_range(axial_mask_scale)
        temporal_mask_scale_ = _to_range(temporal_mask_scale)
        aspect_ratio_scale_hw_ = _to_range(aspect_ratio_scale_hw)

        kwargs = dict(
            layout=layout,
            input_format=input_format,
            input_shape=input_shape,
            patch_shape=patch_shape,
            mask_mode=maskmode,
            time_downsample_pattern=pattern,
            random_masking_ratio=random_masking_ratio,
            lateral_mask_scale=lateral_mask_scale_,
            axial_mask_scale=axial_mask_scale_,
            temporal_mask_scale=temporal_mask_scale_,
            aspect_ratio_scale_hw=aspect_ratio_scale_hw_,
            num_blocks=num_blocks,
            device=torch.device("cpu"),
        )
        if mask_unit_size is not None:
            kwargs["mask_unit_size"] = mask_unit_size
        if q_stride is not None:
            kwargs["q_stride"] = q_stride
        if q_pool is not None:
            kwargs["q_pool"] = q_pool
        return MaskGenerator(**kwargs)

    return _make


@pytest.mark.parametrize(
    ("time_length", "pattern"),
    [
        (4, [0, 1, 0, 1]),
        (10, [1, 0, 1, 1, 1, 0]),
    ],
)
@pytest.mark.parametrize(
    ("batch_size", "layout"),
    [
        [1, MULTICHANNEL_HYPERCUBE.TYXC],
        [3, MULTICHANNEL_HYPERCUBE.TYXC],
        [1, MULTICHANNEL_HYPERCUBE.CTYX],
        [3, MULTICHANNEL_HYPERCUBE.CTYX],
        [1, MULTICHANNEL_HYPERCUBE.TZYXC],
        [3, MULTICHANNEL_HYPERCUBE.TZYXC],
        [1, MULTICHANNEL_HYPERCUBE.CTZYX],
        [3, MULTICHANNEL_HYPERCUBE.CTZYX],
    ],
)
def test_blocked_pattern_mask(
    make_mask_generator, time_length, pattern, batch_size, layout, maskmode=MaskModes.BLOCKED_PATTERNED
):
    """
    Validate BLOCKED_PATTERNED mode:
        - tests that the mask repeats `pattern` along the time axis.
    """
    if layout == MULTICHANNEL_HYPERCUBE.CTZYX or layout == MULTICHANNEL_HYPERCUBE.TZYXC:
        spatial_shape = (128, 128, 128)  # Z, Y, X
        temporal_patch_size = 1
        lateral_patch_size = 16
        axial_patch_size = 16
    elif layout == MULTICHANNEL_HYPERCUBE.TYXC or layout == MULTICHANNEL_HYPERCUBE.CTYX:
        spatial_shape = (128, 128)  # Y, X
        temporal_patch_size = 1
        lateral_patch_size = 16
        axial_patch_size = 16
    else:
        raise ValueError(f"Unknown layout {layout}")

    mg = make_mask_generator(
        time_length=time_length,
        spatial_shape=spatial_shape,
        temporal_patch_size=temporal_patch_size,
        lateral_patch_size=lateral_patch_size,
        axial_patch_size=axial_patch_size,
        pattern=pattern,
        layout=layout,
        maskmode=maskmode,
        aspect_ratio_scale_hw=(1.0, 1.0),
    )

    out = mg(batch_size=batch_size)
    masks = out["masks"]
    orig_idx = out["original_patch_indices"]

    expected_length = mg.time * mg.depth * mg.height * mg.width
    assert masks.shape == (batch_size, expected_length)

    pat_len = len(pattern)
    slice_len = mg.depth * mg.height * mg.width

    for b in range(batch_size):
        for t in range(mg.time):
            expected_val = pattern[t % pat_len]
            t_slice = masks[b, t * slice_len : (t + 1) * slice_len]
            unique_vals = torch.unique(t_slice)
            assert unique_vals.numel() == 1, "mixed values within a time slice"
            assert unique_vals.item() == expected_val, f"time {t}: expected {expected_val} but saw {unique_vals.item()}"


@pytest.mark.parametrize(
    ("batch_size", "layout"),
    [
        (1, MULTICHANNEL_HYPERCUBE.ZYXC),
        (3, MULTICHANNEL_HYPERCUBE.ZYXC),
        (1, MULTICHANNEL_HYPERCUBE.CZYX),
        (3, MULTICHANNEL_HYPERCUBE.CZYX),
        (1, MULTICHANNEL_HYPERCUBE.TYXC),
        (3, MULTICHANNEL_HYPERCUBE.TYXC),
        (1, MULTICHANNEL_HYPERCUBE.CTYX),
        (3, MULTICHANNEL_HYPERCUBE.CTYX),
        (1, MULTICHANNEL_HYPERCUBE.TZYXC),
        (1, MULTICHANNEL_HYPERCUBE.CTZYX),
        (3, MULTICHANNEL_HYPERCUBE.TZYXC),
        (3, MULTICHANNEL_HYPERCUBE.CTZYX),
    ],
)
@pytest.mark.parametrize("maskmode", [MaskModes.RANDOM, MaskModes.RANDOM_SPACE_ONLY])
@pytest.mark.parametrize("random_ratio", [0.3, 0.5, 0.7])
def test_random_mask(make_mask_generator, batch_size, layout, maskmode, random_ratio, time_length: int = 4):
    """
    Validate RANDOM masking modes:
        - ratio of target / context within theoretical bounds
        - ctx / tgt index sets match mask values and do not overlap
        - orig_idx is correct permutation of all patch positions
    """
    if layout == MULTICHANNEL_HYPERCUBE.CTZYX or layout == MULTICHANNEL_HYPERCUBE.TZYXC:
        spatial_shape = (128, 128, 128)  # Z, Y, X
        temporal_patch_size = 1
        lateral_patch_size = 16
        axial_patch_size = 16
    elif layout == MULTICHANNEL_HYPERCUBE.ZYXC or layout == MULTICHANNEL_HYPERCUBE.CZYX:
        spatial_shape = (128, 128, 128)  # Z, Y, X
        temporal_patch_size = None
        lateral_patch_size = 16
        axial_patch_size = 16
    elif layout == MULTICHANNEL_HYPERCUBE.TYXC or layout == MULTICHANNEL_HYPERCUBE.CTYX:
        spatial_shape = (128, 128)  # Y, X
        temporal_patch_size = 1
        lateral_patch_size = 16
        axial_patch_size = 16
    else:
        raise ValueError(f"Unknown layout {layout}")

    mg = make_mask_generator(
        spatial_shape=spatial_shape,
        temporal_patch_size=temporal_patch_size,
        lateral_patch_size=lateral_patch_size,
        axial_patch_size=axial_patch_size,
        maskmode=maskmode,
        layout=layout,
        time_length=time_length,
        random_masking_ratio=random_ratio,
        pattern=None,
        aspect_ratio_scale_hw=(1.0, 1.0),
    )

    out = mg(batch_size=batch_size)
    masks = out["masks"]
    ctx_idx = out["context_masks"]
    tgt_idx = out["target_masks"]
    orig_idx = out["original_patch_indices"]

    S = mg.depth * mg.height * mg.width
    total_len = mg.time * S

    if maskmode is MaskModes.RANDOM_SPACE_ONLY:
        ctx_per_slice = int(S * (1 - random_ratio))
        exp_ctx = ctx_per_slice * mg.time
    else:
        exp_ctx = int(total_len * (1 - random_ratio))

    exp_tgt = total_len - exp_ctx

    for b in range(batch_size):
        mask = masks[b]

        # ratio/counts
        zeros = (mask == 0).sum().item()
        ones = (mask == 1).sum().item()
        assert zeros == exp_ctx
        assert ones == exp_tgt

        # ctx / tgt index correctness
        assert ctx_idx[b].numel() == exp_ctx
        assert tgt_idx[b].numel() == exp_tgt
        assert torch.all(mask[ctx_idx[b]] == 0)
        assert torch.all(mask[tgt_idx[b]] == 1)

        # check no overlap between ctx and tgt
        assert torch.isin(ctx_idx[b], tgt_idx[b]).sum() == 0

        # orig_idx is correct permutation
        sorted_idx = torch.sort(orig_idx[b]).values
        assert torch.equal(sorted_idx, torch.arange(total_len))

        permuted = torch.cat([ctx_idx[b], tgt_idx[b]])
        reconstructed = permuted[orig_idx[b]]
        assert torch.equal(reconstructed, torch.arange(total_len)), "orig_idx is not a proper inverse permutation"


@pytest.mark.parametrize(
    ("batch_size", "layout"),
    [
        (1, MULTICHANNEL_HYPERCUBE.ZYXC),
        (1, MULTICHANNEL_HYPERCUBE.CZYX),
        (1, MULTICHANNEL_HYPERCUBE.TYXC),
        (1, MULTICHANNEL_HYPERCUBE.CTYX),
        (1, MULTICHANNEL_HYPERCUBE.TZYXC),
        (1, MULTICHANNEL_HYPERCUBE.CTZYX),
    ],
)
@pytest.mark.parametrize(
    ("maskmode", "temporal_scale"),
    [(MaskModes.BLOCKED, (0.2, 0.4)), (MaskModes.BLOCKED_SPACE_ONLY, (1.0, 1.0))],
)
def test_blocked_mask_properties(
    make_mask_generator, temporal_scale, batch_size, layout, maskmode, time_length: int = 16
):
    """
    Validate BLOCKED and BLOCKED_SPACE_ONLY masks:
    - ratio of target / context within theoretical bounds
    - ctx / tgt index sets match mask values and do not overlap
    - orig_idx is correct permutation of all patch positions
    """
    if layout == MULTICHANNEL_HYPERCUBE.CTZYX or layout == MULTICHANNEL_HYPERCUBE.TZYXC:
        spatial_shape = (128, 128, 128)  # Z, Y, X
        temporal_patch_size = 1
        lateral_patch_size = 16
        axial_patch_size = 16
    elif layout == MULTICHANNEL_HYPERCUBE.ZYXC or layout == MULTICHANNEL_HYPERCUBE.CZYX:
        spatial_shape = (128, 128, 128)  # Z, Y, X
        temporal_patch_size = None
        lateral_patch_size = 16
        axial_patch_size = 16
    elif layout == MULTICHANNEL_HYPERCUBE.TYXC or layout == MULTICHANNEL_HYPERCUBE.CTYX:
        spatial_shape = (128, 128)  # Y, X
        temporal_patch_size = 1
        lateral_patch_size = 16
        axial_patch_size = 16
    else:
        raise ValueError(f"Unknown layout {layout}")

    lateral_scale = (0.7, 0.75)
    axial_scale = (0.7, 0.75)

    mg = make_mask_generator(
        spatial_shape=spatial_shape,
        temporal_patch_size=temporal_patch_size,
        lateral_patch_size=lateral_patch_size,
        axial_patch_size=axial_patch_size,
        maskmode=maskmode,
        layout=layout,
        time_length=time_length,
        pattern=None,
        temporal_mask_scale=temporal_scale,
        lateral_mask_scale=lateral_scale,
        axial_mask_scale=axial_scale,
        aspect_ratio_scale_hw=(1.0, 1.0),
        num_blocks=1,
    )

    out = mg(batch_size=batch_size)
    masks = out["masks"]
    ctx_idx = out["context_masks"]
    tgt_idx = out["target_masks"]
    orig_idx = out["original_patch_indices"]

    t_lo, t_hi = mg.temporal_mask_scale
    a_lo, a_hi = mg.axial_mask_scale
    l_lo, l_hi = mg.lateral_mask_scale

    T, D, H, W = mg.time, mg.depth, mg.height, mg.width

    if T > 1:
        t_min = max(1, int((T * t_lo)))
        t_max = min(T, int((T * t_hi)))
        ts = list(range(t_min, t_max + 1))
    else:
        ts = [1]

    if D > 1:
        d_min = max(1, int((D * l_lo)))
        d_max = min(D, int((D * l_hi)))
        ds = list(range(d_min, d_max + 1))
    else:
        ds = [1]

    hw_lo = a_lo * H * W
    hw_hi = a_hi * H * W
    h_min = w_min = int(round(math.sqrt(hw_lo)))
    h_max = w_max = int(round(math.sqrt(hw_hi)))
    all_hw = [(h, w) for h in range(h_min, h_max + 1) for w in range(w_min, w_max + 1)]

    possible_volumes = {t * d * h * w for t in ts for d in ds for h, w in all_hw}
    min_vol, max_vol = min(possible_volumes), max(possible_volumes)
    min_ratio = min_vol / (T * D * H * W)
    max_ratio = max_vol / (T * D * H * W)

    tgt_per_sample = tgt_idx[0].numel()
    ratio = tgt_per_sample / (T * D * H * W)
    assert min_ratio <= ratio <= max_ratio, f"observed {ratio:.3f} not in [{min_ratio:.3f}, {max_ratio:.3f}]"

    total_len = mg.time * mg.depth * mg.height * mg.width
    for b in range(batch_size):
        mask = masks[b]

        # ctx / tgt counts (must equal the values in the mask)
        assert ctx_idx[b].numel() == (mask == 0).sum()
        assert tgt_idx[b].numel() == (mask == 1).sum()

        # indices point to correct values
        assert torch.all(mask[ctx_idx[b]] == 0)
        assert torch.all(mask[tgt_idx[b]] == 1)

        # no overlap between ctx and tgt
        assert torch.isin(ctx_idx[b], tgt_idx[b]).sum() == 0

        # orig_idx is a permutation of [0,..., L-1]
        assert torch.equal(torch.sort(orig_idx[b]).values, torch.arange(total_len))

        permuted = torch.cat([ctx_idx[b], tgt_idx[b]])
        reconstructed = permuted[orig_idx[b]]
        assert torch.equal(reconstructed, torch.arange(total_len)), "orig_idx is not a proper inverse permutation"

    grid_shape = (mg.time, mg.depth, mg.height, mg.width)
    for b in range(batch_size):
        mask_reshaped = masks[b].view(*grid_shape)
        tgt_pos = (mask_reshaped == 1).nonzero(as_tuple=False)

        # bounding box of all target indices
        t_min, d_min, h_min, w_min = tgt_pos.min(dim=0).values
        t_max, d_max, h_max, w_max = tgt_pos.max(dim=0).values

        # all positions inside that box must also be target
        box = mask_reshaped[t_min : t_max + 1, d_min : d_max + 1, h_min : h_max + 1, w_min : w_max + 1]

        assert torch.all(box == 1), "masked block is not a block"


def test_blocked_with_random_fill_partitions_patches(make_mask_generator):
    """Every sample is split into exactly round(N * ratio) target patches and the
    rest context; the index sets are disjoint, consistent with the 0/1 mask, and
    original_patch_indices is a permutation of all patches."""
    mg = make_mask_generator(
        maskmode=MaskModes.BLOCKED_WITH_RANDOM_FILL, layout=MULTICHANNEL_HYPERCUBE.TZYXC,
        time_length=4, spatial_shape=(16, 16, 16), lateral_patch_size=16, axial_patch_size=16)
    B, N = 2, mg.time * mg.depth * mg.height * mg.width
    n_tgt = round(N * mg.random_masking_ratio)
    out = mg(batch_size=B)
    assert out["masks"].shape == (B, N) and out["masks"].sum(1).tolist() == [n_tgt] * B
    assert out["context_masks"].shape == (B, N - n_tgt) and out["target_masks"].shape == (B, n_tgt)
    for b in range(B):
        ctx, tgt = out["context_masks"][b], out["target_masks"][b]
        assert not set(ctx.tolist()) & set(tgt.tolist())
        assert (out["masks"][b, tgt] == 1).all() and (out["masks"][b, ctx] == 0).all()
        assert sorted(out["original_patch_indices"][b].tolist()) == list(range(N))
    assert torch.equal(out["patches_used"], torch.arange(N).expand(B, N))
    assert out["mu_mask"] is None


def test_dino_ibot_masks_half_the_batch_at_ratio_range(make_mask_generator):
    """With the defaults (mask_probability 0.5, mask_ratio_range (0.1, 0.5)) one
    of two samples is masked, at the top of the ratio range; the flat index list,
    weights, count and upper bound are all derived from that one mask."""
    mg = make_mask_generator(
        maskmode=MaskModes.DINO_IBOT, layout=MULTICHANNEL_HYPERCUBE.TZYXC,
        time_length=4, spatial_shape=(16, 16, 16), lateral_patch_size=16, axial_patch_size=16)
    B, N = 2, mg.time * mg.depth * mg.height * mg.width
    out = mg(batch_size=B)
    cm = out["collated_masks"]
    assert cm.shape == (B, N) and cm.dtype == torch.bool
    per_sample = cm.sum(1)
    n_masked = int(B * mg.mask_probability)                           # 1 of 2 samples
    assert (per_sample > 0).sum().item() == n_masked
    assert per_sample.max().item() == int(N * mg.mask_ratio_range[1])   # linspace top end for 1 sample
    assert torch.equal(out["mask_indices_list"], cm.flatten().nonzero().flatten())
    assert out["n_masked_patches"].item() == cm.sum().item() == out["upperbound"]
    torch.testing.assert_close(out["masks_weight"],
                               torch.full((int(cm.sum()),), 1.0 / per_sample.max().item()))


def test_hiera_mu_masks_whole_mask_units(make_mask_generator):
    """Masking happens per mask unit: the patch-level mask is constant inside each
    MU, kept MUs are context, and (with one token per MU) tgt_tok_idx lists
    exactly the masked MUs."""
    mg = make_mask_generator(
        maskmode=MaskModes.HIERA_MU, layout=MULTICHANNEL_HYPERCUBE.TZYXC, time_length=4,
        spatial_shape=(16, 16, 16), lateral_patch_size=8, axial_patch_size=8,
        mask_unit_size=(1, 2, 2, 2), q_stride=(1, 2, 2, 2), q_pool=1)
    B = 2
    T, D, H, W = mg.time, mg.depth, mg.height, mg.width              # (4, 2, 2, 2)
    n_mu, mu_flat = T, D * H * W                                     # 4 MUs of 8 patches
    n_tgt = round(n_mu * mg.random_masking_ratio)
    out = mg(batch_size=B)
    mu_mask, keep = out["mu_mask"], out["mu_keep_idx"]
    assert mu_mask.shape == (B, n_mu) and mu_mask.dtype == torch.bool
    assert mu_mask.sum(1).tolist() == [n_tgt] * B
    assert keep.shape == (B, n_mu - n_tgt)
    for b in range(B):
        assert not mu_mask[b, keep[b]].any()                          # kept MUs are context
        per_mu = out["masks"][b].view(n_mu, mu_flat)                  # patch grid is T-major
        assert torch.equal(per_mu.all(1), mu_mask[b]) and torch.equal(per_mu.any(1), mu_mask[b])
        assert sorted(out["tgt_tok_idx"][b].tolist()) == mu_mask[b].nonzero().flatten().tolist()


def test_fresh_generators_replay_the_same_stream(make_mask_generator):
    """step() seeds the block-SIZE draw from a per-instance counter that starts at
    -1; block placement uses the global torch RNG (V-JEPA style). With the global
    RNG pinned, two fresh generators therefore produce identical first masks, and
    a second call on one of them (step seed advanced 0 -> 1) produces a different mask."""
    def make():
        return make_mask_generator(
            maskmode=MaskModes.BLOCKED, layout=MULTICHANNEL_HYPERCUBE.TZYXC, time_length=8,
            spatial_shape=(128, 128, 128), lateral_patch_size=16, axial_patch_size=16,
            temporal_mask_scale=(0.3, 0.6), axial_mask_scale=(0.3, 0.6),
            lateral_mask_scale=(0.3, 0.6), aspect_ratio_scale_hw=(1.0, 1.0))
    a, b = make(), make()
    torch.manual_seed(0)
    first_a = a(batch_size=2)["masks"]
    torch.manual_seed(0)
    first_b = b(batch_size=2)["masks"]
    assert first_a.any()                                  # a non-trivial block was drawn
    assert torch.equal(first_a, first_b)                 # same step() seed on first call
    torch.manual_seed(0)
    second_a = a(batch_size=2)["masks"]
    assert not torch.equal(second_a, first_a)            # only the step seed changed -> new block size
