import math
from typing import Optional

import pytest

import torch

from data.masking.mask_generator import MaskGenerator, MaskModes
from data.data_shapes import MULTICHANNEL_HYPERCUBE


@pytest.fixture
def make_mask_generator():
    def _make(maskmode,
              layout,
              time_length: int,
              pattern: Optional[list] = None,
              input_channels: int = 2,
              spatial_shape: int = (16,16,16),
              temporal_patch_size: int = 1,
              lateral_patch_size: int = 16,
              axial_patch_size: int = 16,
              random_masking_ratio: float = 0.5,
              lateral_mask_scale: float = 0.1,
              axial_mask_scale: float = 0.1,
              temporal_mask_scale: float = 0.1,
              aspect_ratio_scale_hw: float = 0.1,
              num_blocks: int = 1
              ):
        if layout == MULTICHANNEL_HYPERCUBE.CTZYX or \
            layout == MULTICHANNEL_HYPERCUBE.CTYX:
            input_shape = (input_channels, time_length, *spatial_shape)
        elif layout == MULTICHANNEL_HYPERCUBE.TZYXC or \
            layout == MULTICHANNEL_HYPERCUBE.TYXC:
            input_shape = (time_length, *spatial_shape, input_channels)
        elif layout == MULTICHANNEL_HYPERCUBE.ZYXC:
            input_shape = (*spatial_shape, input_channels)
        elif layout == MULTICHANNEL_HYPERCUBE.CZYX:
            input_shape = (input_channels, *spatial_shape)
        else:
            raise ValueError(f"Unknown layout {layout}")

        return MaskGenerator(
            layout=layout,
            input_shape=input_shape,
            temporal_patch_size=temporal_patch_size,
            lateral_patch_size=lateral_patch_size,
            axial_patch_size=axial_patch_size,
            mask_mode=maskmode,
            time_downsample_pattern=pattern,
            random_masking_ratio=random_masking_ratio,
            lateral_mask_scale=lateral_mask_scale,
            axial_mask_scale=axial_mask_scale,
            temporal_mask_scale=temporal_mask_scale,
            aspect_ratio_scale_hw=aspect_ratio_scale_hw,
            num_blocks=num_blocks,
            device=torch.device("cpu"),
        )

    return _make

@pytest.mark.parametrize(
    ("time_length", "pattern"),
    [
        (4, [0, 1, 0, 1]),
        (10, [1, 0, 1, 1, 1, 0]),
    ],
)
@pytest.mark.parametrize(("batch_size", "layout"), [[1, MULTICHANNEL_HYPERCUBE.TYXC],
                                                    [3, MULTICHANNEL_HYPERCUBE.TYXC],
                                                    [1, MULTICHANNEL_HYPERCUBE.CTYX],
                                                    [3, MULTICHANNEL_HYPERCUBE.CTYX],
                                                    [1, MULTICHANNEL_HYPERCUBE.TZYXC],
                                                    [3, MULTICHANNEL_HYPERCUBE.TZYXC],
                                                    [1, MULTICHANNEL_HYPERCUBE.CTZYX],
                                                    [3, MULTICHANNEL_HYPERCUBE.CTZYX],]
)
def test_blocked_pattern_mask(make_mask_generator, 
                              time_length, 
                              pattern, 
                              batch_size, 
                              layout,
                              maskmode=MaskModes.BLOCKED_PATTERNED
):
    """
    Validate BLOCKED_PATTERNED mode: 
        - tests that the mask repeats `pattern` along the time axis.
    """
    if layout == MULTICHANNEL_HYPERCUBE.CTZYX \
        or layout == MULTICHANNEL_HYPERCUBE.TZYXC:
        spatial_shape = (128, 128, 128)  # Z, Y, X
        temporal_patch_size = 1
        lateral_patch_size = 16
        axial_patch_size = 16
    elif layout == MULTICHANNEL_HYPERCUBE.TYXC \
        or layout == MULTICHANNEL_HYPERCUBE.CTYX:
        spatial_shape = (128, 128)  # Y, X
        temporal_patch_size = 1
        lateral_patch_size = 16
        axial_patch_size = 16
    else:
        raise ValueError(f"Unknown layout {layout}")

    mg = make_mask_generator(time_length=time_length,
                             spatial_shape=spatial_shape,
                             temporal_patch_size=temporal_patch_size,
                             lateral_patch_size=lateral_patch_size,
                             axial_patch_size=axial_patch_size,
                             pattern=pattern,
                             layout=layout,
                             maskmode=maskmode
                             )

    masks, ctx, tgt, orig_idx, _ = mg(batch_size=batch_size)

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
            assert unique_vals.item() == expected_val, (
                f"time {t}: expected {expected_val} but saw {unique_vals.item()}"
            )


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
def test_random_mask(make_mask_generator,
                     batch_size, layout,
                     maskmode, random_ratio,
                     time_length: int = 4):
    """
    Validate RANDOM masking modes: 
        - ratio of target / context within theoretical bounds
        - ctx / tgt index sets match mask values and do not overlap  
        - orig_idx is correct permutation of all patch positions
    """
    if layout == MULTICHANNEL_HYPERCUBE.CTZYX \
        or layout == MULTICHANNEL_HYPERCUBE.TZYXC:
        spatial_shape = (128, 128, 128)  # Z, Y, X
        temporal_patch_size = 1
        lateral_patch_size = 16
        axial_patch_size = 16
    elif layout == MULTICHANNEL_HYPERCUBE.ZYXC \
        or layout == MULTICHANNEL_HYPERCUBE.CZYX:
        spatial_shape = (128, 128, 128)  # Z, Y, X
        temporal_patch_size = None
        lateral_patch_size = 16
        axial_patch_size = 16
    elif layout == MULTICHANNEL_HYPERCUBE.TYXC \
        or layout == MULTICHANNEL_HYPERCUBE.CTYX:
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
    )

    masks, ctx_idx, tgt_idx, orig_idx, _ = mg(batch_size=batch_size)

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
        ones  = (mask == 1).sum().item()
        assert zeros == exp_ctx
        assert ones  == exp_tgt

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
        assert torch.equal(reconstructed, torch.arange(total_len)), (
                "orig_idx is not a proper inverse permutation"
        )


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
    [
        (MaskModes.BLOCKED, (0.2, 0.4)),
        (MaskModes.BLOCKED_SPACE_ONLY, (1.0, 1.0))
    ],
)
def test_blocked_mask_properties(make_mask_generator,
                                 temporal_scale,
                                 batch_size, layout, maskmode,
                                 time_length: int = 16):
    """
    Validate BLOCKED and BLOCKED_SPACE_ONLY masks:
    - ratio of target / context within theoretical bounds
    - ctx / tgt index sets match mask values and do not overlap
    - orig_idx is correct permutation of all patch positions
    """
    if layout == MULTICHANNEL_HYPERCUBE.CTZYX \
        or layout == MULTICHANNEL_HYPERCUBE.TZYXC:
        spatial_shape = (128, 128, 128)  # Z, Y, X
        temporal_patch_size = 1
        lateral_patch_size = 16
        axial_patch_size = 16
    elif layout == MULTICHANNEL_HYPERCUBE.ZYXC \
        or layout == MULTICHANNEL_HYPERCUBE.CZYX:
        spatial_shape = (128, 128, 128)  # Z, Y, X
        temporal_patch_size = None
        lateral_patch_size = 16
        axial_patch_size = 16
    elif layout == MULTICHANNEL_HYPERCUBE.TYXC \
        or layout == MULTICHANNEL_HYPERCUBE.CTYX:
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

    masks, ctx_idx, tgt_idx, orig_idx, _ = mg(batch_size=batch_size)

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
    min_ratio = min_vol / (T*D*H*W)
    max_ratio = max_vol / (T*D*H*W)

    tgt_per_sample = tgt_idx[0].numel()
    ratio = tgt_per_sample / (T*D*H*W)
    assert min_ratio <= ratio <= max_ratio, (
        f"observed {ratio:.3f} not in [{min_ratio:.3f}, {max_ratio:.3f}]"
    )

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
        assert torch.equal(torch.sort(orig_idx[b]).values,
                           torch.arange(total_len))
        
        permuted = torch.cat([ctx_idx[b], tgt_idx[b]])
        reconstructed = permuted[orig_idx[b]]
        assert torch.equal(reconstructed, torch.arange(total_len)), (
                "orig_idx is not a proper inverse permutation"
        )

    grid_shape = (mg.time, mg.depth, mg.height, mg.width)
    for b in range(batch_size):
        mask_reshaped = masks[b].view(*grid_shape)
        tgt_pos = (mask_reshaped == 1).nonzero(as_tuple=False)

        # bounding box of all target indices
        t_min, d_min, h_min, w_min = tgt_pos.min(dim=0).values
        t_max, d_max, h_max, w_max = tgt_pos.max(dim=0).values

        # all positions inside that box must also be target
        box = mask_reshaped[t_min : t_max + 1,
                d_min : d_max + 1,
                h_min : h_max + 1,
                w_min : w_max + 1]

        assert torch.all(box == 1), "masked block is not a block"