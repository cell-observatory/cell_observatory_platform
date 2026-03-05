import math
import random
from enum import Enum
from multiprocessing import Value
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import torch
from hydra.utils import get_method

from cell_observatory_platform.data.data_shapes import MULTICHANNEL_HYPERCUBE
from cell_observatory_platform.training.helpers import get_patch_sizes


# BLOCKED: mask out a block of patches in the input tensor
# BLOCKED_TIME_ONLY: mask out a block of patches with variation
#                    in terms of mask/not mask across time only
#                    (subset of BLOCKED with temporal_mask_scale=1.0)
# BLOCKED_SPACE_ONLY: mask out a block of patches with variation
#                    in terms of mask/not mask across space only
#                    (subset of BLOCKED with spatial_mask_scale=1.0)
# BLOCKED_PATTERNED: mask out a block of patches with a fixed pattern
#                    that repeats across time if necessary, used
#                    primarily for upsampling finetuning task
# RANDOM: randomly mask out patches in the input tensor
# RANDOM_SPACE_ONLY: randomly mask out patches in the input tensor
#                    with variation in terms of mask/not mask across
#                    space only (mask all time patches)


class MaskModes(Enum):
    BLOCKED = "blocked"
    BLOCKED_TIME_ONLY = "blocked_time_only"
    BLOCKED_SPACE_ONLY = "blocked_space_only"
    BLOCKED_PATTERNED = "blocked_patterned"
    RANDOM = "random"
    RANDOM_SPACE_ONLY = "random_space_only"
    BLOCKED_WITH_RANDOM_FILL = "blocked_with_random_fill"
    DINO_IBOT = "dino_ibot"
    HIERA_MU = "hiera_mu"
    HIERA_MU_BLOCKED = "hiera_mu_blocked"


def _scale_to_tuple(scale: tuple | list | float | int) -> tuple[float, float]:
    """Expand scalar to (value, value); pass through (min, max) sequence of length 2 as-is."""
    if isinstance(scale, (int, float)):
        v = float(scale)
        return (v, v)
    t = tuple(scale)
    if len(t) != 2:
        raise ValueError(
            f"Scale must be a scalar or a (min, max) sequence of length 2; got {type(scale).__name__} of len {len(t)}"
        )
    return (float(t[0]), float(t[1]))


class MaskGenerator(object):
    def __init__(
        self,
        layout: MULTICHANNEL_HYPERCUBE,
        batch_size: int = 1,
        input_format: str = "TZYXC",
        input_shape: tuple = (128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        lateral_mask_scale: tuple | float = 0.5,
        axial_mask_scale: tuple | float = 0.5,
        temporal_mask_scale: tuple | float = 0.5,
        aspect_ratio_scale_hw: tuple | float = (0.2, 0.4),
        num_blocks: int = 2,
        random_masking_ratio: float = 0.7,
        channels_to_mask: Optional[Sequence[int]] = None,
        time_downsample_pattern: Optional[Sequence[int]] = None,
        mask_mode: MaskModes = MaskModes.RANDOM,
        device: str = "cuda",
        # DINO/iBOT-specific parameters (only used when mask_mode == DINO_IBOT)
        mask_ratio_range: tuple = (0.1, 0.5),
        mask_probability: float = 0.5,
        random_circular_shift: bool = False,
        # HIERA_MU-specific: mask_unit_size must divide token grid evenly
        mask_unit_size: Optional[Sequence[int]] = None,
        q_stride: Optional[Sequence[int]] = None,
        q_pool: Optional[int] = None,
        skip_patch_mask_generation: bool = False,
        multiscale: bool = False,
    ):
        self.device = device

        if isinstance(layout, str):
            try:
                layout = MULTICHANNEL_HYPERCUBE[layout]
            except KeyError:
                layout = MULTICHANNEL_HYPERCUBE(layout)
        self.layout = layout

        self.input_shape = input_shape
        self.temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
            input_format=input_format, patch_shape=patch_shape
        )

        self.axial_mask_scale = _scale_to_tuple(axial_mask_scale)
        self.lateral_mask_scale = _scale_to_tuple(lateral_mask_scale)
        self.temporal_mask_scale = _scale_to_tuple(temporal_mask_scale)
        self.aspect_ratio_scale_hw = _scale_to_tuple(aspect_ratio_scale_hw)

        self.random_masking_ratio = random_masking_ratio
        self.num_blocks = num_blocks

        self.channels_to_mask = channels_to_mask
        self.time_downsample_pattern = time_downsample_pattern

        self.mask_mode = mask_mode

        # DINO/iBOT-specific
        self.mask_ratio_range = tuple(mask_ratio_range)
        self.mask_probability = mask_probability
        self.random_circular_shift = random_circular_shift

        # HIERA_MU-specific
        self.mask_unit_size = tuple(mask_unit_size) if mask_unit_size else None
        self.q_stride = tuple(q_stride) if q_stride else None
        self.q_pool = int(q_pool) if q_pool is not None else None

        self.multiscale = multiscale
        # tokens per MU at fusion output (mask_unit_size / q_stride^q_pool) for JEPA tgt_tok_idx
        if self.mask_unit_size is not None and self.q_stride is not None and self.q_pool is not None:
            qs = self.q_stride
            if len(qs) != len(self.mask_unit_size):
                raise ValueError(f"q_stride must have the same length as mask_unit_size; got {len(qs)} and {len(self.mask_unit_size)}")
            q_stride_pow = tuple(int(s) ** self.q_pool for s in qs)
            self.tok_in_mu_final = tuple(
                max(1, s // qsp) for s, qsp in zip(self.mask_unit_size, q_stride_pow)
            )
            self.tok_prod = int(math.prod(self.tok_in_mu_final))

            if self.multiscale:
                self.tok_prods_per_level = []
                for lvl in range(self.q_pool + 1):
                    pools = min(lvl, q_pool)
                    tok_in_mu_lvl = tuple(
                        max(1, mu // (s ** pools))
                        for mu, s in zip(self.mask_unit_size, qs)
                    )
                    self.tok_prods_per_level.append(int(math.prod(tok_in_mu_lvl)))
            else:
                self.tok_prods_per_level = None
        else:
            self.tok_prod = None
            self.tok_prods_per_level = None

        # the iteration counter impacts the RNG state
        # for a given mask generation collator
        # since all processes will step in unison
        # this ensures that each process generates
        # block sizes of the same size for each step
        # strictly speaking we don't need to ensure
        # that the block sizes are the same across
        # GPU workers however this is
        # the strategy utilized in V-JEPA
        self._itr_counter = Value("i", -1)

        self.time, self.depth, self.height, self.width = self._get_input_shape_patches(
            input_shape=self.input_shape,
            temporal_patch_size=self.temporal_patch_size,
            axial_patch_size=self.axial_patch_size,
            lateral_patch_size=self.lateral_patch_size,
            layout=self.layout,
        )
        self.input_shape_patches = (self.time, self.depth, self.height, self.width)

        self.skip_patch_mask_generation = bool(skip_patch_mask_generation)

    def _get_input_shape_patches(self, input_shape, temporal_patch_size, axial_patch_size, lateral_patch_size, layout):
        axis_to_value = dict(zip(layout.value, input_shape))

        t = axis_to_value.get("T", 1)
        z = axis_to_value.get("Z", 1)
        y = axis_to_value.get("Y", 1)
        x = axis_to_value.get("X", 1)

        if t > 1 and z > 1:
            time = t // temporal_patch_size
            depth = z // axial_patch_size
            height = y // lateral_patch_size
            width = x // lateral_patch_size
        elif t > 1 and z == 1:
            time = t // temporal_patch_size
            depth = 1
            height = y // lateral_patch_size
            width = x // lateral_patch_size
        elif t == 1 and z > 1:
            time = 1
            depth = z // axial_patch_size
            height = y // lateral_patch_size
            width = x // lateral_patch_size
        else:
            raise ValueError(
                f"Invalid input shape {input_shape} and patch shape "
                f"{(temporal_patch_size, axial_patch_size, lateral_patch_size, lateral_patch_size)} for layout {layout}. "
                "Expected at least one of time or depth to be greater than 1."
            )

        return time, depth, height, width

    # we step with iteration counter
    # once per step
    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator):
        # sample temporal block mask scale
        if self.time > 1:
            _rand = torch.rand(1, generator=generator).item()
            min_t, max_t = self.temporal_mask_scale
            temporal_mask_scale = min_t + _rand * (max_t - min_t)
            t = int(self.time * temporal_mask_scale)
            t = min(t, self.time)
        else:
            t = 1

        _rand_axial = torch.rand(1, generator=generator).item()
        min_axial, max_axial = self.axial_mask_scale
        axial_mask_scale = min_axial + _rand_axial * (max_axial - min_axial)
        axial_num_mask = int(self.height * self.width * axial_mask_scale)

        # sample block aspect-ratio
        # TODO: we may consider other ways to sample blocks of
        #       different shapes/sizes in the future
        _rand_ar = torch.rand(1, generator=generator).item()
        min_ar, max_ar = self.aspect_ratio_scale_hw
        aspect_ratio_hw = min_ar + _rand_ar * (max_ar - min_ar)

        h = int(round(math.sqrt(axial_num_mask * aspect_ratio_hw)))
        w = int(round(math.sqrt(axial_num_mask / aspect_ratio_hw)))

        if self.depth > 1:
            _rand_lateral = torch.rand(1, generator=generator).item()
            min_lateral, max_lateral = self.lateral_mask_scale
            lateral_mask_scale = min_lateral + _rand_lateral * (max_lateral - min_lateral)
            d = int(self.depth * lateral_mask_scale)
            d = min(d, self.depth)
        else:
            d = 1

        h = min(h, self.height)
        w = min(w, self.width)
        return (t, d, h, w)

    def _sample_block_mask(self, block_size):
        starts = [
            None if dim in (None, 1) else torch.randint(0, dim - sz + 1, ()).item()
            for dim, sz in zip(self.input_shape_patches, block_size)
        ]

        slices = [slice(st, st + sz) if st is not None else slice(None) for st, sz in zip(starts, block_size)]

        shape = [1 if dim in (None, 0, 1) else dim for dim in self.input_shape_patches]
        block_mask = torch.ones(shape, dtype=torch.int32, device=self.device)

        block_mask[tuple(slices)] = 0
        block_mask = block_mask.squeeze()
        return block_mask

    # adapted from:
    # https://github.com/facebookresearch/jepa/blob/main/src/masks/multiblock3d.py
    def _generate_batched_blocked_mask(self, batch_size: int, generator):
        # we sample the block size once per batch
        # to ensure that all samples in the batch
        # have the same block size, they may still not
        # have the same number of patches masked out
        # since the block is sampled randomly
        # across the batch and taking a union may
        # result in differences in the number of patches
        block_size = self._sample_block_size(generator)

        masks_target, masks_context = [], []
        # the largest our minimum context/target patch sequences can be
        # will be shortened to the minimum across all samples after the loop
        min_keep_ctx = min_keep_target = self.time * self.depth * self.height * self.width
        for _ in range(batch_size):
            empty_context = True
            while empty_context:

                # we use the opposite convention here for masking/unmasking where
                # the mask is 1 for unmasked patches
                if self.time > 1 and self.depth > 1:
                    mask_ctx = torch.ones(
                        (self.time, self.depth, self.height, self.width), dtype=torch.int32, device=self.device
                    )
                elif self.time > 1 and self.depth == 1:
                    mask_ctx = torch.ones((self.time, self.height, self.width), dtype=torch.int32, device=self.device)
                elif self.time == 1 and self.depth > 1:
                    mask_ctx = torch.ones((self.depth, self.height, self.width), dtype=torch.int32, device=self.device)
                else:
                    raise ValueError(
                        "Invalid input shape for masking. "
                        "Expected at least one of time or depth to be greater than 1."
                    )

                for _ in range(self.num_blocks):
                    mask_ctx *= self._sample_block_mask(block_size)
                mask_ctx = mask_ctx.flatten()

                # we include this step to ensure we maintain the same
                # convention for all block modes i.e. that the mask is 1
                # for unmasked patches and 0 for masked patches
                mask_ctx = 1 - mask_ctx

                context_idx = torch.nonzero(mask_ctx == 0, as_tuple=False).squeeze(1)
                target_idx = torch.nonzero(mask_ctx == 1, as_tuple=False).squeeze(1)

                empty_context = len(context_idx) == 0
                if not empty_context:
                    min_keep_ctx = min(min_keep_ctx, len(context_idx))
                    min_keep_target = min(min_keep_target, len(target_idx))
                    masks_target.append(target_idx)
                    masks_context.append(context_idx)

        masks, ctx_list, target_list, original_indices_list = [], [], [], []
        for ctx, target in zip(masks_context, masks_target):
            ctx = ctx[:min_keep_ctx]
            target = target[:min_keep_target]
            perm = torch.cat([ctx, target])
            orig_idx = torch.argsort(perm)

            # TODO: is this really necessary? seems like all
            # we currently use mask for is to do masks.sum()?
            mask = torch.ones_like(perm, dtype=torch.int32, device=self.device)
            mask[: len(ctx)] = 0
            mask = mask[orig_idx]

            ctx_list.append(ctx)
            target_list.append(target)
            original_indices_list.append(orig_idx)
            masks.append(mask)

        masks = torch.utils.data.default_collate(masks)
        collated_masks_context = torch.utils.data.default_collate(ctx_list)
        collated_masks_target = torch.utils.data.default_collate(target_list)
        original_patch_indices = torch.utils.data.default_collate(original_indices_list)
        return masks, collated_masks_context, collated_masks_target, original_patch_indices

    def _try_add_block_bool(
        self,
        tgt_grid: torch.Tensor,
        block_size: tuple[int, int, int, int],
        max_to_add: int,
        generator: torch.Generator,
        tries: int = 10,
    ) -> int:
        T, D, H, W = tgt_grid.shape
        bt, bd, bh, bw = block_size

        for _ in range(tries):
            st = 0 if T <= bt else int(torch.randint(0, T - bt + 1, (), generator=generator).item())
            sd = 0 if D <= bd else int(torch.randint(0, D - bd + 1, (), generator=generator).item())
            sh = 0 if H <= bh else int(torch.randint(0, H - bh + 1, (), generator=generator).item())
            sw = 0 if W <= bw else int(torch.randint(0, W - bw + 1, (), generator=generator).item())

            sl = (slice(st, st + bt), slice(sd, sd + bd), slice(sh, sh + bh), slice(sw, sw + bw))
            region = tgt_grid[sl]

            # how many *new* masked patches would this add?
            delta = int((~region).sum().item())
            if 0 < delta <= max_to_add:
                tgt_grid[sl] = True
                return delta

        return 0

    def _complete_target_count(
        self,
        tgt_flat: torch.Tensor,
        num_target: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        # Ensure EXACTLY num_target True entries
        device = tgt_flat.device
        cur = int(tgt_flat.sum().item())
        if cur == num_target:
            return tgt_flat

        if cur < num_target:
            need = num_target - cur
            avail = (~tgt_flat).nonzero(as_tuple=False).squeeze(1).cpu()
            if avail.numel() < need:
                raise RuntimeError(f"Not enough available positions to add: need={need}, avail={avail.numel()}")
            pick = avail[torch.randperm(avail.numel(), generator=generator)[:need]].to(device)
            tgt_flat[pick] = True
        else:
            drop = cur - num_target
            masked = tgt_flat.nonzero(as_tuple=False).squeeze(1).cpu()
            pick = masked[torch.randperm(masked.numel(), generator=generator)[:drop]].to(device)
            tgt_flat[pick] = False

        return tgt_flat

    def _mu_grid_shapes(self):
        """Return (mu_size_tuple, mu_grid_shape) for MU-based masking."""
        if self.mask_unit_size is None:
            raise ValueError(
                "mask_unit_size must be set for MU-based masking (HIERA_MU, HIERA_MU_BLOCKED)."
            )
        mu_t, mu_d, mu_h, mu_w = self.mask_unit_size
        T, D, H, W = self.time, self.depth, self.height, self.width
        assert T % mu_t == 0 and D % mu_d == 0 and H % mu_h == 0 and W % mu_w == 0, (
            f"mask_unit_size {self.mask_unit_size} must divide input_shape_patches {(T,D,H,W)}"
        )
        Tg = T // mu_t
        Dg = D // mu_d
        Hg = H // mu_h
        Wg = W // mu_w
        return (mu_t, mu_d, mu_h, mu_w), (Tg, Dg, Hg, Wg)

    # TODO: unify with existing block sampling logic into one helper function
    def _sample_block_size_on_grid(
        self, grid_shape: Tuple[int, int, int, int], generator
    ) -> Tuple[int, int, int, int]:
        """Sample block size (t,d,h,w) for a 4D grid using existing scale logic."""
        Tg, Dg, Hg, Wg = grid_shape

        if Tg > 1:
            r = torch.rand(1, generator=generator).item()
            mn, mx = self.temporal_mask_scale
            t = max(1, int(Tg * (mn + r * (mx - mn))))
            t = min(t, Tg)
        else:
            t = 1

        r = torch.rand(1, generator=generator).item()
        mn, mx = self.axial_mask_scale
        area = int(Hg * Wg * (mn + r * (mx - mn)))
        area = max(1, min(area, Hg * Wg))

        r = torch.rand(1, generator=generator).item()
        mn, mx = self.aspect_ratio_scale_hw
        ar = mn + r * (mx - mn)
        h = int(round(math.sqrt(area * ar)))
        w = int(round(math.sqrt(area / ar)))
        h = max(1, min(h, Hg))
        w = max(1, min(w, Wg))

        if Dg > 1:
            r = torch.rand(1, generator=generator).item()
            mn, mx = self.lateral_mask_scale
            d = max(1, int(Dg * (mn + r * (mx - mn))))
            d = min(d, Dg)
        else:
            d = 1

        return (t, d, h, w)

    def _shrink_block_to_fit(
        self, block_size: Tuple[int, int, int, int], max_volume: int
    ) -> Tuple[int, int, int, int]:
        """Shrink block size until volume <= max_volume."""
        bt, bd, bh, bw = block_size
        sizes = [bt, bd, bh, bw]
        while sizes[0] * sizes[1] * sizes[2] * sizes[3] > max_volume:
            k = max(range(4), key=lambda i: sizes[i])
            if sizes[k] == 1:
                break
            sizes[k] -= 1
        return tuple(sizes)

    def _mu_patch_offsets(
        self,
        mu_size: Tuple[int, int, int, int],
        patch_grid_shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Flat patch indices within one MU (for base 0,0,0,0). Shape (flat_mu_size,)."""
        mu_t, mu_d, mu_h, mu_w = mu_size
        T, D, H, W = patch_grid_shape

        tt = torch.arange(mu_t, device=self.device, dtype=torch.long)
        dd = torch.arange(mu_d, device=self.device, dtype=torch.long)
        hh = torch.arange(mu_h, device=self.device, dtype=torch.long)
        ww = torch.arange(mu_w, device=self.device, dtype=torch.long)
        grid = torch.stack(torch.meshgrid(tt, dd, hh, ww, indexing="ij"), dim=-1)

        off = (
            grid[..., 0] * (D * H * W)
            + grid[..., 1] * (H * W)
            + grid[..., 2] * W
            + grid[..., 3]
        )
        return off.reshape(-1)

    def _mu_indices_to_patch_indices(
        self,
        mu_idx: torch.Tensor,
        mu_size: Tuple[int, int, int, int],
        mu_grid_shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Convert flat MU indices to flat patch indices."""
        mu_t, mu_d, mu_h, mu_w = mu_size
        Tg, Dg, Hg, Wg = mu_grid_shape
        T, D, H, W = self.time, self.depth, self.height, self.width

        mu_idx = mu_idx.to(torch.long).to(self.device)

        w = mu_idx % Wg
        tmp = mu_idx // Wg
        h = tmp % Hg
        tmp = tmp // Hg
        d = tmp % Dg
        t = tmp // Dg

        # get patch indices for MU
        t0 = t * mu_t
        d0 = d * mu_d
        h0 = h * mu_h
        w0 = w * mu_w

        # start patch index of MU
        base = t0 * (D * H * W) + d0 * (H * W) + h0 * W + w0

        # get patch offsets within MU
        offsets = self._mu_patch_offsets(mu_size, (T, D, H, W))
        # get patch indices for patches within MU
        patches = base[:, None] + offsets[None, :]
        return patches.reshape(-1)

    def _generate_single_blocked_mu_mask(
        self,
        num_target_mus: int,
        block_size_mu: Tuple[int, int, int, int],
        mu_grid_shape: Tuple[int, int, int, int],
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Flat (num_mus,) bool, True = target MU."""
        Tg, Dg, Hg, Wg = mu_grid_shape
        num_mus = Tg * Dg * Hg * Wg

        if num_target_mus <= 0:
            raise ValueError(f"num_target_mus ({num_target_mus}) must be greater than 0")
        if num_target_mus >= num_mus:
            raise ValueError(f"num_target_mus ({num_target_mus}) must be less than num_mus ({num_mus})")

        tgt_grid = torch.zeros((Tg, Dg, Hg, Wg), dtype=torch.bool, device=self.device)
        masked = 0

        for _ in range(self.num_blocks):
            remaining = num_target_mus - masked
            if remaining <= 0:
                break
            delta = self._try_add_block_bool(
                tgt_grid, block_size_mu, remaining, generator, tries=10
            )
            masked += delta

        tgt_flat = tgt_grid.flatten()
        tgt_flat = self._complete_target_count(tgt_flat, num_target_mus, generator)
        return tgt_flat

    def _generate_single_blocked_mask(
        self,
        num_target: int,
        block_size: Tuple[int, int, int, int],
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Generate a single flat bool mask with exactly `num_target` True entries.
        Uses block placement followed by random fill to reach the exact target count.

        Args:
            num_target: Number of patches to mask (True entries).
            block_size: (t, d, h, w) block dimensions for initial placement.
            generator: Torch RNG generator for reproducibility.

        Returns:
            Flat (N,) bool tensor where True = masked/target.
        """
        T, D, H, W = self.time, self.depth, self.height, self.width
        N = T * D * H * W

        if num_target == 0:
            return torch.zeros(N, dtype=torch.bool, device=self.device)
        if num_target >= N:
            return torch.ones(N, dtype=torch.bool, device=self.device)

        tgt_grid = torch.zeros((T, D, H, W), dtype=torch.bool, device=self.device)
        masked = 0

        for _b in range(self.num_blocks):
            remaining = num_target - masked
            if remaining <= 0:
                break
            delta = self._try_add_block_bool(tgt_grid, block_size, remaining, generator, tries=10)
            masked += delta

        tgt_flat = tgt_grid.flatten()
        tgt_flat = self._complete_target_count(tgt_flat, num_target, generator)
        return tgt_flat

    def _generate_blocked_with_random_fill(self, batch_size: int, generator: torch.Generator):
        # grid shape always 4D (with possible 1s)
        T, D, H, W = self.time, self.depth, self.height, self.width
        N = T * D * H * W

        num_target = int(round(N * self.random_masking_ratio))
        num_ctx = N - num_target

        # sample block size once per batch
        block_size = self._sample_block_size(generator)

        masks_list, ctx_list, tgt_list, orig_list = [], [], [], []
        for _ in range(batch_size):
            tgt_flat = self._generate_single_blocked_mask(num_target, block_size, generator)

            tgt_idx = tgt_flat.nonzero(as_tuple=False).squeeze(1)        # (num_target,)
            ctx_idx = (~tgt_flat).nonzero(as_tuple=False).squeeze(1)     # (num_ctx,)

            # sanity
            if tgt_idx.numel() != num_target or ctx_idx.numel() != num_ctx:
                raise RuntimeError("Target/context counts mismatched after completion.")

            perm = torch.cat([ctx_idx, tgt_idx], dim=0)
            orig = torch.argsort(perm, dim=0, stable=True)

            # mask in ORIGINAL patch order, 0 for ctx, 1 for tgt
            m = torch.ones((N,), dtype=torch.int32, device=self.device)
            m[:num_ctx] = 0
            m = m[orig]

            masks_list.append(m)
            ctx_list.append(ctx_idx)
            tgt_list.append(tgt_idx)
            orig_list.append(orig)

        masks = torch.utils.data.default_collate(masks_list)                 # (B, N)
        context_masks = torch.utils.data.default_collate(ctx_list)           # (B, num_ctx)
        target_masks = torch.utils.data.default_collate(tgt_list)            # (B, num_target)
        original_patch_indices = torch.utils.data.default_collate(orig_list) # (B, N)

        return masks, context_masks, target_masks, original_patch_indices

    def _generate_dino_ibot_masks(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Generate DINO/iBOT-style masks with per-sample variable masking ratios.

        Mirrors the masking logic from DINOv3's collate_data_and_cast:
          - Only a fraction (mask_probability) of the batch is masked.
          - Masked samples get linearly spaced ratios from mask_ratio_range.
          - Uses block placement + random fill for each sample.
          - Optionally applies random circular shift in XY.

        Args:
            batch_size: Total number of images to mask (typically n_global_crops * B).
            generator: Torch RNG generator for reproducibility.

        Returns:
            Dict with:
                collated_masks:    (batch_size, N) bool -- True = masked (target for iBOT)
                mask_indices_list: (total_masked,) long  -- flat indices of all True positions
                masks_weight:      (total_masked,) float -- 1/per_sample_mask_count per position
                upperbound:        int                   -- sum of per-sample target counts
                n_masked_patches:  (1,) long tensor
        """
        T, D, H, W = self.time, self.depth, self.height, self.width
        N = T * D * H * W
        B = batch_size

        n_samples_masked = int(B * self.mask_probability)

        # Linearly spaced ratios from min to max across the masked samples
        probs = torch.linspace(
            self.mask_ratio_range[0], self.mask_ratio_range[1], n_samples_masked + 1
        )

        # Sample block size once for the whole batch
        block_size = self._sample_block_size(generator)

        upperbound = 0
        masks_list = []

        # Generate masks for samples that WILL be masked (variable ratio per sample)
        for i in range(n_samples_masked):
            prob_max = probs[i + 1].item()
            num_target = int(N * prob_max)

            mask = self._generate_single_blocked_mask(num_target, block_size, generator)

            if self.random_circular_shift:
                # Reshape to (T, D, H, W) and shift only in XY (last two dims)
                mask_4d = mask.view(T, D, H, W)
                shift_h = torch.randint(0, max(1, H), (), generator=generator).item()
                shift_w = torch.randint(0, max(1, W), (), generator=generator).item()
                mask_4d = torch.roll(mask_4d, shifts=(shift_h, shift_w), dims=(-2, -1))
                mask = mask_4d.flatten()

            masks_list.append(mask)
            upperbound += num_target

        # Generate empty (all-False) masks for samples that will NOT be masked
        for _ in range(n_samples_masked, B):
            masks_list.append(torch.zeros(N, dtype=torch.bool, device=self.device))

        # Shuffle so masked/unmasked are randomly distributed across the batch
        random.shuffle(masks_list)

        # Stack into (B, N) and compute derived quantities
        collated_masks = torch.stack(masks_list)  # (B, N) bool
        mask_indices_list = collated_masks.flatten().nonzero(as_tuple=False).flatten()  # (total_masked,)

        # Per-sample inverse-count weights, broadcast to masked positions only
        per_sample_count = collated_masks.sum(dim=-1).clamp(min=1.0).float()  # (B,)
        masks_weight = (
            (1.0 / per_sample_count)
            .unsqueeze(-1)
            .expand_as(collated_masks)[collated_masks]
        )  # (total_masked,)

        return {
            "collated_masks": collated_masks,
            "mask_indices_list": mask_indices_list,
            "masks_weight": masks_weight,
            "upperbound": upperbound,
            "n_masked_patches": torch.full(
                (1,), fill_value=mask_indices_list.shape[0], dtype=torch.long
            ),
        }

    def _generate_blocked_mask_patterned(self, batch_size: int):
        if self.time_downsample_pattern is None:
            raise ValueError("time_downsample_pattern must be provided for BLOCKED_PATTERNED.")

        pattern = torch.as_tensor(self.time_downsample_pattern, dtype=torch.bool, device=self.device)
        if pattern.ndim != 1 or pattern.numel() == 0:
            raise ValueError("time_downsample_pattern must be a 1D non-empty sequence of 0/1 values.")

        K = pattern.numel()
        time_idx = torch.arange(self.time, device=self.device) % K
        time_mask = pattern[time_idx]

        if self.depth > 1:
            base = time_mask.view(self.time, 1, 1, 1).expand(-1, self.depth, self.height, self.width)
        else:
            base = time_mask.view(self.time, 1, 1).expand(-1, self.height, self.width)

        masks = base.reshape(1, -1).expand(batch_size, -1).to(torch.int32)  # (B, L)

        num_tgt = masks.sum(dim=1)
        num_ctx = masks.shape[1] - num_tgt
        if (num_tgt == 0).any() or (num_ctx == 0).any():
            raise ValueError(
                "Pattern produced empty context or target in at least one sample. "
                "Ensure pattern has at least one 0 and one 1."
            )

        ctx_pos = (masks == 0).nonzero(as_tuple=False)  # (N_ctx_total, 2)
        tgt_pos = (masks == 1).nonzero(as_tuple=False)  # (N_tgt_total, 2)

        ctx_list, tgt_list, orig_list = [], [], []
        for b in range(batch_size):
            ctx_idx = ctx_pos[ctx_pos[:, 0] == b][:, 1]  # (L_ctx,)
            tgt_idx = tgt_pos[tgt_pos[:, 0] == b][:, 1]  # (L_tgt,)

            perm = torch.cat([ctx_idx, tgt_idx], dim=0)  # (L,)
            orig = torch.argsort(perm, dim=0, stable=True)  # (L,)

            ctx_list.append(ctx_idx)
            tgt_list.append(tgt_idx)
            orig_list.append(orig)

        context_masks = torch.utils.data.default_collate(ctx_list)  # (B, L_ctx)
        target_masks = torch.utils.data.default_collate(tgt_list)  # (B, L_tgt)
        original_patch_indices = torch.utils.data.default_collate(orig_list)  # (B, L)

        return masks, context_masks, target_masks, original_patch_indices

    def _generate_random_mask(self, batch_size: int, space_only=False, device="cuda"):
        B, T = batch_size, self.time
        # works no matter the data layout since we set d=1 for 2D data
        S = self.depth * self.height * self.width
        # total number of patches
        N = T * S

        # standard MAE masking logic
        def _mask_sequence(axis_len: int):
            ctx_len = int(axis_len * (1 - self.random_masking_ratio))
            noise = torch.rand(B, axis_len, device=device)
            shuffle = torch.argsort(noise, dim=1)
            orig_idx = torch.argsort(shuffle, dim=1)

            ctx_idx = shuffle[:, :ctx_len]
            tgt_idx = shuffle[:, ctx_len:]

            base = torch.ones_like(noise, device=device)
            base[:, :ctx_len] = 0
            base = torch.gather(base, 1, orig_idx)
            return base, ctx_idx, tgt_idx, orig_idx

        if space_only:
            base, ctx, tgt, orig = _mask_sequence(S)
            # repeats the base mask across the time dimension
            # so that each time slice has the same mask
            # i.e. we do [a,b,c] -> [a,b,c,a,b,c] if we had
            # T=2 and S=3 where a,b,c are mask/not mask values
            masks = base.repeat(1, T)

            # offsets: [0,...,T] -> [0,1*S,2*S,...,T*S]
            time_offsets = torch.arange(T, device=device)[None, :, None] * S
            # (B, 1, S) + (1, T, 1) -> (B, T, S) -> (B, T*S)
            # where for (B, 1:2) all values are offset by S since the
            # stride per T time step is S
            ctx_idx = (ctx[:, None, :] + time_offsets).reshape(B, -1)
            tgt_idx = (tgt[:, None, :] + time_offsets).reshape(B, -1)

            perm = torch.cat([ctx_idx, tgt_idx], dim=1)
            orig_idx = torch.argsort(perm, dim=1)

            return masks, ctx_idx, tgt_idx, orig_idx

        else:
            masks, context_masks, target_masks, original_patch_indices = _mask_sequence(N)

            return masks, context_masks, target_masks, original_patch_indices

    def _collate_tgt_tok_idx(self, tgt_tok_idx_list):
        if not tgt_tok_idx_list:
            return None
        if self.multiscale and self.tok_prods_per_level:
            num_levels = len(self.tok_prods_per_level)
            return [
                torch.utils.data.default_collate(
                    [sample[lvl] for sample in tgt_tok_idx_list]
                )
                for lvl in range(num_levels)
            ]
        elif self.tok_prod is not None:
            return torch.utils.data.default_collate(tgt_tok_idx_list)
        else:
            return None

    def _generate_hiera_mu_mask(
        self, batch_size: int, generator: Optional[torch.Generator] = None
    ) -> Tuple:
        """
        Generate MU-level masks and expand to patch-level metadata.
        Returns 7-tuple including mu_mask for Hiera encoder.
        """
        if self.mask_unit_size is None:
            raise ValueError(
                "mask_unit_size must be provided when mask_mode=HIERA_MU. "
                "It should match Hiera's mask_unit_size (e.g. from q_stride^q_pool)."
            )
        T, D, H, W = self.time, self.depth, self.height, self.width
        mu_size = tuple(self.mask_unit_size)
        if len(mu_size) != 4:
            raise ValueError(
                f"mask_unit_size must have 4 elements (T,D,H,W) for input_shape_patches "
                f"{(T,D,H,W)}; got {mu_size}"
            )
        mu_t, mu_d, mu_h, mu_w = mu_size
        assert T % mu_t == 0 and D % mu_d == 0 and H % mu_h == 0 and W % mu_w == 0, (
            f"mask_unit_size {mu_size} must divide input_shape_patches {(T,D,H,W)}"
        )
        num_mus_t, num_mus_d = T // mu_t, D // mu_d
        num_mus_h, num_mus_w = H // mu_h, W // mu_w
        num_mus = num_mus_t * num_mus_d * num_mus_h * num_mus_w
        flat_mu_size = mu_t * mu_d * mu_h * mu_w
        N = T * D * H * W

        num_target_mus = int(round(num_mus * self.random_masking_ratio))
        num_ctx_mus = num_mus - num_target_mus
        if num_target_mus <= 0 or num_ctx_mus <= 0:
            raise ValueError(f"num_target_mus ({num_target_mus}) or num_ctx_mus ({num_ctx_mus}) must be greater than 0")

        mu_size, mu_grid_shape = self._mu_grid_shapes()

        masks_list, ctx_list, tgt_list, orig_list = [], [], [], []
        mu_masks_list = []
        mu_keep_idx_list = []
        tgt_tok_idx_list = []

        for _ in range(batch_size):
            perm_mus = torch.randperm(num_mus, generator=generator, device=self.device)
            ctx_mus = perm_mus[:num_ctx_mus]
            tgt_mus = perm_mus[num_ctx_mus : num_ctx_mus + num_target_mus]

            mu_mask = torch.ones(num_mus, dtype=torch.bool, device=self.device)
            mu_mask[ctx_mus] = False
            mu_masks_list.append(mu_mask)
            mu_keep_idx_list.append(ctx_mus)

            if not self.skip_patch_mask_generation:
                ctx_patches = self._mu_indices_to_patch_indices(ctx_mus, mu_size, mu_grid_shape)
                tgt_patches = self._mu_indices_to_patch_indices(tgt_mus, mu_size, mu_grid_shape)
                perm = torch.cat([ctx_patches, tgt_patches], dim=0)
                orig = torch.argsort(perm, dim=0, stable=True)
                m = torch.ones(N, dtype=torch.int32, device=self.device)
                m[: len(ctx_patches)] = 0
                m = m[orig]
                masks_list.append(m)
                ctx_list.append(ctx_patches)
                tgt_list.append(tgt_patches)
                orig_list.append(orig)

            if self.multiscale and self.tok_prods_per_level:
                per_level_tgt = []
                for tp in self.tok_prods_per_level:
                    base = tgt_mus.unsqueeze(-1).long() * tp
                    offs = torch.arange(tp, device=self.device, dtype=base.dtype).view(1, -1)
                    per_level_tgt.append((base + offs).reshape(-1))
                tgt_tok_idx_list.append(per_level_tgt)
            elif self.tok_prod is not None:
                base = tgt_mus.unsqueeze(-1).long() * self.tok_prod
                offs = torch.arange(self.tok_prod, device=self.device, dtype=base.dtype).view(1, -1)
                tgt_tok_idx_list.append((base + offs).reshape(-1))

        if self.skip_patch_mask_generation:
            mu_mask = torch.stack(mu_masks_list)
            mu_keep_idx = torch.utils.data.default_collate(mu_keep_idx_list)
            tgt_tok_idx = self._collate_tgt_tok_idx(tgt_tok_idx_list)
            return {
                "mu_mask": mu_mask,
                "mu_keep_idx": mu_keep_idx,
                "tgt_tok_idx": tgt_tok_idx,
                "channels_to_mask": self.channels_to_mask,
            }

        min_ctx = min(len(c) for c in ctx_list)
        min_tgt = min(len(t) for t in tgt_list)
        min_ctx_mus = min_ctx // flat_mu_size
        ctx_list = [c[:min_ctx] for c in ctx_list]
        tgt_list = [t[:min_tgt] for t in tgt_list]
        mu_keep_idx_list = [m[:min_ctx_mus] for m in mu_keep_idx_list]

        masks = torch.utils.data.default_collate(masks_list)
        context_masks = torch.utils.data.default_collate(ctx_list)
        target_masks = torch.utils.data.default_collate(tgt_list)
        original_patch_indices = torch.utils.data.default_collate(orig_list)
        mu_mask = torch.stack(mu_masks_list)
        mu_keep_idx = torch.utils.data.default_collate(mu_keep_idx_list)

        perm = torch.cat([context_masks, target_masks], dim=1)
        patches_used, _ = torch.sort(perm, dim=1)

        tgt_tok_idx = self._collate_tgt_tok_idx(tgt_tok_idx_list)

        return {
            "masks": masks,
            "context_masks": context_masks,
            "target_masks": target_masks,
            "original_patch_indices": original_patch_indices,
            "channels_to_mask": self.channels_to_mask,
            "patches_used": patches_used,
            "mu_mask": mu_mask,
            "mu_keep_idx": mu_keep_idx,
            "tgt_tok_idx": tgt_tok_idx,
        }

    def _generate_hiera_mu_blocked_mask(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> Dict[str, Any]:
        """MU-aligned blocked masking. Same output dict as HIERA_MU."""
 
        mu_size, mu_grid_shape = self._mu_grid_shapes()
        mu_t, mu_d, mu_h, mu_w = mu_size
        Tg, Dg, Hg, Wg = mu_grid_shape

        num_mus = Tg * Dg * Hg * Wg
        num_target_mus = int(round(num_mus * self.random_masking_ratio))
        num_target_mus = max(1, min(num_target_mus, num_mus - 1))
        num_ctx_mus = num_mus - num_target_mus

        block_size_mu = self._sample_block_size_on_grid(mu_grid_shape, generator)
        block_size_mu = self._shrink_block_to_fit(
            block_size_mu, max_volume=num_target_mus
        )

        flat_mu = mu_t * mu_d * mu_h * mu_w
        N = self.time * self.depth * self.height * self.width
        assert N == num_mus * flat_mu

        masks_list, ctx_list, tgt_list, orig_list, mu_masks_list = (
            [],
            [],
            [],
            [],
            [],
        )
        mu_keep_idx_list = []
        tgt_tok_idx_list = []

        for _ in range(batch_size):
            tgt_mu_flat = self._generate_single_blocked_mu_mask(
                num_target_mus=num_target_mus,
                block_size_mu=block_size_mu,
                mu_grid_shape=mu_grid_shape,
                generator=generator,
            )
            tgt_mus = tgt_mu_flat.nonzero(as_tuple=False).squeeze(1)
            ctx_mus = (~tgt_mu_flat).nonzero(as_tuple=False).squeeze(1)

            mu_mask = tgt_mu_flat.clone()
            mu_masks_list.append(mu_mask)
            mu_keep_idx_list.append(ctx_mus)

            if not self.skip_patch_mask_generation:
                ctx_patches = self._mu_indices_to_patch_indices(
                    ctx_mus, mu_size, mu_grid_shape
                )
                tgt_patches = self._mu_indices_to_patch_indices(
                    tgt_mus, mu_size, mu_grid_shape
                )
                perm = torch.cat([ctx_patches, tgt_patches], dim=0)
                orig = torch.argsort(perm, dim=0, stable=True)

                m = torch.ones(N, dtype=torch.int32, device=self.device)
                m[: ctx_patches.numel()] = 0
                m = m[orig]
                masks_list.append(m)
                ctx_list.append(ctx_patches)
                tgt_list.append(tgt_patches)
                orig_list.append(orig)

            if self.multiscale and self.tok_prods_per_level:
                per_level_tgt = []
                for tp in self.tok_prods_per_level:
                    base = tgt_mus.unsqueeze(-1).long() * tp
                    offs = torch.arange(tp, device=self.device, dtype=base.dtype).view(1, -1)
                    per_level_tgt.append((base + offs).reshape(-1))
                tgt_tok_idx_list.append(per_level_tgt)
            elif self.tok_prod is not None:
                base = tgt_mus.unsqueeze(-1).long() * self.tok_prod
                offs = torch.arange(self.tok_prod, device=self.device, dtype=base.dtype).view(1, -1)
                tgt_tok_idx_list.append((base + offs).reshape(-1))

        if self.skip_patch_mask_generation:
            mu_mask = torch.stack(mu_masks_list)
            mu_keep_idx = torch.utils.data.default_collate(mu_keep_idx_list)
            tgt_tok_idx = self._collate_tgt_tok_idx(tgt_tok_idx_list)
            return {
                "mu_mask": mu_mask,
                "mu_keep_idx": mu_keep_idx,
                "tgt_tok_idx": tgt_tok_idx,
                "channels_to_mask": self.channels_to_mask,
            }

        masks = torch.utils.data.default_collate(masks_list)
        context_masks = torch.utils.data.default_collate(ctx_list)
        target_masks = torch.utils.data.default_collate(tgt_list)
        original_patch_indices = torch.utils.data.default_collate(orig_list)
        
        mu_mask = torch.stack(mu_masks_list)
        mu_keep_idx = torch.utils.data.default_collate(mu_keep_idx_list)

        perm = torch.cat([context_masks, target_masks], dim=1)
        patches_used, _ = torch.sort(perm, dim=1)

        tgt_tok_idx = self._collate_tgt_tok_idx(tgt_tok_idx_list)

        return {
            "masks": masks,
            "context_masks": context_masks,
            "target_masks": target_masks,
            "original_patch_indices": original_patch_indices,
            "channels_to_mask": self.channels_to_mask,
            "patches_used": patches_used,
            "mu_mask": mu_mask,
            "mu_keep_idx": mu_keep_idx,
            "tgt_tok_idx": tgt_tok_idx,
        }

    def __call__(self, batch_size):
        """
        Generate masks for the given batch size.

        Returns:
            Dict with optional keys (unused keys are omitted or None):
            - JEPA/MAE: masks, context_masks, target_masks, original_patch_indices,
              channels_to_mask, patches_used, mu_mask (HIERA_MU only)
            - DINO_IBOT: collated_masks, mask_indices_list, masks_weight,
              upperbound, n_masked_patches
        """
        if self.mask_mode in (
            MaskModes.BLOCKED,
            MaskModes.BLOCKED_TIME_ONLY,
            MaskModes.BLOCKED_SPACE_ONLY,
            MaskModes.BLOCKED_WITH_RANDOM_FILL,
            MaskModes.DINO_IBOT,
            MaskModes.HIERA_MU,
            MaskModes.HIERA_MU_BLOCKED,
        ):
            seed = self.step()
            dev = str(self.device)
            gen_dev = "cuda" if dev.startswith("cuda") else "cpu"
            g = torch.Generator(device=gen_dev)
            g.manual_seed(seed)

        # ---- DINO/iBOT mode: returns a dict ----
        if self.mask_mode == MaskModes.DINO_IBOT:
            return self._generate_dino_ibot_masks(batch_size, generator=g)

        # ---- All other modes: return 6-tuple ----
        if self.mask_mode == MaskModes.BLOCKED:
            masks, context_masks, target_masks, original_patch_indices = self._generate_batched_blocked_mask(
                generator=g, batch_size=batch_size
            )
        elif self.mask_mode == MaskModes.BLOCKED_TIME_ONLY:
            assert (
                self.axial_mask_scale == (1.0, 1.0)
                and self.lateral_mask_scale == (1.0, 1.0)
                and self.aspect_ratio_scale_hw == (1.0, 1.0)
            ), "Axial, lateral, and aspect ratio mask scales must be 1.0 for BLOCKED_TIME_ONLY mode."
            masks, context_masks, target_masks, original_patch_indices = self._generate_batched_blocked_mask(
                generator=g, batch_size=batch_size
            )
        elif self.mask_mode == MaskModes.BLOCKED_SPACE_ONLY:
            assert self.temporal_mask_scale == (
                1.0,
                1.0,
            ), "Temporal mask scale must be 1.0 for BLOCKED_SPACE_ONLY mode."
            masks, context_masks, target_masks, original_patch_indices = self._generate_batched_blocked_mask(
                generator=g, batch_size=batch_size
            )
        elif self.mask_mode == MaskModes.BLOCKED_PATTERNED:
            masks, context_masks, target_masks, original_patch_indices = self._generate_blocked_mask_patterned(
                batch_size=batch_size
            )
        elif self.mask_mode == MaskModes.RANDOM:
            masks, context_masks, target_masks, original_patch_indices = self._generate_random_mask(
                batch_size=batch_size, space_only=False, device=self.device
            )
        elif self.mask_mode == MaskModes.RANDOM_SPACE_ONLY:
            masks, context_masks, target_masks, original_patch_indices = self._generate_random_mask(
                batch_size=batch_size, space_only=True, device=self.device
            )
        elif self.mask_mode == MaskModes.BLOCKED_WITH_RANDOM_FILL:
            masks, context_masks, target_masks, original_patch_indices = self._generate_blocked_with_random_fill(
                batch_size=batch_size, generator=g
            )
        elif self.mask_mode == MaskModes.HIERA_MU:
            return self._generate_hiera_mu_mask(batch_size=batch_size, generator=g)
        elif self.mask_mode == MaskModes.HIERA_MU_BLOCKED:
            return self._generate_hiera_mu_blocked_mask(batch_size=batch_size, generator=g)
        else:
            raise ValueError(f"Unknown mask mode: {self.mask_mode}")

        # perm: [B, patches_used]
        perm = torch.cat([context_masks, target_masks], dim=1)
        patches_used, _ = torch.sort(perm, dim=1)

        return {
            "masks": masks,
            "context_masks": context_masks,
            "target_masks": target_masks,
            "original_patch_indices": original_patch_indices,
            "channels_to_mask": self.channels_to_mask,
            "patches_used": patches_used,
            "mu_mask": None,
        }


def apply_masks(x, masks, concat=True):
    if isinstance(masks, list):
        output = []
        for m in masks:
            mask_keep = m.unsqueeze(-1)
            if x.dim() > 2:
                mask_keep = mask_keep.expand(-1, -1, x.size(-1))

            output += [torch.gather(x, dim=1, index=mask_keep)]
        if not concat:
            return output

        return torch.cat(output, dim=0)
    else:
        indices = masks.unsqueeze(-1)
        if x.dim() > 2:
            indices = indices.expand(-1, -1, x.size(-1))

        return torch.gather(x, dim=1, index=indices)


def apply_masks_rope(x, masks, type="mixed"):
    B, N_masked = masks.shape

    if type == "mixed":
        out = []
        # x = [t_t, t_z, t_y, t_x]
        for pos in x:
            if pos is None:
                out.append(None)
                continue
            # pos: [N_full]
            if pos.dim() == 1:
                pos = pos.unsqueeze(0).expand(B, -1)
            out.append(torch.gather(pos, dim=1, index=masks.to(pos.device)))
        t_t, t_z, t_y, t_x = out
        return t_t, t_z, t_y, t_x

    elif type == "axial":
        # x: [N_full, J]
        if x.dim() == 2:
            # xb: [B, N_full, J] -> [B, N_masked, J]
            xb = x.unsqueeze(0).expand(B, -1, -1)
            # idx: [B, N_masked] -> [B, N_masked, J]
            idx = masks.unsqueeze(-1).expand(-1, -1, xb.size(-1))
            return torch.gather(xb, dim=1, index=idx.to(xb.device))
        else:
            raise ValueError(f"Unexpected axial freqs_cis shape: {x.shape}")

    else:
        raise ValueError(f"Unknown RoPE mask type: {type}")
