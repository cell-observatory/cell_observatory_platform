import math
from enum import Enum
from multiprocessing import Value
from typing import Literal, Optional, Sequence, Tuple

import torch
from hydra.utils import get_method

from cell_observatory_platform.data.data_shapes import MULTICHANNEL_HYPERCUBE
from cell_observatory_platform.training.helpers import get_patch_sizes


# DEPRECATED
# adapted from: https://github.com/facebookresearch/jepa/blob/main/src/masks/multiblock3d.py
class MaskCollator(object):

    def __init__(self, mask_generators, base_collator):
        super(MaskCollator, self).__init__()
        self.base_collator = get_method(base_collator)
        self.mask_generators = mask_generators

    def step(self):
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch):
        batch_size = len(batch)
        # returns a DataSample object with collated
        # data tensor and collated metainfo
        collated_batch = self.base_collator(batch)

        collated_batch["metainfo"]["masks"] = []
        collated_batch["metainfo"]["context_masks"] = []
        collated_batch["metainfo"]["target_masks"] = []
        collated_batch["metainfo"]["channels_to_mask"] = []
        collated_batch["metainfo"]["original_patch_indices"] = []
        for mask_generator in self.mask_generators:
            masks, context_masks, target_masks, original_patch_indices, channels_to_mask = mask_generator(batch_size)

            collated_batch["metainfo"]["masks"].append(masks)
            collated_batch["metainfo"]["context_masks"].append(context_masks)
            collated_batch["metainfo"]["target_masks"].append(target_masks)
            collated_batch["metainfo"]["channels_to_mask"].append(channels_to_mask)
            collated_batch["metainfo"]["original_patch_indices"].append(original_patch_indices)

        # collated batch now is a DataSample object with
        # a batched data tensor and a metainfo dictionary
        # containing lists of batched masks where each
        # list element (B,L) is from a different mask generator
        # if the user wants to train on multiple masks variations
        # per batch (different scales etc., see V-JEPA paper)
        return collated_batch


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


class MaskGenerator(object):
    def __init__(
        self,
        layout: MULTICHANNEL_HYPERCUBE,
        batch_size: int = 1,
        input_format: str = "TZYXC",
        input_shape: Tuple[int, int, int, int, int] = (128, 128, 128, 2),
        patch_shape: Tuple[int, int, int] = (4, 16, 16, 16),
        lateral_mask_scale: Tuple[float, float] = 0.5,
        axial_mask_scale: Tuple[float, float] = 0.5,
        temporal_mask_scale: Tuple[float, float] = 0.5,
        aspect_ratio_scale_hw: Tuple[float, float] = (0.2, 0.4),
        input_format: str = "TZYXC",
        input_shape: Tuple[int, int, int, int, int] = (128, 128, 128, 2),
        patch_shape: Tuple[int, int, int] = (4, 16, 16, 16),
        lateral_mask_scale: Tuple[float, float] = 0.5,
        axial_mask_scale: Tuple[float, float] = 0.5,
        temporal_mask_scale: Tuple[float, float] = 0.5,
        aspect_ratio_scale_hw: Tuple[float, float] = (0.2, 0.4),
        num_blocks: int = 2,
        random_masking_ratio: float = 0.7,
        channels_to_mask: Optional[Sequence[int]] = None,
        time_downsample_pattern: Optional[Sequence[int]] = None,
        mask_mode: MaskModes = MaskModes.RANDOM,
        device: str = "cuda",
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

        self.axial_mask_scale = axial_mask_scale
        self.temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
            input_format=input_format, patch_shape=patch_shape
        )

        self.axial_mask_scale = axial_mask_scale
        self.lateral_mask_scale = lateral_mask_scale
        self.temporal_mask_scale = temporal_mask_scale
        self.aspect_ratio_scale_hw = tuple(aspect_ratio_scale_hw)
        self.aspect_ratio_scale_hw = tuple(aspect_ratio_scale_hw)

        self.random_masking_ratio = random_masking_ratio
        self.num_blocks = num_blocks

        self.channels_to_mask = channels_to_mask
        self.time_downsample_pattern = time_downsample_pattern

        self.mask_mode = mask_mode

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

    def __call__(self, batch_size):
        if self.mask_mode in (
            MaskModes.BLOCKED,
            MaskModes.BLOCKED_TIME_ONLY,
            MaskModes.BLOCKED_SPACE_ONLY,
        ):
            seed = self.step()
            g = torch.Generator()
            g.manual_seed(seed)

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
        else:
            raise ValueError(f"Unknown mask mode: {self.mask_mode}")

        # perm: [B, patches_used]
        perm = torch.cat([context_masks, target_masks], dim=1)
        patches_used, _ = torch.sort(perm, dim=1)

        return masks, context_masks, target_masks, original_patch_indices, self.channels_to_mask, patches_used
        # perm: [B, patches_used]
        perm = torch.cat([context_masks, target_masks], dim=1)
        patches_used, _ = torch.sort(perm, dim=1)

        return masks, context_masks, target_masks, original_patch_indices, self.channels_to_mask, patches_used


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
