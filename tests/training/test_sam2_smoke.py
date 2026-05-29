"""End-to-end SAM2 forward smoke test.

Builds a small SAM2 via the existing BUILD factory, drives a synthetic batch
through `SAM2VideoPreprocessor` -> `SAM2.forward`, and asserts the loss is
finite + grads flow. This validates that the labelmap target view, criterion
path, prompt sampling rewire, and low-res multimask consumer all hold up
when wired into the actual decoder + memory attention stack.

Run inside the apptainer image so ops3d (RoIAlign3D, FlashDeformAttn) is
available, e.g.

    apptainer exec --nv \\
      -B /clusterfs/nvme/martinalvarez -B /global/home/users/martinalvarezkuglen \\
      /clusterfs/nvme/martinalvarez/apptainer_images/feature_local_db_torch_26_01.sif \\
      bash -c "PYTHONPATH=/global/home/users/martinalvarezkuglen/.cursor/worktrees/sam2-pointrend-7f3e9a2c \\
        python -m pytest tests/training/test_sam2_smoke.py -v"
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

CUDA_AVAILABLE = torch.cuda.is_available()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configs() -> Path:
    return _repo_root() / "configs"


def _compose_smoke_cfg(
    T: int = 1,
    Z: int = 32,
    Y: int = 32,
    X: int = 32,
    C_in: int = 1,
    max_masks: int = 4,
):
    """Programmatic OmegaConf compose that bypasses hydra search paths.

    Loads the SAM2 single-scale model config + dependencies via OmegaConf.load,
    inserts the small `datasets`/`dataset_layout_order`/`quantization`/`seed`
    refs the interpolations rely on, and overrides the masked_encoder backbone
    to the cheapest config (`base`). All shapes are downsized so the smoke
    fits on a single GPU in seconds.
    """
    cfg_root = _configs()

    sam_cfg = OmegaConf.load(cfg_root / "models/meta_arch/sam/base_single_scale.yaml")
    sam_backbone_cfg = OmegaConf.load(cfg_root / "models/backbones/sam_backbone/sam_backbone.yaml")
    masked_encoder_cfg = OmegaConf.load(cfg_root / "models/backbones/masked_encoder/base.yaml")

    # Tiny masked_encoder overrides keep param count + compute small.
    masked_encoder_cfg.embed_dim = 96
    masked_encoder_cfg.depth = 2
    masked_encoder_cfg.num_heads = 4
    masked_encoder_cfg.mlp_ratio = 2.0

    # Train shape mirrors input shape sans mask channel; bbox channel added by
    # last index. dataset_layout_order = TZYXC -> [T, Z, Y, X, C].
    train_shape = [T, Z, Y, X, C_in]
    input_shape = [T, Z, Y, X, C_in + 1]  # +1 for the labelmap channel
    # Patch shape = (T_patch, Z_patch, Y_patch, X_patch, channel_patch). SAM's
    # 3D mask decoder upscales feature tokens by 4x and the prompt encoder's
    # mask_input_size = input/mask_downsample_factor (=4). For the iterative
    # correction loop to skip the (broken-for-3D, antialias=True) resize, the
    # backbone stride must equal mask_downsample_factor * decoder_upscale =
    # 4 * 4 = 16. So we set Z/Y/X patch to 16. Inputs must be multiples of 16.
    patch_shape = [1, 16, 16, 16, None]

    # SAM embed dim must be /3 (mem-encoder pos-enc constraint) and a multiple
    # of mask_decoder transformer_num_heads (default 8). 96 satisfies both.
    sam_cfg.sam_embed_dim = 96
    sam_cfg.mask_decoder_args.transformer_num_heads = 4
    sam_cfg.mask_decoder_args.transformer_mlp_dim = 192
    sam_cfg.mask_decoder_args.transformer_depth = 1
    sam_cfg.memory_attention_args.num_layers = 1
    sam_cfg.memory_attention_args.dim_feedforward = 192
    sam_cfg.criterion_args.num_points = 64
    sam_cfg.criterion_args.activation_checkpoint = False

    cfg = OmegaConf.create(
        {
            "dataset_layout_order": "TZYXC",
            "quantization": "float32",
            "seed": 0,
            "datasets": {
                "train_shape": train_shape,
                "input_shape": input_shape,
                "patch_shape": patch_shape,
                "mask_channel_idx": -1,
            },
            "models": {
                "meta_arch": {"sam": sam_cfg},
                "backbones": {
                    "sam_backbone": sam_backbone_cfg,
                    "masked_encoder": masked_encoder_cfg,
                },
            },
        }
    )

    # Resolve once so downstream BUILD sees plain values.
    OmegaConf.resolve(cfg)
    return cfg, train_shape, input_shape, patch_shape


def _make_data_sample(
    B: int,
    T: int,
    Z: int,
    Y: int,
    X: int,
    C_in: int,
    max_masks: int,
    device: torch.device,
    seed: int = 0,
) -> dict:
    """Build a `{"data_tensor", "metainfo"}` dict matching the collator output.

    `data_tensor`: (B, T, Z, Y, X, C_in + 1). Last channel is the integer
    labelmap (uint16-style ids), preceding channels are image floats. Two
    distinct nonzero ids appear in each volume.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    image = torch.randn(B, T, Z, Y, X, C_in, generator=g)
    labelmap = torch.zeros(B, T, Z, Y, X, 1, dtype=torch.float32)
    # Plant two cubes per (b, t) so the preprocessor has real ids to sample.
    for b in range(B):
        for t in range(T):
            labelmap[b, t, 4:8, 4:8, 4:8, 0] = 7
            labelmap[b, t, 18:24, 18:24, 18:24, 0] = 11

    data_tensor = torch.cat([image, labelmap], dim=-1).to(device)

    targets = []
    for b in range(B):
        targets.append(
            {
                "boxes": torch.tensor(
                    [
                        [4.0, 4.0, 4.0, 8.0, 8.0, 8.0],  # zyxzyx for id 7
                        [18.0, 18.0, 18.0, 24.0, 24.0, 24.0],  # zyxzyx for id 11
                    ],
                    dtype=torch.float32,
                    device=device,
                ),
                "mask_ids": torch.tensor([7, 11], dtype=torch.long, device=device),
                "labels": torch.tensor([0, 0], dtype=torch.long, device=device),
            }
        )

    return {"data_tensor": data_tensor, "metainfo": {"targets": targets}}


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="SAM2 smoke needs CUDA")
@pytest.mark.parametrize(
    "use_point_sampling",
    [
        pytest.param(True, id="pointrend"),
        pytest.param(False, id="dense"),
    ],
)
def test_sam2_forward_smoke(use_point_sampling: bool):
    """Drive both loss paths end-to-end. The preprocessor materializes the
    K=max_masks binary-mask subset; the criterion flag selects PointRend
    point-sampling (`True`) vs dense per-voxel (`False`)."""
    from cell_observatory_platform.models.meta_arch.sam import BUILD as BUILD_SAM2
    from cell_observatory_platform.models.layers.preprocessor import SAM2VideoPreprocessor

    torch.manual_seed(0)
    device = torch.device("cuda")

    B = 1
    T = 1
    Z, Y, X, C_in = 32, 32, 32, 1
    max_masks = 4

    cfg, train_shape, input_shape, patch_shape = _compose_smoke_cfg(
        T=T, Z=Z, Y=Y, X=X, C_in=C_in, max_masks=max_masks
    )
    cfg.models.meta_arch.sam.criterion_args.use_point_sampling = use_point_sampling

    model = BUILD_SAM2(cfg).to(device).train()

    # The preprocessor materializes the per-instance binary-mask subset and the
    # labelmap target view on-device (the collator never builds masks on CPU).
    preprocessor = SAM2VideoPreprocessor(
        transforms_list=None,
        with_masking=False,
        mask_generator=None,
        patch_shape=tuple(patch_shape[:4]),
        dtype="float32",
        input_format="TZYXC",
        input_shape=tuple(input_shape),
        seed=0,
        mask_channel_idx=-1,
        expect_mask_channel=True,
        max_masks=max_masks,
        require_targets=True,
        bbox_format="zyxzyx",
    )

    data_sample = _make_data_sample(
        B=B, T=T, Z=Z, Y=Y, X=X, C_in=C_in, max_masks=max_masks, device=device, seed=0
    )

    processed = preprocessor.forward(data_sample, data_time=0.0, idx=0)

    target_view = processed["metainfo"]["targets"]
    assert "labelmaps" in target_view, "preprocessor must emit labelmap target view"
    assert "instance_ids" in target_view and "valid" in target_view and "presence_t" in target_view

    losses, _ = model(processed)

    for key in ("loss_mask", "loss_dice", "loss_iou", "loss_class"):
        assert key in losses, f"missing {key} in loss dict"
        v = losses[key]
        assert torch.is_tensor(v) and v.ndim == 0, f"{key} expected scalar"
        assert torch.isfinite(v).item(), f"{key} not finite: {v.item()}"

    total = losses[model.criterion.core_loss_key]
    assert torch.is_tensor(total) and torch.isfinite(total).item()
    total.backward()

    # Verify at least one trainable param picked up a gradient.
    grad_found = any(
        p.grad is not None and torch.any(p.grad != 0).item()
        for p in model.parameters()
        if p.requires_grad
    )
    assert grad_found, "no gradients flowed through SAM2 from the labelmap loss"
