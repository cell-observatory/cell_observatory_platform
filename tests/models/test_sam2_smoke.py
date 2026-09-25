"""End-to-end SAM2 forward smoke test.

Builds a small SAM2 via the existing BUILD factory, drives a synthetic batch
through `SAM2VideoPreprocessor` -> `SAM2.forward`, and asserts the loss is
finite + grads flow. This validates that the labelmap target view, criterion
path, prompt sampling rewire, and low-res multimask consumer all hold up
when wired into the actual decoder + memory attention stack.

Needs a GPU and the ops3d extension (RoIAlign3D, FlashDeformAttn).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configs() -> Path:
    return _repo_root() / "configs"


def _load_packaged(path: Path, node: str):
    """`OmegaConf.load` a yaml that carries a hydra `# @package` directive.

    `OmegaConf.load` does not interpret `# @package`, so the loaded document
    keeps its in-file nesting (e.g. `meta_arch.sam: {...}`). Reach through to
    `node` explicitly and assert it exists so a future config restructure fails
    loudly here instead of silently producing an empty/misplaced config.
    """
    doc = OmegaConf.load(path)
    selected = OmegaConf.select(doc, node)
    assert selected is not None, f"expected node '{node}' in {path}"
    return selected


def _compose_smoke_cfg(
    T: int = 1,
    Z: int = 32,
    Y: int = 32,
    X: int = 32,
    C_in: int = 1,
    max_masks: int = 4,
    masked_encoder_overrides: dict | None = None,
):
    """Programmatic OmegaConf compose that bypasses hydra search paths.

    Loads the SAM2 single-scale model config + dependencies via OmegaConf.load,
    inserts the small `datasets`/`dataset_layout_order`/`quantization`/`seed`
    refs the interpolations rely on, and overrides the masked_encoder backbone
    to the cheapest config (`base`). All shapes are downsized so the smoke
    fits on a single GPU in seconds.
    """
    cfg_root = _configs()

    # base_single_scale.yaml starts with `# @package models` and nests its body
    # under `meta_arch.sam:`; the other two yamls carry no package directive and
    # are already flat, so they load as-is.
    sam_cfg = _load_packaged(
        cfg_root / "models/meta_arch/sam/base_single_scale.yaml", "meta_arch.sam"
    )
    sam_backbone_cfg = OmegaConf.load(cfg_root / "models/backbones/sam_backbone/sam_backbone.yaml")
    masked_encoder_cfg = OmegaConf.load(cfg_root / "models/backbones/masked_encoder/base.yaml")

    # Tiny masked_encoder overrides keep param count + compute small.
    masked_encoder_cfg.embed_dim = 96
    masked_encoder_cfg.depth = 2
    masked_encoder_cfg.num_heads = 4
    masked_encoder_cfg.mlp_ratio = 2.0
    if masked_encoder_overrides:
        # applied BEFORE resolve: sam_backbone.backbone_args and the meta-arch's
        # backbone_wrapper_args interpolate this node, so late edits would not propagate
        masked_encoder_cfg = OmegaConf.merge(masked_encoder_cfg, OmegaConf.create(masked_encoder_overrides))

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

    `metainfo` carries a `channel_mapping` naming the labelmap channel's ROLE.
    It is REQUIRED: the role table comes from the DB's
    channel_type/annotation_type arrays, and nothing infers the labelmap from
    its position. A fixture that omits the role gets both channels partitioned
    as model INPUT, which surfaces downstream as a patchify channel-count error.
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

    return {
        "data_tensor": data_tensor,
        "metainfo": {
            "targets": targets,
            "channel_mapping": {C_in: "instance_masks"},
        },
    }


# Build the (use_point_sampling, low_res_multimasks, pred_obj_scores) matrix.
# low_res_multimasks=True requires use_point_sampling=True (asserted at
# MultiStepMultiMasksAndIousLoss.__init__), so filter those rows out instead of
# letting the build fail loudly inside the test.
_SMOKE_PARAMS = [
    pytest.param(
        (use_point_sampling, low_res_multimasks, pred_obj_scores),
        id=(
            ("pointrend" if use_point_sampling else "dense")
            + ("-lowres" if low_res_multimasks else "-highres")
            + ("-objscore" if pred_obj_scores else "-noobj")
        ),
    )
    for use_point_sampling in (True, False)
    for low_res_multimasks in (True, False)
    for pred_obj_scores in (True, False)
    if not (low_res_multimasks and not use_point_sampling)
]


@pytest.mark.cuda
@pytest.mark.parametrize("flags", _SMOKE_PARAMS)
def test_sam2_forward_smoke(flags):
    """Drive both loss paths end-to-end. The preprocessor materializes the
    K=max_masks binary-mask subset; the criterion flag selects PointRend
    point-sampling (`True`) vs dense per-voxel (`False`).

    Additional axes:
      * `low_res_multimasks` - route focal/dice through the low-res mask
        stream. Requires `use_point_sampling=True`.
      * `pred_obj_scores`    - exercise the SAM2 object-score head + the
        criterion's `loss_class` branch. Without this case the obj-score
        path silently drops out of the gradient graph (loss_class weight
        defaults to 0 in the YAML).
    """
    use_point_sampling, low_res_multimasks, pred_obj_scores = flags

    # REGISTRY is populated purely by importing component modules
    # (utils/registry.py has no autodiscovery). `utils._register` walk-imports
    # every component root at import time; without it BUILD_SAM2's
    # REGISTRY.build("backbone"/"criterion", "sam", ...) sees an empty registry.
    import cell_observatory_platform.utils._register  # noqa: F401
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
    cfg.models.meta_arch.sam.criterion_args.low_res_multimasks = low_res_multimasks
    # `pred_obj_scores` is read by BOTH the meta-arch (to build the obj-score
    # head inside the mask decoder) and the criterion (to enable the
    # `loss_class` branch). Keep them in sync, and bump loss_class's weight so
    # the obj-score head actually receives gradient when enabled.
    cfg.models.meta_arch.sam.pred_obj_scores = pred_obj_scores
    cfg.models.meta_arch.sam.criterion_args.pred_obj_scores = pred_obj_scores
    if pred_obj_scores:
        cfg.models.meta_arch.sam.criterion_args.weight_dict.loss_class = 1.0

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
        # `mask_channel_idx` is no longer a constructor arg: the labelmap is
        # always the last channel of `data_tensor`.
        expect_mask_channel=True,
        max_masks=max_masks,
        require_targets=True,
        bbox_format="zyxzyx",
    )

    data_sample = _make_data_sample(
        B=B, T=T, Z=Z, Y=Y, X=X, C_in=C_in, max_masks=max_masks, device=device, seed=0
    )

    processed = preprocessor.forward(data_sample, data_time=0.0, idx=0)

    # `metainfo["targets"]` is now the platform-contract per-image list of dicts;
    # the SAM2-private per-frame view (labelmaps/instance_ids/valid/presence_t)
    # lives under `metainfo["sam2_views"]`.
    target_view = processed["metainfo"]["sam2_views"]
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

    # Per-head gradient assertion. The original `any(...)` check would pass
    # even if an entire head silently lost its connection to the loss graph
    # (e.g. the obj-score head when loss_class weight is zero, or the prompt
    # encoder when a refactor drops the prompt path). Here we require that
    # AT LEAST ONE trainable parameter under each required prefix has a
    # non-zero gradient -- the "head is exercised" contract. We avoid
    # requiring *every* sub-param because the SAM2 prompt encoder/mask
    # decoder include structurally-unused submodules in our config (box-prompt
    # embeddings 2/3 with prob_to_use_box_input=0; the second mask token's
    # hypernetwork MLP when num_multimask_outputs=1 and multimask_output_in_sam
    # =False) that legitimately receive no gradient.
    #
    # Memory encoder/attention are NOT in the required set because with T=1
    # they don't contribute to the loss (their outputs only feed future frames
    # via the memory bank).
    required_prefixes = ["image_encoder", "sam_prompt_encoder", "sam_mask_decoder"]
    if getattr(model, "pred_obj_scores", False):
        # MaskDecoder builds `pred_obj_score_head` only when pred_obj_scores=True.
        required_prefixes.append("sam_mask_decoder.pred_obj_score_head")

    for prefix in required_prefixes:
        params = [(n, p) for n, p in model.named_parameters()
                  if p.requires_grad and n.startswith(prefix)]
        assert params, f"prefix {prefix!r} matched zero trainable params"
        assert any(p.grad is not None and torch.any(p.grad != 0).item() for _, p in params), \
            f"prefix {prefix!r} received no gradient on any of {len(params)} params"


@pytest.mark.cuda
def test_sam2_forward_smoke_channel_adaptive():
    """End-to-end with the channel-adaptive patch embed: the preprocessor maps
    channel_tokens to [B, C, 2] ids from the frozen vocab, SAM2 threads them
    through the backbone, and the token embeddings receive gradient."""
    import cell_observatory_platform.utils._register  # noqa: F401
    from cell_observatory_platform.models.meta_arch.sam import BUILD as BUILD_SAM2
    from cell_observatory_platform.models.layers.preprocessor import SAM2VideoPreprocessor

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, T, Z, Y, X, C_in, max_masks = 1, 1, 32, 32, 32, 2, 4
    vocab = {
        "localization": {"<unk>": 0, "cytosol": 1, "membrane": 2},
        "fluorophore": {"<unk>": 0, "electra2": 1, "mstaygold": 2},
    }

    cfg, train_shape, input_shape, patch_shape = _compose_smoke_cfg(
        T=T, Z=Z, Y=Y, X=X, C_in=C_in, max_masks=max_masks,
        masked_encoder_overrides={
            "patch_embed_type": "channel_adaptive",
            "patch_embed_args": {
                "channel_fusion": "attn_pool", "attn_pool_num_heads": 4,
                "channel_embed": "factorized", "channel_vocab": vocab, "vocab_extra_slots": 2,
            },
        },
    )
    model = BUILD_SAM2(cfg).to(device).train()
    pe = model.image_encoder.backbone.patch_embedding
    assert pe.localization_embed is not None and pe.fluorophore_embed is not None

    preprocessor = SAM2VideoPreprocessor(
        transforms_list=None, with_masking=False, mask_generator=None,
        patch_shape=tuple(patch_shape[:4]), dtype="float32", input_format="TZYXC",
        input_shape=tuple(input_shape), seed=0, expect_mask_channel=True,
        max_masks=max_masks, require_targets=True, bbox_format="zyxzyx",
        channel_vocab=vocab, unknown_policy="error",
    )
    data_sample = _make_data_sample(B=B, T=T, Z=Z, Y=Y, X=X, C_in=C_in, max_masks=max_masks, device=device)
    data_sample["metainfo"]["channel_tokens"] = [[["membrane", "mstaygold"], ["cytosol", "electra2"], None]]

    processed = preprocessor.forward(data_sample, data_time=0.0, idx=0)
    assert processed["metainfo"]["channel_ids"].tolist() == [[[2, 2], [1, 1]]]

    losses, _ = model(processed)
    total = losses[model.criterion.core_loss_key]
    assert torch.isfinite(total).item()
    total.backward()
    assert pe.localization_embed.weight.grad is not None and pe.localization_embed.weight.grad.abs().sum() > 0
    assert pe.fluorophore_embed.weight.grad is not None and pe.fluorophore_embed.weight.grad.abs().sum() > 0
    # only the looked-up rows (2 and 1) receive gradient
    assert pe.localization_embed.weight.grad[0].abs().sum() == 0
