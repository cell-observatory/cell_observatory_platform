import re
import inspect
import logging
from typing import Any, Dict, Literal, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from cell_observatory_platform.models.layers.mlp import get_mlp
from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.training.losses import get_loss_fn
from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.models.layers.attention import RopeAttention
from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.data.masking.mask_generator import apply_masks
from cell_observatory_platform.models.backbones.maskedencoder import MaskedEncoder
from cell_observatory_platform.models.heads.maskedpredictor import MaskedPredictor
from cell_observatory_platform.models.layers.patch_embeddings import calc_num_patches
from cell_observatory_platform.models.backbones.masked_hiera_encoder import MaskedHieraEncoder
from cell_observatory_platform.models.heads.masked_hiera_predictor import MaskedHieraPredictor
from cell_observatory_platform.training.helpers import get_masked_input_data, get_nparams_and_flops
from cell_observatory_platform.models.meta_arch import utils as mo

logger = logging.getLogger(__name__)


CONFIGS = {
    "mae-tiny": {
        "embed_dim": 192,
        "decoder_embed_dim": 96,
        "depth": 12,
        "decoder_depth": 3,
        "num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4,
    },
    "mae-small": {
        "embed_dim": 384,
        "decoder_embed_dim": 192,
        "depth": 12,
        "decoder_depth": 6,
        "num_heads": 6,
        "decoder_num_heads": 6,
        "mlp_ratio": 4,
    },
    "mae-base": {
        "embed_dim": 768,
        "decoder_embed_dim": 256,
        "depth": 12,
        "decoder_depth": 8,
        "num_heads": 12,
        "decoder_num_heads": 8,
        "mlp_ratio": 4,
    },
    "mae-large": {
        "embed_dim": 1024,
        "decoder_embed_dim": 512,
        "depth": 24,
        "decoder_depth": 8,
        "num_heads": 16,
        "decoder_num_heads": 8,
        "mlp_ratio": 4,
    },
    "mae-huge": {
        "embed_dim": 1280,
        "decoder_embed_dim": 512,
        "depth": 32,
        "decoder_depth": 8,
        "num_heads": 16,
        "decoder_num_heads": 8,
        "mlp_ratio": 4,
    },
    "mae-2billion": {
        "embed_dim": 2560,
        "decoder_embed_dim": 512,
        "depth": 24,
        "decoder_depth": 8,
        "num_heads": 32,
        "decoder_num_heads": 8,
        "mlp_ratio": 4,
    },
    "mae-6billion": {
        "embed_dim": 4096,
        "decoder_embed_dim": 512,
        "depth": 32,
        "decoder_depth": 8,
        "num_heads": 32,
        "decoder_num_heads": 8,
        "mlp_ratio": 4,
    },
    "mae-giant": {
        "embed_dim": 1408,
        "decoder_embed_dim": 512,
        "depth": 40,
        "decoder_depth": 8,
        "num_heads": 16,
        "decoder_num_heads": 8,
        "mlp_ratio": 48 / 11,
    },
    "mae-gigantic": {
        "embed_dim": 1664,
        "decoder_embed_dim": 1024,
        "depth": 48,
        "decoder_depth": 16,
        "num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 64 / 13,
    },
    "mae-enormous": {
        "embed_dim": 1792,
        "decoder_embed_dim": 1024,
        "depth": 56,
        "decoder_depth": 16,
        "num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 8.5714285714,
    },
}


class MaskedAutoEncoder(nn.Module):
    def __init__(
        self,
        model_template: Literal[
            "mae",  # custom use `embed_dim`, `decoder_embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            "mae-tiny",
            "mae-small",
            "mae-base",
            "mae-large",
            "mae-huge",
            "mae-giant",
            "mae-gigantic",
        ] = "mae",
        input_fmt="TZYXC",
        input_shape: tuple = (16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        embed_dim=768,
        decoder_embed_dim=256,
        depth=12,
        decoder_depth=8,
        num_heads=12,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        proj_drop_rate=0.0,
        att_drop_rate=0.0,
        drop_path_rate=0.1,
        init_std=0.02,
        fixed_dropout_depth=False,
        norm_layer: Union[nn.Module, Literal["RmsNorm", "LayerNorm", "SyncBatchNorm", "GroupNorm"]] = "RmsNorm",
        act_layer: Union[nn.Module, Literal["GELU", "SiLU", "LeakyReLU", "GLU", "Sigmoid", "Tanh"]] = "SiLU",
        mlp_layer: Union[nn.Module, Literal["Mlp", "SwiGLU"]] = "SwiGLU",
        abs_sincos_enc: bool = False,
        rope_pos_enc: bool = True,
        rope_random_rotation_per_head: bool = True,
        rope_type: Literal["mixed", "axial", "custom"] = "axial",
        rope_theta: float = 10.0,
        weight_init_type: str = "mae",
        mlp_wide_silu: bool = False,
        loss_fn: str = "l2_masked",
        with_auxiliary_loss: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        backbone_type: Literal["vit", "hiera"] = "vit",
        # Hiera-specific parameters
        hiera_q_pool: int = 3,
        # int default expands to the token-grid rank inside Hiera; the old
        # rank-2 tuple default zip-truncated against 3-D/4-D grids (Hiera
        # now validates rank loudly).
        hiera_q_stride: Union[tuple, int] = 2,
        hiera_stages: tuple = (2, 3, 16, 3),
        hiera_mask_unit_size: tuple = (8,8,8),
        # Deformable Attention parameters
        use_deformable_attn: bool = False,
        da_n_points: int = 4,
        da_n_levels: int = 1,
        buffer_device: str = "cuda",
        output_metadata: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()

        if model_template in CONFIGS.keys():
            config = CONFIGS[model_template]
            self.depth = config["depth"]
            self.decoder_depth = config["decoder_depth"]
            self.embed_dim = config["embed_dim"]
            self.decoder_embed_dim = config["decoder_embed_dim"]
            self.num_heads = config["num_heads"]
            self.decoder_num_heads = config["decoder_num_heads"]
            self.mlp_ratio = config["mlp_ratio"]
        else:
            self.depth = depth
            self.decoder_depth = decoder_depth
            self.embed_dim = embed_dim
            self.decoder_embed_dim = decoder_embed_dim
            self.num_heads = num_heads
            self.decoder_num_heads = decoder_num_heads
            self.mlp_ratio = mlp_ratio

        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        self.input_fmt = input_fmt
        self.input_shape = input_shape
        # Inference contract: inference_step returns full-context per-patch token
        # FEATURES (not saved; VLM/downstream). Metadata is lazy-built in
        # get_output_metadata() because it needs the encoder (built below); a config
        # may still override by passing output_metadata.
        self._output_metadata_override = output_metadata
        self.output_metadata = None

        axis_to_value = dict(zip(input_fmt, input_shape))
        self.in_chans = axis_to_value["C"]
        self.num_frames = axis_to_value.get("T", None)

        self.patch_shape = patch_shape

        self.proj_drop_rate = proj_drop_rate
        self.att_drop_rate = att_drop_rate
        self.drop_path_rate = drop_path_rate
        self.fixed_dropout_depth = fixed_dropout_depth

        self.init_std = init_std

        self.norm_layer = get_norm(norm_layer)
        self.act_layer = get_activation(act_layer)
        self.mlp_layer = get_mlp(mlp_layer)

        # positional encoding parameters
        self.abs_sincos_enc = abs_sincos_enc
        self.rope_pos_enc = rope_pos_enc
        self.rope_type = rope_type
        self.rope_theta = rope_theta
        self.wide_silu = mlp_wide_silu
        self.rope_random_rotation_per_head = rope_random_rotation_per_head
        self.backbone_type = backbone_type

        if use_deformable_attn:
            raise NotImplementedError(
                "use_deformable_attn on the MASKED ENCODER is not supported: the "
                "DA scatter path needs spatial_kwargs['mask_indices']/'full_seq_len' "
                "and no producer exists -- the context-subset sequence would be "
                "interpreted against the full token grid (broadcast crash at best, "
                "silent OOB kernel reads at worst). JEPA's predictor-side DA and "
                "the Hiera DA path are unaffected (full-length sequences)."
            )

        if use_deformable_attn:
            _, token_shape = calc_num_patches(
                input_fmt=input_fmt, input_shape=input_shape, patch_shape=patch_shape,
            )
            assert self.input_fmt in ["ZYXC"], f"Input format {self.input_fmt} not supported yet."
            spatial_dims = [s for s in token_shape[:-1] if s is not None]
            if len(spatial_dims) == 4:
                raise ValueError(
                    "Deformable attention is not supported for 4D (T,Z,Y,X) token grids. "
                    f"Got token_shape={token_shape}."
                )

        if backbone_type == "vit":
            self.masked_encoder = MaskedEncoder(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
                channels=self.in_chans,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                proj_drop_rate=self.proj_drop_rate,
                att_drop_rate=self.att_drop_rate,
                drop_path_rate=self.drop_path_rate,
                fixed_dropout_depth=self.fixed_dropout_depth,
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
                mlp_layer=self.mlp_layer,
                init_std=self.init_std,
                abs_sincos_enc=self.abs_sincos_enc,
                rope_pos_enc=self.rope_pos_enc,
                rope_random_rotation_per_head=self.rope_random_rotation_per_head,
                rope_type=self.rope_type,
                rope_theta=self.rope_theta,
                mlp_wide_silu=mlp_wide_silu,
                dtype=self.dtype,
                use_deformable_attn=use_deformable_attn,
                da_n_points=da_n_points,
                da_n_levels=da_n_levels,
            )
            self.masked_decoder = MaskedPredictor(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
                input_embed_dim=self.embed_dim,
                output_embed_dim=self.masked_encoder.patch_embedding.pixels_per_patch,
                embed_dim=self.decoder_embed_dim,
                depth=self.decoder_depth,
                num_heads=self.decoder_num_heads,
                mlp_ratio=self.mlp_ratio,
                proj_drop_rate=self.proj_drop_rate,
                att_drop_rate=self.att_drop_rate,
                drop_path_rate=self.drop_path_rate,
                fixed_dropout_depth=self.fixed_dropout_depth,
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
                mlp_layer=self.mlp_layer,
                init_std=self.init_std,
                abs_sincos_enc=self.abs_sincos_enc,
                rope_pos_enc=self.rope_pos_enc,
                rope_random_rotation_per_head=self.rope_random_rotation_per_head,
                rope_type=self.rope_type,
                rope_theta=self.rope_theta,
                mlp_wide_silu=mlp_wide_silu,
                dtype=self.dtype,
                use_deformable_attn=use_deformable_attn,
                da_n_points=da_n_points,
                da_n_levels=da_n_levels,
            )
        elif backbone_type == "hiera":
            # MAE Hiera: encoder -> fuse (BlockFusionHeadND) -> single-level predictor -> pixels -> loss
            # Multiscale / return_intermediates not supported; only single-level DA or SA.
            self.masked_encoder = MaskedHieraEncoder(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path_rate=self.drop_path_rate,
                q_pool=hiera_q_pool,
                q_stride=hiera_q_stride,
                stages=hiera_stages,
                mask_unit_size=hiera_mask_unit_size,
                norm_layer=self.norm_layer,
                channel_proj_type="fusion",
            )
            self.masked_decoder = MaskedHieraPredictor(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
                encoder_dim_out=self.masked_encoder.encoder.blocks[-1].dim_out,
                decoder_embed_dim=self.decoder_embed_dim,
                decoder_depth=self.decoder_depth,
                decoder_num_heads=self.decoder_num_heads,
                decoder_spec=self.masked_encoder.get_decoder_spec(),
                mlp_ratio=self.mlp_ratio,
                norm_layer=self.norm_layer,
                use_deformable_attn=use_deformable_attn,
                da_n_points=da_n_points,
                da_n_levels=1,
                prediction_mode="pixels",
                # NOTE: output_embed_dim used for lowest_level prediction mode
                output_embed_dim=None,
            )
        else:
            raise ValueError(f"Unsupported backbone_type={backbone_type}")

        self.weight_init_type = weight_init_type

        self.loss_fn = get_loss_fn(loss_fn)
        self.with_auxiliary_loss = with_auxiliary_loss

        self._init_model_weights(buffer_device=buffer_device)

    def _init_model_weights(self, buffer_device: str | None = None):
        # MAE model init adapted from:
        # https://github.com/facebookresearch/mae/blob/main/models_mae.py
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        if self.backbone_type == "vit":
            w = self.masked_encoder.patch_embedding.proj.weight
            torch.nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        elif self.backbone_type == "hiera":
            w = self.masked_encoder.encoder.patch_embed.proj.weight.data
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        if hasattr(self.masked_decoder, "token_param"):
            torch.nn.init.normal_(self.masked_decoder.token_param, std=0.02)

        for mod in self.modules():
            if isinstance(mod, RopeAttention):
                mod.init_rope_parameters(device=buffer_device)

    def get_param_groups(
        self,
        weight_decay: float,
        enc_layer_decay: float = 1.0,
        dec_layer_decay: float = 1.0,
        **kwargs,
    ) -> List[Dict]:
        """Layer-wise LR decay param groups for MAE pretraining.

        Encoder and decoder each get per-layer groups with decaying LR scales.
        Biases, norms, pos_embedding, cls_token, and token_param get zero WD.
        
        Adapted from https://github.com/facebookresearch/mae/util/lr_decay.py
        """
        # "pos_embed" (no trailing "ding") is Hiera's positional embedding param
        # name — without it the Hiera pos_embed silently received full WD.
        NO_WD_KEYWORDS = ("pos_embedding", "pos_embed", "cls_token", "token_param")

        enc_L = self.masked_encoder.get_num_layers()
        dec_L = self.masked_decoder.get_num_layers()

        enc_scales = [enc_layer_decay ** (enc_L - i) for i in range(enc_L + 1)]
        dec_scales = [dec_layer_decay ** (dec_L - i) for i in range(dec_L + 1)]

        def _layer_id(suffix: str, L: int) -> int:
            if suffix.startswith(("patch_embedding", "pos_embedding", "cls_token",
                                  "token_param", "patch_projection")):
                return 0
            m = re.search(r"transformer_blocks\.(\d+)", suffix)
            if m:
                return int(m.group(1)) + 1
            return L

        def _is_no_wd(name: str, p) -> bool:
            if p.ndim == 1:
                return True
            return any(kw in name for kw in NO_WD_KEYWORDS)

        groups: Dict[str, Dict] = {}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue

            if n.startswith("masked_encoder."):
                side = "enc"
                suffix = n[len("masked_encoder."):]
                L, scales = enc_L, enc_scales
            elif n.startswith("masked_decoder."):
                side = "dec"
                suffix = n[len("masked_decoder."):]
                L, scales = dec_L, dec_scales
            else:
                side, suffix, L, scales = "other", n, 0, [1.0]

            wd = 0.0 if _is_no_wd(n, p) else weight_decay
            decay_tag = "no_decay" if wd == 0.0 else "decay"
            lid = _layer_id(suffix, L)
            lr_scale = scales[min(lid, len(scales) - 1)]

            gname = f"{side}_layer_{lid}_{decay_tag}"
            if gname not in groups:
                groups[gname] = {"lr_scale": lr_scale, "weight_decay": wd, "params": []}
            groups[gname]["params"].append(p)

        return list(groups.values())

    # TODO: implement for each meta_arch
    # @torch.jit.ignore
    # def _get_nparams_and_flops(
    #     self, batch_size: int, device: Literal["cuda", "meta"] = "cuda", masking_ratio: float = 0.0
    # ):
    #     if device == "cuda":
    #         # TODO: test this path more thoroughly
    #         with torch.cuda.device(device):
    #             input_shape = (batch_size, *self.input_shape)
    #             data_sample = get_masked_input_data(
    #                 self,
    #                 inputs=input_shape,
    #                 device="cuda",
    #                 mask_ratio=masking_ratio,
    #             )
    #             seq_len = int(self.get_num_patches()) * (1 - masking_ratio)
    #             model_summary = get_nparams_and_flops(self, data_sample, seq_len)
    #             model_param_count, num_flops_per_token = (
    #                 model_summary["total_params"],
    #                 model_summary["training_flops"],
    #             )
    #     elif device == "meta":
    #         print(f"Warning: using 'meta' device for flops/nparams calculation is not yet supported.")
    #         return -1, -1
    #     else:
    #         # TODO: add support for meta device calculation for other backends
    #         raise ValueError(f"Unsupported device for flops/nparams calculation: {device}")
    #     return model_param_count, num_flops_per_token

    @torch.jit.ignore
    def get_patch_embedding(self):
        return self.masked_encoder.patch_embedding

    @torch.jit.ignore
    def get_encoder(self):
        return self.masked_encoder

    @torch.jit.ignore
    def get_decoder(self):
        return self.masked_decoder

    @torch.jit.ignore
    def get_num_patches(self):
        if self.backbone_type == "hiera":
            return self.masked_encoder.get_num_patches()
        elif self.backbone_type == "vit":
            if self.abs_sincos_enc:
                return self.masked_encoder.pos_embedding.num_patches
            num_patches, _ = calc_num_patches(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
            )
            return num_patches
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")
    
    @torch.jit.ignore
    def get_output_metadata(self):
        # Lazy-built (needs the encoder): declares the per-level token FEATURES that
        # inference_step returns. A config-provided override wins.
        if self.output_metadata is None:
            self.output_metadata = (
                self._output_metadata_override
                if self._output_metadata_override is not None
                else mo.output_metadata(**mo.build_feature_metadata(self, self.masked_encoder))
            )
        return self.output_metadata

    def forward(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]
        if self.backbone_type == "vit":
            return self._forward_vit(inputs, meta)
        elif self.backbone_type == "hiera":
            return self._forward_hiera(inputs, meta)
        else:
            raise ValueError(f"Unsupported backbone_type={self.backbone_type}")

    def _forward_vit(self, inputs: torch.Tensor, meta: dict):
        masks, spatial_kwargs = meta["masks"][0], meta.get("spatial_kwargs", None)
        context_masks, patches_used = meta["context_masks"][0], meta["patches_used"][0]
        target_masks, original_patch_indices = meta["target_masks"][0], meta["original_patch_indices"][0]

        x, patches = self.masked_encoder(
            inputs, masks=context_masks, spatial_kwargs=spatial_kwargs,
        )
        x = self.masked_decoder(
            x,
            original_patch_indices=original_patch_indices,
            target_masks=target_masks,
            patches_used=patches_used,
            spatial_kwargs=spatial_kwargs,
        )

        if patches_used is not None:
            target_idx_in_patches_used = torch.searchsorted(patches_used, target_masks)
        else:
            target_idx_in_patches_used = target_masks
        targets = apply_masks(patches, masks=target_masks)
        predictions = apply_masks(x, masks=target_idx_in_patches_used)

        if self.with_auxiliary_loss:
            aux_loss_meta = {
                "targets": patches,
                "predictions": x,
                "patches_used": patches_used,
                "target_masks": target_masks,
                "prediction_masks": target_idx_in_patches_used,
            }
        else:
            aux_loss_meta = None

        loss, aux_losses = self.loss_fn(targets, predictions, masks.sum(), aux_loss_meta)
        loss_dict = {"step_loss": loss, **(aux_losses or {})}
        return loss_dict, predictions

    def _forward_hiera(self, inputs: torch.Tensor, meta: dict):
        masks, target_masks, patches_used = meta["masks"][0], meta["target_masks"][0], meta["patches_used"][0]
        mu_mask, mu_keep_idx = meta.get("mu_mask", [None])[0], meta.get("mu_keep_idx", [None])[0]
        spatial_kwargs = meta.get("spatial_kwargs", None)

        if mu_mask is None or mu_keep_idx is None:
            raise ValueError(
                "Hiera backbone requires mu_mask and mu_keep_idx in meta. "
                "Use mask_mode=HIERA_MU or HIERA_MU_BLOCKED."
            )

        x, patches = self.masked_encoder(
            inputs, masks=mu_mask, ctx_idx=mu_keep_idx, spatial_kwargs=spatial_kwargs,
            with_intermediates=True,
            with_fusion_heads=True,
            # channel_proj_type="fusion" hard-requires windowed [B, N, *mu, C]
            # intermediates (masked_hiera_encoder asserts); the fusion heads
            # emit un-windowed output, so the decoder contract is unchanged.
            return_windowed=True,
        )
        x = self.masked_decoder(
            x, mu_mask=mu_mask, ctx_idx=mu_keep_idx, spatial_kwargs=spatial_kwargs,
        )

        if patches_used is not None:
            target_idx_in_patches_used = torch.searchsorted(patches_used, target_masks)
        else:
            target_idx_in_patches_used = target_masks
        targets = apply_masks(patches, masks=target_masks)
        predictions = apply_masks(x, masks=target_idx_in_patches_used)

        if self.with_auxiliary_loss:
            aux_loss_meta = {
                "targets": patches,
                "predictions": x,
                "patches_used": patches_used,
                "target_masks": target_masks,
                "prediction_masks": target_idx_in_patches_used,
            }
        else:
            aux_loss_meta = None

        loss, aux_losses = self.loss_fn(targets, predictions, masks.sum(), aux_loss_meta)

        loss_dict = {"step_loss": loss, **(aux_losses or {})}
        return loss_dict, predictions

    def evaluate_step(self, data_sample: dict) -> dict:
        """EVAL — consumed by PretrainEvaluator (loss-only). Returns the loss dict;
        metric keys (e.g. ``"step_loss"``) are read directly. The masked-target
        patch payload is dropped (unused by the metric, and 'reconstruction' was a
        misnomer for masked patch embeddings)."""
        loss_dict, _ = self.forward(data_sample)
        return dict(loss_dict)

    @torch.no_grad()
    def inference_step(self, data_sample: dict) -> dict:
        """INFERENCE — full-context per-patch token FEATURES (VLM/downstream); NOT saved.
        No masking, no unpatchify, no raster un-window, no scatter -- flat [B, N, C]
        per level (single for vit/single-scale hiera; every level for multiscale).
        See :func:`meta_arch.utils.extract_token_features`."""
        return mo.extract_token_features(self.masked_encoder, self, data_sample)


def _extract_model_kwargs(cfg: Mapping[str, Any]) -> dict:
    sig = inspect.signature(MaskedAutoEncoder.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    ignore = {"_target_", "BUILD", "name"}
    kwargs = {}
    for k in cfg.keys():
        if k in ignore or k not in allowed:
            continue
        kwargs[k] = cfg[k]
    return kwargs


from cell_observatory_platform.utils.registry import REGISTRY


@REGISTRY.register("model", "mae")
def BUILD(cfg: Mapping[str, Any]) -> MaskedAutoEncoder:
    model_cfg = cfg.models.meta_arch.mae
    return MaskedAutoEncoder(**_extract_model_kwargs(model_cfg))
