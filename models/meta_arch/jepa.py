import sys
import inspect
import logging
from copy import deepcopy
from typing import Any, Literal, Mapping, Optional, Union, List

import torch
import torch.nn as nn
from deepspeed.runtime.zero import GatheredParameters
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from cell_observatory_platform.models.layers.mlp import get_mlp
from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.training.losses import get_loss_fn
from cell_observatory_platform.models.layers.attention import RopeAttention
from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.data.masking.mask_generator import apply_masks
from cell_observatory_platform.models.heads.maskedpredictor import MaskedPredictor
from cell_observatory_platform.models.backbones.maskedencoder import MaskedEncoder
from cell_observatory_platform.models.layers.patch_embeddings import calc_num_patches
from cell_observatory_platform.models.backbones.masked_hiera_encoder import MaskedHieraEncoder
from cell_observatory_platform.models.heads.masked_hiera_predictor import MaskedHieraPredictor
from cell_observatory_platform.training.helpers import get_masked_input_data, get_nparams_and_flops, init_weights

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


CONFIGS = {
    "jepa-tiny": {
        "embed_dim": 192,
        "predictor_embed_dim": 96,
        "depth": 12,
        "predictor_depth": 3,
        "num_heads": 3, 
        "predictor_num_heads": 3,
        "mlp_ratio": 4,
    },
    "jepa-small": {
        "embed_dim": 384,
        "predictor_embed_dim": 192,
        "depth": 12,
        "predictor_depth": 6,
        "num_heads": 6,
        "predictor_num_heads": 6,
        "mlp_ratio": 4,
    },
    "jepa-base": {
        "embed_dim": 768,
        "predictor_embed_dim": 384,
        "depth": 12,
        "predictor_depth": 12,
        "num_heads": 12,
        "predictor_num_heads": 12,
        "mlp_ratio": 4,
    },
    "jepa-large": {
        "embed_dim": 1024,
        "predictor_embed_dim": 384,
        "depth": 24,
        "predictor_depth": 12,
        "num_heads": 16,
        "predictor_num_heads": 12,
        "mlp_ratio": 4,
    },
    "jepa-huge": {
        "embed_dim": 1280,
        "predictor_embed_dim": 384,
        "depth": 32,
        "predictor_depth": 12,
        "num_heads": 16,
        "predictor_num_heads": 12,
        "mlp_ratio": 4,
    },
    "jepa-2billion": {
        "embed_dim": 2560,
        "predictor_embed_dim": 512,
        "depth": 24,
        "predictor_depth": 8,
        "num_heads": 32,
        "predictor_num_heads": 8,
        "mlp_ratio": 4,
    },
    "jepa-6billion": {
        "embed_dim": 4096,
        "predictor_embed_dim": 512,
        "depth": 32,
        "predictor_depth": 8,
        "num_heads": 32,
        "predictor_num_heads": 8,
        "mlp_ratio": 4,
    },
    "jepa-giant": {
        "embed_dim": 1408,
        "predictor_embed_dim": 512,
        "depth": 40,
        "predictor_depth": 12,
        "num_heads": 16,
        "predictor_num_heads": 12,
        "mlp_ratio": 48 / 11,
    },
    "jepa-gigantic": {
        "embed_dim": 1664,
        "predictor_embed_dim": 1024,
        "depth": 48,
        "predictor_depth": 16,
        "num_heads": 16,
        "predictor_num_heads": 16,
        "mlp_ratio": 64 / 13,
    },
    "jepa-enormous": {
        "embed_dim": 1792,
        "predictor_embed_dim": 1024,
        "depth": 56,
        "predictor_depth": 16,
        "num_heads": 16,
        "predictor_num_heads": 16,
        "mlp_ratio": 8.5714285714,
    },
}


class JEPA(nn.Module):
    def __init__(
        self,
        model_template: Literal[
            "jepa",  # custom use `embed_dim`, `predictor_embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            "jepa-tiny",
            "jepa-small",
            "jepa-base",
            "jepa-large",
            "jepa-huge",
            "jepa-giant",
            "jepa-gigantic",
        ] = "jepa",
        input_fmt="TZYXC",
        input_shape: tuple = (16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        embed_dim=768,
        predictor_embed_dim=256,
        depth=12,
        predictor_depth=8,
        num_heads=12,
        predictor_num_heads=8,
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
        weight_init_type: str = "vjepa2",
        mlp_wide_silu: bool = False,
        loss_fn: str = "l1_masked",
        dtype: torch.dtype = torch.bfloat16,
        backbone_type: Literal["vit", "hiera"] = "vit",
        # Hiera-specific parameters
        hiera_q_pool: int = 3,
        hiera_q_stride: tuple = (2, 2),
        hiera_stages: tuple = (2, 3, 16, 3),
        hiera_mask_unit_size: Optional[tuple] = None,
        buffer_device: str = "cuda",
        # Deformable Attention parameters
        use_deformable_attn: bool = False,
        da_n_points: int = 4,
        da_n_levels: int = 1,
        # Multiscale (Hiera only)
        multiscale: bool = False,
        multiscale_out_dim: Optional[int] = None,
        multiscale_level_indices: Optional[List[int]] = None,
        target_only_predictor: bool = False,
        **kwargs,
    ):
        super().__init__()

        if model_template in CONFIGS.keys():
            config = CONFIGS[model_template]
            self.depth = config["depth"]
            self.predictor_depth = config["predictor_depth"]
            self.embed_dim = config["embed_dim"]
            self.predictor_embed_dim = config["predictor_embed_dim"]
            self.num_heads = config["num_heads"]
            self.predictor_num_heads = config["predictor_num_heads"]
            self.mlp_ratio = config["mlp_ratio"]
        else:
            self.depth = depth
            self.predictor_depth = predictor_depth
            self.embed_dim = embed_dim
            self.predictor_embed_dim = predictor_embed_dim
            self.num_heads = num_heads
            self.predictor_num_heads = predictor_num_heads
            self.mlp_ratio = mlp_ratio

        self.input_fmt = input_fmt
        self.input_shape = input_shape
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
        self.mlp_wide_silu = mlp_wide_silu
        self.rope_random_rotation_per_head = rope_random_rotation_per_head
        self.backbone_type = backbone_type

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
            # JEPA ViT: encoder -> single-level predictor (SA or DA)
            self.multiscale = False
            self.input_encoder = MaskedEncoder(
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
                dtype=dtype
            )
            self.target_predictor = MaskedPredictor(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
                input_embed_dim=self.embed_dim,
                output_embed_dim=self.embed_dim,
                embed_dim=self.predictor_embed_dim,
                depth=self.predictor_depth,
                num_heads=self.predictor_num_heads,
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
                dtype=dtype,
                # Deformable Attention parameters: DA or SA
                use_deformable_attn=use_deformable_attn,
                da_n_points=da_n_points,
                da_n_levels=1,
            )
        elif backbone_type == "hiera":
            self.multiscale = multiscale
            self.target_only_predictor = target_only_predictor
            self.multiscale_level_indices = multiscale_level_indices
            self.input_encoder = MaskedHieraEncoder(
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
                # NOTE: for one-level prediction (i.e. multiscale=False), we fuse
                #       all feature maps and predict at the lowest level.
                #       For multi-level prediction (i.e. multiscale=True), we equalize
                #       number of channels and predict at all levels.
                channel_proj_type="equalization" if multiscale else "fusion",
                multiscale_out_dim=multiscale_out_dim if multiscale else None,
                multiscale_level_indices=multiscale_level_indices if multiscale else None,
            )

            # Hiera predictor options: N-level multiscale (DA or SA) or 1-level single-scale (DA or SA).
            # - multiscale=True: N levels, all_levels prediction; DA concatenates levels, SA processes per-level (slower).
            # - multiscale=False: 1 level (fusion), lowest_level prediction.
            if multiscale:
                if multiscale_out_dim is None:
                    raise ValueError("multiscale_out_dim must be set when multiscale=True")
                encoder_dim_out = multiscale_out_dim
                decoder_specs = self.input_encoder.get_decoder_specs_per_level()
                prediction_mode = "all_levels"
            else:
                encoder_dim_out = self.input_encoder.encoder.blocks[-1].dim_out
                decoder_specs = self.input_encoder.get_decoder_spec()
                prediction_mode = "lowest_level"

            self.target_predictor = MaskedHieraPredictor(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
                encoder_dim_out=encoder_dim_out,
                decoder_embed_dim=self.predictor_embed_dim,
                decoder_depth=self.predictor_depth,
                decoder_num_heads=self.predictor_num_heads,
                decoder_spec=decoder_specs,
                mlp_ratio=self.mlp_ratio,
                norm_layer=self.norm_layer,
                prediction_mode=prediction_mode,
                # NOTE: output_embed_dim used for lowest_level prediction mode
                output_embed_dim=encoder_dim_out,
                use_deformable_attn=use_deformable_attn,
                da_n_points=da_n_points,
                da_n_levels=da_n_levels,
                target_only_predictor=target_only_predictor,
            )

        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        self.weight_init_type = weight_init_type

        self.model_initialized = False
        self.init_model_weights(buffer_device=buffer_device)

        # NOTE: do deepcopy after weight init
        self.target_encoder = deepcopy(self.input_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.loss_fn = get_loss_fn(loss_fn)

    # see training/hooks.py for usage
    def ema_update(self, beta=0.99):
        def collect_params(params):
            return [p for p in params if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]

        with torch.no_grad():
            for iparam, tparam in zip(self.input_encoder.parameters(), self.target_encoder.parameters()):
                fetch = collect_params([iparam, tparam])
                # fetches parameters from other ranks if needed
                with GatheredParameters(fetch, enabled=len(fetch) > 0):
                    # input_encoder*B + (target_encoder - input_encoder)*(1-B) = target_encoder*B + input_encoder*(1-B)
                    tparam.data.copy_(torch.lerp(iparam.data, tparam.data, beta))

    def init_model_weights(self, buffer_device: str | None = None):
        # FIXME: hack until we move model inits back into each model class
        if not self.model_initialized:
            # TODO: move model inits back into each model class
            init_weights(self, weight_init_type=self.weight_init_type)
            for mod in self.modules():
                if isinstance(mod, RopeAttention):
                    mod.init_rope_parameters(device=buffer_device)
            self.model_initialized = True
    
    @torch.jit.ignore
    def _get_nparams_and_flops(
        self, batch_size: int, device: Literal["cuda", "meta"] = "cuda", masking_ratio: float = 0.0
    ):
        if device == "cuda":
            # TODO: test this path more thoroughly
            with torch.cuda.device(device):
                input_shape = (batch_size, *self.input_shape)
                data_sample = get_masked_input_data(
                    self,
                    inputs=input_shape,
                    device="cuda",
                    mask_ratio=masking_ratio,
                )
                seq_len = int(self.get_num_patches()) * (1 - masking_ratio)
                model_summary = get_nparams_and_flops(self, data_sample, seq_len)
                model_param_count, num_flops_per_token = (
                    model_summary["total_params"],
                    model_summary["training_flops"],
                )
        elif device == "meta":
            print(f"Warning: using 'meta' device for flops/nparams calculation is not yet supported.")
            return -1, -1
        else:
            # TODO: add support for meta device calculation for other backends
            raise ValueError(f"Unsupported device for flops/nparams calculation: {device}")
        return model_param_count, num_flops_per_token

    @torch.jit.ignore
    def get_patch_embedding(self):
        return self.input_encoder.patch_embedding

    @torch.jit.ignore
    def get_input_encoder(self):
        return self.input_encoder

    @torch.jit.ignore
    def get_target_encoder(self):
        return self.target_encoder

    @torch.jit.ignore
    def get_predictor(self):
        return self.target_predictor

    @torch.jit.ignore
    def get_num_patches(self):
        if self.backbone_type == "hiera":
            return self.input_encoder.get_num_patches()
        elif self.backbone_type == "vit":
            if self.abs_sincos_enc:
                return self.input_encoder.pos_embedding.num_patches
            num_patches, _ = calc_num_patches(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
            )
            return num_patches
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")

    @staticmethod
    def _select_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, N, C]
        idx: [B, M] long
        ->   [B, M, C]
        """
        return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    def forward(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]
        if self.backbone_type == "vit":
            return self._forward_vit(inputs, meta)
        elif self.backbone_type == "hiera":
            return self._forward_hiera(inputs, meta)
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")

    def _forward_vit(self, inputs: torch.Tensor, meta: dict):
        masks, spatial_kwargs = meta["masks"][0], meta.get("spatial_kwargs", None)
        context_masks, patches_used = meta["context_masks"][0], meta["patches_used"][0]
        target_masks, original_patch_indices = meta["target_masks"][0], meta["original_patch_indices"][0]

        embedding, patches = self.input_encoder(
            inputs, masks=context_masks, spatial_kwargs=spatial_kwargs,
        )
        predictions = self.target_predictor(
            embedding,
            original_patch_indices=original_patch_indices,
            target_masks=target_masks,
            patches_used=patches_used,
            spatial_kwargs=spatial_kwargs,
        )

        with torch.no_grad():
            targets, _ = self.target_encoder(inputs, spatial_kwargs=spatial_kwargs)

        if patches_used is not None:
            target_idx_in_patches_used = torch.searchsorted(patches_used, target_masks)
        else:
            target_idx_in_patches_used = target_masks
        targets = apply_masks(targets, masks=target_masks)
        predictions = apply_masks(predictions, masks=target_idx_in_patches_used)
        loss, aux_losses = self.loss_fn(targets, predictions, masks.sum())

        loss_dict = {"step_loss": loss, **(aux_losses or {})}
        return loss_dict, predictions

    def _forward_hiera(self, inputs: torch.Tensor, meta: dict):
        mu_mask, spatial_kwargs = meta.get("mu_mask", [None])[0], meta.get("spatial_kwargs", None)
        mu_keep_idx, tgt_tok_idx = meta.get("mu_keep_idx", [None])[0], meta.get("tgt_tok_idx", [None])[0]

        if mu_mask is None or mu_keep_idx is None or tgt_tok_idx is None:
            raise ValueError(
                "JEPA+Hiera requires mu_mask, mu_keep_idx, tgt_tok_idx in meta. "
                "Use mask_mode=HIERA_MU or HIERA_MU_BLOCKED with q_stride and q_pool."
            )

        # Slice tgt_tok_idx to match selected multiscale levels
        if self.multiscale and self.multiscale_level_indices is not None:
            if isinstance(tgt_tok_idx, list):
                tgt_tok_idx = [tgt_tok_idx[i] for i in self.multiscale_level_indices]

        ctx_out, _ = self.input_encoder(
            inputs,
            masks=mu_mask,
            ctx_idx=mu_keep_idx,
            with_intermediates=True,
            with_fusion_heads=not self.multiscale,
            return_windowed=True,
            spatial_kwargs=spatial_kwargs,
        )

        if isinstance(ctx_out, list):
            pred_out = self.target_predictor(
                ctx_out,
                mu_mask=[mu_mask] * len(ctx_out),
                ctx_idx=[mu_keep_idx] * len(ctx_out),
                tgt_idx_list=tgt_tok_idx if self.target_only_predictor else None,
                spatial_kwargs=spatial_kwargs,
            )
        else:
            pred_out = self.target_predictor(
                ctx_out,
                mu_mask=mu_mask,
                ctx_idx=mu_keep_idx,
                spatial_kwargs=spatial_kwargs,
            )

        with torch.no_grad():
            tgt_out, _ = self.target_encoder(
                inputs,
                masks=None,
                ctx_idx=None,
                with_intermediates=True,
                with_fusion_heads=not self.multiscale,
                return_windowed=True,
                spatial_kwargs=spatial_kwargs,
            )

        if not isinstance(pred_out, list):
            pred_out = [pred_out]
        if not isinstance(tgt_out, list):
            tgt_out = [tgt_out]
        if not isinstance(tgt_tok_idx, list):
            tgt_tok_idx = [tgt_tok_idx]

        pred_sels, tgt_sels = [], []
        for lvl in range(len(pred_out)):
            tgt_tokens = tgt_out[lvl].reshape(
                tgt_out[lvl].shape[0], -1, tgt_out[lvl].shape[-1],
            )
            if self.target_only_predictor:
                pred_sels.append(pred_out[lvl])
            else:
                pred_sels.append(self._select_tokens(pred_out[lvl], tgt_tok_idx[lvl]))
            tgt_sels.append(self._select_tokens(tgt_tokens, tgt_tok_idx[lvl]))

        total_count = sum(idx.numel() for idx in tgt_tok_idx)

        if len(pred_sels) == 1:
            loss, aux = self.loss_fn(tgt_sels[0], pred_sels[0], total_count)
        else:
            loss, aux = self.loss_fn(tgt_sels, pred_sels, total_count)

        loss_dict = {"step_loss": loss, **(aux or {})}
        return loss_dict, pred_sels[0] if len(pred_sels) == 1 else pred_sels


def _extract_model_kwargs(cfg: Mapping[str, Any]) -> dict:
    sig = inspect.signature(JEPA.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    ignore = {"_target_", "BUILD"}
    kwargs = {}
    for k in cfg.keys():
        if k in ignore or k not in allowed:
            continue
        kwargs[k] = cfg[k]
    return kwargs


def BUILD(cfg: Mapping[str, Any]) -> JEPA:
    model_cfg = cfg.models.meta_arch.jepa
    return JEPA(**_extract_model_kwargs(model_cfg))
