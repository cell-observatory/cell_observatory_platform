"""
Adapted from:
https://github.com/facebookresearch/dinov3/dinov3/models/vision_transformer.py
"""

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from timm.layers import RmsNorm
from cell_observatory_platform.models.layers.mlp import get_mlp
from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.training.helpers import named_apply
from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.models.layers.transformer import Transformer
from cell_observatory_platform.models.layers.layer_scale import LayerScale
from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding
from cell_observatory_platform.models.layers.positional_encoding import RopePositionEmbedding

logger = logging.getLogger(__name__)

# ffn_layer_dict = {
#     "mlp": Mlp,
#     "swiglu": SwiGLUFFN,
#     "swiglu32": partial(SwiGLUFFN, align_to=32),
#     "swiglu64": partial(SwiGLUFFN, align_to=64),
#     "swiglu128": partial(SwiGLUFFN, align_to=128),
# }

# norm_layer_dict = {
#     "layernorm": partial(nn.LayerNorm, eps=1e-6),
#     "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
#     "rmsnorm": RMSNorm,
# }

# dtype_dict = {
#     "fp32": torch.float32,
#     "fp16": torch.float16,
#     "bf16": torch.bfloat16,
# }


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    # TODO: add reset_parameters for PatchEmbedding
    # if isinstance(module, PatchEmbedding):
    #     module.reset_parameters()
    # TODO: add reset_parameters for RmsNorm
    # if isinstance(module, RmsNorm):
    #     module.reset_parameters()


class DinoEncoder(nn.Module):
    def __init__(
        self,
        input_format: str = "TZYXC",
        input_shape: tuple = (16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "LayerNorm",
        qk_norm: bool = False,
        ffn_layer: str = "Mlp_ListFwdMixin",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        rope_type: Literal["mixed", "axial", "custom"] = "custom",
        **ignored_kwargs,
    ):
        super().__init__()
        
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = get_norm(norm_layer)

        # NOTE: # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.num_features = self.embed_dim = embed_dim

        self.input_fmt = input_format
        self.input_shape = input_shape
        self.patch_shape = patch_shape

        axis_to_value = dict(zip(input_format, input_shape))
        self.in_chans = axis_to_value["C"]
        self.num_frames = axis_to_value.get("T", None)
        
        self.patch_embedding = PatchEmbedding(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=self.patch_shape,
            embed_dim=self.embed_dim,
            channels=self.in_chans,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))

        # apply_rope_v1 (axial/mixed) does not slice register/cls tokens off the
        # sequence before rotating; only custom (apply_rope_v2) computes
        # prefix = N - sin.shape[-2] and passes prefix tokens through unrotated.
        # This encoder ALWAYS carries prefix tokens (1 cls + n_storage_tokens
        # registers), so axial/mixed would rotate them with grid frequencies
        # (silent wrong PE or shape mismatch). Restrict to custom until v1
        # prefix slicing is implemented.
        if rope_type != "custom":
            raise ValueError(
                f"DinoEncoder carries {1 + self.n_storage_tokens} prefix tokens and "
                f"requires rope_type='custom' (axial/mixed do not slice prefix tokens "
                f"before rotating); got rope_type={rope_type!r}."
            )

        self.rope_embed = RopePositionEmbedding(
            input_fmt=self.input_fmt,
            embed_dim=embed_dim,
            num_heads=num_heads,
            theta=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=TORCH_DTYPES[pos_embed_rope_dtype].value if isinstance(pos_embed_rope_dtype, str) else pos_embed_rope_dtype,
            device=device,
        )

        ffn_layer_cls = get_mlp(ffn_layer)
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            Transformer(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                mlp_layer=ffn_layer_cls,
                layer_scale_init_values=layerscale_init,
                ffn_mask_k_bias=mask_k_bias,
                rope_type=rope_type
                # device=device,
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)

        # NOTE: this norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # NOTE: when untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # NOTE: when untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None

        # NOTE: initialized in meta_arch/dino.py
        self.head = nn.Identity()
        
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        self._init_model_weights()

    def _init_model_weights(self):
        self.rope_embed._init_model_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def _get_input_shape(self, x: Tensor) -> Tuple[int, tuple]:
        if self.input_fmt == "TZYXC":
            B, T, Z, Y, X, C = x.shape
            # NOTE: patch shape matches input shape so we include channel dim
            Tp, Zp, Yp, Xp, _ = self.patch_shape
            return B, (T // Tp, Z // Zp, Y // Yp, X // Xp)
        elif self.input_fmt == "ZYXC":
            B, Z, Y, X, C = x.shape
            # NOTE: patch shape matches input shape so we include channel dim
            Zp, Yp, Xp, _ = self.patch_shape
            return B, (Z // Zp, Y // Yp, X // Xp)
        else:
            raise ValueError(f"Invalid input format: {self.input_fmt}")

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        B, input_shape = self._get_input_shape(x)
        x = self.patch_embedding(x, shape=tuple(x.shape[1:]))

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token

        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, input_shape

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x, rope = [], []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, input_shape = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(input_shape)

        # Compute rope ONCE per forward: the embedding draws random train-time
        # coordinate augmentations (shift/jitter/rescale), which must be shared
        # by every block (DINOv3 semantics), not resampled per layer — and the
        # sin/cos rebuild is depth× redundant.
        if self.rope_embed is not None:
            rope_sincos = [self.rope_embed(shape=input_shape) for input_shape in rope]
        else:
            rope_sincos = [None for _ in rope]
        for blk in self.blocks:
            x = blk(x, pos_enc=rope_sincos)

        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        
        return output

    def forward_features(
        self, 
        x: Tensor | List[Tensor], 
        masks: Optional[Tensor] = None
    ) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, input_shape = self.prepare_tokens_with_masks(x)

        # NOTE: If n is an int, we take the n last blocks. If it's a list, we take those layers.
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        # Rope once per forward (shared random augs across blocks) — see
        # forward_features_list.
        if self.rope_embed is not None:
            rope_sincos = self.rope_embed(shape=input_shape)
        else:
            rope_sincos = None
        for i, blk in enumerate(self.blocks):
            x = blk(x, pos_enc=rope_sincos)
            if i in blocks_to_take:
                output.append(x)

        assert len(output) == len(blocks_to_take), f"Only {len(output)} / {len(blocks_to_take)} blocks found!"
        return output

    def _reshape_output(self, output: List[Tensor], data_tensor_shape: tuple) -> List[Tensor]:
        if self.input_fmt == "TZYXC":
            B, T, Z, Y, X, C = data_tensor_shape
            Tp, Zp, Yp, Xp, _ = self.patch_shape
            return [
                out.reshape(B, T // Tp, Z // Zp, Y // Yp, X // Xp, -1).permute(0, 5, 1, 2, 3, 4).contiguous()
                for out in output
            ]
        elif self.input_fmt == "ZYXC":
            B, Z, Y, X, C = data_tensor_shape
            Zp, Yp, Xp, _ = self.patch_shape
            return [
                out.reshape(B, Z // Zp, Y // Yp, X // Xp, -1).permute(0, 4, 1, 2, 3).contiguous()
                for out in output
            ]
        else:
            raise ValueError(f"Invalid input format: {self.input_fmt}")

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        # NOTE: Layers or n last layers to take.
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)

        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed

        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]

        if reshape:
            outputs = self._reshape_output(outputs, self.input_shape)

        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        out = self.forward_features(*args, **kwargs)
        if is_training:
            return out
        else:
            return self.head(out["x_norm_clstoken"])