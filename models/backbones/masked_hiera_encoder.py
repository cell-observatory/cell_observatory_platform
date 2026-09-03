"""
Adapted from:
https://github.com/facebookresearch/hiera/blob/main/hiera/hiera_mae.py
"""

import math
from typing import Tuple, Optional, Union, List, Literal

import torch
import torch.nn as nn

from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.models.backbones.hiera import Hiera
from cell_observatory_platform.models.layers.patch_embeddings import calc_num_patches


class BlockFusionHeadND(nn.Module):
    """
    Channel-last block downsample + Linear projection.
    Input:  [N, S1, S2, ..., SD, C_in]
    Output: [N, O1, O2, ..., OD, C_out], where Oi = Si // Ki and Ki = kernel_i = stride_i
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.kernel = tuple(int(k) for k in kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.block_elems = int(math.prod(self.kernel))
        self.proj = nn.Linear(self.in_channels * self.block_elems, self.out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, *S, C]
        N = x.shape[0]
        *spatial, C = x.shape[1:]
        D = len(spatial)
        if D != len(self.kernel):
            raise ValueError(f"D mismatch: spatial={spatial} kernel={self.kernel}")
        if C != self.in_channels:
            raise ValueError(f"C mismatch: got {C}, expected {self.in_channels}")

        out_spatial = []
        for s, k in zip(spatial, self.kernel):
            if s % k != 0:
                raise ValueError(f"Spatial dim {s} not divisible by kernel {k}")
            out_spatial.append(s // k)
        out_spatial = tuple(out_spatial)

        # [N, S1,...,SD, C] -> [N, O1, K1, O2, K2, ..., OD, KD, C]
        shape = [N]
        for o, k in zip(out_spatial, self.kernel):
            shape += [o, k]
        shape += [C]
        x = x.reshape(*shape)

        # Permute to group K dims next to C:
        # [N, O1, K1, O2, K2, ..., OD, KD, C] -> [N, O1,...,OD, K1,...,KD, C]
        O_idx = [1 + 2*i for i in range(D)]
        K_idx = [2 + 2*i for i in range(D)]
        x = x.permute([0] + O_idx + K_idx + [x.dim() - 1])

        # Flatten (K1..KD,C) into last dim: [N, *O, (prodK*C)]
        x = x.reshape(N, *out_spatial, self.block_elems * C)

        # Linear over last dim -> [N, *O, C_out]
        x = self.proj(x)
        return x


def apply_fusion_head(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, M, *S, C]  ->  head([B*M, *S, C])  ->  [B, M, *O, C']
    """
    if isinstance(head, nn.Identity):
        return x
    B, M = x.shape[:2]
    x = x.reshape(B * M, *x.shape[2:])   # [N, *S, C]
    x = head(x)                          # [N, *O, C']
    x = x.reshape(B, M, *x.shape[1:])    # [B, M, *O, C']
    return x


class MaskedHieraEncoder(nn.Module):
    def __init__(
        self,
        input_fmt: str,
        input_shape: tuple,
        patch_shape: tuple,
        embed_dim: int,
        num_heads: int,
        drop_path_rate: float,
        q_pool: int,
        q_stride: Union[tuple, int],
        stages: tuple,
        norm_layer: nn.Module,
        mask_unit_size: Optional[Union[tuple, list]] = None,
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        # --- multiscale backbone mode ---
        channel_proj_type: Literal["none", "equalization", "fusion"] = "none",
        multiscale_out_dim: Optional[int] = None,
        multiscale_level_indices: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()

        self.input_fmt = input_fmt
        self.input_shape = input_shape
        self.patch_shape = patch_shape

        self.encoder = Hiera(
            input_fmt=input_fmt,
            input_shape=input_shape,
            patch_shape=patch_shape,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            q_pool=q_pool,
            q_stride=q_stride,
            stages=stages,
            mask_unit_size=tuple(mask_unit_size) if mask_unit_size is not None else None,
            dim_mul=dim_mul,
            head_mul=head_mul,
            norm_layer=norm_layer,
            **kwargs,
        )

        encoder_dim_out = self.encoder.blocks[-1].dim_out

        # --- multiscale backbone settings ---
        self.multiscale_out_dim = int(multiscale_out_dim or encoder_dim_out)

        q_pool = int(self.encoder.q_pool)
        total_levels = q_pool + 1
        all_stage_in_dims = [self.encoder.blocks[i].dim_out for i in self.encoder.stage_ends[:q_pool]]
        all_stage_in_dims.append(encoder_dim_out)

        if multiscale_level_indices is not None:
            self.multiscale_level_indices = list(multiscale_level_indices)
            for idx in self.multiscale_level_indices:
                if not (0 <= idx < total_levels):
                    raise ValueError(
                        f"multiscale_level_indices entry {idx} out of range [0, {total_levels - 1}]"
                    )
        else:
            self.multiscale_level_indices = list(range(total_levels))

        stage_in_dims = [all_stage_in_dims[i] for i in self.multiscale_level_indices]

        q_stride_pow = tuple(int(s) ** int(q_pool) for s in self.encoder.q_stride)
        self.mask_unit_spatial_shape_final = tuple(
            max(1, s // qs) for s, qs in zip(self.encoder.mask_unit_size, q_stride_pow)
        )

        self.channel_proj_type = channel_proj_type
        if self.channel_proj_type == "equalization":
            self.with_intermediates = True
            # Channel equalization only
            self.multiscale_channel_projs = nn.ModuleList(
                [nn.Linear(in_ch, self.multiscale_out_dim, bias=True) for in_ch in stage_in_dims]
            )
            self.multi_scale_fusion_heads = nn.ModuleList()
        
        elif self.channel_proj_type == "fusion":
            self.with_intermediates = True
            # Fusion: BlockFusionHeadND (spatial downsample + channel proj)
            self.multiscale_channel_projs = nn.ModuleList()
            self.multi_scale_fusion_heads = nn.ModuleList()
            for lvl in self.multiscale_level_indices:
                if lvl == q_pool:
                    # Final level: no spatial downsample needed
                    self.multi_scale_fusion_heads.append(nn.Identity())
                else:
                    lvl_mu_size = tuple(
                        max(1, int(mu // (s ** lvl)))
                        for mu, s in zip(self.encoder.mask_unit_size, self.encoder.q_stride)
                    )
                    kernel = tuple(
                        max(1, c // f)
                        for c, f in zip(lvl_mu_size, self.mask_unit_spatial_shape_final)
                    )
                    stage_idx = self.encoder.stage_ends[lvl]
                    in_ch = self.encoder.blocks[stage_idx].dim_out
                    out_ch = encoder_dim_out
                    head = BlockFusionHeadND(in_ch, out_ch, kernel_size=kernel)
                    self.multi_scale_fusion_heads.append(head)
        
        else:
            self.with_intermediates = False

        # Norm: use output channel dim (multiscale_out_dim when projecting, else encoder_dim_out)
        norm_dim = (
            self.multiscale_out_dim
            if self.channel_proj_type == "equalization"
            else encoder_dim_out
        )
        self.encoder_norm = get_norm(norm_layer)(norm_dim)

        self.num_patches, _ = calc_num_patches(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=self.patch_shape,
        )
        self.patch_embedding = self.encoder.patch_embed
        self._init_model_weights()

    def get_decoder_spec(self) -> dict:
        D = len(self.encoder.tokens_spatial_shape)
        mu_grid = tuple(
            self.encoder.tokens_spatial_shape[i] // self.encoder.mask_unit_size[i]
            for i in range(D)
        )
        return {
            "mu_grid": mu_grid,
            # cur_mu_shape in undo_windowing: tokens per MU along each axis
            "tok_in_mu": tuple(self.mask_unit_spatial_shape_final),
            # initial patches per MU at input resolution
            "mu_window_patches": tuple(self.encoder.mask_unit_size),
            "pixels_per_patch": int(self.patch_embedding.pixels_per_patch),
            "in_chans": int(self.input_shape[self.input_fmt.index("C")]),
        }

    def get_decoder_spec_for_level(self, lvl: int) -> dict:
        """
        Decoder spec for level lvl in [0, q_pool].
        Level 0 = first intermediate (1 pool), q_pool = final.
        """
        D = len(self.encoder.tokens_spatial_shape)
        q_pool = int(self.encoder.q_pool)
        q_stride = self.encoder.q_stride
        mask_unit_size = self.encoder.mask_unit_size

        pools = min(lvl, q_pool)
        # total number of tokens in the mask unit at the current level
        tok_in_mu = tuple(
            max(1, int(mu // (s ** pools)))
            for mu, s in zip(mask_unit_size, q_stride)
        )
        # number of mask units along each axis at the current level
        mu_grid = tuple(
            self.encoder.tokens_spatial_shape[i] // mask_unit_size[i]
            for i in range(D)
        )
        stage_idx = self.encoder.stage_ends[lvl] if lvl < q_pool else -1
        dim_out = self.encoder.blocks[stage_idx].dim_out

        return {
            "mu_grid": mu_grid,
            "tok_in_mu": tok_in_mu,
            "mu_window_patches": tuple(mask_unit_size),
            "pixels_per_patch": int(self.patch_embedding.pixels_per_patch),
            "in_chans": int(self.input_shape[self.input_fmt.index("C")]),
            "dim_out": dim_out,
        }

    def get_decoder_specs_per_level(self) -> List[dict]:
        """Decoder specs for selected multiscale levels."""
        return [self.get_decoder_spec_for_level(lvl) for lvl in self.multiscale_level_indices]

    def _init_model_weights(self):
        self.apply(self._mae_init_weights)
        w = self.encoder.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _mae_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_patches(self) -> int:
        return self.num_patches

    def forward_features(self, x: torch.Tensor):
        """Backbone-wrapper contract: SAMBackbone / MaskDinoBackbone /
        Mask2FormerBackbone all call ``backbone.forward_features(tensor)`` on
        every wrapped encoder. Unmasked full-grid forward returning
        channels-FIRST feature maps (one per multiscale level), matching the
        wrappers' ``backbone_output_format: feature_map`` configs.

        With ``masks=None`` and un-windowed intermediates, ``Reroll`` emits
        spatial channels-last ``[B, *grid, C]`` per level -- permute to
        ``[B, C, *grid]`` (squeezing a leading T=1 grid axis for TZYXC).
        """
        if self.channel_proj_type not in ("equalization",):
            raise NotImplementedError(
                "forward_features (backbone-wrapper contract) requires the "
                "multiscale backbone mode (channel_proj_type='equalization'); "
                f"got {self.channel_proj_type!r}. 'fusion' returns windowed "
                "MU-layout tensors (MAE path) and 'none' returns the raw "
                "MU-major sequence -- neither is a spatial feature map."
            )
        x_list, _ = self.forward(x, masks=None, ctx_idx=None, return_windowed=False)
        out = []
        for feat in x_list:
            # [B, *grid, C] -> [B, C, *grid]
            feat = torch.movedim(feat, -1, 1)
            if feat.dim() == 6:
                if feat.shape[2] != 1:
                    raise NotImplementedError(
                        f"wrapper contract expects 5D [B,C,Z,Y,X] maps; got a "
                        f"4D token grid with T={feat.shape[2]} != 1"
                    )
                feat = feat.squeeze(2)            # drop T=1 -> [B, C, Z, Y, X]
            out.append(feat)
        return out

    @torch.jit.ignore
    def get_num_layers(self) -> int:
        return len(self.encoder.blocks)

    def forward(
        self,
        inputs: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        ctx_idx: Optional[torch.Tensor] = None,
        with_intermediates: bool = True,
        with_fusion_heads: bool = True,
        return_windowed: bool = False,
        spatial_kwargs: Optional[dict] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
        if self.with_intermediates:
            x, intermediates, patches = self.encoder(
                inputs, mask=masks, ctx_idx=ctx_idx,
                return_intermediates=True,
                return_windowed_intermediates=return_windowed,
            )

            if self.channel_proj_type == "equalization":
                # Channel equalization only, no fusion. Returns list.
                all_intermediates = intermediates[: self.encoder.q_pool] + intermediates[-1:]
                intermediates_to_eq = [all_intermediates[i] for i in self.multiscale_level_indices]
                x_list = [
                    self.encoder_norm(proj(interm))
                    for proj, interm in zip(self.multiscale_channel_projs, intermediates_to_eq)
                ]
                return x_list, patches
            
            elif self.channel_proj_type == "fusion":
                assert return_windowed, (
                    "channel_proj_type='fusion' requires return_windowed=True "
                    "(intermediates must be in [B, N, *cur_mu_shape, C] format for fusion heads)"
                )
                all_intermediates = intermediates[: self.encoder.q_pool] + intermediates[-1:]
                intermediates_to_fuse = [all_intermediates[i] for i in self.multiscale_level_indices]
                x = 0.0
                for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates_to_fuse):
                    x = x + apply_fusion_head(head, interm_x)
                x = self.encoder_norm(x)
                return x, patches
            
            else:
                x = [self.encoder_norm(interm) for interm in intermediates]
                return x, patches

        else:
            x, patches = self.encoder(inputs, mask=masks, ctx_idx=ctx_idx, return_intermediates=False)
            x = self.encoder_norm(x)
            return x, patches


from cell_observatory_platform.utils.registry import REGISTRY


@REGISTRY.register("backbone", "masked_hiera")
def BUILD(cfg) -> "MaskedHieraEncoder":
    ignore = {"_target_", "BUILD", "name"}
    kwargs = {k: v for k, v in cfg.items() if k not in ignore}
    return MaskedHieraEncoder(**kwargs)