"""
Adapted from:
https://github.com/facebookresearch/hiera/blob/main/hiera/hiera_mae.py
"""

import math
from typing import Dict, Any, Literal, Optional, List, Union

import torch
import torch.nn as nn

from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.models.backbones.encoder import Encoder
from cell_observatory_platform.models.layers.utils import get_reference_points


class MaskedHieraPredictor(nn.Module):
    def __init__(
        self,
        input_fmt: str,
        input_shape: tuple,
        patch_shape: tuple,
        encoder_dim_out: Union[int, List[int]],
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        decoder_spec: Union[Dict[str, Any], List[Dict[str, Any]]],
        mlp_ratio: float = 4.0,
        norm_layer: str = "LayerNorm",
        prediction_mode: Literal["pixels", "lowest_level", "all_levels"] = "all_levels",
        output_embed_dim: Optional[int] = None,
        # Deformable Attention parameters
        use_deformable_attn: bool = False,
        da_n_points: int = 4,
        da_n_levels: int = 1,
        target_only_predictor: bool = False,
    ):
        super().__init__()

        self.input_fmt = input_fmt
        self.input_shape = input_shape
        self.patch_shape = patch_shape

        self.prediction_mode = prediction_mode
        self.use_deformable_attn = use_deformable_attn
        self.target_only_predictor = target_only_predictor

        if isinstance(decoder_spec, list):
            self.decoder_specs = decoder_spec
            self.num_levels = len(decoder_spec)
        else:
            self.decoder_specs = [decoder_spec]
            self.num_levels = 1

        if isinstance(encoder_dim_out, (list, tuple)):
            encoder_dims = list(encoder_dim_out)
        else:
            encoder_dims = [encoder_dim_out] * self.num_levels

        assert len(encoder_dims) == self.num_levels, "encoder_dim_out must be a list of length num_levels"

        # NOTE: all_levels mode requires multi-level decoder_spec (DA or SA both supported)
        if prediction_mode == "all_levels":
            assert self.num_levels > 1, "all_levels requires multi-level decoder_spec"

        # NOTE: use last spec for single-level attrs
        last_spec = self.decoder_specs[-1]
        self.mu_grid = tuple(int(x) for x in last_spec["mu_grid"])
        self.tok_in_mu = tuple(int(x) for x in last_spec["tok_in_mu"])
        self.tok_in_mu_prod = int(math.prod(self.tok_in_mu))
        # mu_window_patches: initial patches per MU at input resolution (mask_unit_size)
        self.mu_window_patches = tuple(int(x) for x in last_spec["mu_window_patches"])
        self.pixels_per_patch = int(last_spec["pixels_per_patch"])
        self.num_decoder_tokens = int(math.prod(self.mu_grid) * self.tok_in_mu_prod)
        
        # patches per decoder token along each axis: mu_window_patches / tok_in_mu
        patches_per_token_shape = []
        for d in range(len(self.mu_grid)):
            w = self.mu_window_patches[d]
            t = self.tok_in_mu[d]
            if w % t != 0:
                raise ValueError(
                    f"mu_window_patches[{d}]={w} must be divisible by tok_in_mu[{d}]={t}"
                )
            patches_per_token_shape.append(w // t)
        self.patches_per_token_shape = tuple(patches_per_token_shape)
        self.patches_per_token = int(math.prod(self.patches_per_token_shape))
        self.pixels_per_decoder_token = self.patches_per_token * self.pixels_per_patch

        # Per-level derived quantities
        self._level_tok_in_mu = []
        self._level_tok_in_mu_prod = []
        self._level_num_tokens = []
        for spec in self.decoder_specs:
            tim = tuple(int(x) for x in spec["tok_in_mu"])
            mg = tuple(int(x) for x in spec["mu_grid"])
            mwp = tuple(int(x) for x in spec["mu_window_patches"])
            self._level_tok_in_mu.append(tim)
            self._level_tok_in_mu_prod.append(int(math.prod(tim)))
            self._level_num_tokens.append(int(math.prod(mg) * math.prod(tim)))

        # Per-level modules
        self.decoder_embeds = nn.ModuleList([
            nn.Linear(dim, decoder_embed_dim) for dim in encoder_dims
        ])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ntok, decoder_embed_dim))
            for ntok in self._level_num_tokens
        ])

        if self.num_levels > 1:
            self.level_embed = nn.Parameter(torch.zeros(self.num_levels, decoder_embed_dim))
        else:
            self.level_embed = None

        Norm = get_norm(norm_layer)

        self.decoder = Encoder(
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=Norm,
            input_fmt=input_fmt,
            input_shape=input_shape,
            patch_shape=patch_shape,
            rope_pos_enc=False,
            rope_random_rotation_per_head=False,
            use_deformable_attn=use_deformable_attn,
            da_n_points=da_n_points,
            da_n_levels=da_n_levels,
        )
        self.decoder_norm = Norm(decoder_embed_dim)

        # Prediction heads
        if prediction_mode == "pixels":
            self.decoder_pred = nn.Linear(decoder_embed_dim, self.pixels_per_decoder_token)
        elif prediction_mode == "lowest_level":
            self.decoder_pred = nn.Linear(decoder_embed_dim, int(output_embed_dim))
        elif prediction_mode == "all_levels":
            dims_out = [int(encoder_dims[i]) for i in range(self.num_levels)]
            self.decoder_preds = nn.ModuleList([
                nn.Linear(decoder_embed_dim, d) for d in dims_out
            ])
        else:
            raise NotImplementedError(f"prediction_mode={prediction_mode}")

        self._init_model_weights()

    def _init_model_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for pe in self.decoder_pos_embeds:
            nn.init.trunc_normal_(pe, std=0.02)
        self.apply(self._mae_init_weights)

    def _mae_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _scatter_context_to_full_sequence(
        self,
        tokens: torch.Tensor,
        ctx_idx: torch.Tensor,
        mask_token: torch.Tensor,
        tok_in_mu_prod: int,
        num_decoder_tokens: int,
    ) -> torch.Tensor:
        B, C = tokens.shape[0], tokens.shape[-1]
        x_seq = mask_token.view(1, 1, -1).expand(B, num_decoder_tokens, C).clone()
        src = tokens.reshape(B, -1, C)
        base = ctx_idx.unsqueeze(-1).long() * tok_in_mu_prod
        offs = torch.arange(tok_in_mu_prod, device=tokens.device, dtype=ctx_idx.dtype).view(1, 1, -1)
        seq_idx = (base + offs).view(B, -1).long()
        index = seq_idx.unsqueeze(-1).expand(-1, -1, C)
        x_seq.scatter_(dim=1, index=index, src=src)
        return x_seq

    def _tokens_to_patch_tokens(
        self,
        x: torch.Tensor,
        mu_grid: tuple,
        tok_in_mu: tuple,
        patches_per_token_shape: tuple,
        pixels_per_patch: int,
    ) -> torch.Tensor:
        B, N, K = x.shape
        D = len(mu_grid)
        pp = pixels_per_patch
        sub = patches_per_token_shape

        full_tok = tuple(mu_grid[d] * tok_in_mu[d] for d in range(D))
        
        expected_N = int(math.prod(full_tok))
        expected_K = int(math.prod(sub) * pp)
        if N != expected_N or K != expected_K:
            raise ValueError(f"_tokens_to_patch_tokens: N={N} vs {expected_N}, K={K} vs {expected_K}")

        # [B, N, sub_prod*pp] -> [B, *full_tok, *sub, pp]
        x = x.view(B, *full_tok, *sub, pp)

        # Interleave (full_tok_i, sub_i) per axis
        # current: [B, full0..full{D-1}, sub0..sub{D-1}, pp]
        # want:    [B, full0, sub0, full1, sub1, ..., pp]
        perm = [0]
        full_start = 1
        sub_start = 1 + D
        for i in range(D):
            perm += [full_start + i, sub_start + i]
        perm += [1 + 2 * D]  # pp

        x = x.permute(perm).contiguous()

        patch_grid = [full_tok[i] * sub[i] for i in range(D)]
        x = x.view(B, *patch_grid, pp)
        return x.view(B, int(math.prod(patch_grid)), pp)

    def _build_spatial_kwargs(
        self,
        decoder_specs: List[dict],
        device: torch.device,
        batch_size: int,
    ) -> dict:
        spatial_shapes_list = []
        tokens_per_level = []
        for spec in decoder_specs:
            mg = tuple(int(x) for x in spec["mu_grid"])
            tim = tuple(int(x) for x in spec["tok_in_mu"])
            full_tok = tuple(mg[d] * tim[d] for d in range(len(mg)))
            n_tok = int(math.prod(full_tok))
            tokens_per_level.append(n_tok)
            spatial_shapes_list.append(list(full_tok))

        spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long, device=device)
        level_start_index = torch.zeros(len(decoder_specs), dtype=torch.long, device=device)
        level_start_index[1:] = torch.tensor(tokens_per_level, device=device).cumsum(0)[:-1]
        valid_ratios = torch.ones(batch_size, len(decoder_specs), len(self.mu_grid), device=device)

        return {
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
            "tokens_per_level": tokens_per_level,
        }

    def _forward(
        self,
        inputs: torch.Tensor,
        mu_mask: torch.Tensor,
        ctx_idx: torch.Tensor,
        spatial_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        tokens = self.decoder_embeds[-1](inputs)
        x_seq = self._scatter_context_to_full_sequence(
            tokens, ctx_idx.to(inputs.device), self.mask_token,
            self.tok_in_mu_prod, self.num_decoder_tokens,
        )
        x = x_seq + self.decoder_pos_embeds[-1]
        if self.level_embed is not None:
            x = x + self.level_embed[-1].view(1, 1, -1)
        x = self.decoder(x, masks=None, pos_enc=None, spatial_kwargs=spatial_kwargs)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        if self.prediction_mode == "pixels":
            return self._tokens_to_patch_tokens(
                x, self.mu_grid, self.tok_in_mu,
                self.patches_per_token_shape, self.pixels_per_patch,
            )
        return x

    def _forward_list(
        self,
        inputs_list: List[torch.Tensor],
        mu_mask_list: List[torch.Tensor],
        ctx_idx_list: List[torch.Tensor],
        tgt_idx_list: Optional[List[torch.Tensor]] = None,
        spatial_kwargs: Optional[dict] = None,
    ) -> List[torch.Tensor]:
        """
        Multilevel predictor forward.
        - Always builds dense per-level value_maps (full token grid) from context tokens + mask_token.
        - If target_only_predictor=True, builds queries only at target indices.
        - If target_only_predictor=False, runs decoder on full sequences (target and context tokens).
        Returns list of per-level outputs.
        """
        assert self.prediction_mode == "all_levels", "_forward_list requires prediction_mode='all_levels'"
        if self.target_only_predictor and tgt_idx_list is None:
            raise ValueError("target_only_predictor=True requires tgt_idx_list to be passed")
        if (not self.target_only_predictor) and (tgt_idx_list is not None):
            raise ValueError(
                "target_only_predictor=False but tgt_idx_list was passed; "
                "pass tgt_idx_list only when target_only_predictor=True"
            )

        device = inputs_list[0].device
        B = inputs_list[0].shape[0]
        Cdec = self.mask_token.shape[-1]

        # Build dense per-level value maps (full token grids)
        value_maps: List[torch.Tensor] = []
        for lvl, (inp, ctx) in enumerate(zip(inputs_list, ctx_idx_list)):
            tok_in_mu_prod = self._level_tok_in_mu_prod[lvl]
            num_tok = self._level_num_tokens[lvl]

            # tokens: [B, N_ctx, Cdec]
            tokens = self.decoder_embeds[lvl](inp)
            # x_seq: [B, num_tok, Cdec]
            x_seq = self._scatter_context_to_full_sequence(
                tokens=tokens,
                ctx_idx=ctx.to(inp.device),
                mask_token=self.mask_token,
                tok_in_mu_prod=tok_in_mu_prod,
                num_decoder_tokens=num_tok,
            )

            # x: [B, num_tok, Cdec]
            x = x_seq + self.decoder_pos_embeds[lvl]
            x = x + self.level_embed[lvl].view(1, 1, -1)
            # value_maps: [B, num_tok, Cdec]
            value_maps.append(x)

        sp_kw = self._build_spatial_kwargs(self.decoder_specs, device, B)

        # Build decoder inputs
        if self.target_only_predictor:
            # DA cross-attn consumes dense value maps as memory
            sp_kw["value_maps"] = value_maps
            sp_kw["tgt_idx_list"] = tgt_idx_list
            sp_kw["value_flatten"] = torch.cat(value_maps, dim=1)

            # Cache reference points for target queries once (reused by every decoder layer)
            spatial_shapes = sp_kw["spatial_shapes"]
            valid_ratios = sp_kw["valid_ratios"]
            tokens_per_level = sp_kw["tokens_per_level"]

            # full_ref: [B, sum(full_len), n_levels, 3]
            full_ref = get_reference_points(spatial_shapes, valid_ratios, device=device)
            
            ref_parts = []
            offset = 0
            n_levels = spatial_shapes.shape[0]
            for i in range(n_levels):
                full_len_i = tokens_per_level[i]
                level_ref = full_ref[:, offset:offset + full_len_i]
                offset += full_len_i

                idx = tgt_idx_list[i]
                if idx.dim() == 1:
                    idx = idx.unsqueeze(0).expand(B, -1)
                idx = idx.to(device=device, dtype=torch.long)
                idx_exp = idx[:, :, None, None].expand(-1, -1, n_levels, 3)
                ref_parts.append(level_ref.gather(1, idx_exp))
            # reference_points: [B, sum(Ntgt), n_levels, 3]
            sp_kw["reference_points"] = torch.cat(ref_parts, dim=1)

            # queries: mask_token + pos_embed[tgt] + level_embed
            x_list: List[torch.Tensor] = []
            for lvl, tgt_idx in enumerate(tgt_idx_list):
                if tgt_idx.dim() == 1:
                    tgt_idx = tgt_idx.unsqueeze(0).expand(B, -1)
                tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
                # pe: # [1, Nfull, Cdec]
                pe = self.decoder_pos_embeds[lvl]
                # idx: [B, Ntgt, Cdec]
                idx = tgt_idx.unsqueeze(-1).expand(-1, -1, Cdec)
                # pe_tgt: [B, Ntgt, Cdec]
                pe_tgt = pe.expand(B, -1, -1).gather(1, idx)

                q = self.mask_token.expand(B, tgt_idx.shape[1], Cdec) + pe_tgt
                q = q + self.level_embed[lvl].view(1, 1, -1)
                x_list.append(q)
        else:
            x_list = value_maps

        out_list = self.decoder(x_list, masks=None, pos_enc=None, spatial_kwargs=sp_kw)

        outputs: List[torch.Tensor] = []
        for lvl, out in enumerate(out_list):
            out = self.decoder_norm(out)
            out = self.decoder_preds[lvl](out)
            outputs.append(out)
        return outputs

    def forward(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor]],
        mu_mask: Union[torch.Tensor, List[torch.Tensor]],
        ctx_idx: Union[torch.Tensor, List[torch.Tensor]],
        tgt_idx_list: Optional[List[torch.Tensor]] = None,
        spatial_kwargs: Optional[dict] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(inputs, (list, tuple)):
            return self._forward_list(inputs, mu_mask, ctx_idx, tgt_idx_list, spatial_kwargs)
        elif isinstance(inputs, torch.Tensor):
            return self._forward(inputs, mu_mask, ctx_idx, spatial_kwargs)
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")

    @torch.jit.ignore
    def get_num_layers(self) -> int:
        return self.decoder.get_num_layers()


from cell_observatory_platform.utils.registry import REGISTRY


@REGISTRY.register("head", "hiera_decoder")
def BUILD(cfg) -> "MaskedHieraPredictor":
    ignore = {"_target_", "BUILD", "name"}
    kwargs = {k: v for k, v in cfg.items() if k not in ignore}
    return MaskedHieraPredictor(**kwargs)