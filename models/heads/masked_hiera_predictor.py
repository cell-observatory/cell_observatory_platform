"""
Adapted from:
https://github.com/facebookresearch/hiera/blob/main/hiera/hiera_mae.py
"""

import math
from typing import Dict, Any, Literal, Optional

import torch
import torch.nn as nn

from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.models.backbones.encoder import Encoder


class MaskedHieraPredictor(nn.Module):
    def __init__(
        self,
        input_fmt: str,
        input_shape: tuple,
        patch_shape: tuple,
        encoder_dim_out: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        decoder_spec: Dict[str, Any],
        mlp_ratio: float = 4.0,
        norm_layer: str = "LayerNorm",
        prediction_mode: Literal["pixels", "lowest_level"] = "pixels",
        output_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.input_fmt = input_fmt
        self.input_shape = input_shape
        self.patch_shape = patch_shape

        Norm = get_norm(norm_layer)

        # ---------------------------------------------------------------------
        # decoder_spec:
        # Required keys:
        #   mu_grid:            tuple[int,...]  # number of mask-units/windows along each axis
        #   tok_in_mu:          tuple[int,...]  # number of tokens inside each MU along each axis
        #   mu_window_patches:  tuple[int,...]  # patches per MU at input (mask_unit_size)
        #   pixels_per_patch:   int             # pp
        # Optional:
        #   in_chans:           int
        # ---------------------------------------------------------------------
        self.mu_grid = tuple(int(x) for x in decoder_spec["mu_grid"])
        self.tok_in_mu = tuple(int(x) for x in decoder_spec["tok_in_mu"])
        self.tok_prod = int(math.prod(self.tok_in_mu))
        self.mu_window_patches = tuple(int(x) for x in decoder_spec["mu_window_patches"])
        self.pixels_per_patch = int(decoder_spec["pixels_per_patch"])
        self.in_chans = int(decoder_spec.get("in_chans", 0))

        D = len(self.mu_grid)
        if D != len(self.tok_in_mu) or D != len(self.mu_window_patches):
            raise ValueError(
                f"Dim mismatch: mu_grid={self.mu_grid}, tok_in_mu={self.tok_in_mu}, "
                f"mu_window_patches={self.mu_window_patches}"
            )

        # number of decoder tokens (sequence length after undo_windowing + flatten)
        self.num_decoder_tokens = int(math.prod(self.mu_grid) * math.prod(self.tok_in_mu))

        # patches per decoder token along each axis: mu_window_patches / tok_in_mu
        patches_per_token_shape = []
        for d in range(D):
            w = self.mu_window_patches[d]
            t = self.tok_in_mu[d]
            if w % t != 0:
                raise ValueError(
                    f"mu_window_patches[{d}]={w} must be divisible by tok_in_mu[{d}]={t}"
                )
            patches_per_token_shape.append(w // t)
        self.patches_per_token_shape = tuple(patches_per_token_shape)
        self.patches_per_token = int(math.prod(self.patches_per_token_shape))

        # prediction width per decoder token
        self.pixels_per_decoder_token = int(self.patches_per_token * self.pixels_per_patch)

        self.prediction_mode = prediction_mode

        # ---------------------------------------------------------------------
        # Modules
        # ---------------------------------------------------------------------
        self.decoder_embed = nn.Linear(encoder_dim_out, decoder_embed_dim)

        # mask token that will be broadcast over [B, #MUs_all, *tok_in_mu, Cdec]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # learned positional embedding over the flattened full token grid
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_decoder_tokens, decoder_embed_dim)
        )

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
        )
        self.decoder_norm = Norm(decoder_embed_dim)

        if self.prediction_mode == "pixels":
            self.decoder_pred = nn.Linear(decoder_embed_dim, self.pixels_per_decoder_token)
        elif self.prediction_mode == "lowest_level":
            if output_embed_dim is None:
                raise ValueError(
                    "output_embed_dim is required when prediction_mode='lowest_level'"
                )
            self.decoder_pred = nn.Linear(decoder_embed_dim, int(output_embed_dim))
        else:
            raise NotImplementedError(f"prediction_mode={prediction_mode}")

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._mae_init_weights)

    def _mae_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _tokens_to_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert per-token patch-block vectors into patch-token grid.

        Input:
            x: [B, N_tokens, patches_per_token * pp]
            where N_tokens = prod(mu_grid) * prod(tok_in_mu)

        Output:
            [B, N_patches, pp]
            where patch_grid[d] = mu_grid[d] * tok_in_mu[d] * patches_per_token_shape[d]
                               = mu_grid[d] * mu_window_patches[d]
        """
        B, N, K = x.shape
        D = len(self.mu_grid)
        pp = self.pixels_per_patch
        sub = self.patches_per_token_shape

        # full token grid after undo_windowing: full_tok[d] = mu_grid[d] * tok_in_mu[d]
        full_tok = tuple(self.mu_grid[d] * self.tok_in_mu[d] for d in range(D))

        expected_N = int(math.prod(full_tok))
        expected_K = int(math.prod(sub) * pp)
        if N != expected_N:
            raise ValueError(f"_tokens_to_patch_tokens: expected N={expected_N}, got {N}")
        if K != expected_K:
            raise ValueError(f"_tokens_to_patch_tokens: expected K={expected_K}, got {K}")

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

    def forward(
        self,
        inputs: torch.Tensor,
        mu_mask: torch.Tensor,
        ctx_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        inputs:   [B, N_ctx_mu, *tok_in_mu, encoder_dim_out]
        mu_mask:  [B, N_all_mu]  (True means masked MU)
        ctx_idx:  [B, K]  (indices of kept MUs, precomputed by mask generator)
        returns:  [B, N_patches_total, pp]
        """
        tokens = self.decoder_embed(inputs)  # [B, K, *tok_in_mu, Cdec]
        ctx_idx = ctx_idx.to(inputs.device)

        B = tokens.shape[0]
        K = ctx_idx.shape[1]
        Cdec = tokens.shape[-1]
        N_tokens = self.num_decoder_tokens

        # Build sequence directly: all mask tokens, then scatter context
        x_seq = self.mask_token.view(1, 1, -1).expand(B, N_tokens, Cdec).clone()

        src = tokens.view(B, K, self.tok_prod, Cdec)
        base = ctx_idx.unsqueeze(-1) * self.tok_prod  # [B, K, 1]
        offs = torch.arange(self.tok_prod, device=tokens.device, dtype=ctx_idx.dtype)
        offs = offs.view(1, 1, -1)  # [1, 1, self.tok_prod]
        seq_idx = (base + offs).view(B, K * self.tok_prod).long()  # [B, K*self.tok_prod]

        index = seq_idx.unsqueeze(-1).expand(-1, -1, Cdec)
        x_seq.scatter_(dim=1, index=index, src=src.reshape(B, K * self.tok_prod, Cdec))

        # TODO: add more positional encoding schemes
        x = x_seq + self.decoder_pos_embed
        x = self.decoder(x, masks=None, pos_enc=None)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        if self.prediction_mode == "pixels":
            return self._tokens_to_patch_tokens(x)
        else:
            return x  # [B, N_tokens, C_out]

    @torch.jit.ignore
    def get_num_layers(self) -> int:
        return self.decoder.get_num_layers()


def BUILD(cfg) -> "MaskedHieraPredictor":
    ignore = {"_target_", "BUILD"}
    kwargs = {k: v for k, v in cfg.items() if k not in ignore}
    return MaskedHieraPredictor(**kwargs)