"""
Adapted from:
https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam/prompt_encoder.py
"""

from typing import Optional, Tuple, Type, List

import torch
from torch import nn

from cell_observatory_platform.models.layers.norm import LayerNorm3D
from cell_observatory_platform.models.layers.patch_embeddings import calc_num_patches
from cell_observatory_platform.models.layers.positional_encoding import PositionEmbeddingRandom


class DownShuffle3d(nn.Module):
    """``Conv3d(kernel=stride=s)`` as space-to-depth + one GEMM.

    With kernel == stride each ``s x s x s`` block is read exactly once and
    contributes to exactly one output token, so the convolution is a
    space-to-depth reshape followed by ``Linear(c_in * s**3, c_out)``. The tiny
    channel counts used by the mask prompt (1 -> 4 -> 16) are the worst case for
    cuDNN's 3D kernels, so this is both faster and layout-friendly.

    Shapes: ``(B, Z, Y, X, c_in) -> (B, Z/s, Y/s, X/s, c_out)`` (channels-last
    in and out).

    The flattened input to the GEMM is ordered ``(dz, dy, dx, c)`` with the
    channel fastest; ``load_from_conv`` permutes the conv weight to match.
    """

    def __init__(self, c_in: int, c_out: int, s: int = 2):
        super().__init__()
        self.s = s
        self.c_in = c_in
        self.c_out = c_out
        self.proj = nn.Linear(c_in * s ** 3, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, Z, Y, X, c_in)
        B, Z, Y, X, C = x.shape
        s = self.s
        if Z % s or Y % s or X % s:
            raise ValueError(
                f"spatial extent {(Z, Y, X)} is not divisible by the stride {s}"
            )
        x = x.reshape(B, Z // s, s, Y // s, s, X // s, s, C)
        # (B, Z/s, dz, Y/s, dy, X/s, dx, C) -> (B, Z/s, Y/s, X/s, dz, dy, dx, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
            B, Z // s, Y // s, X // s, s ** 3 * C
        )
        return self.proj(x)

    @torch.no_grad()
    def load_from_conv(self, conv: nn.Conv3d) -> "DownShuffle3d":
        """Copy the weights of an equivalent ``Conv3d(k=s=self.s)``."""
        s, C_out, C_in = self.s, self.c_out, self.c_in
        assert tuple(conv.weight.shape) == (C_out, C_in, s, s, s), (
            f"expected Conv3d weight {(C_out, C_in, s, s, s)}, "
            f"got {tuple(conv.weight.shape)}"
        )
        # conv.weight: (c_out, c_in, dz, dy, dx)
        #  -> (c_out, dz, dy, dx, c_in), flattened to cols ((dz*s+dy)*s+dx)*C_in + c
        w = conv.weight.permute(0, 2, 3, 4, 1).reshape(C_out, s ** 3 * C_in)
        self.proj.weight.copy_(w)
        if conv.bias is not None:
            self.proj.bias.copy_(conv.bias)
        else:
            self.proj.bias.zero_()
        return self


@torch.no_grad()
def convert_mask_downscaling_state_dict(
    state_dict: dict,
    prefix: str = "mask_downscaling.",
    s: int = 2,
) -> dict:
    """Remap a ``mask_downscaling="conv"`` state dict onto ``"linear_shuffle"``.

    ==============================  ===============================
    conv                            linear_shuffle
    ==============================  ===============================
    ``0.weight`` (Conv3d k=s=2)     ``0.proj.weight``
    ``0.bias``                      ``0.proj.bias``
    ``1.ln.{weight,bias}``          ``1.{weight,bias}`` (LayerNorm)
    ``3.weight`` (Conv3d k=s=2)     ``3.proj.weight``
    ``3.bias``                      ``3.proj.bias``
    ``4.ln.{weight,bias}``          ``4.{weight,bias}`` (LayerNorm)
    ``6.weight`` (Conv3d k=1)       ``6.weight`` (Linear, squeezed)
    ``6.bias``                      ``6.bias``
    ==============================  ===============================

    Every other key is passed through untouched.
    """
    out = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            out[k] = v
            continue
        tail = k[len(prefix):]
        if tail in ("0.weight", "3.weight"):
            c_out, c_in = v.shape[0], v.shape[1]
            out[prefix + tail.replace(".weight", ".proj.weight")] = (
                v.permute(0, 2, 3, 4, 1).reshape(c_out, s ** 3 * c_in).contiguous()
            )
        elif tail in ("0.bias", "3.bias"):
            out[prefix + tail.replace(".bias", ".proj.bias")] = v
        elif tail.startswith("1.ln.") or tail.startswith("4.ln."):
            out[prefix + tail[0] + "." + tail[len("1.ln."):]] = v
        elif tail == "6.weight":
            # 1x1x1 Conv3d weight (embed, C, 1, 1, 1) -> Linear weight (embed, C)
            out[k] = v.reshape(v.shape[0], v.shape[1]).contiguous()
        else:
            out[k] = v
    return out


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mask_in_chans: int,
        mask_downsample_factor: int,
        input_shape: Optional[List[int]] = [128, 256, 512, 2],
        patch_shape: Optional[List[int]] = [16, 32, 32, None],
        input_format: Optional[str] = "TZYXC",
        activation: Type[nn.Module] = nn.GELU,
        # how to build `mask_downscaling`.
        #   "conv"           -> Conv3d(k=s=2) x2 + LayerNorm3D + Conv3d 1x1 (today)
        #   "linear_shuffle" -> DownShuffle3d x2 + nn.LayerNorm + nn.Linear,
        #                       channels-last internally (identical math).
        mask_downscaling: str = "conv",
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.
        """
        super().__init__()

        self.embed_dim = embed_dim

        self.input_shape = input_shape
        self.patch_shape = patch_shape
        self.input_format = input_format
        _,self.token_shape = calc_num_patches(
            input_fmt=self.input_format,
            input_shape=self.input_shape,
            patch_shape=patch_shape,
        )
        if self.input_format == "TZYXC":
            t, z, y, x, c = self.token_shape
            self.token_shape = [z, y, x]
        else:
            raise NotImplementedError(f"Input format {self.input_format} not supported yet.")

        assert embed_dim % 2 == 0, (
            f"embed_dim={embed_dim} must be even for PositionEmbeddingRandom (sin/cos)."
        )
        num_pos_feats = embed_dim // 2
        self.pe_layer = PositionEmbeddingRandom(
            input_fmt=self.input_format,
            num_pos_feats=num_pos_feats, 
            time_separable=True
        )

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        if mask_downscaling not in ("conv", "linear_shuffle"):
            raise ValueError(
                f"mask_downscaling must be 'conv' or 'linear_shuffle', got {mask_downscaling!r}"
            )
        # `mask_downscaling` is also the name of the nn.Sequential built below,
        # so the MODE lives under a distinct attribute.
        self.mask_downscaling_mode = mask_downscaling

        if self.input_format == "TZYXC":
            T, Z, Y, X, C = self.input_shape
            self.spatial_shape = (Z, Y, X)
            self.mask_input_size = (
                Z // mask_downsample_factor,
                Y // mask_downsample_factor,
                X // mask_downsample_factor,
            )
            assert mask_downsample_factor == 4, "Mask downsample factor must be 4."
            if mask_downscaling == "conv":
                self.mask_downscaling = nn.Sequential(
                    nn.Conv3d(1, mask_in_chans // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    LayerNorm3D(mask_in_chans // 4),
                    activation(),
                    nn.Conv3d(mask_in_chans // 4, mask_in_chans, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                    LayerNorm3D(mask_in_chans),
                    activation(),
                    nn.Conv3d(mask_in_chans, embed_dim, kernel_size=(1, 1, 1)),
                )
            else:
                # Same seven stages, channels-last: `_embed_masks` permutes in
                # and out so the module's external contract is unchanged.
                self.mask_downscaling = nn.Sequential(
                    DownShuffle3d(1, mask_in_chans // 4, s=2),
                    nn.LayerNorm(mask_in_chans // 4),
                    activation(),
                    DownShuffle3d(mask_in_chans // 4, mask_in_chans, s=2),
                    nn.LayerNorm(mask_in_chans),
                    activation(),
                    nn.Linear(mask_in_chans, embed_dim),
                )
        else:
            raise NotImplementedError(f"Input format {self.input_format} not supported yet.")
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.
        """
        return self.pe_layer(self.token_shape).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 3), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.spatial_shape
        )

        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 0).unsqueeze(-1),
            point_embedding + self.point_embeddings[0].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 1).unsqueeze(-1),
            point_embedding + self.point_embeddings[1].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 2).unsqueeze(-1),
            point_embedding + self.point_embeddings[2].weight,
            point_embedding,
        )
        point_embedding = torch.where(
            (labels == 3).unsqueeze(-1),
            point_embedding + self.point_embeddings[3].weight,
            point_embedding,
        )
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        if self.input_format == "TZYXC":
            # boxes: [B, 6] -> [B, 2, 3] where coords[:, 0, :] = (x0, y0, z0)
            coords = boxes.reshape(-1, 2, 3)
            corner_embedding = self.pe_layer.forward_with_coords(
                coords, self.spatial_shape
            )
            corner_embedding[:, 0, :] += self.point_embeddings[2].weight
            corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        else:
            raise NotImplementedError(f"Input format {self.input_format} not supported yet.")
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs.

        In/out layout is channels-first ``(B, 1, Z, Y, X) -> (B, embed, Z/4,
        Y/4, X/4)`` for BOTH `mask_downscaling` modes; the linear_shuffle stack
        works channels-last internally and is bracketed by one permute each way.
        """
        if self.mask_downscaling_mode == "linear_shuffle":
            x = masks.permute(0, 2, 3, 4, 1)         # (B, Z, Y, X, 1)
            x = self.mask_downscaling(x)             # (B, Z/4, Y/4, X/4, embed)
            return x.permute(0, 4, 1, 2, 3)          # back to channels-first
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            if self.input_format == "TZYXC":
                dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(
                    bs, -1, *self.token_shape
                )
            else:
                raise NotImplementedError(f"Input format {self.input_format} not supported yet.")

        return sparse_embeddings, dense_embeddings