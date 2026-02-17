"""
Adapted from:
https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam/prompt_encoder.py
"""

from typing import Optional, Tuple, Type, List

import torch
from torch import nn

from cell_observatory_platform.models.layers.norm import LayerNorm3d
from cell_observatory_platform.models.layers.patch_embeddings import calc_num_patches
from cell_observatory_platform.models.layers.positional_encoding import PositionEmbeddingRandom


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mask_in_chans: int,
        mask_downsample_factor: int,
        input_shape: Optional[List[int]] = [128, 256, 512, 2],
        patch_shape: Optional[List[int]] = [16, 32, 32, None],
        input_format: Optional[str] = "ZYXC",        
        activation: Type[nn.Module] = nn.GELU,
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
        if self.input_format == "ZYXC":
            t, z, y, x, c = self.token_shape
            self.token_shape = [z, y, x]
        else:
            raise NotImplementedError(f"Input format {self.input_format} not supported yet.")

        # NOTE: embed_dim-dim positional encodings per axis
        num_pos_feats = embed_dim // 3
        self.pe_layer = PositionEmbeddingRandom(input_fmt=self.input_format, num_pos_feats=num_pos_feats)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        if self.input_format == "ZYXC":
            Z, Y, X, C = self.input_shape
            self.mask_input_size = (
                Z // mask_downsample_factor,
                Y // mask_downsample_factor,
                X // mask_downsample_factor,
            )
            assert mask_downsample_factor == 4, "Mask downsample factor must be 4."
            self.mask_downscaling = nn.Sequential(
                nn.Conv3d(1, mask_in_chans // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                LayerNorm3d(mask_in_chans // 4),
                activation(),
                nn.Conv3d(mask_in_chans // 4, mask_in_chans, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                LayerNorm3d(mask_in_chans),
                activation(),
                nn.Conv3d(mask_in_chans, embed_dim, kernel_size=(1, 1, 1)),
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
            points, self.input_shape[:3]
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
        if self.input_format == "ZYXC":
            # boxes: [B, 6] -> [B, 2, 3] where coords[:, 0, :] = (x0, y0, z0)
            coords = boxes.reshape(-1, 2, 3)
            corner_embedding = self.pe_layer.forward_with_coords(
                coords, self.input_shape[:3]
            )
            corner_embedding[:, 0, :] += self.point_embeddings[2].weight
            corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        else:
            raise NotImplementedError(f"Input format {self.input_format} not supported yet.")
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
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
            if self.input_format == "ZYXC":
                dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(
                    bs, -1, *self.token_shape
                )
            else:
                raise NotImplementedError(f"Input format {self.input_format} not supported yet.")

        return sparse_embeddings, dense_embeddings