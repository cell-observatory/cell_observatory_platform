"""
Adapted from:
# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""

from typing import List, Optional
import functools

import torch
import torch.nn.functional as F
from hydra.utils import get_method
from torch import nn

from cell_observatory_platform.models.layers.norm import LayerNorm3D
from cell_observatory_platform.models.layers.patch_embeddings import calc_num_patches


class MaskDinoBackbone(nn.Module):
    def __init__(
        self,
        backbone_args: dict,
        adapter_args: Optional[dict],
        backbone_embed_dims: List[int],
        train_backbone: bool,
        blocks_to_train: Optional[List[str]] = None,
        use_layernorm: bool = True,
        adapter_out_layers: Optional[List[int]] = None,
        backbone_output_format: str = "feature_map",
        input_shape: Optional[List[int]] = [128, 256, 512, 2],
        patch_shape: Optional[List[int]] = [16, 32, 32, None],
        input_format: Optional[str] = "ZYXC",
    ):
        super().__init__()

        self.backbone = REGISTRY.build("backbone", backbone_args.name, backbone_args)

        self.blocks_to_train = blocks_to_train

        for _, (name, parameter) in enumerate(self.backbone.named_parameters()):
            train_condition = any(f".{b}." in name for b in self.blocks_to_train) if self.blocks_to_train else True
            if (not train_backbone) or "mask_token" in name or (not train_condition):
                parameter.requires_grad_(False)

        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.layer_norms = nn.ModuleList([LayerNorm3D(embed_dim) for embed_dim in backbone_embed_dims])

        if adapter_args is not None:
            self.with_backbone_adapter = True
            self.adapter = REGISTRY.build("adapter", adapter_args.name, adapter_args)
        else:
            # TODO: implement logic to handle positional encodings without adapter
            self.with_backbone_adapter = False

        self.adapter_out_layers = adapter_out_layers

        if len(backbone_embed_dims) > 1:
            self.multi_scale_features = True
        else:
            self.multi_scale_features = False

        self.input_shape = input_shape
        self.patch_shape = patch_shape
        self.input_format = input_format
        _, token_shape = calc_num_patches(
            input_fmt=self.input_format,
            input_shape=self.input_shape,
            patch_shape=patch_shape,
        )
        if self.input_format == "ZYXC":
            t, z, y, x, c = token_shape
            self.token_shape = [z, y, x]
        else:
            raise NotImplementedError(f"Input format {self.input_format} not supported yet.")

        assert self.input_format[-1] == "C", "The last dimension of input_format must be 'C'."
        self.out_channels = self.input_shape[-1]
        self.backbone_output_format = backbone_output_format
        self.backbone_returns_sequence = self.backbone_output_format == "sequence"
        
    def _unpatchify_if_sequence(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # feats: list of either [B, N, C] or [B, C, D, H, W]
        if not self.backbone_returns_sequence:
            return feats

        out = []
        for feat in feats:
            if feat.dim() == 3:  # [B, N, C]
                B, N, C = feat.shape
                out.append(feat.transpose(1, 2).reshape(B, C, *self.token_shape))
            else:
                out.append(feat)
        return out

    def _to_feature_dict(self, feats: List[torch.Tensor]):
        # Ensure finest->coarsest mapping for keys "1","2","3","4",...
        feats = [f for f in feats if f is not None]
        assert self.input_format == "ZYXC", f"Expected input_format 'ZYXC', got {self.input_format}"
        assert all(f.dim() == 5 for f in feats), f"Expected 5D feature maps [B,C,D,H,W], got {[f.shape for f in feats]}"
        feats = sorted(feats, key=lambda t: t.shape[-3] * t.shape[-2] * t.shape[-1], reverse=True)
        return {str(i + 1): f for i, f in enumerate(feats)}

    def forward(self, data_sample: dict):
        feats = self.backbone.forward_features(data_sample["data_tensor"])

        adapter_keys = None
        if self.with_backbone_adapter:
            feats_dict = self.adapter(data_sample["data_tensor"], feats)
            feats_dict = {str(k): v for k, v in feats_dict.items()}  # ensure string keys
            adapter_keys = sorted(feats_dict.keys(), key=lambda s: int(s))
            feats_list = [feats_dict[k] for k in adapter_keys]
        else:
            if isinstance(feats, (list, tuple)):
                feats_list = list(feats)
            else:
                feats_list = [feats]

        feats_list = self._unpatchify_if_sequence(feats_list)
        if self.use_layernorm:
            assert len(self.layer_norms) == len(feats_list), (
                f"layer_norms ({len(self.layer_norms)}) != feats_list ({len(feats_list)})"
            )
            feats_list = [ln(f).contiguous() for ln, f in zip(self.layer_norms, feats_list)]

        if self.adapter_out_layers is not None:
            feats_list = [feats_list[i] for i in self.adapter_out_layers]
            if adapter_keys is not None:
                adapter_keys = [adapter_keys[i] for i in self.adapter_out_layers]

        if adapter_keys is not None:
            # keep adapter’s semantic keys ("1","2","3","4" etc)
            return {k: v for k, v in zip(adapter_keys, feats_list)}
        else:
            return self._to_feature_dict(feats_list)


from cell_observatory_platform.utils.registry import REGISTRY


@REGISTRY.register("backbone", "maskdino")
def BUILD(backbone_wrapper_args: dict, adapter_args: Optional[dict] = None) -> nn.Module:
    out_layers = backbone_wrapper_args.get("out_layers", None)
    if out_layers is not None:
        backbone_wrapper_args["backbone_args"]["out_layers"] = out_layers

    model = MaskDinoBackbone(
        backbone_args=backbone_wrapper_args["backbone_args"],
        adapter_args=adapter_args,
        backbone_embed_dims=backbone_wrapper_args["backbone_embed_dims"],
        train_backbone=backbone_wrapper_args["train_backbone"],
        blocks_to_train=backbone_wrapper_args.get("blocks_to_train"),
        use_layernorm=backbone_wrapper_args.get("use_layernorm", True),
        adapter_out_layers=backbone_wrapper_args.get("adapter_out_layers"),
        backbone_output_format=backbone_wrapper_args.get("backbone_output_format"),
        input_shape=backbone_wrapper_args.get("input_shape"),
        input_format=backbone_wrapper_args.get("input_format"),
        patch_shape=backbone_wrapper_args.get("patch_shape"),
    )
    return model