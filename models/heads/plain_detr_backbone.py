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

import torch
import torch.nn.functional as F
from hydra.utils import get_method
from torch import nn

from cell_observatory_platform.models.layers.norm import LayerNorm3D


class PlainDETRBackbone(nn.Module):
    def __init__(
        self,
        backbone_args: dict,
        adapter_args: Optional[dict],
        backbone_embed_dims: List[int],
        train_backbone: bool,
        blocks_to_train: Optional[List[str]] = None,
        use_layernorm: bool = True,
        adapter_out_layers: Optional[List[int]] = None,
    ):
        super().__init__()

        BUILD_BACKBONE = get_method(backbone_args["BUILD"])
        self.backbone = BUILD_BACKBONE(backbone_args)

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
            BUILD_ADAPTER = get_method(adapter_args["BUILD"])
            self.adapter = BUILD_ADAPTER(adapter_args)
        else:
            # TODO: implement logic to handle positional encodings without adapter
            raise NotImplementedError("Backbone adapter must be specified for PlainDETRBackbone.")
            # self.with_backbone_adapter = False

        self.adapter_out_layers = adapter_out_layers

    def forward(self, data_sample: dict) -> List[dict]:
        features = self.backbone.forward_features(data_sample["data_tensor"])

        if self.with_backbone_adapter:
            # dict[str, Tensor]: {"1": f1, ...}
            features_dict = self.adapter(data_sample["data_tensor"], features)
            # turn into ordered list [f1, f2, f3, f4] assuming keys "1","2","3","4"
            features_list = [features_dict[k] for k in sorted(features_dict.keys(), key=int)]
        else:
            # we assume backbone.forward_features already returns a list of feature maps
            features_list = features

        if self.use_layernorm:
            features_list = [ln(x).contiguous() for ln, x in zip(self.layer_norms, features_list)]

        if self.adapter_out_layers is not None:
            features_list = [features_list[i] for i in self.adapter_out_layers]

        out = []
        m = data_sample["metainfo"]["padding_mask"]  # [B, Z, Y, X]
        assert m is not None, "data_sample should include padding mask"
        for lvl, x in enumerate(features_list):
            # x: [B, C, D, H, W]
            mask = F.interpolate(m[None].float(), size=x.shape[-3:], mode="nearest").to(torch.bool)[0]
            out.append({"x": x, "mask": mask})

        return out


def BUILD(backbone_wrapper_args: dict, adapter_args: Optional[dict]) -> nn.Module:
    out_layers = backbone_wrapper_args.get("out_layers", None)
    if out_layers is not None:
        backbone_wrapper_args["backbone_args"]["out_layers"] = out_layers

    model = PlainDETRBackbone(
        backbone_args=backbone_wrapper_args["backbone_args"],
        adapter_args=adapter_args,
        backbone_embed_dims=backbone_wrapper_args["backbone_embed_dims"],
        train_backbone=backbone_wrapper_args["train_backbone"],
        blocks_to_train=backbone_wrapper_args.get("blocks_to_train"),
        use_layernorm=backbone_wrapper_args.get("use_layernorm", True),
        adapter_out_layers=backbone_wrapper_args.get("adapter_out_layers"),
    )
    return model
