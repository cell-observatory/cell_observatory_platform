import copy
import inspect
import math
from typing import Any, Dict, Literal, Mapping, Optional, List

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from hydra.utils import get_method
from torch import nn

from cell_observatory_platform.models.backbones.mednext import MedNeXt
from cell_observatory_platform.models.backbones.convnext import ConvNeXtV2
from cell_observatory_platform.training.losses import MultiLabelBinaryPredictionLoss

class Unet(nn.Module):
    def __init__(
        self,
        backbone: MedNeXt | ConvNeXtV2,
        criterion: MultiLabelBinaryPredictionLoss,
    ):
        super().__init__()
        self.backbone = backbone
        self.criterion = criterion

    def init_model_weights(self, buffer_device: str | None = None):
        # FIXME: Implement this
        # MedNeXt/ConvNeXt backbones use default PyTorch init; no special handling needed
        pass

    def forward(self, data_sample: dict):
        features = self.backbone(data_sample) # (B, N_classes, spatial)
        losses = self.criterion(features, data_sample["metainfo"]["targets"][0])
        losses["step_loss"] = sum(
            losses[k] * self.criterion.loss_weight_dict[k] 
            for k in losses.keys() 
            if k in self.criterion.loss_weight_dict
        )
        return losses, features # FIXME: features here is just the predicted masks
    
    def predict(self, data_sample: dict):
        features = self.backbone(data_sample) # (B, N_classes, spatial)
        # NOTE: we permute channels back because we need to return the 
        # predicted masks in the same format as the data sample (B, Z, Y, X, C)
        features["pred_masks"] = features["pred_masks"].permute(0, 2, 3, 4, 1) # (B, Z, Y, X, N_classes)
        return features # FIXME: features here is just the predicted masks
    
    
def _extract_kwargs(cfg: Mapping[str, Any], extra_ignores: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Drop Hydra/meta keys like _target_, BUILD, and any explicitly ignored keys.
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    ignore = {"_target_", "BUILD"}
    if extra_ignores:
        ignore.update(extra_ignores)
    return {k: v for k, v in cfg.items() if k not in ignore}


def BUILD(cfg: Mapping[str, Any]) -> nn.Module:
    model_cfg = cfg.models.meta_arch.unet

    # ------------------------------------------------------------------
    # 1) Build backbone
    # ------------------------------------------------------------------

    bw_cfg = model_cfg["backbone_wrapper_args"]
    build_backbone_wrapper = get_method(bw_cfg.BUILD)

    adapter_cfg = model_cfg.get("adapter_args", None)
    if adapter_cfg is not None:
        backbone = build_backbone_wrapper(bw_cfg, adapter_cfg)
    else:
        backbone = build_backbone_wrapper(bw_cfg, None)

    # ------------------------------------------------------------------
    # 2) Build criterion
    # ------------------------------------------------------------------
    # TODO: Make a BUILD function for MultiLabelBinaryPredictionLoss
    criterion = MultiLabelBinaryPredictionLoss(**_extract_kwargs(model_cfg["criterion_args"]))

    return Unet(backbone, criterion)