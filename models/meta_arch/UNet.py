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
from cell_observatory_platform.models.meta_arch import utils as mo

class Unet(nn.Module):
    def __init__(
        self,
        backbone: MedNeXt | ConvNeXtV2,
        criterion: MultiLabelBinaryPredictionLoss,
        output_metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.criterion = criterion

        # Inference contract (see meta_arch/utils.py): inference_step emits a single
        # semantic label map (argmax+1 / bg-0). Spatial extents are placeholders; the
        # inferencer grows dense buffers to the DB per-table maxima at restore.
        # TODO: Unet uses MultiLabelBinaryPredictionLoss (per-class binary).
        self.output_metadata = mo.output_metadata(
            masks_labelmap=mo.semantic_map((1, 1, 1)),   # (Z,Y,X,1) uint16, extents grown at restore
        )
        if output_metadata is not None:
            self.output_metadata.merge_with(output_metadata)

    def _init_model_weights(self, buffer_device: str | None = None):
        # FIXME: Implement this
        # MedNeXt/ConvNeXt backbones use default PyTorch init; no special handling needed
        pass

    @torch.jit.ignore
    def get_output_metadata(self):
        return self.output_metadata

    def forward(self, data_sample: dict):
        features = self.backbone(data_sample) # (B, N_classes, spatial)
        losses = self.criterion(features, data_sample["metainfo"]["targets"])
        losses["step_loss"] = sum(
            losses[k] * self.criterion.loss_weight_dict[k] 
            for k in losses.keys() 
            if k in self.criterion.loss_weight_dict
        )
        return losses, features # FIXME: features here is just the predicted masks
    
    def inference_step(self, data_sample: dict):
        # INFERENCE — collapse per-class scores to a semantic label map
        # (class+1 / bg-0), the fixed saveable artifact. Strips the backbone's
        # auxiliary_outputs (deep-supervision list, not a save tensor).
        # TODO: multi-label (sigmoid) collapsed via argmax is an approximation.
        # Thresholded in LOGIT space: sigmoid is monotonic, so argmax is unchanged
        # by it and `sigmoid(x) > 0.5` is `x > 0`. Skipping it avoids materializing
        # a second (B, N_classes, Z, Y, X) float volume per tile.
        logits = self.backbone(data_sample)["pred_masks"]      # (B, N_classes, Z, Y, X)
        return {"masks_labelmap": mo.collapse_to_semantic_map(logits, threshold=0.0)}

    @torch.no_grad()
    def evaluate_step(self, data_sample: dict) -> List[Dict[str, Any]]:
        """EVAL — consumed by SemanticSegmentationEvaluator.process(): a per-sample
        list of {"labelmap": (Z, Y, X) long}, class+1 / bg-0 (matches gt_semantic_map).
        """
        lm = self.inference_step(data_sample)["masks_labelmap"]   # (B, Z, Y, X, 1) uint16
        return [{"labelmap": lm[b, ..., 0].long()} for b in range(lm.shape[0])]
    
    
def _extract_kwargs(cfg: Mapping[str, Any], extra_ignores: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Drop Hydra/meta keys like _target_, BUILD, and any explicitly ignored keys.
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    ignore = {"_target_", "BUILD", "name"}
    if extra_ignores:
        ignore.update(extra_ignores)
    return {k: v for k, v in cfg.items() if k not in ignore}


from cell_observatory_platform.utils.registry import REGISTRY


@REGISTRY.register("model", "unet")
def BUILD(cfg: Mapping[str, Any]) -> nn.Module:
    model_cfg = cfg.models.meta_arch.unet

    # ------------------------------------------------------------------
    # 1) Build backbone
    # ------------------------------------------------------------------

    bw_cfg = model_cfg["backbone_wrapper_args"]
    adapter_cfg = model_cfg.get("adapter_args", None)
    backbone = REGISTRY.build("backbone", bw_cfg.name, bw_cfg, adapter_args=adapter_cfg)

    # ------------------------------------------------------------------
    # 2) Build criterion
    # ------------------------------------------------------------------
    # TODO: Make a BUILD function for MultiLabelBinaryPredictionLoss
    criterion = MultiLabelBinaryPredictionLoss(**_extract_kwargs(model_cfg["criterion_args"]))

    return Unet(backbone, criterion)