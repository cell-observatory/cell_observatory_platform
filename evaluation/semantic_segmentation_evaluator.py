"""Semantic-segmentation evaluator (per-class Jaccard / mean IoU).

Consumes per-image integer label maps from ``model.evaluate_step`` (each returns
``[{"labelmap": (D, H, W)}]`` argmax maps in the ``class + 1`` / background ``0`` convention)
and pairs each with a GT label map built from the per-batch targets. Per-class
intersection/union is accumulated across the dataset by :class:`MaskMIoUMetric` in ``"semantic"``
mode (true streaming Jaccard, distributed-reduction safe). Prediction-based test flow, so
``loss_dict`` is ``None``.
"""

from typing import Any, Dict, Optional

import torch

import cell_observatory_platform.evaluation.evaluate_postprocess as ep
from cell_observatory_platform.evaluation.evaluator import DatasetEvaluator
from cell_observatory_platform.evaluation.metrics import build_metrics
from cell_observatory_platform.utils.registry import REGISTRY


class SemanticSegmentationEvaluator(DatasetEvaluator):
    """Mean-IoU semantic-segmentation evaluator.

    Args:
        num_classes: number of label values (incl. background) the metric iterates over
            (``classes + 1`` in the ``class + 1`` / background ``0`` convention); ``None`` infers
            per image.
        ignore_index: label value to skip in the mIoU average (e.g. ``0`` for background).
        gt_mask_source: how the per-image GT is built -- ``"masks"`` (scatter ``label + 1`` from
            per-instance binary masks) or ``"label_map"`` (map instance ids -> ``class + 1``).
        result_name: key under which the aggregated mIoU is reported.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        ignore_index: Optional[int] = None,
        gt_mask_source: str = "masks",
        result_name: str = "mask_miou_semantic",
    ):
        if gt_mask_source not in ("label_map", "masks"):
            raise ValueError(
                f"gt_mask_source must be 'label_map' or 'masks'; got {gt_mask_source!r}"
            )
        self.gt_mask_source = gt_mask_source
        self.num_classes = num_classes
        self.result_name = result_name
        self.metrics = build_metrics([{
            "name": "mask_miou",
            "key": result_name,
            "mode": "semantic",
            "num_classes": num_classes,
            "ignore_index": ignore_index,
        }])
        # Single-metric convenience handle used by process().
        self.metric = self.metrics[result_name]
        self._results: Dict[str, Optional[float]] = {result_name: None}

    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()
        self._results = {self.result_name: None}

    @torch.no_grad()
    def process(self, data_sample: dict, outputs: Any, loss_dict=None) -> None:
        # model.evaluate_step returns List[{"labelmap": (D, H, W) long}].
        pred_maps = [item["labelmap"] for item in outputs]

        # Taxonomy parity: the preprocessor declares semantic_classes in config and
        # num_classes is declared separately here. Nothing ties them together, so a
        # mismatch would silently score the wrong taxonomy -- fail loudly instead.
        classes = data_sample["metainfo"].get("semantic_classes")
        if classes is not None and self.num_classes is not None:
            expected = len(classes) + 1  # + background
            if expected != self.num_classes:
                raise ValueError(
                    f"evaluator num_classes={self.num_classes} does not match the data "
                    f"taxonomy {classes} (expected {expected} = len(classes) + 1). "
                    "Fix the evaluator config or the preprocessor's semantic_classes."
                )

        targets = ep.extract_targets(data_sample)
        if len(targets) != len(pred_maps):
            raise RuntimeError(
                f"batch size mismatch: outputs has {len(pred_maps)} samples but "
                f"metainfo['targets'] has {len(targets)}"
            )
        for pred_map, target in zip(pred_maps, targets):
            gt_map = ep.gt_semantic_map(target, size=pred_map.shape[-3:], source=self.gt_mask_source)
            self.metric(pred_map.to(gt_map.device).long(), gt_map)

    # evaluate() is inherited from DatasetEvaluator: gather() pools the per-class
    # intersection/union across ranks (no-op at world_size==1) and aggregate()
    # computes a single global Jaccard, keyed by result_name.


from cell_observatory_platform.utils.config import register_class as _register_class
_register_class("evaluator", "semantic_segmentation", SemanticSegmentationEvaluator)
