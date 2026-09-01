"""The shipped instance-evaluator YAMLs resolve to the registered evaluator and
build through `REGISTRY.build` exactly as the trainer builds them."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cell_observatory_platform.utils.registry import REGISTRY
from cell_observatory_platform.evaluation.instance_segmentation_evaluator import (  # noqa: F401  registers
    InstanceSegmentationEvaluator,
)
from cell_observatory_platform.evaluation.metrics import PredictedIoUEvalMetric


_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "evaluation"
_BASE_METRICS = {"box_map", "box_miou", "box_f1", "mask_map", "mask_miou"}


def _load(name: str):
    path = _CONFIG_DIR / f"{name}_instance_evaluator.yaml"
    assert path.is_file(), f"{path} not found -- did tests/evaluation move?"
    return OmegaConf.load(path).evaluator


def test_maskdino_config_resolves():
    cfg = _load("maskdino")
    assert cfg.name == "instance_segmentation" and REGISTRY.has("evaluator", cfg.name)
    assert OmegaConf.to_container(cfg.metrics, resolve=True) == [
        "box_map", "box_miou", "box_f1", "mask_map", {"name": "mask_miou", "mode": "instance"},
    ]
    assert cfg.mask_chunk_size > 0 and cfg.match_labels is True


def test_sam2_config_resolves():
    cfg = _load("sam2")
    assert cfg.name == "instance_segmentation" and REGISTRY.has("evaluator", cfg.name)
    # SAM2 is class-agnostic (sentinel class id -1): match_labels=True raises in process().
    assert cfg.match_labels is False
    assert cfg.gt_box_format == "cxcyczwhd" and cfg.gt_boxes_normalized is True


@pytest.mark.parametrize("name,metric_keys,match_labels", [
    ("maskdino", _BASE_METRICS, True),
    ("sam2", _BASE_METRICS | {"pred_iou_eval"}, False),
])
def test_config_builds_like_the_trainer(name, metric_keys, match_labels):
    cfg = _load(name)
    evaluator = REGISTRY.build("evaluator", cfg.name, cfg)
    assert isinstance(evaluator, InstanceSegmentationEvaluator)
    assert evaluator.match_labels is match_labels
    assert set(evaluator.metrics) == metric_keys
    assert evaluator.metrics["mask_miou"].mode == "instance"
    if name == "sam2":
        assert isinstance(evaluator.metrics["pred_iou_eval"], PredictedIoUEvalMetric)
