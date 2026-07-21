from pathlib import Path

from omegaconf import OmegaConf

from cell_observatory_platform.utils.registry import REGISTRY
from cell_observatory_platform.evaluation.instance_segmentation_evaluator import (  # noqa: F401  registers
    InstanceSegmentationEvaluator,
)
from cell_observatory_platform.evaluation.metrics import PredictedIoUEvalMetric


def _load_cfg():
    repo_root = Path(__file__).resolve().parents[2]
    return OmegaConf.load(repo_root / "configs/evaluation/sam2_instance_evaluator.yaml")


def test_sam2_instance_evaluator_config_name_resolves():
    cfg = _load_cfg()

    assert cfg.evaluator.name == "instance_segmentation"
    assert REGISTRY.has("evaluator", cfg.evaluator.name)
    # SAM2 is class-agnostic (sentinel class id -1); match_labels MUST be False
    # or the evaluator raises and every metric collapses to ~0.
    assert cfg.evaluator.match_labels is False
    assert cfg.evaluator.gt_box_format == "cxcyczwhd"
    assert cfg.evaluator.gt_boxes_normalized is True


def test_sam2_instance_evaluator_config_builds_like_the_trainer():
    cfg = _load_cfg()
    evaluator = REGISTRY.build("evaluator", cfg.evaluator.name, cfg.evaluator)

    assert isinstance(evaluator, InstanceSegmentationEvaluator)
    assert evaluator.match_labels is False
    assert set(evaluator.metrics) == {
        "box_map",
        "box_miou",
        "box_f1",
        "mask_map",
        "mask_miou",
        "pred_iou_eval",
    }
    assert isinstance(evaluator.metrics["pred_iou_eval"], PredictedIoUEvalMetric)
