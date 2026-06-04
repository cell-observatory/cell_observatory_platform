from pathlib import Path

from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from cell_observatory_platform.evaluation.instance_segmentation_evaluator import (
    InstanceSegmentationEvaluator,
)
from cell_observatory_platform.evaluation.metrics import PredictedIoUEvalMetric


def _load_cfg():
    repo_root = Path(__file__).resolve().parents[2]
    return OmegaConf.load(repo_root / "configs/evaluation/sam2_instance_evaluator.yaml")


def test_sam2_instance_evaluator_config_target_resolves():
    cfg = _load_cfg()

    assert get_class(cfg.evaluator._target_) is InstanceSegmentationEvaluator
    # SAM2 is class-agnostic (sentinel class id -1); match_labels MUST be False
    # or the evaluator raises and every metric collapses to ~0.
    assert cfg.evaluator.match_labels is False
    assert cfg.evaluator.gt_box_format == "cxcyczwhd"
    assert cfg.evaluator.gt_boxes_normalized is True


def test_sam2_instance_evaluator_config_instantiates_like_the_trainer():
    cfg = _load_cfg()
    evaluator = instantiate(cfg.evaluator)

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
