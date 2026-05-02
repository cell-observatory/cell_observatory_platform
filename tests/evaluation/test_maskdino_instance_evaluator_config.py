from pathlib import Path

from hydra.utils import get_class
from omegaconf import OmegaConf

from cell_observatory_platform.evaluation.instance_segmentation_evaluator import (
    InstanceSegmentationEvaluator,
)


def test_maskdino_instance_evaluator_config_target_resolves():
    repo_root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(repo_root / "configs/evaluation/maskdino_instance_evaluator.yaml")

    evaluator_cls = get_class(cfg.evaluator._target_)

    assert evaluator_cls is InstanceSegmentationEvaluator
    assert cfg.evaluator.metrics == [
        "box_map",
        "box_miou",
        "box_f1",
        "mask_map",
        "mask_miou",
    ]
    assert cfg.evaluator.mask_chunk_size > 0
