from pathlib import Path

from omegaconf import OmegaConf

from cell_observatory_platform.utils.registry import REGISTRY
from cell_observatory_platform.evaluation.instance_segmentation_evaluator import (  # noqa: F401  registers
    InstanceSegmentationEvaluator,
)


def test_maskdino_instance_evaluator_config_name_resolves():
    repo_root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(repo_root / "configs/evaluation/maskdino_instance_evaluator.yaml")

    assert cfg.evaluator.name == "instance_segmentation"
    assert REGISTRY.has("evaluator", cfg.evaluator.name)
    assert OmegaConf.to_container(cfg.evaluator.metrics, resolve=True) == [
        "box_map",
        "box_miou",
        "box_f1",
        "mask_map",
        {"name": "mask_miou", "mode": "instance"},
    ]
    assert cfg.evaluator.mask_chunk_size > 0
