from typing import Dict, List, Optional

import torch

import cell_observatory_platform.evaluation.evaluate_postprocess as ep
from cell_observatory_platform.evaluation.evaluator import DatasetEvaluator
from cell_observatory_platform.evaluation.metrics import build_metrics
from cell_observatory_platform.utils.registry import REGISTRY
from cell_observatory_platform.utils.config import registers_as


@registers_as("evaluator", "reconstruction")
class AutomatedBenchmarkEvaluator(DatasetEvaluator):
    """
    Reconstruction-style evaluator computing pixel-wise metrics (NRMSE, MAE, ...) between
    ``model.evaluate_step`` predictions and ground-truth targets, in the prediction-based test
    flow. When ``model.evaluate_step`` returns a dict, ``pred_key`` selects the tensor entry to
    evaluate against ``data_sample["metainfo"][target_key][0]``.

    When the target is in patchified token space (e.g. ``metainfo["targets"] = (B, N, ppp*C)``)
    but the prediction is a dense image, set ``unpatchify_targets=True`` and provide
    ``patch_shape`` / ``input_shape`` / ``input_format`` so the evaluator builds its own
    ``unpatchify`` from config (no model reach-in) to lift the target to image space.
    """

    def __init__(
        self,
        metric_reductions: List[Dict[str, str]],
        pred_key: Optional[str] = None,
        target_key: str = "data_tensor",
        unpatchify_targets: bool = False,
        patch_shape=None,
        input_shape=None,
        input_format=None,
    ):
        self.metrics = build_metrics(metric_reductions)
        self.pred_key = pred_key
        self.target_key = target_key
        # Built from config, exactly like RayPreprocessor.pe_patchify -- no model, no injection.
        self._unpatchify_fn = (
            ep.build_unpatchify(patch_shape, input_shape, input_format)
            if unpatchify_targets else None
        )
        self._results = {name: None for name in self.metrics}

    def reset(self):
        for m in self.metrics.values():
            m.reset()
        self._results = {k: None for k in self._results}

    @torch.no_grad()
    def process(self, data_sample, outputs, loss_dict=None):
        pred = outputs[self.pred_key] if self.pred_key is not None else outputs
        target = data_sample["metainfo"][self.target_key][0]

        # Lift a patchified target into image space to match the dense prediction.
        # out_channels comes from pred so denoising (C=1) and channel_split /
        # upsample_space (C=2) resolve without task-specific bookkeeping.
        if self._unpatchify_fn is not None:
            target = self._unpatchify_fn(target, out_channels=pred.shape[-1])

        pred = pred.float()
        target = target.float().to(device=pred.device)
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction and target shape mismatch: pred={tuple(pred.shape)} "
                f"target={tuple(target.shape)} (target_key={self.target_key!r})."
            )
        for metric in self.metrics.values():
            metric(pred, target, None)

    # evaluate() is inherited from DatasetEvaluator (gather no-op + aggregate over
    # the accumulated NRMSE/MAE reduce buffers).
