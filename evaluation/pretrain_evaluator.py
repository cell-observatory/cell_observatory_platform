"""Loss-based evaluator for pretraining models (MAE / JEPA).

MAE and JEPA have no instance/semantic prediction contract; their meaningful
evaluation signal is the self-supervised pretraining loss. Under ``job_type=test``,
``TestTrainer.run_test_step`` calls ``model.evaluate_step(data_sample)`` (which
returns a single dict containing both the loss entries and the decoder/predictor
output) and this evaluator accumulates the requested loss keys.  Because
``evaluate_step`` returns a plain dict, ``_extract_losses`` passes it through as-is
and the loss keys (e.g. ``"step_loss"``) are looked up directly in that dict.

It also works unchanged on the *validation* flow (``run_validation_step`` passes
a real ``loss_dict``): so a dataset configured with a validation split > 0
evaluates MAE/JEPA with zero extra code -- validation forwards the loss dict
directly, test extracts it from ``evaluate_step``'s returned dict.
"""

from typing import Any, Dict, List

import torch

from cell_observatory_platform.evaluation.evaluator import DatasetEvaluator
from cell_observatory_platform.evaluation.metrics import build_metrics


from cell_observatory_platform.utils.registry import REGISTRY


class PretrainEvaluator(DatasetEvaluator):
    """Accumulate pretraining losses for MAE/JEPA under ``job_type=test``.

    Args:
        training_metrics: list of ``{loss_key: reduce_method}`` dicts (same
            shape as :class:`BaseEvaluator`), e.g. ``[{"step_loss": "mean"}]``.
            Each ``loss_key`` must appear in the model's ``evaluate_step`` dict.
    """

    def __init__(self, training_metrics: List[Dict[str, str]]):
        # Each {loss_key: reduce_method} pair builds a TrainLosses keyed by the
        # loss name (build_metrics routes unknown names to TrainLosses).
        self.metrics = build_metrics(training_metrics)
        self._results: Dict[str, Any] = {m: None for m in self.metrics}

    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()
        self._results = {m: None for m in self._results.keys()}

    @torch.no_grad()
    def process(self, data_sample, outputs, loss_dict=None) -> None:
        losses = loss_dict if loss_dict is not None else self._extract_losses(outputs)
        for metric, metric_impl in self.metrics.items():
            if metric not in losses:
                raise KeyError(
                    f"PretrainEvaluator metric {metric!r} not found in loss dict "
                    f"(available: {sorted(losses.keys())})"
                )
            metric_impl(outputs, data_sample, losses[metric])

    @staticmethod
    def _extract_losses(outputs: Any) -> Dict[str, torch.Tensor]:
        """Pull the loss dict out of ``model.evaluate_step``'s return (test flow).

        MAE/JEPA ``evaluate_step`` returns a plain dict containing loss keys
        (e.g. ``"step_loss"``) alongside output keys; this method returns it
        directly.  A ``(loss_dict, predictions)`` tuple is also accepted.
        Anything else is a flow/config error.
        """
        if (
            isinstance(outputs, (tuple, list))
            and outputs
            and isinstance(outputs[0], dict)
        ):
            return outputs[0]
        if isinstance(outputs, dict):
            return outputs
        raise TypeError(
            "PretrainEvaluator expected model.evaluate_step to return a loss dict "
            f"(or a (loss_dict, ...) tuple); got {type(outputs).__name__}. Ensure "
            "the model is a pretraining model (MAE/JEPA)."
        )

    # evaluate() is inherited from DatasetEvaluator (gather no-op + aggregate
    # over the accumulated per-loss TrainLosses).


from cell_observatory_platform.utils.config import register_class as _register_class
_register_class("evaluator", "pretrain", PretrainEvaluator)
