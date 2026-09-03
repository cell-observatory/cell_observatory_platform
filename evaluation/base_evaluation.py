from typing import Dict, List

from cell_observatory_platform.evaluation.evaluator import DatasetEvaluator
from cell_observatory_platform.evaluation.metrics import build_metrics


from cell_observatory_platform.utils.registry import REGISTRY
from cell_observatory_platform.utils.config import registers_as


@registers_as("evaluator", "base")
class BaseEvaluator(DatasetEvaluator):
    """
    Evaluate model loss on validation dataset.
    """

    def __init__(self, training_metrics: List[Dict[str, str]]):
        # Metric spec list: each entry is a registered metric name or
        # {"name": ..., "key"?: ..., **ctor_kwargs} (see metrics._build_one_metric),
        # e.g. [{"name": "train_loss", "key": "step_loss", "reduce_method": "mean"}].
        # These accumulate per-step losses and aggregate epoch stats written to
        # the event-writer backends (TensorBoard/WandB/disk).
        self.metrics = build_metrics(training_metrics)
        self._results = {m: None for m in self.metrics}

    # reset _results AND the accumulated per-metric state: without the metric
    # reset, epoch E+1's aggregate silently included every epoch <= E's losses
    # (every sibling evaluator already resets its metrics here).
    def reset(self):
        self._results = {m: None for m in self._results.keys()}
        for m in self.metrics.values():
            m.reset()

    # for each metric after each step we process the
    # loss_dict and append each loss metric to the
    # corresponding TrainLosses instance in self.metrics
    def process(self, data_sample, outputs, loss_dict):
        # Only `loss_dict` is unconditionally required: this method dispatches
        # to metrics by subscripting it (loss_dict[metric]), which silently
        # raised an opaque TypeError when None was passed in. data_sample and
        # outputs may legitimately be None on validation paths where the metric
        # impl doesn't need them; any metric that does will fail with its own
        # meaningful error.
        if loss_dict is None:
            raise TypeError("loss_dict=None; BaseEvaluator.process requires a loss_dict to dispatch on metric keys")

        for metric, metric_impl in self.metrics.items():
            metric_impl(outputs, data_sample, loss_dict[metric])

    # evaluate() is inherited from DatasetEvaluator: it gathers (no-op for
    # TrainLosses) + aggregates each metric into a flat dict passed to the event
    # writer before writing to the backend (e.g. TensorBoard, WandB, disk).
