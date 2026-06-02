from typing import Dict, List

from cell_observatory_platform.evaluation.evaluator import DatasetEvaluator
from cell_observatory_platform.evaluation.metrics import TrainLosses


class BaseEvaluator(DatasetEvaluator):
    """
    Evaluate model loss on validation dataset.
    """

    def __init__(self, training_metrics: List[Dict[str, str]]):
        # _results is a dictionary that will hold the final results
        # after evaluation, where each key is a metric name
        # and the value is the aggregated result for that metric
        self._results = {metric: None for training_metric in training_metrics for metric in training_metric.keys()}

        # for a loss dict: {"metric1": loss, "metric2": loss}
        # we create a TrainLosses instance for each metric
        # with the specified reduction method
        # e.g. {"metric1": "mean", "metric2": "min"}
        # this object will accumulate the losses
        # for a given metric across steps and then
        # aggregate epoch statistics to write to
        # eventWriter backends for logging
        self.metrics = {
            metric: TrainLosses(reduce_method=reduce_op)
            for training_metric in training_metrics
            for metric, reduce_op in training_metric.items()
        }

    # reset _results for each metric
    def reset(self):
        self._results = {m: None for m in self._results.keys()}

    # for each metric after each step we process the
    # loss_dict and append each loss metric to the
    # corresponding TrainLosses instance in self.metrics
    def process(self, data_sample, outputs, loss_dict):
        if data_sample is None:
            raise TypeError("data_sample=None; BaseEvaluator.process requires a data_sample to forward to metric implementations")
        if outputs is None:
            raise TypeError("outputs=None; BaseEvaluator.process requires outputs to forward to metric implementations")
        if loss_dict is None:
            raise TypeError("loss_dict=None; BaseEvaluator.process requires a loss_dict to dispatch on metric keys")

        for metric, metric_impl in self.metrics.items():
            metric_impl(outputs, data_sample, loss_dict[metric])

    # calls the Metric object aggregate method
    # to compute the final metric value for each metric
    # which is stored in self._results
    def aggregate(self):
        for metric, metric_impl in self.metrics.items():
            self._results[metric] = float(metric_impl.aggregate())

    # calls the aggregate method to compute the final results
    # and returns the _results dictionary which is passed to
    # the event writer before writing to the backend
    # e.g. TensorBoard, WandB, disk, etc.
    def evaluate(self):
        self.aggregate()
        return self._results
