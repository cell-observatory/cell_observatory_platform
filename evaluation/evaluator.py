"""
https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/evaluator.py#L224

Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import abc
from typing import Dict, Mapping


class DatasetEvaluator(metaclass=abc.ABCMeta):
    """
    Base class for a dataset evaluator.

    A ``DatasetEvaluator`` is fed by both flows:
      * Validation (``EpochBasedTrainer.run_validation_step``): ``outputs`` are
        raw ``model.forward`` outputs and ``loss_dict`` is the loss dict from
        the same forward pass.
      * Prediction-based test (``TestTrainer.run_test_step``): ``outputs`` are
        postprocessed predictions from ``model.evaluate_step`` in target space,
        and ``loss_dict`` is ``None``.

    Each evaluator decides which flow(s) it supports. Loss-based evaluators
    (e.g. :class:`BaseEvaluator`) must reject ``loss_dict=None``. Prediction-
    based evaluators (e.g. :class:`AutomatedBenchmarkEvaluator`) ignore
    ``loss_dict`` and operate on ``outputs`` against the ground truth in
    ``data_sample["metainfo"]``.

    Concrete evaluators build ``self.metrics`` (an ordered ``{name: Metric}``
    dict, via :func:`evaluation.metrics.build_metrics`) and accumulate per-step
    state in :meth:`process`. The shared :meth:`evaluate` gathers + aggregates
    every metric and flattens any ``Mapping`` return into ``f"{name}/{subkey}"``
    so every evaluator yields a flat ``dict[str, float]``.
    """

    # Concrete evaluators populate this in __init__ (via build_metrics).
    metrics: Dict[str, object] = {}

    @abc.abstractmethod
    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    @abc.abstractmethod
    def process(self, data_sample, outputs, loss_dict):
        """
        Process one (data_sample, outputs[, loss_dict]) tuple.

        Args:
            data_sample (dict): the inputs that were used to call the model.
                Must contain ``metainfo`` for evaluators that need targets.
            outputs:
                - validation flow -> raw return value of ``model(inputs)``
                - test flow -> return value of ``model.evaluate_step(inputs)``
                already in target space.
            loss_dict (Optional[dict]):a dictionary of losses
                e.g. {"metric1": loss, "metric2": loss}
                where each metric is a string and loss is a float.
        """
        pass

    def evaluate(self) -> Dict[str, float]:
        """Gather + aggregate every metric into a flat ``dict[str, float]``.

        Cross-rank reduction happens by gathering each metric's compact
        sufficient statistics (``gather()`` is idempotent and a no-op at
        ``world_size == 1``) and aggregating once over the pooled multiset. A
        metric whose ``aggregate()`` returns a ``Mapping`` (e.g.
        :class:`PredictedIoUEvalMetric`) is flattened under ``f"{name}/{sub}"``
        -- unless the metric sets ``flat_result_keys``, in which case its
        mapping keys are already fully-qualified log keys and are written
        verbatim (e.g. :class:`BoxMIoUMetric`'s ``box_miou`` +
        ``box_match_recall``, keeping the historical flat key for dashboards
        and ``val_metric`` selection).
        """
        results: Dict[str, float] = {}
        for name, metric in self.metrics.items():
            metric.gather()
            value = metric.aggregate()
            if isinstance(value, Mapping):
                bare = bool(getattr(metric, "flat_result_keys", False))
                for subkey, subval in value.items():
                    results[subkey if bare else f"{name}/{subkey}"] = float(subval)
            else:
                results[name] = float(value)
        self._results = results
        return results