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


class DatasetEvaluator(metaclass=abc.ABCMeta):
    """
    Base class for a dataset evaluator.

    A ``DatasetEvaluator`` is fed by both flows:
      * Validation (``EpochBasedTrainer.run_validation_step``): ``outputs`` are
        raw ``model.forward`` outputs and ``loss_dict`` is the loss dict from
        the same forward pass.
      * Prediction-based test (``TestTrainer.run_test_step``): ``outputs`` are
        postprocessed predictions from ``model.predict`` (in target space) and
        ``loss_dict`` is ``None`` because ``predict`` does not compute losses.

    Each evaluator decides which flow(s) it supports. Loss-based evaluators
    (e.g. :class:`BaseEvaluator`) must reject ``loss_dict=None``. Prediction-
    based evaluators (e.g. :class:`AutomatedBenchmarkEvaluator`) ignore
    ``loss_dict`` and operate on ``outputs`` against
    ``data_sample["metainfo"][target_key]``.

    The class accumulates per-step state via :meth:`process` and produces the
    final aggregated metrics via :meth:`evaluate`.
    """

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
                - test flow -> return value of ``model.predict(inputs)`` already
                in target space.
            loss_dict (Optional[dict]):a dictionary of losses
                e.g. {"metric1": loss, "metric2": loss}
                where each metric is a string and loss is a float.
        """
        pass

    @abc.abstractmethod
    def evaluate(self):
        """
        Evaluate/summarize the performance, after
        processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                The dict should have the following structure:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass