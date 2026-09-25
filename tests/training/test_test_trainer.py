"""TestTrainer.run_test_step dispatches model.evaluate_step and feeds the
postprocessed predictions to the evaluator with loss_dict=None; a model without
evaluate_step fails loudly. CPU-only."""

from unittest.mock import Mock

import pytest

import cell_observatory_platform.training.loops as loops


def _make_test_trainer(model, evaluator):
    trainer = object.__new__(loops.TestTrainer)
    trainer.model = model
    trainer.evaluator = evaluator
    trainer.before_test_step = lambda: None
    trainer.after_test_step = lambda **kwargs: None
    trainer._iter = 0
    return trainer


def test_run_test_step_calls_evaluate_step():
    """evaluate_step's output is handed to evaluator.process with loss_dict=None
    and the step counter advances."""
    model = Mock()
    model.evaluate_step = Mock(return_value=[{"eval": True}])
    evaluator = Mock(spec=["process"])
    trainer = _make_test_trainer(model, evaluator)
    data_sample = {"metainfo": {"targets": []}}

    trainer.run_test_step(idx=0, data_sample=data_sample)

    model.evaluate_step.assert_called_once_with(data_sample)
    evaluator.process.assert_called_once_with(data_sample, [{"eval": True}], loss_dict=None)
    assert trainer._iter == 1


def test_run_test_step_missing_evaluate_step_raises():
    """A model that does not implement evaluate_step raises an AttributeError
    naming the missing method."""
    class ModelWithoutEvaluateStep:
        pass

    model = ModelWithoutEvaluateStep()
    evaluator = Mock(spec=["process"])
    trainer = _make_test_trainer(model, evaluator)

    with pytest.raises(AttributeError, match="evaluate_step"):
        trainer.run_test_step(idx=0, data_sample={"metainfo": {}})
