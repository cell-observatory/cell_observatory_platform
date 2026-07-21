from unittest.mock import Mock

import pytest

from cell_observatory_platform.training.loops import TestTrainer


def _make_test_trainer(model, evaluator):
    trainer = TestTrainer.__new__(TestTrainer)
    trainer.model = model
    trainer.evaluator = evaluator
    trainer.before_test_step = lambda: None
    trainer.after_test_step = lambda **kwargs: None
    trainer._iter = 0
    return trainer


def test_run_test_step_calls_evaluate_step():
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
    class ModelWithoutEvaluateStep:
        pass

    model = ModelWithoutEvaluateStep()
    evaluator = Mock(spec=["process"])
    trainer = _make_test_trainer(model, evaluator)

    with pytest.raises(AttributeError, match="evaluate_step"):
        trainer.run_test_step(idx=0, data_sample={"metainfo": {}})
