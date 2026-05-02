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


def test_run_test_step_default_calls_predict():
    model = Mock()
    model.predict = Mock(return_value={"ok": True})
    # spec so getattr(evaluator, "predict_method", "predict") resolves to "predict"
    # (bare Mock would synthesize predict_method as another Mock)
    evaluator = Mock(spec=["process"])
    trainer = _make_test_trainer(model, evaluator)
    data_sample = {"metainfo": {"targets": []}}

    trainer.run_test_step(idx=0, data_sample=data_sample)

    model.predict.assert_called_once_with(data_sample)
    evaluator.process.assert_called_once_with(data_sample, {"ok": True}, loss_dict=None)
    assert trainer._iter == 1


def test_run_test_step_custom_predict_method():
    model = Mock()
    model.predict = Mock(return_value={"legacy": True})
    model.predict_for_eval = Mock(return_value=[{"eval": True}])
    evaluator = Mock()
    evaluator.predict_method = "predict_for_eval"
    trainer = _make_test_trainer(model, evaluator)
    data_sample = {"metainfo": {"targets": []}}

    trainer.run_test_step(idx=0, data_sample=data_sample)

    model.predict.assert_not_called()
    model.predict_for_eval.assert_called_once_with(data_sample)
    evaluator.process.assert_called_once_with(data_sample, [{"eval": True}], loss_dict=None)
    assert trainer._iter == 1


def test_run_test_step_missing_predict_method_raises():
    class ModelWithoutRequestedMethod:
        pass

    model = ModelWithoutRequestedMethod()
    evaluator = Mock()
    evaluator.predict_method = "missing_predict"
    trainer = _make_test_trainer(model, evaluator)

    with pytest.raises(AttributeError, match="missing_predict"):
        trainer.run_test_step(idx=0, data_sample={"metainfo": {}})
