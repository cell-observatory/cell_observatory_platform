"""Unit tests for utils/context.py: rank/world-size fallbacks outside any
process group, the single-rank reducer shared by loggers/hooks/evaluators,
gather_and_reduce on a gloo world of one, and inference_context mode
restoration. CPU-only."""

import pytest
import torch
import torch.distributed as dist

from cell_observatory_platform.utils.context import (
    barrier,
    gather_and_reduce,
    get_local_world_size,
    get_world_size,
    inference_context,
    is_main_process,
    local_rank,
    process_rank,
    reduce_values,
)


@pytest.fixture()
def no_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()
    yield


def test_single_process_rank_and_world_size(no_process_group):
    """Outside Ray Train and torch.distributed the helpers describe a single
    main process and barrier is a no-op."""
    assert get_world_size() == 1 and get_local_world_size() == 1
    assert process_rank() == 0 and local_rank() == 0
    assert is_main_process() is True
    assert barrier() is None


def test_gather_and_reduce_without_process_group_returns_independent_clone(no_process_group):
    """Without a process group the input is returned as an independent clone;
    the caller's tensor is never aliased or mutated."""
    t = torch.tensor([1.0, 2.0])
    out = gather_and_reduce(t, "sum")
    assert out is not t and torch.equal(out, t)
    out += 1
    assert torch.equal(t, torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("op", ["sum", "mean", "max", "min", "median", "MEAN"])
def test_gather_and_reduce_world_one_all_ops_are_identity(gloo_pg, op):
    """On a world of one every supported op (case-insensitive) reduces to the
    input itself, through the real all_reduce / all_gather branches."""
    t = torch.tensor([3.0, -1.0])
    out = gather_and_reduce(t, op)
    assert torch.equal(out, t) and out is not t


def test_gather_and_reduce_rejects_unknown_op(gloo_pg):
    """An op outside the supported set raises rather than silently all-reducing."""
    with pytest.raises(ValueError, match="Unsupported op"):
        gather_and_reduce(torch.tensor(1.0), "variance")


@pytest.mark.parametrize("op,values,expected", [
    ("sum", [1.0, 2.0, 4.0], 7.0), ("mean", [1.0, 2.0, 6.0], 3.0),
    ("median", [5.0, 1.0, 3.0], 3.0), ("median", [4.0, 1.0, 3.0, 2.0], 2.5),
    ("max", [1.0, 9.0, 3.0], 9.0), ("min", [1.0, 9.0, -3.0], -3.0),
    ("mean", [], 0.0), ("max", [], 0.0),
])
def test_reduce_values(op, values, expected):
    """reduce_values implements sum/mean/median/max/min over a flat list and
    returns 0.0 for empty input."""
    assert reduce_values(op, values) == pytest.approx(expected)


def test_reduce_values_rejects_unknown_method():
    """An unknown reduce method raises."""
    with pytest.raises(ValueError, match="Unknown reduce method"):
        reduce_values("p95", [1.0])


@pytest.mark.parametrize("was_training", [True, False])
def test_inference_context_restores_previous_mode(was_training):
    """inference_context puts the whole module tree into eval mode and restores
    the previous mode (including children) on exit."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    model.train(was_training)
    with inference_context(model):
        assert model.training is False
        assert all(not m.training for m in model.modules())
    assert model.training is was_training
    assert model[1].training is was_training
