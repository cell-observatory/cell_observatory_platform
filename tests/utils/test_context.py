import pytest

import torch

from omegaconf import open_dict

from tests.conftest import distributed_test, config


def _test_context(config):
    import torch
    import math
    from ray.train import report
    from utils.context import (
        OpMap, process_rank, get_world_size, barrier,
        gather_and_reduce, inference_context
    )

    # basic sanity — OpMap values equal torch.distributed enums
    assert OpMap.SUM.value == torch.distributed.ReduceOp.SUM
    assert OpMap.MAX.value == torch.distributed.ReduceOp.MAX
    assert OpMap.MIN.value == torch.distributed.ReduceOp.MIN
    assert OpMap.MEAN.value == torch.distributed.ReduceOp.SUM

    # ranks and world size under Ray
    rank  = process_rank()
    world = get_world_size()
    assert 0 <= rank < world
    assert world > 1

    # barrier must not dead-lock / raise
    barrier()

    # gather_and_reduce: SUM & MEAN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.tensor(float(rank + 1), device=device)

    # SUM 
    tsum = gather_and_reduce(t.clone(), "sum")
    expected_sum = sum(float(i + 1) for i in range(world))
    assert math.isclose(tsum.item(), expected_sum, rel_tol=1e-6)

    # MEAN
    tmean = gather_and_reduce(t.clone(), "mean")
    expected_mean = expected_sum / world
    assert math.isclose(tmean.item(), expected_mean, rel_tol=1e-6)

    # inference_context restores training flag
    model = torch.nn.Linear(4, 4).to(device)
    model.train()
    assert model.training is True

    with inference_context(model):
        assert model.training is False
    assert model.training is True

    report({"success": True})


def test_context(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("At least 2 GPUs are required for this test")


    with open_dict(config):
        config.experiment_name = "test_context"
        config.paths.resume_checkpointdir = None
        
        config.clusters.worker_nodes = 1
        config.clusters.gpus_per_worker = 2
        config.clusters.cpus_per_gpu = 4

    metrics = distributed_test(cfg=config, test="tests.utils.test_context._test_context")
    assert metrics.get("success", True), "Distributed context test failed"