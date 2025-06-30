# import os
# import pytest

# import torch

# from omegaconf import open_dict
# from hydra.utils import get_class

# from tests.conftest import distributed_test, config

# def _test_loops_dist(config):
#     trainer_cls = get_class(config.trainer)
#     trainer = trainer_cls(config)
    
#     trainer.run()
#     trainer.test()
    
#     return {"success": True}

# def test_loops(config):
#     if not torch.cuda.is_available():
#         pytest.skip("No GPUs available for testing")

#     with open_dict(config):
#         config.experiment_name = "test_hooks"
#         config.paths.resume_checkpointdir = None
        
#         config.clusters.worker_nodes = 1
#         config.clusters.gpus_per_worker = torch.cuda.device_count() 
#         config.clusters.cpus_per_gpu = 4
#         config.clusters.mem_per_cpu = 31000

#     metrics = distributed_test(cfg=config, test="tests.training.test_loops._test_loops_dist")
#     assert metrics.get("success", False), "Distributed loops test failed"