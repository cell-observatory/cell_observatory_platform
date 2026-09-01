"""Single-GPU FSDP2 smoke test for the torch-native path: fully_shard per
block + root, bf16 MixedPrecisionPolicy, per-block in-place compile, real
forward/backward/clip/step, and a sharded DCP save/load roundtrip.

Requires one CUDA device (marked ``cuda``; conftest skips without the
toolkit). The multi-GPU + Ray + FP8 path is exercised by the real experiment
config (configs/experiments/.../torch_native.yaml)."""

import math

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

pytestmark = pytest.mark.cuda


class _Block(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class _Encoder(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(_Block(dim) for _ in range(depth))

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        return x


class _TinyTransformer(nn.Module):
    def __init__(self, dim=64, depth=3):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.encoder = _Encoder(dim, depth)
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        return self.head(self.encoder(self.embed(x)))


@pytest.fixture()
def single_gpu_pg(monkeypatch):
    from cell_observatory_platform.tests.conftest import free_tcp_port

    if dist.is_initialized():
        dist.destroy_process_group()
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(free_tcp_port()))
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def test_fsdp2_forward_backward_step_and_dcp_roundtrip(single_gpu_pg, tmp_path):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    from cell_observatory_platform.parallelism.parallelize import apply_fsdp
    from cell_observatory_platform.training.checkpoint import DCPCheckpointManager
    from cell_observatory_platform.training.optimizers import OptimizersContainer

    torch.manual_seed(0)
    with torch.device("cuda"):
        model = _TinyTransformer()
    # in-place per-block compile keeps FQNs canonical for DCP
    for block in model.encoder.transformer_blocks:
        block.compile(backend="inductor")

    dp_mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp_shard",))
    apply_fsdp(
        model,
        dp_mesh,
        module_blocks=[("encoder", "transformer_blocks")],
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # params are sharded DTensors, FQNs canonical (no _orig_mod)
    params = dict(model.named_parameters())
    assert all(isinstance(p, DTensor) for p in params.values())
    assert not any("_orig_mod" in k for k in params)

    optimizers = OptimizersContainer(
        [model], torch.optim.AdamW,
        {"lr": 1e-3, "betas": (0.9, 0.99), "eps": 1e-8, "weight_decay": 0.01,
         "fused": False, "foreach": True},
    )

    from torchtitan.distributed import utils as dist_utils

    losses = []
    for _ in range(3):
        optimizers.zero_grad()
        loss = model(torch.randn(2, 16, 64, device="cuda")).float().pow(2).mean()
        loss.backward()
        grad_norm = dist_utils.clip_grad_norm_(
            list(model.parameters()), 1.0, foreach=True
        )
        assert torch.isfinite(grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm)
        optimizers.step()
        losses.append(float(loss.detach()))
    assert all(math.isfinite(v) for v in losses)

    # sharded DCP save -> mutate -> load roundtrip
    manager = DCPCheckpointManager(
        model_parts=[model], optimizers=optimizers,
        save_checkpointdir=tmp_path / "ckpt",
    )
    assert manager.save(curr_step=3, last_step=True)
    want = {k: v.full_tensor().clone() for k, v in model.state_dict().items()}

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    manager2 = DCPCheckpointManager(
        model_parts=[model], optimizers=optimizers,
        save_checkpointdir=tmp_path / "ckpt2",
        resume_checkpointdir=tmp_path / "ckpt",
    )
    loaded_step, _ = manager2.load()
    assert loaded_step == 3
    for k, v in model.state_dict().items():
        assert torch.allclose(v.full_tensor(), want[k]), f"weight mismatch after DCP load: {k}"
