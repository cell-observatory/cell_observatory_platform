from typing import Any, Dict, List, Sequence, Tuple, Iterable

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor, Replicate, Placement
from torch.distributed.tensor.parallel import PrepareModuleInput, PrepareModuleOutput


# Pytorch Issue: https://github.com/pytorch/pytorch/issues/121020
# Put unsharded weights on the same device mesh as tp_mesh.
# This does not introduce communication overhead since run_check=False by default
def replicate_unsharded_modules(tp_mesh, model):
    # Create a pre-forward hook to convert input tensors into DTensors
    pre_forward_hook = PrepareModuleInput(
        input_layouts=Replicate(), desired_input_layouts=Replicate(), use_local_output=False
    )
    post_forward_hook = PrepareModuleOutput(
        output_layouts=Replicate(), desired_output_layouts=Replicate(), use_local_output=True
    )

    hooks = (pre_forward_hook, post_forward_hook)

    # Replicate non-tp modules onto the mesh
    def replicate_module(parallelize_hooks, module, tp_mesh):
        # If this module does not have any parameters, skip
        # (i.e., do NOT attach hooks or attempt replication).
        if not any(True for _ in module.parameters(recurse=False)):
            return
        # If any parameter of the module is already a DTensor, it is on the tp_mesh and skip replication.
        if any(isinstance(p, DTensor) for p in module.parameters(recurse=False)):
            assert all([param.device_mesh.mesh_dim_names == ("tp",) for param in module.parameters(recurse=False)])
            return

        # Attach forward hooks
        for hook in parallelize_hooks:
            hook._apply(module, tp_mesh)

        # Replicate all parameters
        for name, param in module.named_parameters(recurse=False):
            dist_param = torch.nn.Parameter(distribute_tensor(param, tp_mesh, [Replicate()]))
            module.register_parameter(name, dist_param)

    # apply to all modules in the model
    model.apply(lambda module: replicate_module(hooks, module, tp_mesh))


# --- --- --- --- alternative but similar strategies --- --- --- --- #


def replicate_module_params_on_mesh(module: nn.Module, tp_mesh: DeviceMesh) -> None:
    if module is None:
        return

    for name, param in list(module.named_parameters(recurse=False)):
        if isinstance(param, DTensor):
            continue

        dist_param = torch.nn.Parameter(
            distribute_tensor(param, tp_mesh, [Replicate()])
        )
        module.register_parameter(name, dist_param)


def replicate_parameter_on_mesh(param: nn.Parameter, tp_mesh: DeviceMesh) -> nn.Parameter:
    if isinstance(param, DTensor):
        return param

    return torch.nn.Parameter(
        distribute_tensor(param, tp_mesh, [Replicate()])
    )


def distribute_param_on_mesh(
    module: nn.Module,
    param_name: str,
    mesh: DeviceMesh,
    placements: list[Placement],
) -> None:
    if module is None:
        return

    # Grab the existing parameter
    param = getattr(module, param_name, None)
    if param is None:
        raise AttributeError(f"Module {module.__class__.__name__} has no param '{param_name}'")

    # If it's already a DTensor, do nothing
    if isinstance(param, DTensor):
        return

    dt = distribute_tensor(param, mesh, placements)
    dist_param = nn.Parameter(dt)

    module.register_parameter(param_name, dist_param)


def get_cp_buffers(
    data_sample: Dict[str, Any],
    model_parts: Sequence[nn.Module],
    disable_load_balance: bool = True,
) -> Tuple[List[torch.Tensor], List[int]]:
    raise NotImplementedError(
        "CP is currently disable pending a larger parallelism refactor."
    )

    if disable_load_balance:
        disable_cp_load_balance()

    if len(model_parts) != 1:
        raise NotImplementedError(
            "get_cp_buffers currently only supports single-module models."
        )

    model = model_parts[0]

    assert hasattr(model, "get_pos_embedding"), (
        "Model must implement get_pos_embedding() to use context parallelism."
    )

    inputs = data_sample["data_tensor"]
    targets = data_sample["metainfo"]["targets"][0]
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("data_sample['data_tensor'] must be a torch.Tensor.")

    cp_buffers: List[torch.Tensor] = [inputs, targets]
    cp_seq_dims: List[int] = [1, 1]

    pos_embs = model.get_pos_embedding()
    if pos_embs is None:
        return cp_buffers, cp_seq_dims

    # normalize to list
    if isinstance(pos_embs, torch.Tensor):
        pos_embs = [pos_embs]

    for pe in pos_embs:
        if pe is None:
            continue
        assert pe.ndim >= 2, "Positional embeddings must have at least 2 dims."
        cp_buffers.append(pe)
        cp_seq_dims.append(-2)

    return cp_buffers, cp_seq_dims


# HACK: Disable load balancing in context parallel attention.
def disable_cp_load_balance():
    from torch.distributed.tensor.experimental import _attention as cp_attn
    cp_attn._cp_options.enable_load_balance = False


# based on: https://github.com/pytorch/torchtitan/torchtitan/distributed/utils.py
@torch.no_grad()
def compute_grad_norm(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )
    return total_norm