"""
Low-precision (FP8 / MX-format) training converters for the torch-native trainer.

Partly adapted from:
https://github.com/pytorch/torchtitan/torchtitan/components/quantization/float8.py

Converters swap eligible ``nn.Linear`` modules for torchao low-precision
variants AFTER the model is materialized and BEFORE activation checkpointing /
torch.compile / fully_shard are applied (see parallelism/parallelize.py for the
ordering contract). Each converter hardware-gates itself at construction and
raises rather than silently falling back to bf16.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("ray")
logger.setLevel(logging.INFO)


def has_cuda_capability(major: int, minor: int) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


class Float8Converter:
    """FP8 linears via torchao (stable API).

    Recipes:
      - ``tensorwise`` (default torchao config; fastest on H100-class parts;
        optionally with FP8 FSDP2 param all-gather + the post-optimizer-step
        dynamic-scale precompute hook)
      - ``rowwise`` / ``rowwise_with_gw_hp`` (better numerics; comms stay bf16)

    Hardware gate: SM89+ (Ada/Hopper/Blackwell).
    """

    def __init__(self, quantize_cfg):
        if not has_cuda_capability(8, 9):
            raise ValueError(
                "float8 training is only supported on SM89 or later "
                f"(H100/H200/B200-class GPUs); found "
                f"{torch.cuda.get_device_name() if torch.cuda.is_available() else 'no CUDA device'}. "
                "Disable parallelism.quantize (or fix the hardware selector)."
            )
        from torchao.float8 import Float8LinearConfig

        self.recipe = quantize_cfg.recipe
        self.fsdp_float8_all_gather = False
        if self.recipe == "tensorwise":
            self.fsdp_float8_all_gather = bool(
                quantize_cfg.get("fsdp_float8_all_gather", False)
            )
            self.torchao_config = Float8LinearConfig(
                enable_fsdp_float8_all_gather=self.fsdp_float8_all_gather,
            )
        elif self.recipe in ("rowwise", "rowwise_with_gw_hp"):
            self.torchao_config = Float8LinearConfig.from_recipe_name(self.recipe)
            # inductor precision-cast workaround for rowwise numerics under
            # torch.compile (pytorch#150859; torchtitan applies the same flag)
            torch._inductor.config.emulate_precision_casts = True
        else:
            raise ValueError(
                f"Unknown float8 recipe: {self.recipe!r}. "
                "Valid recipes: tensorwise, rowwise, rowwise_with_gw_hp."
            )
        self.filter_fqns = list(quantize_cfg.get("filter_fqns", []) or [])

    def module_filter(self, module: nn.Module, fqn: str) -> bool:
        # Exact type check, NOT isinstance: LinearKMaskedBias/LinearMaskedBias
        # (models/layers/attention.py) subclass nn.Linear, and torchao's
        # Float8Linear swap would silently drop their bias-mask semantics.
        if type(module) is not nn.Linear:
            return False
        # fp8 tensor cores require dims % 16 == 0
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            return False
        return not any(f in fqn for f in self.filter_fqns)

    def convert(self, model: nn.Module) -> nn.Module:
        from torchao.float8 import convert_to_float8_training

        convert_to_float8_training(
            model, config=self.torchao_config, module_filter_fn=self.module_filter
        )
        n_swapped = sum(
            1 for m in model.modules() if type(m).__name__ == "Float8Linear"
        )
        if n_swapped == 0:
            raise ValueError(
                "Float8Converter matched no Linear layers — check "
                "parallelism.quantize.filter_fqns and layer dims (% 16)."
            )
        logger.info(
            f"[quantize] swapped {n_swapped} nn.Linear -> Float8Linear "
            f"(recipe={self.recipe}, fsdp_fp8_all_gather={self.fsdp_float8_all_gather})"
        )
        return model

    def post_optimizer_hook(self, model_parts) -> None:
        """Recompute FP8 dynamic scales after the optimizer step.

        Only needed for tensorwise + FP8 FSDP all-gather (amax precompute in a
        single fused kernel instead of per-param at the next forward).
        """
        if not self.fsdp_float8_all_gather:
            return
        from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

        for model in model_parts:
            precompute_float8_dynamic_scale_for_fsdp(model)


class MXConverter:
    """MX-format (block-scaled) linears via torchao prototype. Blackwell only.

    Recipes (torchao MXLinearRecipeName): ``mxfp8_cublas`` (default),
    ``mxfp8_cublas_rceil``, and experimental ``mxfp4_cutlass``. MX formats use
    1x32 block scaling; requires SM100+ (B200-class) hardware.

    NOTE: torchao marks MX training as prototype — pin/verify the torchao
    version when first enabling this on Blackwell.
    """

    _EMULATED_RECIPES = ("mxfp8_emulated", "mxfp4_emulated")

    def __init__(self, quantize_cfg):
        recipe = quantize_cfg.recipe
        if not has_cuda_capability(10, 0) and recipe not in self._EMULATED_RECIPES:
            raise ValueError(
                "MX-format training requires SM100 or later (B200-class GPUs); "
                f"found {torch.cuda.get_device_name() if torch.cuda.is_available() else 'no CUDA device'}."
            )
        from torchao.prototype.mx_formats import MXLinearConfig

        self.recipe = recipe
        self.torchao_config = MXLinearConfig.from_recipe_name(recipe)
        self.block_size = int(self.torchao_config.block_size)
        self.filter_fqns = list(quantize_cfg.get("filter_fqns", []) or [])

    def module_filter(self, module: nn.Module, fqn: str) -> bool:
        if type(module) is not nn.Linear:
            return False
        # MX block scaling needs dims % block_size (32) == 0
        if (
            module.in_features % self.block_size != 0
            or module.out_features % self.block_size != 0
        ):
            return False
        return not any(f in fqn for f in self.filter_fqns)

    def convert(self, model: nn.Module) -> nn.Module:
        from torchao.quantization import quantize_

        quantize_(model, self.torchao_config, filter_fn=self.module_filter)
        n_swapped = sum(1 for m in model.modules() if type(m).__name__ == "MXLinear")
        if n_swapped == 0:
            raise ValueError(
                "MXConverter matched no Linear layers — check "
                "parallelism.quantize.filter_fqns and layer dims "
                f"(% {self.block_size})."
            )
        logger.info(
            f"[quantize] swapped {n_swapped} nn.Linear -> MXLinear (recipe={self.recipe})"
        )
        return model

    def post_optimizer_hook(self, model_parts) -> None:
        pass


_CONVERTERS = {
    "float8": Float8Converter,
    "mx": MXConverter,
    # "nvfp4": torchao has no NVFP4 *training* support yet (inference/QAT only
    # as of 2026-07); slot reserved.
}


def build_quantize_converter(parallelism_cfg):
    """Build the configured quantization converter, or None when disabled.

    Unsupported hardware raises at construction (fail loud, not silent bf16).
    """
    quantize_cfg = parallelism_cfg.get("quantize", None)
    if quantize_cfg is None or not quantize_cfg.get("enable", False):
        return None
    backend = quantize_cfg.backend
    if backend not in _CONVERTERS:
        raise ValueError(
            f"Unknown quantize backend: {backend!r}. "
            f"Valid backends: {sorted(_CONVERTERS)}."
        )
    return _CONVERTERS[backend](quantize_cfg)
