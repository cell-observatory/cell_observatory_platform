"""Unit tests for parallelism/quantize.py: converter module filters, hardware
gates, and the build_quantize_converter registry entry point.

CPU-only: converters are exercised via object.__new__ (their __init__ hardware
gate is monkeypatched where needed), never by running a real torchao swap.
"""

import pytest
import torch.nn as nn
from omegaconf import OmegaConf

import cell_observatory_platform.parallelism.quantize as quantize
from cell_observatory_platform.models.layers.attention import (
    LinearKMaskedBias,
    LinearMaskedBias,
)
from cell_observatory_platform.parallelism.quantize import (
    Float8Converter,
    MXConverter,
    build_quantize_converter,
)


def _float8_converter(filter_fqns=()):
    conv = object.__new__(Float8Converter)  # bypass __init__ (needs SM89 GPU)
    conv.filter_fqns = list(filter_fqns)
    return conv


def _mx_converter(filter_fqns=(), block_size=32):
    conv = object.__new__(MXConverter)  # bypass __init__ (needs SM100 GPU)
    conv.filter_fqns = list(filter_fqns)
    conv.block_size = block_size
    return conv


class TestFloat8ModuleFilter:
    def test_plain_linear_multiple_of_16_is_converted(self):
        assert _float8_converter().module_filter(nn.Linear(64, 64), "blocks.0.mlp.fc1")

    def test_masked_bias_linear_subclasses_are_never_converted(self):
        # torchao's swap would silently drop the bias-mask semantics
        # (LinearKMaskedBias is a fused-qkv projection: out_features % 3 == 0)
        conv = _float8_converter()
        assert not conv.module_filter(LinearKMaskedBias(64, 48), "blocks.0.att.qkv")
        assert not conv.module_filter(LinearMaskedBias(64, 64), "blocks.0.att.proj")

    def test_non_linear_modules_rejected(self):
        assert not _float8_converter().module_filter(nn.LayerNorm(64), "blocks.0.norm1")

    def test_dims_not_multiple_of_16_rejected(self):
        conv = _float8_converter()
        assert not conv.module_filter(nn.Linear(30, 64), "head")
        assert not conv.module_filter(nn.Linear(64, 30), "head")

    def test_filter_fqn_substring_excludes(self):
        conv = _float8_converter(filter_fqns=["head"])
        assert not conv.module_filter(nn.Linear(64, 64), "decoder.head")
        assert conv.module_filter(nn.Linear(64, 64), "decoder.blocks.0.mlp")


class TestMXModuleFilter:
    def test_dims_must_be_multiple_of_block_size(self):
        conv = _mx_converter()
        assert conv.module_filter(nn.Linear(64, 64), "blocks.0.mlp")
        assert not conv.module_filter(nn.Linear(48, 64), "blocks.0.mlp")

    def test_masked_bias_linear_subclasses_are_never_converted(self):
        assert not _mx_converter().module_filter(
            LinearKMaskedBias(64, 48), "blocks.0.att.qkv"
        )


class TestHardwareGates:
    def test_float8_raises_on_unsupported_hardware(self, monkeypatch):
        monkeypatch.setattr(quantize, "has_cuda_capability", lambda *a: False)
        cfg = OmegaConf.create(
            {"recipe": "tensorwise", "fsdp_float8_all_gather": True, "filter_fqns": []}
        )
        with pytest.raises(ValueError, match="SM89"):
            Float8Converter(cfg)

    def test_mx_raises_on_unsupported_hardware(self, monkeypatch):
        monkeypatch.setattr(quantize, "has_cuda_capability", lambda *a: False)
        cfg = OmegaConf.create({"recipe": "mxfp8_cublas", "filter_fqns": []})
        with pytest.raises(ValueError, match="SM100"):
            MXConverter(cfg)

    def test_float8_unknown_recipe_raises(self, monkeypatch):
        monkeypatch.setattr(quantize, "has_cuda_capability", lambda *a: True)
        cfg = OmegaConf.create({"recipe": "blockwise", "filter_fqns": []})
        with pytest.raises(ValueError, match="Unknown float8 recipe"):
            Float8Converter(cfg)


class TestBuildQuantizeConverter:
    def test_disabled_returns_none(self):
        assert build_quantize_converter(OmegaConf.create({"quantize": {"enable": False}})) is None

    def test_missing_section_returns_none(self):
        assert build_quantize_converter(OmegaConf.create({})) is None

    def test_unknown_backend_raises(self):
        cfg = OmegaConf.create({"quantize": {"enable": True, "backend": "nvfp4"}})
        with pytest.raises(ValueError, match="Unknown quantize backend"):
            build_quantize_converter(cfg)
