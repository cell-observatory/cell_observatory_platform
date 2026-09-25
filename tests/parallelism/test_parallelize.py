"""Unit tests for the CPU-testable helpers of parallelism/parallelize.py:
the reshard-after-forward policy mapping and FSDP module-block discovery."""

import pytest
from omegaconf import OmegaConf

from cell_observatory_platform.parallelism.parallelize import (
    _fsdp_module_blocks,
    _resolve_reshard_after_forward,
)


class TestParallelizeHelpers:
    def test_reshard_policy_mapping(self):
        assert _resolve_reshard_after_forward("default") is True
        assert _resolve_reshard_after_forward("always") is True
        assert _resolve_reshard_after_forward("never") is False
        with pytest.raises(ValueError, match="Invalid fsdp_reshard_after_forward"):
            _resolve_reshard_after_forward("sometimes")

    def test_fsdp_module_blocks_prefers_dedicated_list(self):
        node = OmegaConf.create(
            {
                "torch_compile": {"modules": [["enc", "transformer_blocks"]]},
                "fsdp": {"modules": [["enc", "blocks"], ["target_enc", "blocks"]]},
            }
        )
        assert list(_fsdp_module_blocks(node)) == [
            ["enc", "blocks"],
            ["target_enc", "blocks"],
        ]

    def test_fsdp_module_blocks_falls_back_to_compile_list(self):
        node = OmegaConf.create(
            {"torch_compile": {"modules": [["enc", "transformer_blocks"]]}}
        )
        assert list(_fsdp_module_blocks(node)) == [["enc", "transformer_blocks"]]
