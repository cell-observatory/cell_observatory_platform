"""Per-test Ray fixtures for all tests/inference/ tests.

Ray cluster starts and shuts down for every test that requests ``ray_ctx``.
Slower than session-scoped but fully isolated — no cross-test cluster state.
The parent conftest autouse (_reset_ray_and_cuda_before_test) is compatible:
ray_ctx teardown runs first (inner scope), Ray is already shut down when
the parent autouse fires, so ray.is_initialized() == False → no double-shutdown.
"""
from __future__ import annotations

import uuid

import pytest
import ray

from tests.ray_init_helpers import init_ray_like_training


@pytest.fixture
def ray_ctx():
    """Start a fresh local Ray cluster for this test, tear it down after."""
    init_ray_like_training(num_cpus=4, num_gpus=0)
    yield
    ray.shutdown()


@pytest.fixture
def ray_node_id(ray_ctx):
    """Hex node-ID of the single-node test cluster."""
    return ray.nodes()[0]["NodeID"]


@pytest.fixture
def unique_suffix():
    """Per-test unique suffix for actor / pool names to avoid collisions."""
    return uuid.uuid4().hex[:8]
