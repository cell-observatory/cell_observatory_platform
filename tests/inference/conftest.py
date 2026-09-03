"""Session-scoped Ray cluster for ``tests/inference/`` + per-test scrub of
test-owned named actors and the SHM segments they leave behind (``ray.kill``
skips ``HostMemoryBuffer``'s atexit unlink). Tests that never request
``ray_ctx``/``ray_node_id`` never start Ray.

The SHM scrub is scoped to segments that either belonged to an actor killed
by the scrub itself or appeared during the test: segments owned by a
concurrent pytest session or a live inference run on the same host are never
touched.
"""
from __future__ import annotations

import os
import uuid
from multiprocessing import shared_memory
from pathlib import Path

import pytest
import ray
from ray.util import list_named_actors

from tests.ray_init_helpers import init_ray_like_training


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SHM_DIR = Path("/dev/shm")
# CPython ``multiprocessing.shared_memory`` auto-names segments with these
# prefixes (``psm_`` historically, ``wnsm_`` on modern Linux Python).
_TEST_SHM_PREFIXES = ("wnsm_", "psm_")
# Namespaces whose detached / named actors are owned exclusively by tests
# in this directory; safe to mass-kill on per-test scrub.
_TEST_OWNED_NS_PREFIXES = ("buffers_node_",)
_TEST_OWNED_NS_EXACT = frozenset({"saver", "visualizer", "test_buffers"})

_RAY_NUM_CPUS = 4
_RAY_NUM_GPUS = 0


def _init_ray_for_tests() -> None:
    init_ray_like_training(num_cpus=_RAY_NUM_CPUS, num_gpus=_RAY_NUM_GPUS)


def _test_shm_names() -> set[str]:
    """Multiprocessing-default segment names owned by this uid, right now."""
    try:
        entries = list(_SHM_DIR.iterdir())
    except OSError:
        return set()
    try:
        euid = os.geteuid()
    except AttributeError:
        return set()
    names: set[str] = set()
    for p in entries:
        if not p.name.startswith(_TEST_SHM_PREFIXES):
            continue
        try:
            if p.stat().st_uid == euid:
                names.add(p.name)
        except OSError:
            pass
    return names


# ---------------------------------------------------------------------------
# Session-scoped Ray cluster
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _ray_session_init_once():
    """Initialize Ray once per pytest session for this directory."""
    baseline = _test_shm_names()  # pre-existing segments are never ours
    _init_ray_for_tests()
    try:
        yield
    finally:
        owned: set[str] = set()
        if ray.is_initialized():
            try:
                owned = _kill_test_named_actors()
            except Exception:
                pass
            try:
                ray.shutdown()
            except Exception:
                pass
        _unlink_shm_segments(owned | (_test_shm_names() - baseline))


# ---------------------------------------------------------------------------
# Public fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ray_ctx(_ray_session_init_once):
    """Session cluster; re-inits if a sibling directory's autouse tore it down.

    The parent ``tests/conftest.py`` autouse calls ``ray.shutdown()`` after
    every test. That autouse is overridden below for inference tests, but
    sibling directories' tests can still tear it down between inference
    batches (e.g. ``pytest tests/inference tests/models tests/inference``).
    """
    if not ray.is_initialized():
        _init_ray_for_tests()
    yield


@pytest.fixture
def ray_node_id(ray_ctx):
    """Hex node-ID of the single-node test cluster."""
    return ray.nodes()[0]["NodeID"]


@pytest.fixture
def unique_suffix():
    """Per-test unique suffix for actor / pool names to avoid collisions."""
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Per-test scrub autouse (overrides parent)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_ray_and_cuda_before_test():
    """Per-test scrub for inference tests.

    Overrides ``tests/conftest.py::_reset_ray_and_cuda_before_test`` so the
    session-scoped Ray cluster is **not** torn down between tests. Instead:

      1. hard-kill any named actors left behind in test-owned namespaces
         (``buffers_node_*``, ``buffers``-style test namespaces, ``saver``,
         ``visualizer``)
      2. unlink ONLY the ``/dev/shm`` segments that (a) belonged to an actor
         killed in step 1 or (b) appeared during this test -- never a
         concurrent session's or a live run's pools
      3. flush CUDA caches (matches parent behavior)

    All three steps are best-effort; failures are swallowed so a flaky
    cleanup never masks the real test failure.
    """
    before = _test_shm_names()
    yield
    owned: set[str] = set()
    if ray.is_initialized():
        try:
            owned = _kill_test_named_actors()
        except Exception:
            pass
    _unlink_shm_segments(owned | (_test_shm_names() - before))
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Internal cleanup helpers
# ---------------------------------------------------------------------------


def _kill_test_named_actors() -> set[str]:
    """Hard-kill every named actor in test-owned namespaces.

    Returns the SHM segment names those actors owned, read via
    ``get_config`` *before* the kill (stub ``saver``/``visualizer`` actors
    have no ``get_config`` and contribute nothing).

    Idempotent: repeat invocations are no-ops once the actor is gone.
    """
    owned: set[str] = set()
    if not ray.is_initialized():
        return owned
    try:
        actors = list_named_actors(all_namespaces=True)
    except Exception:
        return owned
    for entry in actors:
        ns = entry.get("namespace") or ""
        name = entry.get("name") or ""
        if not name:
            continue
        if not (ns.startswith(_TEST_OWNED_NS_PREFIXES) or ns in _TEST_OWNED_NS_EXACT):
            continue
        try:
            handle = ray.get_actor(name, namespace=ns)
        except Exception:
            continue
        try:
            owned.add(ray.get(handle.get_config.remote(), timeout=5)["name"])
        except Exception:
            pass
        try:
            ray.kill(handle, no_restart=True)
        except Exception:
            pass
    return owned


def _unlink_shm_segments(names: set[str]) -> int:
    """Unlink the given POSIX SHM segments (best-effort, idempotent)."""
    removed = 0
    for name in names:
        try:
            shm = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            continue
        except Exception:
            try:
                (_SHM_DIR / name).unlink()
                removed += 1
            except OSError:
                pass
            continue
        try:
            shm.close()
            shm.unlink()
            removed += 1
        except (FileNotFoundError, OSError):
            pass
    return removed
