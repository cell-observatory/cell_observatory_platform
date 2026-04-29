"""Session-scoped Ray fixtures for ``tests/inference/``.

One Ray cluster per pytest session (instead of per test) — amortizes the
~3-5s ``ray.init``/``ray.shutdown`` cycle that previously dominated CI time.

Robustness to running subsets of tests is preserved by:

  * **Per-test ``unique_suffix``** (uuid4) — every actor / pool name is unique
    so cross-test name collisions in detached namespaces cannot happen.
  * **Per-test scrub autouse** — overrides the parent
    ``tests/conftest.py::_reset_ray_and_cuda_before_test`` autouse so the
    session cluster survives between tests, and additionally hard-kills any
    leaked named actors and unlinks any leaked ``/dev/shm`` segments left
    behind by ``ray.kill`` (which skips ``atexit`` handlers).
  * **Lazy session init** — pure utility tests (Tier-0) that never request
    ``ray_ctx`` / ``ray_node_id`` never trigger ``ray.init`` at all.
  * **Lazy re-init guard** — if a sibling test directory's autouse tore the
    cluster down between inference batches, the next inference test
    transparently re-initializes.

Test files are unchanged: the public fixture API (``ray_ctx``, ``ray_node_id``,
``unique_suffix``) is identical to the previous per-test implementation.
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
_TEST_OWNED_NS_EXACT = frozenset({"saver", "visualizer"})

_RAY_NUM_CPUS = 4
_RAY_NUM_GPUS = 0


def _init_ray_for_tests() -> None:
    init_ray_like_training(num_cpus=_RAY_NUM_CPUS, num_gpus=_RAY_NUM_GPUS)


# ---------------------------------------------------------------------------
# Session-scoped Ray cluster
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _ray_session_init_once():
    """Initialize Ray once per pytest session for this directory."""
    _init_ray_for_tests()
    try:
        yield
    finally:
        if ray.is_initialized():
            try:
                _kill_test_named_actors()
            except Exception:
                pass
            try:
                ray.shutdown()
            except Exception:
                pass
        _unlink_orphan_test_shm_segments()


@pytest.fixture
def _ensure_ray(_ray_session_init_once):
    """Re-init if a non-inference test's autouse tore the session cluster down.

    The parent ``tests/conftest.py`` autouse calls ``ray.shutdown()`` after
    every test. That autouse is overridden below for inference tests, but
    sibling directories' tests can still tear it down between inference
    batches (e.g. ``pytest tests/inference tests/models tests/inference``).
    """
    if not ray.is_initialized():
        _init_ray_for_tests()
    yield


# ---------------------------------------------------------------------------
# Public fixtures (identical signatures to the previous per-test version)
# ---------------------------------------------------------------------------


@pytest.fixture
def ray_ctx(_ensure_ray):
    """Backwards-compatible alias for the per-test ``ray_ctx`` fixture.

    No per-test ``ray.init``/``ray.shutdown``: the cluster is session-scoped.
    Existing tests that request this fixture continue to work unchanged.
    """
    yield


@pytest.fixture
def ray_node_id(_ensure_ray):
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
         (``buffers_node_*``, ``saver``, ``visualizer``)
      2. unlink any orphan ``/dev/shm`` segments owned by the current uid
         with the multiprocessing-default prefixes — ``ray.kill`` skips
         ``atexit`` handlers so ``HostMemoryBuffer._cleanup`` does not run
         and the OS segment leaks otherwise
      3. flush CUDA caches (matches parent behavior)

    All three steps are best-effort; failures are swallowed so a flaky
    cleanup never masks the real test failure.
    """
    yield
    if ray.is_initialized():
        try:
            _kill_test_named_actors()
        except Exception:
            pass
    _unlink_orphan_test_shm_segments()
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


def _kill_test_named_actors() -> None:
    """Hard-kill every named actor in test-owned namespaces.

    Targets:
      * ``HostMemoryBuffer`` (``lifetime="detached"``) in
        ``buffers_node_<NodeID>`` namespaces — survive driver exits, so a
        failed test would leak them across the session otherwise.
      * Stub ``saver`` / ``visualizer`` actors used by
        ``test_inferencer_worker.py`` and ``test_transfer_benchmark.py``.

    Idempotent: repeat invocations are no-ops once the actor is gone.
    """
    if not ray.is_initialized():
        return
    try:
        actors = list_named_actors(all_namespaces=True)
    except Exception:
        return
    for entry in actors:
        ns = entry.get("namespace") or ""
        name = entry.get("name") or ""
        if not name:
            continue
        if not (
            any(ns.startswith(p) for p in _TEST_OWNED_NS_PREFIXES)
            or ns in _TEST_OWNED_NS_EXACT
        ):
            continue
        try:
            handle = ray.get_actor(name, namespace=ns)
        except Exception:
            continue
        try:
            ray.kill(handle, no_restart=True)
        except Exception:
            pass


def _unlink_orphan_test_shm_segments() -> int:
    """Unlink leaked POSIX SHM segments left by hard-killed test actors.

    ``ray.kill`` short-circuits ``atexit``, so ``HostMemoryBuffer._cleanup``
    (which unlinks the underlying segment) does not run. Without this scrub
    ``/dev/shm`` accumulates one ``wnsm_*`` segment per killed actor across
    the session.

    Scoped to the current uid + the multiprocessing-default prefixes so a
    shared host's other users' segments are never touched.
    """
    if not _SHM_DIR.is_dir():
        return 0
    try:
        euid = os.geteuid()
    except AttributeError:
        return 0
    try:
        entries = list(_SHM_DIR.iterdir())
    except OSError:
        return 0
    removed = 0
    for p in entries:
        if not any(p.name.startswith(pref) for pref in _TEST_SHM_PREFIXES):
            continue
        try:
            if p.stat().st_uid != euid:
                continue
        except (FileNotFoundError, OSError):
            continue
        try:
            shm = shared_memory.SharedMemory(name=p.name)
        except FileNotFoundError:
            continue
        except Exception:
            try:
                p.unlink()
                removed += 1
            except (FileNotFoundError, PermissionError, OSError):
                pass
            continue
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except (PermissionError, OSError):
            pass
    return removed
