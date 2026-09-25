"""Ray initialization aligned with ``training.runner.initialize_session``.

By default uses ``RuntimeEnv(py_modules=...)`` and **does not** set
``working_dir`` to the full repository: packaging ``working_dir`` on a large
NFS tree can make ``ray.init`` appear to hang for a long time. Callers that
need full parity with ``runner.py`` (which sets ``working_dir``) can pass
``package_working_dir=True``.
"""
from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from ray import init
from ray.runtime_env import RuntimeEnv


@contextlib.contextmanager
def local_cwd_for_ray_start():
    """Run ``ray.init`` from a local-disk cwd, then restore the caller's cwd.

    The raylet and every worker it spawns inherit the driver's cwd at init
    time, and ``setup_worker.py`` calls ``os.getcwd()`` on startup. On the
    NFS-backed repo path (/clusterfs) that dentry is occasionally invalidated
    for minutes at a time (``/proc/<pid>/cwd`` shows the unchanged directory
    as ``(deleted)``); new workers then die with FileNotFoundError and any
    test waiting on a fresh worker hangs until the mount revalidates (a
    32-minute stall was observed on 2026-08-20). Starting Ray from /tmp
    keeps worker start-up independent of the network filesystem. Absolute
    ``py_modules`` paths are unaffected.
    """
    prev = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        yield
    finally:
        os.chdir(prev)


def repository_root() -> Path:
    """Project root (parent of ``tests/``)."""
    return Path(__file__).resolve().parent.parent


def init_ray_like_training(
    *,
    num_cpus: int = 4,
    num_gpus: int | None = 0,
    object_store_memory: int | None = None,
    package_working_dir: bool = False,
) -> None:
    """Start or attach to Ray with env + ``py_modules`` like ``runner.py``.

    ``py_modules`` lets workers import the same tree as the driver without
    zipping the whole repo unless ``package_working_dir`` is true.
    """
    root = str(repository_root())
    env_spec: dict = dict(
        env_vars={k: v for k, v in os.environ.items()},
        py_modules=[root],
    )
    if package_working_dir:
        env_spec["working_dir"] = root
    runtime_env = RuntimeEnv(**env_spec)

    kwargs: dict = dict(
        log_to_driver=True,
        runtime_env=runtime_env,
        num_cpus=num_cpus,
        ignore_reinit_error=True,
    )
    # Optional: set RAY_TMPDIR to local SSD (e.g. /tmp) so Ray metadata is not on NFS.
    if os.environ.get("RAY_TMPDIR"):
        kwargs["_temp_dir"] = os.environ["RAY_TMPDIR"]
    if num_gpus is not None:
        kwargs["num_gpus"] = num_gpus
    if object_store_memory is not None:
        kwargs["object_store_memory"] = object_store_memory
    with local_cwd_for_ray_start():
        init(**kwargs)
