"""Config helpers for registry factories.

``instantiate_as`` lets a component that is *selected* by a registry ``name:`` still be
*constructed* by Hydra ``instantiate`` — so nested ``_target_`` children (e.g. a
preprocessor's transforms and ``mask_generator``) are still recursively instantiated
exactly as before. It works on both a legacy ``_target_`` config and a migrated
``name:`` config (it injects the class as ``_target_`` only when one is absent).
"""

from __future__ import annotations

from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from cell_observatory_platform.utils.registry import REGISTRY

# Registry selector keys that must NOT reach the component's __init__ as kwargs.
_SELECTORS = ("name", "BUILD")


# TODO: decide if we want to keep this instantiation pattern.
def instantiate_as(cls: type, cfg: Any = None, **overrides: Any) -> Any:
    """Hydra-instantiate ``cfg`` as ``cls`` (recursive; parent interpolations preserved).

    Mutates ``cfg`` in place — drops the registry selector keys and injects ``cls`` as
    ``_target_`` when the config doesn't already carry one — so the node stays attached
    to its parent tree and ``${...}`` interpolations still resolve. The node is consumed
    once (at build), so the in-place edit is harmless.
    """
    if cfg is None:
        return cls(**overrides)
    with open_dict(cfg):
        for k in _SELECTORS:
            cfg.pop(k, None)
        if "_target_" not in cfg:
            cfg["_target_"] = f"{cls.__module__}.{cls.__qualname__}"
    return instantiate(cfg, **overrides)


def build_kwargs(cfg: Any, *, drop=()) -> dict:
    """Drop registry/Hydra selector keys (``_target_``/``BUILD``/``name`` + ``drop``)
    before splatting a config into a class constructor."""
    ignore = {"_target_", "BUILD", "name", *drop}
    return {k: v for k, v in cfg.items() if k not in ignore}


def register_class(role: str, name: str, cls: type) -> None:
    """Register ``cls`` under ``(role, name)`` with a factory that Hydra-instantiates it.

    Use for components whose ``__init__`` takes unpacked config kwargs (preprocessors,
    evaluators, inferencer) rather than the sub-config object.
    """
    REGISTRY.register(role, name)(lambda cfg=None, **ov: instantiate_as(cls, cfg, **ov))


def registers_as(role: str, name: str):
    """Class-decorator form of :func:`register_class`; returns the class UNCHANGED.

    Keeps the registry name next to the class it names::

        @registers_as("preprocessor", "sam2_video")
        class SAM2VideoPreprocessor(BaseFinetunePreprocessor):
            ...

    Registration happens at class-definition time, so it fires exactly when the module
    is imported -- same reachability as a bottom-of-file registration table.

    For components whose ``__init__`` takes unpacked config kwargs. Use
    :func:`registers_flat_as` when the config node must stay pristine.
    """
    def _decorator(cls: type) -> type:
        register_class(role, name, cls)
        return cls
    return _decorator


def register_flat_class(role: str, name: str, cls: type) -> None:
    """Register ``cls`` under ``(role, name)`` with a NON-mutating splat factory.

    Unlike ``register_class``/``instantiate_as``, this leaves the config node untouched
    (no ``_target_`` injection, no selector-key popping) and splats the kwargs directly:
    ``cls(**build_kwargs(cfg), **overrides)``. For a flat config (no nested ``_target_``)
    this is behaviourally identical to Hydra ``instantiate`` but keeps the node pristine,
    so it can be re-read later by ``name`` — e.g. the wd-scheduler cross-check that scans
    ``config.hooks.hooks_list`` for ``name == "weight_decay_schedule"`` after the hooks
    were already built. Use for hooks and event writers (flat, class-is-the-factory).
    """
    def _factory(cfg: Any = None, **overrides: Any) -> Any:
        if cfg is None:
            return cls(**overrides)
        return cls(**build_kwargs(cfg), **overrides)
    REGISTRY.register(role, name)(_factory)


def registers_flat_as(role: str, name: str):
    """Class-decorator form of :func:`register_flat_class`; returns the class UNCHANGED.

    Non-mutating splat: the config node keeps no injected ``_target_``, so it can be
    re-read later by ``name`` -- the wd-scheduler cross-check rescans
    ``config.hooks.hooks_list`` after the hooks were already built. Use for hooks and
    event writers.
    """
    def _decorator(cls: type) -> type:
        register_flat_class(role, name, cls)
        return cls
    return _decorator
