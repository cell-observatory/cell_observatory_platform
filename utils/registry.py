"""Component registry: ``(role, name) -> factory``.

Every constructable component (models, backbones, adapters, heads, criterions,
preprocessors, evaluators, inferencer, save/viz handlers, and the training-layer
swap points: hooks, event writers, optimizers, schedulers) registers itself with a
``@REGISTRY.register(role, name)`` decorator. Construction goes through
``REGISTRY.build(role, name, cfg, **overrides)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, Optional

logger = logging.getLogger(__name__)


# Roles a component can be registered under. A duplicate (role, name) is a hard
# boot failure (see register), so collisions can never ship silently.
ROLES: FrozenSet[str] = frozenset({
    "model", "preprocessor", "evaluator", "inferencer",
    "backbone", "adapter", "head", "criterion",
    "viz_handler", "save_handler",
    "hook", "event_writer", "optimizer", "scheduler",
})


@dataclass(frozen=True)
class Entry:
    factory: Callable[..., Any]
    role: str
    name: str


class Registry:
    def __init__(self) -> None:
        self._entries: Dict[tuple[str, str], Entry] = {}

    def register(self, role: str, name: str) -> Callable[[Callable], Callable]:
        """Decorator: register ``factory`` under ``(role, name)``; return it UNCHANGED
        (so decorated ``BUILD`` functions / classes stay directly callable).

        Raises on unknown role or duplicate ``(role, name)`` — no silent last-writer-wins.
        """
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}; expected one of {sorted(ROLES)}.")

        def _decorator(factory: Callable) -> Callable:
            key = (role, name)
            if key in self._entries:
                raise ValueError(
                    f"Duplicate registry entry role={role!r} name={name!r}: "
                    f"{self._entries[key].factory!r} vs {factory!r}."
                )
            self._entries[key] = Entry(factory=factory, role=role, name=name)
            return factory

        return _decorator

    def build(self, role: str, name: str, cfg: Any = None, **overrides: Any) -> Any:
        """Construct the component registered under ``(role, name)``.

        ``cfg`` is the component's sub-config (or the full cfg for ``role='model'``).
        ``overrides`` are forwarded to the factory verbatim (runtime args like
        ``adapter_args=``, ``model=``, ``buffer_manager=``, ``params=``, ``opt=``).
        """
        return self.get(role, name).factory(cfg, **overrides)

    def get(self, role: str, name: str) -> Entry:
        try:
            return self._entries[(role, name)]
        except KeyError:
            raise KeyError(
                f"No component role={role!r} name={name!r}. "
                f"Known {role!r}: {sorted(n for (r, n) in self._entries if r == role)}."
            ) from None

    def has(self, role: str, name: str) -> bool:
        return (role, name) in self._entries

    def names(self, role: str) -> FrozenSet[str]:
        return frozenset(n for (r, n) in self._entries if r == role)

    def entries(self, role: Optional[str] = None) -> Iterable[Entry]:
        for e in self._entries.values():
            if role is None or e.role == role:
                yield e


# Module-global singleton.
REGISTRY = Registry()
