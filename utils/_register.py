"""Populate ``REGISTRY`` by walk-importing every component module.

The registry (``utils/registry.py``) imports nothing from component packages, so a
``@REGISTRY.register`` decorator only fires once its module is imported. This module
walk-imports every submodule under the component package roots so all decorators run
— no hand-maintained list. Adding a new component file is enough; it is discovered
automatically.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)

# Package roots that contain @register-decorated components. Adding a brand-new
# top-level package of components is the only edit this file ever needs.
_ROOTS = ("models", "evaluation", "inference", "data", "training")

_DONE = False


def register_all() -> None:
    """Import every submodule under the component roots so decorators execute.

    Idempotent (``import_module`` is cached; the ``_DONE`` guard skips re-walking).
    """
    global _DONE
    if _DONE:
        return
    for root in _ROOTS:
        pkg = importlib.import_module(f"cell_observatory_platform.{root}")
        for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            importlib.import_module(mod.name)
    _DONE = True


register_all()
