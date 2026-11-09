"""
Runtime binding loader for Caliche Python modules.

Native ownership is `caliche_runtime._caliche_py` only. This loader
aliases that single module under the historical import name so callers that
still reference `caliche._caliche_py` share one in-process extension
(dual dlopen of two `.so` copies SIGBUS on macOS).
"""

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType

_CANONICAL = "caliche_runtime._caliche_py"
_ALIAS = "caliche._caliche_py"


def _alias_native(module: ModuleType) -> ModuleType:
    """Point both import paths at the same already-loaded extension module."""
    sys.modules[_CANONICAL] = module
    sys.modules[_ALIAS] = module
    return module


def load_runtime() -> ModuleType:
    """Load the native runtime extension from the runtime package."""
    for name in (_CANONICAL, _ALIAS):
        existing = sys.modules.get(name)
        if existing is not None:
            return _alias_native(existing)

    module = import_module(_CANONICAL)
    return _alias_native(module)
