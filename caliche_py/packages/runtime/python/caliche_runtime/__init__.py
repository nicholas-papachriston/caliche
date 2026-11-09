"""
Composable runtime package for Caliche.
"""

from __future__ import annotations

import sys

try:
    from ._caliche_py import *  # noqa: F403
except ImportError as exc:
    raise ImportError(
        "caliche-runtime native extension not found. "
        "Build/install with: pip install caliche-runtime"
    ) from exc

# Prevent a second dlopen of the root package copy of the same extension.
# Dual loads of caliche_runtime._caliche_py and
# caliche._caliche_py SIGBUS when native types are used.
_native = sys.modules.get("caliche_runtime._caliche_py")
if _native is not None:
    sys.modules.setdefault("caliche._caliche_py", _native)
