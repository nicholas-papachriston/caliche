#!/usr/bin/env python3
"""Generate packages/runtime/python/caliche_runtime/_caliche_py.pyi via introspection."""

from __future__ import annotations

import inspect
from pathlib import Path

from caliche_runtime import _caliche_py as m

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "packages/runtime/python/caliche_runtime/_caliche_py.pyi"
SKIP = {
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "__version__",
}

# Prefer concrete return types for high-traffic APIs used by typed wrappers.
RETURN_OVERRIDES: dict[tuple[str, str], str] = {
    ("Agent", "chat"): "tuple[str, dict[str, Any], dict[str, Any]]",
    ("Agent", "chat_async"): "Any",
    ("Agent", "chat_structured"): "tuple[str, dict[str, Any]]",
    ("Agent", "stream_chat"): "Any",
    ("Agent", "stream_chat_async"): "Any",
    ("Agent", "list_tools"): "list[dict[str, Any]]",
    ("Agent", "model"): "str",
    ("Agent", "provider"): "str",
    ("StreamIterator", "__next__"): "dict[str, Any] | None",
    ("ToolResult", "__init__"): "None",
    ("ToolResult", "success"): "bool",
    ("ToolResult", "content"): "str",
    ("ToolResult", "error"): "str | None",
}


def fmt_annotation(ann: object) -> str:
    if ann is inspect.Parameter.empty:
        return "Any"
    if isinstance(ann, type):
        return ann.__name__ if ann.__module__ == "builtins" else "Any"
    text = str(ann)
    if text.startswith("<"):
        return "Any"
    return text.replace("caliche_runtime._caliche_py.", "")


def sig_for(obj: object, *, class_name: str | None = None, method_name: str | None = None) -> str:
    # PyO3 signatures from inspect are often incomplete; prefer open kwargs so
    # typed wrappers are not rejected for arity mismatches.
    if method_name == "__init__":
        return "(self, *args: Any, **kwargs: Any) -> None"
    if class_name and method_name and (class_name, method_name) in RETURN_OVERRIDES:
        ret = RETURN_OVERRIDES[(class_name, method_name)]
        if method_name in {"provider", "model"}:
            return f"(self) -> {ret}"
        if method_name in {"chat", "chat_async", "stream_chat", "stream_chat_async"}:
            return f"(self, message: Any) -> {ret}"
        if method_name == "chat_structured":
            return f"(self, message: Any, schema: Any) -> {ret}"
        if method_name == "list_tools":
            return f"(self) -> {ret}"
        if method_name == "__next__":
            return f"(self) -> {ret}"
        return f"(self, *args: Any, **kwargs: Any) -> {ret}"
    if method_name in {"__str__", "__repr__"}:
        return "(self) -> str"
    try:
        sig = inspect.signature(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "(self, *args: Any, **kwargs: Any) -> Any" if method_name else "(*args: Any, **kwargs: Any) -> Any"
    has_self = any(name == "self" for name in sig.parameters)
    prefix = "self, " if has_self or (method_name and method_name not in {"__new__"}) else ""
    if method_name and not method_name.startswith("__") and (class_name, method_name) not in RETURN_OVERRIDES:
        return f"({prefix}*args: Any, **kwargs: Any) -> Any"
    ret = fmt_annotation(sig.return_annotation)
    return f"({prefix}*args: Any, **kwargs: Any) -> {ret}"


def main() -> None:
    lines = [
        "# Auto-generated stubs for caliche_runtime._caliche_py",
        "# Regenerate: uv run python scripts/generate_native_stubs.py",
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        "__version__: str",
        "",
    ]
    names = sorted(n for n in dir(m) if n not in SKIP and not n.startswith("__"))
    for name in names:
        obj = getattr(m, name)
        if not inspect.isclass(obj):
            continue
        lines.append(f"class {name}:")
        methods: list[str] = []
        for attr_name, attr in inspect.getmembers(obj):
            if attr_name.startswith("_") and attr_name not in {
                "__init__",
                "__iter__",
                "__next__",
                "__enter__",
                "__exit__",
                "__call__",
                "__str__",
                "__repr__",
            }:
                continue
            if attr_name in {
                "__class__",
                "__init_subclass__",
                "__subclasshook__",
                "__dir__",
                "__format__",
                "__getattribute__",
                "__getstate__",
                "__reduce__",
                "__reduce_ex__",
                "__sizeof__",
                "__eq__",
                "__hash__",
                "__ne__",
            }:
                continue
            unbound = obj.__dict__.get(attr_name, attr)
            if isinstance(unbound, property):
                ret = RETURN_OVERRIDES.get((name, attr_name), "Any")
                methods.append(f"    @property\n    def {attr_name}(self) -> {ret}: ...")
                continue
            if not callable(attr):
                continue
            if isinstance(unbound, staticmethod):
                s = sig_for(unbound.__func__, class_name=name, method_name=attr_name)
                if s.startswith("(self"):
                    s = "(" + s[6:].lstrip(", ")
                methods.append(f"    @staticmethod\n    def {attr_name}{s}: ...")
            elif isinstance(unbound, classmethod):
                methods.append(
                    f"    @classmethod\n    def {attr_name}{sig_for(unbound.__func__, class_name=name, method_name=attr_name)}: ..."
                )
            else:
                methods.append(
                    f"    def {attr_name}{sig_for(attr, class_name=name, method_name=attr_name)}: ..."
                )
        # Explicit ToolResult attribute surface used by SDK/tests.
        if name == "ToolResult":
            methods = [
                "    def __init__(self, success: bool, content: str, error: str | None = ...) -> None: ...",
                "    @property",
                "    def success(self) -> bool: ...",
                "    @property",
                "    def content(self) -> str: ...",
                "    @property",
                "    def error(self) -> str | None: ...",
                "    def __repr__(self) -> str: ...",
                "    def __str__(self) -> str: ...",
            ]
        methods.append("    def __getattr__(self, name: str) -> Any: ...")
        lines.extend(methods)
        lines.append("")

    for name in names:
        obj = getattr(m, name)
        if inspect.isclass(obj) or not callable(obj):
            continue
        lines.append(f"def {name}{sig_for(obj)}: ...")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
