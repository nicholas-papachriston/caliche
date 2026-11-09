"""Lean bindings omit RagService; imports must fail cleanly."""

from __future__ import annotations

import pytest

import caliche


def test_rag_service_absent_from_lean_runtime() -> None:
    runtime = caliche._runtime  # noqa: SLF001 - intentional feature probe
    if hasattr(runtime, "PyRagService"):
        pytest.skip("native extension built with tools-rag")
    assert not hasattr(caliche, "RagService")
    with pytest.raises((ImportError, AttributeError)) as exc_info:
        from caliche import RagService  # noqa: F401
    message = str(exc_info.value)
    assert "tools-rag" in message or "RagService" in message
