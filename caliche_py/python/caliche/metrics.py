"""Metrics conversion helpers for Caliche SDK surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import HarnessDisciplineMetricsDict


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def normalize_harness_discipline_metrics(
    value: Mapping[str, Any] | None,
) -> HarnessDisciplineMetricsDict:
    """Normalize Rust/API-shaped harness discipline metrics into a typed dict."""

    source: Mapping[str, Any] = value or {}
    result: HarnessDisciplineMetricsDict = {
        "tool_calls": _int_or_zero(source.get("tool_calls")),
        "compactions": _int_or_zero(source.get("compactions")),
        "readonly_filter_hits": _int_or_zero(source.get("readonly_filter_hits")),
    }
    for key in ("tokens_per_turn", "tool_calls_per_turn", "estimated_cost_usd"):
        number = _optional_float(source.get(key))
        if number is not None:
            result[key] = number
    summary = source.get("summary")
    if isinstance(summary, str):
        result["summary"] = summary
    return result


__all__ = ["normalize_harness_discipline_metrics"]
