"""TypedDict helpers for Caliche Python SDK surfaces."""

from __future__ import annotations

from typing import TypedDict


class UsageDict(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    embedding_tokens: int


class MetricsDict(TypedDict, total=False):
    total_duration_ms: float
    llm_duration_ms: float
    tool_duration_ms: float
    overhead_ms: float
    step_count: int
    time_to_first_token_ms: float
    steps: list[dict[str, object]]


class HarnessDisciplineMetricsDict(TypedDict, total=False):
    tokens_per_turn: float
    tool_calls: int
    tool_calls_per_turn: float
    compactions: int
    readonly_filter_hits: int
    estimated_cost_usd: float
    summary: str


class StreamEventDict(TypedDict, total=False):
    type: str
    event_id: str | None
    content: str
    finish_reason: str
    usage: UsageDict
    name: str
    id: str


class ChatResultDict(TypedDict, total=False):
    response: str
    usage: UsageDict
    metrics: MetricsDict


class CollectedStreamDict(TypedDict, total=False):
    text: str
    usage: UsageDict
    finish_reason: str


__all__ = [
    "UsageDict",
    "MetricsDict",
    "HarnessDisciplineMetricsDict",
    "StreamEventDict",
    "ChatResultDict",
    "CollectedStreamDict",
]
