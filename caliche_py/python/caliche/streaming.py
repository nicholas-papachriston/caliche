"""Stream collection helpers for Caliche Python SDK."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any


def collect_text(events: Iterable[Any]) -> str:
    """Concatenate text_chunk / content payloads from a stream of events."""
    parts: list[str] = []
    for event in events:
        text = _event_text(event)
        if text:
            parts.append(text)
    return "".join(parts)


def collect_with_usage(events: Iterable[Any]) -> dict[str, Any]:
    """Collect text plus the last usage/finish_reason seen in the stream."""
    parts: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    for event in events:
        text = _event_text(event)
        if text:
            parts.append(text)
        event_usage = _event_field(event, "usage")
        if isinstance(event_usage, dict):
            usage = event_usage
        reason = _event_field(event, "finish_reason")
        if isinstance(reason, str):
            finish_reason = reason
    return {
        "text": "".join(parts),
        "usage": usage,
        "finish_reason": finish_reason,
    }


def chat_text(agent: Any, message: str) -> str:
    """Call agent.chat and return only the response text."""
    result = agent.chat(message)
    if isinstance(result, tuple) and result:
        first = result[0]
        return first if isinstance(first, str) else str(first)
    if isinstance(result, dict):
        response = result.get("response")
        if isinstance(response, str):
            return response
    if isinstance(result, str):
        return result
    return str(result)


def iter_stream_text(events: Iterable[Any]) -> Iterator[str]:
    """Yield text fragments from a stream of events."""
    for event in events:
        text = _event_text(event)
        if text:
            yield text


def _unwrap_event(event: Any) -> Any:
    """Prefer versioned envelope / nested event payloads when present."""
    if not isinstance(event, dict):
        return event
    nested = event.get("event")
    if isinstance(nested, dict):
        event = nested
    if (
        isinstance(event, dict)
        and "payload" in event
        and "version" in event
        and isinstance(event.get("payload"), dict)
    ):
        return event["payload"]
    return event


def _event_field(event: Any, name: str) -> Any:
    event = _unwrap_event(event)
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def _event_text(event: Any) -> str | None:
    if isinstance(event, str):
        return event
    event = _unwrap_event(event)
    event_type = _event_field(event, "type")
    content = _event_field(event, "content")
    if event_type in {None, "text_chunk", "text", "content"} and isinstance(content, str):
        return content
    if isinstance(event, dict) and isinstance(event.get("text"), str):
        return event["text"]
    return None


__all__ = [
    "chat_text",
    "collect_text",
    "collect_with_usage",
    "iter_stream_text",
]
