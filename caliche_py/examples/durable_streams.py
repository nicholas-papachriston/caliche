#!/usr/bin/env python3
"""
Durable Streams Example

Shows how to configure an agent with a durable stream backend so that events
can be persisted for replay and resume (e.g. when using the API or a harness
with SSE and Last-Event-ID).

Run with:
  uv run examples/durable_streams.py

For Redis backend, use DurableStreamBackend.redis("redis://localhost:6379/").
"""

import caliche


def in_memory_backend():
    """Use an in-memory durable stream (ring buffer, last N events per stream)."""
    print("=== In-memory durable stream (capacity 10_000) ===\n")

    backend = caliche.DurableStreamBackend.memory(10_000)
    builder = (
        caliche.AgentBuilder()
        .provider(caliche.ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .durable_stream_backend(backend)
    )
    agent = builder.build()
    print("Agent built with in-memory durable stream.\n")

    response, usage, _ = agent.chat("Say hello in one short sentence.")
    print(f"Response: {response}\n")
    if usage.get("total_tokens"):
        print(f"Tokens: {usage['total_tokens']}\n")


def redis_backend():
    """Use Redis as durable stream backend (requires Redis running)."""
    print("=== Redis durable stream ===\n")

    try:
        backend = caliche.DurableStreamBackend.redis("redis://127.0.0.1:6379/")
    except Exception as e:
        print(f"Redis backend failed (is Redis running?): {e}\n")
        return

    builder = (
        caliche.AgentBuilder()
        .provider(caliche.ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .durable_stream_backend(backend)
    )
    agent = builder.build()
    print("Agent built with Redis durable stream.\n")

    response, usage, _ = agent.chat("Say hello in one short sentence.")
    print(f"Response: {response}\n")
    if usage.get("total_tokens"):
        print(f"Tokens: {usage['total_tokens']}\n")


if __name__ == "__main__":
    in_memory_backend()
    redis_backend()
