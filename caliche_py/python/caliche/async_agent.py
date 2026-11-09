"""
Async wrapper for Caliche

Provides async/await support for Python 3.7+ using asyncio.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Dict, Optional, Tuple

from caliche import Agent, AgentBuilder, ProviderType


class AsyncAgent:
    """Async wrapper around Agent for use with asyncio"""

    def __init__(self, agent: Agent):
        self._agent = agent

    async def chat(self, message: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Chat with the agent asynchronously. Returns (response_text, usage_dict, metrics_dict)."""
        if hasattr(self._agent, "chat_async"):
            return await self._agent.chat_async(message)
        try:
            return await asyncio.to_thread(self._agent.chat, message)
        except AttributeError:
            # Fallback for Python < 3.9
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._agent.chat, message)

    def stream_chat(self, message: str) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat responses asynchronously without blocking the event loop."""
        stream = self._agent.stream_chat(message)
        return _StreamIterator(stream)

    async def clone_for_request(self) -> AsyncAgent:
        """Create an isolated request-scoped agent for concurrent work."""
        try:
            agent = await asyncio.to_thread(self._agent.clone_for_request)
        except AttributeError:
            loop = asyncio.get_running_loop()
            agent = await loop.run_in_executor(None, self._agent.clone_for_request)
        return AsyncAgent(agent)

    @property
    def provider(self) -> str:
        """Get the provider name"""
        return self._agent.provider()

    @property
    def model(self) -> str:
        """Get the model name"""
        return self._agent.model()


class _StreamIterator:
    """Async iterator wrapper for streaming"""

    def __init__(self, stream):
        self._stream = stream

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            # Use asyncio.to_thread for Python 3.9+, fallback to executor for older versions
            try:
                item = await asyncio.to_thread(next, self._stream, None)
            except AttributeError:
                # Fallback for Python < 3.9
                loop = asyncio.get_running_loop()
                item = await loop.run_in_executor(None, next, self._stream, None)

            if item is None:
                raise StopAsyncIteration
            return item
        except StopIteration:
            raise StopAsyncIteration


async def create_agent_async(
    provider: Optional[ProviderType] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[list] = None,
) -> AsyncAgent:
    """Create an agent asynchronously"""
    builder = AgentBuilder()
    if provider:
        builder = builder.provider(provider)
    if model:
        builder = builder.model(model)
    if system_prompt:
        builder = builder.system_prompt(system_prompt)
    if tools:
        for tool in tools:
            builder = builder.enable_tool(tool)

    # Use asyncio.to_thread for Python 3.9+, fallback to executor for older versions
    try:
        agent = await asyncio.to_thread(builder.build)
    except AttributeError:
        # Fallback for Python < 3.9
        loop = asyncio.get_running_loop()
        agent = await loop.run_in_executor(None, builder.build)

    return AsyncAgent(agent)
