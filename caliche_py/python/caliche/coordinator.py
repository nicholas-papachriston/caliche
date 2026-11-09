"""
Agent Coordinator Wrapper

Provides Python-friendly wrappers for multi-agent coordination features.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from caliche_runtime import _caliche_py as _runtime
else:
    from caliche._runtime import load_runtime

    _runtime = load_runtime()

Agent = _runtime.Agent
PyAgentCoordinator = _runtime.PyAgentCoordinator
PyMessageBus = _runtime.PyMessageBus


class AgentCoordinator:
    """Agent coordinator for managing multiple agents."""

    def __init__(self, strategy: Optional[str] = None):
        """Create a new agent coordinator.

        Args:
            strategy: Load balancing strategy ("round_robin", "least_connections", "weighted")
        """
        self._coordinator = PyAgentCoordinator(strategy)

    @classmethod
    def with_strategy(cls, strategy: str) -> AgentCoordinator:
        """Create a coordinator with a specific load balancing strategy."""
        return cls(strategy)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return False

    def register_agent(
        self,
        name: str,
        role: str,
        capabilities: List[str],
        agent: Agent,
    ) -> str:
        """Register an agent with the coordinator.

        Returns:
            The agent ID assigned to the registered agent.
        """
        return self._coordinator.register_agent(name, role, capabilities, agent)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        self._coordinator.unregister_agent(agent_id)

    def delegate_task(self, message: str, capability: Optional[str] = None) -> Dict[str, Any]:
        """Delegate a task to the best available agent (synchronous wrapper).

        Returns:
            Dictionary with 'agent_id', 'response', and 'usage' keys.
        """
        try:
            asyncio.get_running_loop()
            # If we're in an async context, we can't use asyncio.run()
            raise RuntimeError(
                "Cannot call delegate_task from async context. Use delegate_task_async() instead."
            )
        except RuntimeError as e:
            if "Cannot call delegate_task" in str(e):
                # Re-raise our custom error
                raise
            # No running loop, safe to use asyncio.run()
            return asyncio.run(self._delegate_task_impl(message, capability))

    async def delegate_task_async(
        self, message: str, capability: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delegate a task to the best available agent (async version).

        Returns:
            Dictionary with 'agent_id', 'response', and 'usage' keys.
        """
        return await self._delegate_task_impl(message, capability)

    async def _delegate_task_impl(
        self, message: str, capability: Optional[str] = None
    ) -> Dict[str, Any]:
        """Internal implementation of delegate_task."""
        loop = asyncio.get_running_loop()
        return await self._coordinator.delegate_task(loop, message, capability)

    def execute_on_agent(self, agent_id: str, message: str) -> Dict[str, Any]:
        """Execute a task on a specific agent (synchronous wrapper).

        Returns:
            Dictionary with 'response' and 'usage' keys.
        """
        try:
            asyncio.get_running_loop()
            # If we're in an async context, we can't use asyncio.run()
            raise RuntimeError(
                "Cannot call execute_on_agent from async context. Use execute_on_agent_async() instead."
            )
        except RuntimeError as e:
            if "Cannot call execute_on_agent" in str(e):
                # Re-raise our custom error
                raise
            # No running loop, safe to use asyncio.run()
            return asyncio.run(self._execute_on_agent_impl(agent_id, message))

    async def execute_on_agent_async(self, agent_id: str, message: str) -> Dict[str, Any]:
        """Execute a task on a specific agent (async version).

        Returns:
            Dictionary with 'response' and 'usage' keys.
        """
        return await self._execute_on_agent_impl(agent_id, message)

    async def _execute_on_agent_impl(self, agent_id: str, message: str) -> Dict[str, Any]:
        """Internal implementation of execute_on_agent."""
        loop = asyncio.get_running_loop()
        return await self._coordinator.execute_on_agent(loop, agent_id, message)

    def send_message(self, from_agent_id: str, to_agent_id: str, content: str) -> None:
        """Send a message from one agent to another."""
        self._coordinator.send_message(from_agent_id, to_agent_id, content)

    def broadcast(self, from_agent_id: str, content: str) -> None:
        """Broadcast a message to all agents."""
        self._coordinator.broadcast(from_agent_id, content)

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics (synchronous wrapper)."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call get_stats from async context. Use get_stats_async() instead."
            )
        except RuntimeError as e:
            if "Cannot call get_stats" in str(e):
                # Re-raise our custom error
                raise
            return asyncio.run(self._get_stats_impl())

    async def get_stats_async(self) -> Dict[str, Any]:
        """Get coordinator statistics (async version)."""
        return await self._get_stats_impl()

    async def _get_stats_impl(self) -> Dict[str, Any]:
        """Internal implementation of get_stats."""
        loop = asyncio.get_running_loop()
        return await self._coordinator.get_stats(loop)

    def list_agent_ids(self) -> List[str]:
        """List all registered agent IDs."""
        return self._coordinator.list_agent_ids()

    def agent_count(self) -> int:
        """Get the number of registered agents."""
        return self._coordinator.agent_count()

    def get_message_bus(self) -> MessageBus:
        """Get the message bus for inter-agent communication."""
        bus = self._coordinator.get_message_bus()
        return MessageBus(bus)


class MessageBus:
    """Message bus for inter-agent communication."""

    def __init__(self, bus: PyMessageBus):
        self._bus = bus

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics (synchronous wrapper)."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call get_stats from async context. Use get_stats_async() instead."
            )
        except RuntimeError as e:
            if "Cannot call get_stats" in str(e):
                # Re-raise our custom error
                raise
            return asyncio.run(self._get_stats_impl())

    async def get_stats_async(self) -> Dict[str, Any]:
        """Get message bus statistics (async version)."""
        return await self._get_stats_impl()

    async def _get_stats_impl(self) -> Dict[str, Any]:
        """Internal implementation of get_stats."""
        loop = asyncio.get_running_loop()
        return await self._bus.get_stats(loop)

    def queue_size(self) -> int:
        """Get current queue size (synchronous wrapper)."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call queue_size from async context. Use queue_size_async() instead."
            )
        except RuntimeError as e:
            if "Cannot call queue_size" in str(e):
                # Re-raise our custom error
                raise
            return asyncio.run(self._queue_size_impl())

    async def queue_size_async(self) -> int:
        """Get current queue size (async version)."""
        return await self._queue_size_impl()

    async def _queue_size_impl(self) -> int:
        """Internal implementation of queue_size."""
        return await self._bus.queue_size()
