"""
Memory management for Caliche

Provides memory strategies: BufferMemory, SummaryMemory, EntityMemory
and memory persistence backends.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(Enum):
    """Memory strategy types"""

    BUFFER = "buffer"
    SUMMARY = "summary"
    ENTITY = "entity"


class BufferMemoryConfig:
    """Configuration for buffer memory"""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens


class SummaryMemoryConfig:
    """Configuration for summary memory"""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens


class EntityMemoryConfig:
    """Configuration for entity memory"""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens


@dataclass
class MemorySnapshot:
    """
    A snapshot of conversation memory.

    Attributes:
        messages: List of messages in the conversation
        metadata: Additional metadata as a dict
    """

    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class MemoryPersistence(ABC):
    """
    Abstract base class for memory persistence backends.

    To create a custom persistence backend, inherit from this class and implement:
    - save_snapshot(conversation_id, snapshot): Save a memory snapshot
    - load_snapshot(conversation_id): Load a memory snapshot
    - clear_snapshot(conversation_id): Clear a memory snapshot

    All methods can be synchronous or asynchronous (async def).

    Example:
        class FileMemoryPersistence(MemoryPersistence):
            def __init__(self, base_path: str):
                self.base_path = base_path

            async def save_snapshot(self, conversation_id: str, snapshot: MemorySnapshot):
                import aiofiles
                import os
                os.makedirs(self.base_path, exist_ok=True)
                file_path = os.path.join(self.base_path, f"{conversation_id}.json")
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(json.dumps({
                        "messages": snapshot.messages,
                        "metadata": snapshot.metadata
                    }))

            async def load_snapshot(self, conversation_id: str) -> Optional[MemorySnapshot]:
                import aiofiles
                import os
                file_path = os.path.join(self.base_path, f"{conversation_id}.json")
                if os.path.exists(file_path):
                    async with aiofiles.open(file_path, 'r') as f:
                        data = json.loads(await f.read())
                        return MemorySnapshot(
                            messages=data["messages"],
                            metadata=data.get("metadata", {})
                        )
                return None

            async def clear_snapshot(self, conversation_id: str):
                import os
                file_path = os.path.join(self.base_path, f"{conversation_id}.json")
                if os.path.exists(file_path):
                    os.remove(file_path)
    """

    @abstractmethod
    def save_snapshot(self, conversation_id: str, snapshot: MemorySnapshot):
        """
        Save a memory snapshot for the given conversation.

        Args:
            conversation_id: Unique identifier for the conversation
            snapshot: MemorySnapshot containing messages and metadata
        """
        pass

    @abstractmethod
    def load_snapshot(self, conversation_id: str) -> Optional[MemorySnapshot]:
        """
        Load a memory snapshot for the given conversation.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            MemorySnapshot if found, None otherwise
        """
        pass

    @abstractmethod
    def clear_snapshot(self, conversation_id: str):
        """
        Clear a memory snapshot for the given conversation.

        Args:
            conversation_id: Unique identifier for the conversation
        """
        pass


class FileMemoryPersistence(MemoryPersistence):
    """
    Example memory persistence that saves to JSON files.

    This is a simple implementation that stores each conversation
    as a JSON file in a directory.
    """

    def __init__(self, base_path: str):
        """
        Initialize file-based persistence.

        Args:
            base_path: Directory path where conversation files will be stored
        """
        import os

        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_snapshot(self, conversation_id: str, snapshot: MemorySnapshot):
        """Save snapshot to a JSON file."""
        import os

        file_path = os.path.join(self.base_path, f"{conversation_id}.json")
        with open(file_path, "w") as f:
            json.dump({"messages": snapshot.messages, "metadata": snapshot.metadata}, f, indent=2)

    def load_snapshot(self, conversation_id: str) -> Optional[MemorySnapshot]:
        """Load snapshot from a JSON file."""
        import os

        file_path = os.path.join(self.base_path, f"{conversation_id}.json")
        if os.path.exists(file_path):
            with open(file_path) as f:
                data = json.load(f)
                return MemorySnapshot(messages=data["messages"], metadata=data.get("metadata", {}))
        return None

    def clear_snapshot(self, conversation_id: str):
        """Delete the snapshot file."""
        import os

        file_path = os.path.join(self.base_path, f"{conversation_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)

    async def save_snapshot_async(self, conversation_id: str, snapshot: MemorySnapshot):
        """Save snapshot to a JSON file asynchronously using aiofiles."""
        try:
            import os

            import aiofiles
        except ImportError:
            raise ImportError(
                "aiofiles is required for async file operations. Install with: pip install aiofiles"
            )

        os.makedirs(self.base_path, exist_ok=True)
        file_path = os.path.join(self.base_path, f"{conversation_id}.json")
        async with aiofiles.open(file_path, "w") as f:
            await f.write(
                json.dumps({"messages": snapshot.messages, "metadata": snapshot.metadata}, indent=2)
            )

    async def load_snapshot_async(self, conversation_id: str) -> Optional[MemorySnapshot]:
        """Load snapshot from a JSON file asynchronously using aiofiles."""
        try:
            import os

            import aiofiles
        except ImportError:
            raise ImportError(
                "aiofiles is required for async file operations. Install with: pip install aiofiles"
            )

        file_path = os.path.join(self.base_path, f"{conversation_id}.json")
        if os.path.exists(file_path):
            async with aiofiles.open(file_path) as f:
                data = json.loads(await f.read())
                return MemorySnapshot(messages=data["messages"], metadata=data.get("metadata", {}))
        return None

    async def clear_snapshot_async(self, conversation_id: str):
        """Delete the snapshot file asynchronously."""
        import os

        file_path = os.path.join(self.base_path, f"{conversation_id}.json")
        if os.path.exists(file_path):
            # aiofiles doesn't have remove, use os.remove in executor
            import asyncio

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, os.remove, file_path)


class DictMemoryPersistence(MemoryPersistence):
    """
    Example memory persistence that stores snapshots in memory (dict).

    Useful for testing or short-lived sessions. Data is lost when the process exits.
    """

    def __init__(self):
        """Initialize in-memory persistence."""
        self._storage: Dict[str, MemorySnapshot] = {}

    def save_snapshot(self, conversation_id: str, snapshot: MemorySnapshot):
        """Save snapshot to in-memory dict."""
        self._storage[conversation_id] = snapshot

    def load_snapshot(self, conversation_id: str) -> Optional[MemorySnapshot]:
        """Load snapshot from in-memory dict."""
        return self._storage.get(conversation_id)

    def clear_snapshot(self, conversation_id: str):
        """Remove snapshot from in-memory dict."""
        self._storage.pop(conversation_id, None)


__all__ = [
    "MemoryType",
    "BufferMemoryConfig",
    "SummaryMemoryConfig",
    "EntityMemoryConfig",
    "MemoryPersistence",
    "MemorySnapshot",
    "FileMemoryPersistence",
    "DictMemoryPersistence",
]
