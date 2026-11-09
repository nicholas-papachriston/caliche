"""
Prompt template storage for managing saved prompts.
"""

from __future__ import annotations

from typing import Any, List

# Avoid shadowing the method name in return type
ListStr = List[str]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from caliche_runtime import _caliche_py as _runtime
else:
    try:
        from caliche._runtime import load_runtime

        _runtime = load_runtime()
    except ImportError as exc:
        raise ImportError(
            "Prompt storage requires 's3' feature. Please rebuild with: pip install -e ."
        ) from exc

PyPromptTemplateStore = _runtime.PyPromptTemplateStore
PyStorageConfig = _runtime.PyStorageConfig

from caliche.templates import StorageConfig


class PromptTemplateStore:
    """Prompt template store for managing saved prompts."""

    def __init__(self, store: Any) -> None:
        self._store = store

    @classmethod
    def create(cls, config: StorageConfig) -> PromptTemplateStore:
        """Create a prompt template store with storage configuration."""
        store = PyPromptTemplateStore.new(config._config)
        return cls(store)

    def save(self, name: str, template: str) -> None:
        """Save a prompt template."""
        self._store.save(name, template)

    def load(self, name: str) -> str:
        """Load a prompt template."""
        return self._store.load(name)

    def list(self) -> ListStr:
        """List all prompt template names."""
        return self._store.list()

    def delete(self, name: str) -> None:
        """Delete a prompt template."""
        self._store.delete(name)

    def exists(self, name: str) -> bool:
        """Check if a prompt template exists."""
        return self._store.exists(name)
