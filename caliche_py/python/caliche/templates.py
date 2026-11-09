"""
Template registry for managing agent templates with storage support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

# Avoid shadowing the method name in return type
ListAgentTemplate = List["AgentTemplate"]

if TYPE_CHECKING:
    from caliche import AgentBuilder
    from caliche_runtime import _caliche_py as _runtime
else:
    try:
        from caliche._runtime import load_runtime

        _runtime = load_runtime()
    except ImportError as exc:
        raise ImportError(
            "Template storage requires 's3' feature. Please rebuild with: pip install -e ."
        ) from exc

PyAgentTemplate = _runtime.PyAgentTemplate
PyStorageConfig = _runtime.PyStorageConfig
PyTemplateRegistry = _runtime.PyTemplateRegistry


class TemplateRegistry:
    """Template registry for managing agent templates."""

    def __init__(self, registry: Any | None = None) -> None:
        if registry is None:
            self._registry = PyTemplateRegistry()
        else:
            self._registry = registry

    @classmethod
    def create(cls) -> TemplateRegistry:
        """Create a new template registry."""
        return cls()

    @classmethod
    def with_storage(cls, config: StorageConfig) -> TemplateRegistry:
        """Create a template registry with storage configuration."""
        registry = PyTemplateRegistry.with_storage(config._config)
        return cls(registry)

    def add_template_dir(self, dir: str) -> None:
        """Add a template directory to scan."""
        self._registry.add_template_dir(dir)

    def load_templates(self) -> int:
        """Load templates from configured sources."""
        return self._registry.load_templates()

    def get(self, id: str) -> AgentTemplate | None:
        """Get a template by ID."""
        template = self._registry.get(id)
        if template is None:
            return None
        return AgentTemplate(template)

    def list(self) -> ListAgentTemplate:
        """List all templates."""
        templates = self._registry.list()
        return [AgentTemplate(t) for t in templates]


class AgentTemplate:
    """Agent template wrapper."""

    def __init__(self, template: Any) -> None:
        self._template = template

    @property
    def name(self) -> str:
        """Get template name."""
        return self._template.name()

    @property
    def version(self) -> str:
        """Get template version."""
        return self._template.version()

    @property
    def description(self) -> str:
        """Get template description."""
        return self._template.description()

    def instantiate(self) -> AgentBuilder:
        """Instantiate an agent from this template."""
        from caliche import AgentBuilder

        builder = self._template.instantiate()
        return AgentBuilder(builder)


class StorageConfig:
    """Storage configuration for templates and prompts."""

    def __init__(self, config: Any) -> None:
        self._config = config

    @classmethod
    def filesystem(cls, base_path: str) -> StorageConfig:
        """Create a filesystem storage configuration."""
        config = PyStorageConfig(base_path)
        return cls(config)

    @classmethod
    def aws_s3(
        cls,
        bucket: str,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
    ) -> StorageConfig:
        """Create an AWS S3 storage configuration."""
        config = PyStorageConfig.aws_s3(
            bucket, region, access_key_id, secret_access_key, session_token
        )
        return cls(config)

    @classmethod
    def cloudflare_r2(
        cls,
        bucket: str,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        endpoint: str | None = None,
    ) -> StorageConfig:
        """Create a Cloudflare R2 storage configuration."""
        config = PyStorageConfig.cloudflare_r2(
            bucket, account_id, access_key_id, secret_access_key, endpoint
        )
        return cls(config)

    def with_cache(self, enabled: bool) -> StorageConfig:
        """Enable or disable caching."""
        config = self._config.with_cache(enabled)
        return StorageConfig(config)

    def with_cache_ttl(self, ttl_seconds: int) -> StorageConfig:
        """Set cache TTL in seconds."""
        config = self._config.with_cache_ttl(ttl_seconds)
        return StorageConfig(config)

    def with_cache_max_size(self, max_size_mb: int) -> StorageConfig:
        """Set cache max size in MB."""
        config = self._config.with_cache_max_size(max_size_mb)
        return StorageConfig(config)
