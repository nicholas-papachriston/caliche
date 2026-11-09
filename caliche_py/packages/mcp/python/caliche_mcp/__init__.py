"""
Composable MCP (Model Context Protocol) package for Caliche.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from caliche_runtime import _caliche_py as _runtime
else:
    try:
        import caliche_runtime as _runtime
    except Exception as exc:  # pragma: no cover - import-time guard
        raise ImportError(
            "MCP functionality requires caliche-runtime with mcp support. "
            "Install with: pip install caliche-runtime caliche-mcp"
        ) from exc

PyMcpServer = _runtime.PyMcpServer
PyMcpServerBuilder = _runtime.PyMcpServerBuilder
get_latest_protocol_version = _runtime.get_latest_protocol_version
get_supported_protocol_versions = _runtime.get_supported_protocol_versions

try:
    PyInProcessMcpServer = _runtime.PyInProcessMcpServer
    PyInProcessMcpServerBuilder = _runtime.PyInProcessMcpServerBuilder
    _HAS_IN_PROCESS = True
except AttributeError:
    _HAS_IN_PROCESS = False


@runtime_checkable
class McpTool(Protocol):
    """Protocol for MCP tools that can be added to a server."""

    def name(self) -> str: ...

    def description(self) -> str: ...

    def parameters(self) -> dict[str, Any]: ...

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]: ...


class McpServerBuilder:
    """Builder for creating MCP servers."""

    def __init__(self) -> None:
        self._builder = PyMcpServerBuilder()

    def name(self, name: str) -> McpServerBuilder:
        self._builder.name(name)
        return self

    def version(self, version: str) -> McpServerBuilder:
        self._builder.version(version)
        return self

    def enable_tools(self, enabled: bool = True) -> McpServerBuilder:
        self._builder.enable_tools(enabled)
        return self

    def enable_prompts(self, enabled: bool = True) -> McpServerBuilder:
        self._builder.enable_prompts(enabled)
        return self

    def enable_resources(self, enabled: bool = True) -> McpServerBuilder:
        self._builder.enable_resources(enabled)
        return self

    def add_tool(self, tool: McpTool) -> McpServerBuilder:
        self._builder.add_tool(tool)
        return self

    def build(self) -> McpServer:
        return McpServer(self._builder.build())


class McpServer:
    """MCP server instance."""

    def __init__(self, server: PyMcpServer) -> None:
        self._server = server

    def run_stdio(self) -> None:
        self._server.run_stdio()


LATEST_PROTOCOL_VERSION = get_latest_protocol_version()
SUPPORTED_PROTOCOL_VERSIONS = get_supported_protocol_versions()

__all__ = [
    "LATEST_PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "McpServer",
    "McpServerBuilder",
    "McpTool",
]

if _HAS_IN_PROCESS:

    class InProcessMcpServerBuilder:
        """Builder for creating in-process MCP servers."""

        def __init__(self, name: str) -> None:
            self._builder = PyInProcessMcpServerBuilder(name)

        def version(self, version: str) -> InProcessMcpServerBuilder:
            self._builder.version(version)
            return self

        def add_tool(self, tool: McpTool) -> InProcessMcpServerBuilder:
            self._builder.add_tool(tool)
            return self

        def build(self) -> InProcessMcpServer:
            return InProcessMcpServer(self._builder.build())

    class InProcessMcpServer:
        """In-process MCP server instance."""

        def __init__(self, server: PyInProcessMcpServer) -> None:
            self._server = server

        @property
        def name(self) -> str:
            return self._server.name

        @property
        def version(self) -> str:
            return self._server.version

        def list_tools(self) -> list[dict[str, Any]]:
            return self._server.list_tools()

        def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return self._server.call_tool(name, arguments)

    __all__.extend(["InProcessMcpServer", "InProcessMcpServerBuilder"])
