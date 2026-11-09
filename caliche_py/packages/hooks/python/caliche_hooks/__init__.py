"""
Composable hooks/session package for Caliche.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from caliche_core import SharedState, ToolResult
    from caliche_runtime import _caliche_py as _runtime
else:
    import caliche_runtime as _runtime
    from caliche_core import SharedState

PyHookContext = _runtime.PyHookContext
PyHookMatcher = _runtime.PyHookMatcher
PyHookResult = _runtime.PyHookResult
PyPermissionMode = _runtime.PyPermissionMode
PyAgentSession = _runtime.PyAgentSession
PyMutationStyle = _runtime.MutationStyle
PyInspectionAction = _runtime.InspectionAction
PyInspectionResult = _runtime.InspectionResult
PathSecurityInspector = _runtime.PathSecurityInspector
PermissionModeInspector = _runtime.PermissionModeInspector
ToolInspectionManager = _runtime.ToolInspectionManager
InspectionHookAdapter = _runtime.InspectionHookAdapter
default_inspection_manager = _runtime.default_inspection_manager
JsonlSessionHeader = _runtime.JsonlSessionHeader
JsonlSessionEntry = _runtime.JsonlSessionEntry
JsonlSessionTree = _runtime.JsonlSessionTree
JsonlSessionStore = _runtime.JsonlSessionStore
parse_jsonl_session = _runtime.parse_jsonl_session
new_entry_id = _runtime.new_entry_id
ToolRegistry = _runtime.ToolRegistry


class PermissionMode:
    """Permission mode for automatic permission handling."""

    Default = PyPermissionMode.default()
    AcceptEdits = PyPermissionMode.accept_edits()
    AcceptAll = PyPermissionMode.accept_all()
    DenyAll = PyPermissionMode.deny_all()
    Interactive = PyPermissionMode.interactive()

    @staticmethod
    def from_string(mode: str) -> Any:
        return PyPermissionMode(mode)


class MutationStyle:
    """File mutation style: structured fs write/edit XOR freeform apply_patch."""

    EditWrite = PyMutationStyle.edit_write()
    ApplyPatch = PyMutationStyle.apply_patch()

    @staticmethod
    def from_string(style: str) -> Any:
        return PyMutationStyle(style)


class HookMatcher:
    """Matcher for targeting specific tools or all tools."""

    @staticmethod
    def tool(tool_name: str) -> PyHookMatcher:
        return PyHookMatcher.tool(tool_name)

    @staticmethod
    def all() -> PyHookMatcher:
        return PyHookMatcher.all()


@dataclass
class HookContext:
    """Context passed to hooks during tool execution."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str

    @classmethod
    def from_py(cls, py_ctx: PyHookContext) -> HookContext:
        return cls(
            tool_name=py_ctx.tool_name,
            tool_input=py_ctx.tool_input,
            tool_use_id=py_ctx.tool_use_id,
        )


@dataclass
class HookResult:
    """Result returned by lifecycle hooks."""

    permission_decision: dict[str, Any]
    hook_specific_output: Optional[dict[str, Any]] = None

    @staticmethod
    def allow() -> PyHookResult:
        return PyHookResult.allow()

    @staticmethod
    def deny(reason: str) -> PyHookResult:
        return PyHookResult.deny(reason)

    @classmethod
    def from_py(cls, py_result: PyHookResult) -> HookResult:
        return cls(
            permission_decision=py_result.permission_decision,
            hook_specific_output=py_result.hook_specific_output,
        )


@runtime_checkable
class LifecycleHook(Protocol):
    """Protocol for lifecycle hooks that can intercept agent operations."""

    async def on_pre_tool_use(self, ctx: HookContext) -> HookResult: ...

    async def on_post_tool_use(self, ctx: HookContext, result: ToolResult) -> HookResult: ...


class AgentSession:
    """Agent session for conversation state and session forking."""

    def __init__(
        self,
        id: str,
        history: Optional[list[dict[str, Any]]] = None,
        shared_state: Optional[SharedState] = None,
    ) -> None:
        if shared_state is not None:
            self._session = PyAgentSession(id, history=history, shared_state=shared_state._inner)
        elif history is not None:
            self._session = PyAgentSession.with_history(id, history)
        else:
            self._session = PyAgentSession(id)

    @property
    def id(self) -> str:
        return self._session.id

    def fork(self, new_id: str) -> AgentSession:
        fork_session = self._session.fork(new_id)
        result = AgentSession.__new__(AgentSession)
        result._session = fork_session
        return result

    def merge(self, other: AgentSession) -> None:
        self._session.merge(other._session)

    def get_history(self) -> list[dict[str, Any]]:
        return self._session.get_history()

    def add_message(self, message: dict[str, Any]) -> None:
        self._session.add_message(message)

    def clear_history(self) -> None:
        self._session.clear_history()

    @property
    def shared_state(self) -> SharedState:
        py_shared_state = self._session.shared_state()
        result = SharedState.__new__(SharedState)
        object.__setattr__(result, "inner", py_shared_state)
        return result


__all__ = [
    "AgentSession",
    "HookContext",
    "HookMatcher",
    "HookResult",
    "LifecycleHook",
    "PermissionMode",
    "MutationStyle",
    "InspectionAction",
    "InspectionResult",
    "PathSecurityInspector",
    "PermissionModeInspector",
    "ToolInspectionManager",
    "InspectionHookAdapter",
    "default_inspection_manager",
    "JsonlSessionHeader",
    "JsonlSessionEntry",
    "JsonlSessionTree",
    "JsonlSessionStore",
    "parse_jsonl_session",
    "new_entry_id",
    "ToolRegistry",
]

# Friendly aliases matching native class names used in __all__.
InspectionAction = PyInspectionAction
InspectionResult = PyInspectionResult
