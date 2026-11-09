"""
Caliche Python SDK.

Bindings for the Caliche Rust runtime: agents, tools, providers, and
orchestration helpers for Python applications.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, Any

from ._runtime import load_runtime

# Use the stubbed native module under TYPE_CHECKING so re-exports are valid types.
if TYPE_CHECKING:
    from caliche_runtime import _caliche_py as _runtime
else:
    try:
        _runtime = load_runtime()
    except ImportError as exc:
        raise ImportError(
            "_caliche_py native extension not found. "
            "Please build the package with: pip install -e ."
        ) from exc

Agent = _runtime.Agent
AgentBuilder = _runtime.AgentBuilder
DurableStreamBackend = _runtime.DurableStreamBackend
AgentFlow = _runtime.AgentFlow
FlowBuilder = _runtime.FlowBuilder
LlmClient = _runtime.LlmClient
LlmClientBuilder = _runtime.LlmClientBuilder
ProviderType = _runtime.ProviderType
SharedState = _runtime.SharedState
ToolResult = _runtime.ToolResult


# Prefer native runtime symbols for mutation/inspection/session APIs (avoid import cycles).
class MutationStyle:
    """File mutation style: structured fs write/edit XOR freeform apply_patch."""

    EditWrite = _runtime.MutationStyle.edit_write()
    ApplyPatch = _runtime.MutationStyle.apply_patch()

    @staticmethod
    def from_string(style: str):
        return _runtime.MutationStyle(style)


InspectionAction = _runtime.InspectionAction
InspectionResult = _runtime.InspectionResult
PathSecurityInspector = _runtime.PathSecurityInspector
PermissionModeInspector = _runtime.PermissionModeInspector
ToolInspectionManager = _runtime.ToolInspectionManager
InspectionHookAdapter = _runtime.InspectionHookAdapter
default_inspection_manager = _runtime.default_inspection_manager
ToolRegistry = _runtime.ToolRegistry
JsonlSessionHeader = _runtime.JsonlSessionHeader
JsonlSessionEntry = _runtime.JsonlSessionEntry
JsonlSessionTree = _runtime.JsonlSessionTree
JsonlSessionStore = _runtime.JsonlSessionStore
parse_jsonl_session = _runtime.parse_jsonl_session
new_entry_id = _runtime.new_entry_id
create_calculator_tool = _runtime.create_calculator_tool
create_fs_tool = _runtime.create_fs_tool
create_apply_patch_tool = _runtime.create_apply_patch_tool
create_http_tool = _runtime.create_http_tool
create_code_execution_tool = _runtime.create_code_execution_tool

# Local inference fit probes (present when the extension is built with local-inference).
probe_local_inference_host = getattr(_runtime, "probe_local_inference_host", None)
fit_local_inference_catalog = getattr(_runtime, "fit_local_inference_catalog", None)
fit_local_inference_model = getattr(_runtime, "fit_local_inference_model", None)
safe_local_inference_models = getattr(_runtime, "safe_local_inference_models", None)

# Public names. The native extension keeps its own class names.
LLMChain = _runtime.PyLLMChain
SequentialChain = _runtime.PySequentialChain
MapReduceChain = _runtime.PyMapReduceChain
ParallelChain = _runtime.PyParallelChain
ChainBuilder = _runtime.PyChainBuilder
ReActPattern = _runtime.PyReActPattern
PlanExecutePattern = _runtime.PyPlanExecutePattern
ReflexionPattern = _runtime.PyReflexionPattern
SelfAskPattern = _runtime.PySelfAskPattern
TreeOfThoughtsPattern = _runtime.PyTreeOfThoughtsPattern
DebatePattern = _runtime.PyDebatePattern
SupervisorPattern = _runtime.PySupervisorPattern
HierarchicalPattern = _runtime.PyHierarchicalPattern
HierarchicalPatternBuilder = _runtime.PyHierarchicalPatternBuilder
PipelinePattern = _runtime.PyPipelinePattern
PipelinePatternBuilder = _runtime.PyPipelinePatternBuilder
Document = _runtime.PyDocument
TextSplitter = _runtime.PyTextSplitter
CsvLoader = _runtime.PyCsvLoader
JsonLoader = _runtime.PyJsonLoader
MarkdownLoader = _runtime.PyMarkdownLoader
HtmlLoader = _runtime.PyHtmlLoader
JsonOutputParser = _runtime.PyJsonOutputParser
ListOutputParser = _runtime.PyListOutputParser
EnumOutputParser = _runtime.PyEnumOutputParser
StructuredOutputParser = _runtime.PyStructuredOutputParser
BufferMemoryConfig = _runtime.PyBufferMemoryConfig
SummaryMemoryConfig = _runtime.PySummaryMemoryConfig
EntityMemoryConfig = _runtime.PyEntityMemoryConfig
VectorMemoryConfig = _runtime.PyVectorMemoryConfig
Message = _runtime.PyMessage
Attachment = _runtime.PyAttachment
AuthContext = _runtime.PyAuthContext
ApiKeyValidator = _runtime.PyApiKeyValidator
JwtValidator = _runtime.PyJwtValidator
RateLimitConfig = _runtime.PyRateLimitConfig
RateLimiter = _runtime.PyRateLimiter
Permission = _runtime.PyPermission
Role = _runtime.PyRole
RbacEnforcer = _runtime.PyRbacEnforcer
AuditEvent = _runtime.PyAuditEvent
AuditLogger = _runtime.PyAuditLogger
CachePolicy = _runtime.PyCachePolicy
CachedResponse = _runtime.PyCachedResponse
CacheStats = _runtime.PyCacheStats
LlmCache = _runtime.PyLlmCache
ToolStats = _runtime.PyToolStats
ToolMetrics = _runtime.PyToolMetrics
MetricsRecorder = _runtime.PyMetricsRecorder
WorkflowBuilder = _runtime.PyWorkflowBuilder
Workflow = _runtime.PyWorkflow
WorkflowEngine = _runtime.PyWorkflowEngine
AgentCoordinator = _runtime.PyAgentCoordinator
MessageBus = _runtime.PyMessageBus
ExactMatchEvaluator = _runtime.PyExactMatchEvaluator
ContainsEvaluator = _runtime.PyContainsEvaluator
LLMEvaluator = _runtime.PyLLMEvaluator
EvaluationDataset = _runtime.PyEvaluationDataset

_LAZY_MODULES = {
    "async_agent": ".async_agent",
    "chains": ".chains",
    "coordinator": ".coordinator",
    "env": ".env",
    "evaluation": ".evaluation",
    "exceptions": ".exceptions",
    "hooks": ".hooks",
    "memory": ".memory",
    "reasoning": ".reasoning",
    "session": ".session",
    "streaming": ".streaming",
    "tools": ".tools",
    "workflow": ".workflow",
    "prompts": ".prompts",
    "templates": ".templates",
    "types": ".types",
    "structured": ".structured",
    "mcp": ".mcp",
    "rag": ".rag",
}

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "AsyncAgent": (".async_agent", "AsyncAgent"),
    "create_agent_async": (".async_agent", "create_agent_async"),
    "FunctionTool": (".tools", "FunctionTool"),
    "tool": (".tools", "tool"),
    "PermissionMode": (".hooks", "PermissionMode"),
    "HookContext": (".hooks", "HookContext"),
    "HookMatcher": (".hooks", "HookMatcher"),
    "HookResult": (".hooks", "HookResult"),
    "LifecycleHook": (".hooks", "LifecycleHook"),
    "AgentSession": (".session", "AgentSession"),
    "UsageDict": (".types", "UsageDict"),
    "MetricsDict": (".types", "MetricsDict"),
    "HarnessDisciplineMetricsDict": (".types", "HarnessDisciplineMetricsDict"),
    "StreamEventDict": (".types", "StreamEventDict"),
    "ChatResultDict": (".types", "ChatResultDict"),
    "CollectedStreamDict": (".types", "CollectedStreamDict"),
    "configure_builder_from_env": (".env", "configure_builder_from_env"),
    "agent_from_env": (".env", "agent_from_env"),
    "resolve_provider_from_env": (".env", "resolve_provider_from_env"),
    "resolve_model_from_env": (".env", "resolve_model_from_env"),
    "resolve_api_key_from_env": (".env", "resolve_api_key_from_env"),
    "resolve_base_url_from_env": (".env", "resolve_base_url_from_env"),
    "openai_base_url": (".env", "openai_base_url"),
    "collect_text": (".streaming", "collect_text"),
    "collect_with_usage": (".streaming", "collect_with_usage"),
    "chat_text": (".streaming", "chat_text"),
    "iter_stream_text": (".streaming", "iter_stream_text"),
    "normalize_harness_discipline_metrics": (
        ".metrics",
        "normalize_harness_discipline_metrics",
    ),
    "CalicheError": (".exceptions", "CalicheError"),
    "AgentNotInitializedError": (".exceptions", "AgentNotInitializedError"),
    "ProviderError": (".exceptions", "ProviderError"),
    "ToolExecutionError": (".exceptions", "ToolExecutionError"),
    "WorkflowError": (".exceptions", "WorkflowError"),
    "RagError": (".exceptions", "RagError"),
    "ChainError": (".exceptions", "ChainError"),
    "MemoryError": (".exceptions", "MemoryError"),
    "CoordinatorError": (".exceptions", "CoordinatorError"),
    "EvaluationError": (".exceptions", "EvaluationError"),
    "create_workflow_builder": (".workflow", "create_workflow_builder"),
    "RagService": (".rag", "RagService"),
    "RagTool": (".rag", "RagTool"),
    "RagIndexResult": (".rag", "RagIndexResult"),
    "RagQueryResult": (".rag", "RagQueryResult"),
    "EnhancedQueryResult": (".rag", "EnhancedQueryResult"),
    "GraphSearchResult": (".rag", "GraphSearchResult"),
    "RagStats": (".rag", "RagStats"),
    "McpServer": (".mcp", "McpServer"),
    "McpServerBuilder": (".mcp", "McpServerBuilder"),
    "McpTool": (".mcp", "McpTool"),
    "InProcessMcpServer": (".mcp", "InProcessMcpServer"),
    "InProcessMcpServerBuilder": (".mcp", "InProcessMcpServerBuilder"),
}


_RAG_LAZY_ATTRS = {
    "RagService",
    "RagTool",
    "RagIndexResult",
    "RagQueryResult",
    "EnhancedQueryResult",
    "GraphSearchResult",
    "RagStats",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name], __name__)
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        if name in _RAG_LAZY_ATTRS and not hasattr(_runtime, "PyRagService"):
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r} "
                "(RAG Service requires 'tools-rag' feature)"
            )
        mod_path, attr = _LAZY_ATTRS[name]
        try:
            module = importlib.import_module(mod_path, __name__)
            value = getattr(module, attr)
        except ImportError as exc:
            if name in _RAG_LAZY_ATTRS:
                raise AttributeError(
                    f"module {__name__!r} has no attribute {name!r} "
                    "(RAG Service requires 'tools-rag' feature)"
                ) from exc
            raise
        globals()[name] = value
        return value
    if name == "web_scraper":
        module = importlib.import_module(".web_scraper", __name__)
        globals()[name] = module
        return module
    if name == "WebScrapingApp":
        module = importlib.import_module(".web_scraper", __name__)
        value = module.WebScrapingApp
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()) | set(_LAZY_MODULES) | set(_LAZY_ATTRS))


if TYPE_CHECKING:
    from .async_agent import AsyncAgent, create_agent_async
    from .env import (
        agent_from_env,
        configure_builder_from_env,
        openai_base_url,
        resolve_api_key_from_env,
        resolve_base_url_from_env,
        resolve_model_from_env,
        resolve_provider_from_env,
    )
    from .exceptions import (
        AgentNotInitializedError,
        CalicheError,
        ChainError,
        CoordinatorError,
        EvaluationError,
        MemoryError,
        ProviderError,
        RagError,
        ToolExecutionError,
        WorkflowError,
    )
    from .hooks import (
        HookContext,
        HookMatcher,
        HookResult,
        LifecycleHook,
        PermissionMode,
    )
    from .metrics import normalize_harness_discipline_metrics
    from .session import AgentSession
    from .streaming import chat_text, collect_text, collect_with_usage, iter_stream_text
    from .tools import FunctionTool, tool
    from .types import (
        ChatResultDict,
        CollectedStreamDict,
        HarnessDisciplineMetricsDict,
        MetricsDict,
        StreamEventDict,
        UsageDict,
    )
    from .workflow import create_workflow_builder

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "AgentBuilder",
    "DurableStreamBackend",
    "LlmClient",
    "LlmClientBuilder",
    "ProviderType",
    "Message",
    "Attachment",
    "ToolResult",
    "BufferMemoryConfig",
    "SummaryMemoryConfig",
    "EntityMemoryConfig",
    "VectorMemoryConfig",
    "SharedState",
    "FlowBuilder",
    "AgentFlow",
    "LLMChain",
    "SequentialChain",
    "MapReduceChain",
    "ParallelChain",
    "ChainBuilder",
    "ReActPattern",
    "PlanExecutePattern",
    "ReflexionPattern",
    "SelfAskPattern",
    "TreeOfThoughtsPattern",
    "DebatePattern",
    "SupervisorPattern",
    "HierarchicalPattern",
    "HierarchicalPatternBuilder",
    "PipelinePattern",
    "PipelinePatternBuilder",
    "Document",
    "TextSplitter",
    "CsvLoader",
    "JsonLoader",
    "MarkdownLoader",
    "HtmlLoader",
    "JsonOutputParser",
    "ListOutputParser",
    "EnumOutputParser",
    "StructuredOutputParser",
    "AuthContext",
    "ApiKeyValidator",
    "JwtValidator",
    "RateLimitConfig",
    "RateLimiter",
    "Permission",
    "Role",
    "RbacEnforcer",
    "AuditEvent",
    "AuditLogger",
    "CachePolicy",
    "CachedResponse",
    "CacheStats",
    "LlmCache",
    "ToolStats",
    "ToolMetrics",
    "MetricsRecorder",
    "async_agent",
    "memory",
    "reasoning",
    "chains",
    "tools",
    "workflow",
    "coordinator",
    "evaluation",
    "exceptions",
    "hooks",
    "session",
    "types",
    "FunctionTool",
    "tool",
    "AsyncAgent",
    "create_agent_async",
    "MutationStyle",
    "PermissionMode",
    "InspectionAction",
    "InspectionResult",
    "PathSecurityInspector",
    "PermissionModeInspector",
    "ToolInspectionManager",
    "InspectionHookAdapter",
    "default_inspection_manager",
    "ToolRegistry",
    "JsonlSessionHeader",
    "JsonlSessionEntry",
    "JsonlSessionTree",
    "JsonlSessionStore",
    "parse_jsonl_session",
    "new_entry_id",
    "create_calculator_tool",
    "create_fs_tool",
    "create_apply_patch_tool",
    "create_http_tool",
    "create_code_execution_tool",
    "probe_local_inference_host",
    "fit_local_inference_catalog",
    "fit_local_inference_model",
    "safe_local_inference_models",
    "WorkflowBuilder",
    "Workflow",
    "WorkflowEngine",
    "create_workflow_builder",
    "AgentCoordinator",
    "MessageBus",
    "ExactMatchEvaluator",
    "ContainsEvaluator",
    "LLMEvaluator",
    "EvaluationDataset",
    "CalicheError",
    "AgentNotInitializedError",
    "ProviderError",
    "ToolExecutionError",
    "WorkflowError",
    "RagError",
    "ChainError",
    "MemoryError",
    "CoordinatorError",
    "EvaluationError",
    "HookMatcher",
    "HookContext",
    "HookResult",
    "LifecycleHook",
    "AgentSession",
    "UsageDict",
    "MetricsDict",
    "HarnessDisciplineMetricsDict",
    "StreamEventDict",
    "ChatResultDict",
    "CollectedStreamDict",
    "env",
    "streaming",
    "configure_builder_from_env",
    "agent_from_env",
    "resolve_provider_from_env",
    "resolve_model_from_env",
    "resolve_api_key_from_env",
    "resolve_base_url_from_env",
    "openai_base_url",
    "collect_text",
    "collect_with_usage",
    "chat_text",
    "iter_stream_text",
    "normalize_harness_discipline_metrics",
]

# Optional surfaces: keep in __all__ when discoverable without importing heavy modules.
if hasattr(_runtime, "PyPromptTemplateStore") and hasattr(_runtime, "PyTemplateRegistry"):
    __all__.extend(["templates", "prompts"])

if hasattr(_runtime, "PyRagService"):
    __all__.extend(
        [
            "RagService",
            "RagTool",
            "RagIndexResult",
            "RagQueryResult",
            "EnhancedQueryResult",
            "GraphSearchResult",
            "RagStats",
        ]
    )

if importlib.util.find_spec("caliche.web_scraper") is not None:
    __all__.extend(["WebScrapingApp", "web_scraper"])

if importlib.util.find_spec("caliche_mcp") is not None or importlib.util.find_spec(
    "caliche.mcp"
):
    __all__.extend(["mcp", "McpServer", "McpServerBuilder", "McpTool"])
    if hasattr(_runtime, "PyInProcessMcpServer"):
        __all__.extend(["InProcessMcpServer", "InProcessMcpServerBuilder"])

if importlib.util.find_spec("caliche.structured") is not None:
    __all__.extend(["structured"])
