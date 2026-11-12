pub mod adaptive;
#[cfg(feature = "agentfs")]
pub mod agentfs_tool_logger;
pub mod apply_patch;
#[cfg(feature = "tools-ast")]
pub mod ast_tool;
pub mod bash;
pub mod calculator;
pub mod chaining;
pub mod code_execution;
pub mod common;
pub mod config;
pub mod context;
pub mod context_manager;
pub mod control;
pub mod credentials;
pub mod data;
pub mod data_manipulation;
#[cfg(feature = "tools-sql")]
pub mod database_management;
pub mod deduplication;
mod edit_args;
mod edit_file;
pub mod embedding_order;
pub mod execution;
pub(crate) mod failure_breaker;
pub mod file_creation_context;
pub mod file_mutation_queue;
pub mod filtering;
pub mod fs;
pub mod fs_tool;
#[cfg(feature = "tools-git")]
pub mod git;
#[cfg(feature = "tools-grep")]
mod glob;
#[cfg(feature = "tools-graph")]
pub mod graph;
#[cfg(feature = "tools-grep")]
mod grep;
#[cfg(feature = "tools-grep")]
mod grep_pattern;
pub mod hashline;
pub mod http;
pub mod image_generation;
mod list_directory;
#[cfg(feature = "lsp")]
pub mod lsp_diagnostics;
#[cfg(feature = "lsp")]
pub mod lsp_hover;
#[cfg(feature = "lsp")]
pub mod lsp_symbols;
#[cfg(feature = "mcp-client")]
pub mod mcp_remote;
#[cfg(feature = "tools-media")]
pub mod media_understanding;
pub mod metadata;
pub mod names;
pub mod perplexity_search;
#[cfg(feature = "tools-document")]
pub mod process_document;
pub mod process_tracker;
#[cfg(feature = "tools-rag")]
mod rag_tool;
mod read_file;
mod read_window;
pub mod registry;
pub mod research;
pub mod scoring;
pub mod selection;
#[cfg(feature = "tools-rag")]
pub mod semantic_search;
#[cfg(feature = "session-tools")]
pub mod session_tools;
pub mod sql;
#[cfg(feature = "tools-sql")]
pub mod sql_execution;
#[cfg(feature = "tools-rag")]
pub mod sql_generation;
pub mod sql_tenant_filter;
pub mod tool;
pub mod toolkit;
pub mod validators;
pub mod verify_integration;
pub mod web;
pub mod workspace;
mod write_file;

#[cfg(test)]
#[cfg(unix)]
mod gate_race_tests;
#[cfg(test)]
mod git_metadata_guard_tests;
#[cfg(test)]
mod hotpath_perf;

pub use adaptive::AdaptiveSelector;
#[cfg(feature = "agentfs")]
pub use agentfs_tool_logger::{AgentFsToolLogger, ToolLogRecord};
pub use apply_patch::{ApplyPatchTool, MutationStyle, apply_hunks_to_cwd, parse_patch};
#[cfg(feature = "tools-ast")]
pub use ast_tool::AstTool;
pub use bash::BashTool;
pub use calculator::CalculatorTool;
pub use chaining::{
    ChainedToolCall, DependencyGraph, ExecutionPhase, ExecutionPlan, ExecutionPlanner,
};
pub use code_execution::CodeExecutionTool;
pub use context::{ContextGetTool, ContextSetTool};
pub use control::{HANDOFF_TOOL_NAME, HandoffTool};
pub use credentials::{CredentialResolver, ToolCredential, credential_parameters};
pub use data::DataToolkit;
pub use data_manipulation::DataManipulationTool;
#[cfg(feature = "tools-sql")]
pub use database_management::DatabaseManagementTool;
pub use deduplication::{
    CallDeduplicator, ExecutionHistoryStore, ExecutionMetrics, RedundancyReport,
};
pub use fs::FileSystemToolkit;
pub use fs_tool::FsTool;
#[cfg(feature = "tools-git")]
pub use git::{GitBranchTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool, GitTool};
#[cfg(feature = "tools-graph")]
pub use graph::GraphQueryTool;
pub use http::HttpTool;
pub use http::HttpToolkit;
pub use image_generation::ImageGenerationTool;
#[cfg(feature = "lsp")]
pub use lsp_diagnostics::LspDiagnosticsTool;
#[cfg(feature = "lsp")]
pub use lsp_hover::LspHoverTool;
#[cfg(feature = "lsp")]
pub use lsp_symbols::LspSymbolsTool;
#[cfg(feature = "mcp-client")]
pub use mcp_remote::RemoteMcpTool;
#[cfg(feature = "tools-media")]
pub use media_understanding::{
    MediaAttachment, MediaCapabilities, MediaConfig, MediaOutput, MediaType,
    MediaUnderstandingProvider, MediaUnderstandingRunner,
};
pub use metadata::{ToolCategory, ToolCost, ToolDependencies, ToolMetadata, ToolRelevanceScore};
pub use perplexity_search::PerplexitySearchTool;
#[cfg(feature = "tools-document")]
pub use process_document::ProcessDocumentTool;
#[cfg(feature = "tools-rag")]
pub use rag_tool::RagTool;
pub use read_file::clear_file_memory;
pub use registry::ToolRegistry;
pub use research::{
    AgentResearchRunner, ConfigurableResearchServiceFactory, ParallelResearchRunner,
    ResearchRunner, ResearchServiceFactory, ResearchTool, set_research_progress_sink,
};
pub use selection::{
    RelevanceSelector, SelectionConstraints, SimpleSelector, TaskType, ToolExecutionRecord,
    ToolSelectionContext, ToolSelectionResult, ToolSelector,
};
#[cfg(feature = "tools-rag")]
pub use semantic_search::{SemanticSearchConfig, SemanticSearchTool};
pub use sql::SqlToolkit;
#[cfg(feature = "tools-sql")]
pub use sql_execution::{PostgresSqlExecutor, SqlExecutionTool, SqlExecutor};
#[cfg(feature = "tools-rag")]
pub use sql_generation::SqlGenerationTool;
pub use sql_tenant_filter::TenantFilterConfig;
pub use tool::{Tool, ToolCall, ToolCapability, ToolMeta, ToolResult};
pub use toolkit::Toolkit;
pub use verify_integration::VerifyIntegrationTool;
pub use web::{WebFetchTool, WebSearchTool};
pub use workspace::WorkspaceTool;

/// Enumeration of built-in tools available in the agent framework.
///
/// Provides compile-time safety when enabling tools and can be converted to/from strings.
///
/// # Example
///
/// ```rust,no_run
/// use caliche::AgentBuilder;
///
/// let builder = AgentBuilder::new()
///     .enable_tools(&["calculator", "fs"]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BuiltInTool {
    Bash,
    Calculator,
    Fs,
    #[cfg(feature = "tools-ast")]
    Ast,
    #[cfg(feature = "tools-graph")]
    GraphQuery,
    Http,
    WebSearch,
    WebFetch,
    DataManipulation,
    #[cfg(feature = "tools-document")]
    ProcessDocument,
    PerplexitySearch,
    #[cfg(feature = "tools-rag")]
    SqlGeneration,
    #[cfg(feature = "tools-sql")]
    SqlExecution,
    ExecutePython,
    ContextGet,
    ContextSet,
    Handoff,
    Workspace,
    GenerateImage,
    #[cfg(feature = "lsp")]
    LspDiagnostics,
    #[cfg(feature = "lsp")]
    LspHover,
    #[cfg(feature = "lsp")]
    LspSymbols,
    #[cfg(feature = "session-tools")]
    SessionsList,
    #[cfg(feature = "session-tools")]
    SessionsHistory,
    #[cfg(feature = "session-tools")]
    SessionsSend,
    #[cfg(feature = "session-tools")]
    SessionsSpawn,
}

impl BuiltInTool {
    /// Get the string name of the tool
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        if let Some(name) = self.core_name() {
            return name;
        }
        self.optional_name()
    }

    const fn core_name(&self) -> Option<&'static str> {
        match self {
            Self::Bash => Some("bash"),
            Self::Calculator => Some("calculator"),
            Self::Fs => Some(names::TOOL_FILESYSTEM),
            Self::Http => Some(names::TOOL_HTTP),
            Self::WebSearch => Some(names::TOOL_WEB_SEARCH),
            Self::WebFetch => Some(names::TOOL_WEB_FETCH),
            Self::DataManipulation => Some("data_manipulation"),
            Self::PerplexitySearch => Some("perplexity_search"),
            Self::ExecutePython => Some(names::TOOL_EXECUTE_PYTHON),
            Self::ContextGet => Some(names::TOOL_CONTEXT_GET),
            Self::ContextSet => Some(names::TOOL_CONTEXT_SET),
            Self::Handoff => Some("handoff"),
            Self::Workspace => Some(names::TOOL_WORKSPACE),
            Self::GenerateImage => Some(names::TOOL_GENERATE_IMAGE),
            #[cfg_attr(
                not(any(
                    feature = "tools-ast",
                    feature = "tools-graph",
                    feature = "tools-document",
                    feature = "tools-rag",
                    feature = "tools-sql",
                    feature = "lsp",
                    feature = "session-tools"
                )),
                expect(unreachable_patterns, reason = "feature-gated variants reach this arm")
            )]
            _ => None,
        }
    }

    const fn optional_name(&self) -> &'static str {
        match self {
            #[cfg(feature = "tools-ast")]
            Self::Ast => "ast",
            #[cfg(feature = "tools-graph")]
            Self::GraphQuery => "graph_query",
            #[cfg(feature = "tools-document")]
            Self::ProcessDocument => "process_document",
            #[cfg(feature = "tools-rag")]
            Self::SqlGeneration => "sql_generation",
            #[cfg(feature = "tools-sql")]
            Self::SqlExecution => "sql_execution",
            #[cfg(feature = "lsp")]
            Self::LspDiagnostics => "lsp_diagnostics",
            #[cfg(feature = "lsp")]
            Self::LspHover => "lsp_hover",
            #[cfg(feature = "lsp")]
            Self::LspSymbols => "lsp_symbols",
            #[cfg(feature = "session-tools")]
            Self::SessionsList => "sessions_list",
            #[cfg(feature = "session-tools")]
            Self::SessionsHistory => "sessions_history",
            #[cfg(feature = "session-tools")]
            Self::SessionsSend => "sessions_send",
            #[cfg(feature = "session-tools")]
            Self::SessionsSpawn => "sessions_spawn",
            _ => "",
        }
    }

    /// Try to parse a string into a `BuiltInTool`
    #[must_use]
    pub const fn try_from_str(s: &str) -> Option<Self> {
        if let Some(tool) = Self::try_from_core(s) {
            return Some(tool);
        }
        if let Some(tool) = Self::try_from_optional_tools(s) {
            return Some(tool);
        }
        Self::try_from_session_tools(s)
    }

    const fn try_from_core(s: &str) -> Option<Self> {
        if s.eq_ignore_ascii_case("bash") {
            Some(Self::Bash)
        } else if s.eq_ignore_ascii_case("calculator") {
            Some(Self::Calculator)
        } else if s.eq_ignore_ascii_case(names::TOOL_FILESYSTEM) {
            Some(Self::Fs)
        } else if s.eq_ignore_ascii_case(names::TOOL_HTTP) {
            Some(Self::Http)
        } else if s.eq_ignore_ascii_case(names::TOOL_WEB_SEARCH) {
            Some(Self::WebSearch)
        } else if s.eq_ignore_ascii_case(names::TOOL_WEB_FETCH) {
            Some(Self::WebFetch)
        } else if s.eq_ignore_ascii_case("data_manipulation") {
            Some(Self::DataManipulation)
        } else if s.eq_ignore_ascii_case("perplexity_search") {
            Some(Self::PerplexitySearch)
        } else if s.eq_ignore_ascii_case(names::TOOL_EXECUTE_PYTHON) {
            Some(Self::ExecutePython)
        } else if s.eq_ignore_ascii_case(names::TOOL_CONTEXT_GET) {
            Some(Self::ContextGet)
        } else if s.eq_ignore_ascii_case(names::TOOL_CONTEXT_SET) {
            Some(Self::ContextSet)
        } else if s.eq_ignore_ascii_case("handoff") {
            Some(Self::Handoff)
        } else if s.eq_ignore_ascii_case(names::TOOL_WORKSPACE) {
            Some(Self::Workspace)
        } else if s.eq_ignore_ascii_case(names::TOOL_GENERATE_IMAGE) {
            Some(Self::GenerateImage)
        } else {
            None
        }
    }

    const fn try_from_optional_tools(s: &str) -> Option<Self> {
        if s.eq_ignore_ascii_case("ast") {
            #[cfg(feature = "tools-ast")]
            {
                Some(Self::Ast)
            }
            #[cfg(not(feature = "tools-ast"))]
            {
                None
            }
        } else if s.eq_ignore_ascii_case("graph_query") {
            #[cfg(feature = "tools-graph")]
            {
                Some(Self::GraphQuery)
            }
            #[cfg(not(feature = "tools-graph"))]
            {
                None
            }
        } else if s.eq_ignore_ascii_case("process_document") {
            #[cfg(feature = "tools-document")]
            {
                Some(Self::ProcessDocument)
            }
            #[cfg(not(feature = "tools-document"))]
            {
                None
            }
        } else if s.eq_ignore_ascii_case("sql_generation") {
            #[cfg(feature = "tools-rag")]
            {
                Some(Self::SqlGeneration)
            }
            #[cfg(not(feature = "tools-rag"))]
            {
                None
            }
        } else if s.eq_ignore_ascii_case("sql_execution") {
            #[cfg(feature = "tools-sql")]
            {
                Some(Self::SqlExecution)
            }
            #[cfg(not(feature = "tools-sql"))]
            {
                None
            }
        } else {
            Self::try_from_lsp_tools(s)
        }
    }

    const fn try_from_lsp_tools(s: &str) -> Option<Self> {
        if s.eq_ignore_ascii_case("lsp_diagnostics") {
            #[cfg(feature = "lsp")]
            {
                Some(Self::LspDiagnostics)
            }
            #[cfg(not(feature = "lsp"))]
            {
                None
            }
        } else if s.eq_ignore_ascii_case("lsp_hover") {
            #[cfg(feature = "lsp")]
            {
                Some(Self::LspHover)
            }
            #[cfg(not(feature = "lsp"))]
            {
                None
            }
        } else if s.eq_ignore_ascii_case("lsp_symbols") {
            #[cfg(feature = "lsp")]
            {
                Some(Self::LspSymbols)
            }
            #[cfg(not(feature = "lsp"))]
            {
                None
            }
        } else {
            None
        }
    }

    const fn try_from_session_tools(s: &str) -> Option<Self> {
        if s.eq_ignore_ascii_case("sessions_list") {
            #[cfg(feature = "session-tools")]
            {
                Some(Self::SessionsList)
            }
            #[cfg(not(feature = "session-tools"))]
            {
                None
            }
        } else if s.eq_ignore_ascii_case("sessions_history") {
            #[cfg(feature = "session-tools")]
            {
                Some(Self::SessionsHistory)
            }
            #[cfg(not(feature = "session-tools"))]
            {
                None
            }
        } else if s.eq_ignore_ascii_case("sessions_send") {
            #[cfg(feature = "session-tools")]
            {
                Some(Self::SessionsSend)
            }
            #[cfg(not(feature = "session-tools"))]
            {
                None
            }
        } else if s.eq_ignore_ascii_case("sessions_spawn") {
            #[cfg(feature = "session-tools")]
            {
                Some(Self::SessionsSpawn)
            }
            #[cfg(not(feature = "session-tools"))]
            {
                None
            }
        } else {
            None
        }
    }
}

impl From<BuiltInTool> for String {
    fn from(tool: BuiltInTool) -> Self {
        tool.as_str().to_owned()
    }
}

impl std::fmt::Display for BuiltInTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
