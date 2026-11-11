//! # Caliche
//!
//! Rust agent runtime with multi-provider LLM support, tools, and orchestration.
//!
//! `caliche` is the supported Rust engine crate. Shared wire DTOs live in
//! `caliche_contracts` (no runtime dependency). Orchestration primitives
//! (ACL, privilege, quarantine, external harness dispatch) live in
//! `caliche_orchestration` and are re-exported from `caliche::orchestration`.
//! Typed unified policy lives in `caliche_policy` (`caliche::policy`).
//! Shadow mode does not replace live hook allow/deny; set
//! `CALICHE_POLICY_SHADOW=1` to log a comparison.
//!
//! ## Quick Start (hosts)
//!
//! For CLI/API-style hosts, use the high-level `AgentService` API:
//!
//! ```rust,no_run
//! use caliche::{AgentService, AgentServiceConfig, ProviderType};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create an agent with default settings (reads from env vars)
//!     let mut agent = AgentService::new().await?;
//!     
//!     // Or specify provider and model explicitly
//!     let mut agent = AgentService::from_config(
//!         AgentServiceConfig::new()
//!             .with_provider(
//!                 Some(ProviderType::Bedrock),
//!                 Some("anthropic.claude-3-sonnet-20240229-v1:0".to_string()),
//!             )
//!             .with_all_default_tools(),
//!     ).await?;
//!     
//!     // Chat with the agent
//!     let response = agent.chat("What is 42 * 17?").await?;
//!     println!("Agent: {}", response);
//!     println!("Tokens used: {:?}", response.usage.total_tokens);
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Library embedding
//!
//! For custom construction inside another Rust app, use `AgentBuilder`:
//!
//! ```rust,no_run
//! use caliche::{AgentBuilder, ProviderType};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut agent = AgentBuilder::new()
//!         .provider(ProviderType::OpenAI)
//!         .model("gpt-4o".to_string())
//!         .system_prompt("You are a helpful coding assistant.".to_string())
//!         .enable_tool("calculator")
//!         .enable_tool("fs")
//!         .build()
//!         .await?;
//!     
//!     let response = agent.chat("What is 2 + 2?").await?;
//!     println!("{}", response);
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Agent Lifecycle Controls
//!
//! Caliche provides advanced controls for intercepting and managing agent behavior:
//!
//! ### Lifecycle Hooks
//!
//! Intercept and control tool execution with lifecycle hooks:
//!
//! ```rust,no_run
//! use caliche::{AgentBuilder, ProviderType};
//! use caliche::agent::hooks::{HookContext, HookMatcher, HookResult, LifecycleHook};
//! use async_trait::async_trait;
//! use std::sync::Arc;
//!
//! #[derive(Debug)]
//! struct SecurityHook;
//!
//! #[async_trait]
//! impl LifecycleHook for SecurityHook {
//!     async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
//!         // Block dangerous commands
//!         if ctx.tool_name == "Bash" && ctx.tool_input.get("command")
//!             .and_then(|v| v.as_str())
//!             .map(|c| c.contains("rm -rf"))
//!             .unwrap_or(false) {
//!             return HookResult::deny("Dangerous command blocked");
//!         }
//!         HookResult::allow()
//!     }
//! }
//!
//! # async fn example() -> anyhow::Result<()> {
//! let agent = AgentBuilder::new()
//!     .provider(ProviderType::Bedrock)
//!     .hook(HookMatcher::tool("Bash"), Arc::new(SecurityHook))
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Permission Modes
//!
//! Control tool execution permissions:
//!
//! ```rust,no_run
//! use caliche::{AgentBuilder, ProviderType};
//! use caliche::agent::hooks::permission::PermissionMode;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let agent = AgentBuilder::new()
//!     .provider(ProviderType::Bedrock)
//!     .permission_mode(PermissionMode::AcceptEdits)  // Auto-accept file edits
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Allowed Tools
//!
//! Restrict which tools can be used:
//!
//! ```rust,no_run
//! use caliche::{AgentBuilder, ProviderType};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let agent = AgentBuilder::new()
//!     .provider(ProviderType::Bedrock)
//!     .allowed_tools(&["calculator", "fs"])
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### In-Process MCP Servers
//!
//! Register tools directly without subprocess overhead:
//!
//! ```rust,no_run
//! use caliche::{AgentBuilder, ProviderType, Tool, ToolResult};
//! use async_trait::async_trait;
//! use serde_json::json;
//!
//! #[derive(Debug)]
//! struct MyCustomTool;
//!
//! impl caliche::agent::tools::ToolMeta for MyCustomTool {
//!     fn name(&self) -> &str { "my_tool" }
//!     fn description(&self) -> &str { "A custom tool" }
//!     fn parameters(&self) -> serde_json::Value { json!({}) }
//! }
//!
//! #[async_trait]
//! impl Tool for MyCustomTool {
//!     async fn execute(&self, _args: serde_json::Value) -> ToolResult {
//!         ToolResult::success("done".to_string())
//!     }
//! }
//!
//! # async fn example() -> anyhow::Result<()> {
//! #[cfg(feature = "mcp")]
//! {
//!     use caliche::mcp::InProcessMcpServer;
//!     let server = InProcessMcpServer::builder("my-tools")
//!         .add_tool(MyCustomTool)
//!         .build()?;
//!
//!     let agent = AgentBuilder::new()
//!         .provider(ProviderType::Bedrock)
//!         .mcp_server("tools", server)
//!         .build()
//!         .await?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Session Forking
//!
//! Enable parallel work and subagents:
//!
//! ```rust,no_run
//! use caliche::agent::session::AgentSession;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let mut main_session = AgentSession::new("main");
//! let forked = main_session.fork("worker").await.unwrap();
//! // Work in parallel...
//! main_session.merge(&forked).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Custom Tools
//!
//! Create your own tools by implementing the `Tool` trait:
//!
//! ```rust,no_run
//! use caliche::{Tool, ToolResult};
//! use async_trait::async_trait;
//! use serde_json::json;
//! use std::sync::Arc;
//!
//! #[derive(Debug)]
//! struct MyCustomTool;
//!
//! impl caliche::agent::tools::ToolMeta for MyCustomTool {
//!     fn name(&self) -> &str {
//!         "my_custom_tool"
//!     }
//!     
//!     fn description(&self) -> &str {
//!         "A custom tool that does something useful"
//!     }
//!     
//!     fn parameters(&self) -> serde_json::Value {
//!         json!({
//!             "type": "object",
//!             "properties": {
//!                 "input": {
//!                     "type": "string",
//!                     "description": "Input parameter"
//!                 }
//!             },
//!             "required": ["input"]
//!         })
//!     }
//! }
//!
//! #[async_trait]
//! impl Tool for MyCustomTool {
//!     async fn execute(&self, args: serde_json::Value) -> ToolResult {
//!         let input = match args["input"].as_str() {
//!             Some(s) => s,
//!             None => return ToolResult::error("Missing input".to_string()),
//!         };
//!         
//!         let result = json!({"result": "success", "input": input});
//!         ToolResult::success(result.to_string())
//!     }
//! }
//! ```
//!
//! ## Entry points
//!
//! Prefer this ladder; do not treat every surface as an equal quick start:
//!
//! 1. [`AgentBuilder`] / [`Agent`] — library embedding and custom construction
//! 2. [`AgentService`] — CLI and API host path (settings, tools, sessions)
//! 3. [`harness::AgentHarness`] — REPL extras (events, checkpoints, undo)
//! 4. [`harness::AgentRuntime`] — multi-session map over harness
//!
//! The HTTP server lives in the `caliche_server` crate, not in this facade.
//!
//! ## Modules
//!
//! - `agent`: core agent, builder, service, tools, hooks, memory
//! - `llm`: provider abstractions (Bedrock, OpenAI, OpenRouter, and others)
//! - `harness`: session harness and multi-session runtime
//! - `orchestration`: multi-agent workflows and coordinator
//! - `messages`: message and stream types
//! - `rag` / `gateway` / `mcp` / `plugins`: optional capability modules
//!
//! Lint policy lives in root `Cargo.toml` (`[workspace.lints]`) and `clippy.toml`.
//! Prefer `#[expect(clippy::lint, reason = "...")]` over crate-level `allow`.

#![recursion_limit = "256"]

pub mod adapters;
pub mod agent;
pub mod artifacts;

#[cfg(feature = "ast")]
pub mod ast;

pub mod distributed;
pub mod evaluation;
pub mod gateway;
pub mod harness;
pub mod llm;
pub mod messages;
pub mod observability;
pub mod orchestration;

#[cfg(any(test, feature = "tools-rag"))]
mod backpressure;

/// Typed unified policy. Live hook decisions stay on the hook path.
pub use caliche_policy as policy;

pub mod core_domain;
#[cfg(feature = "tools-rag")]
pub mod rag;
pub mod security;
pub mod testing;

#[cfg(feature = "scheduling")]
pub mod scheduling;

#[cfg(feature = "prompts")]
pub mod prompts;

#[cfg(feature = "reasoning-patterns")]
pub mod reasoning;

pub mod output;

#[cfg(feature = "document-loaders")]
pub use data::loaders;

#[cfg(feature = "s3")]
pub use data::storage::object as storage;

#[cfg(feature = "vector-memory")]
pub use data::storage::vector as vecstore;

pub use data::persistence;

#[cfg(feature = "chains")]
pub mod chains;

pub mod config;

// Capabilities module (feature availability)
pub mod capabilities;

// Discovery (embedding provider auto-detection)
pub mod discovery;

/// Workspace warm-start declarations (MCP, repos, skills) without the RAG indexer.
#[path = "workspace/declaration.rs"]
pub mod workspace_declaration;

/// `.gitmodules` parser. The workspace tool uses it without the RAG indexer.
#[path = "workspace/gitmodules.rs"]
pub(crate) mod gitmodules;

#[cfg(feature = "tools-rag")]
pub mod workspace;

#[cfg(feature = "tools-grep")]
pub mod repomap;

// Utilities module
mod utils;

// Re-export utility modules
pub use utils::cost;
pub use utils::health;
pub use utils::user_context::UserContext;

// Re-export caching from llm module for convenience
pub use llm::cache::{CachePolicy, LlmCache};
pub mod caching {
    pub use crate::llm::cache::{CachePolicy, LlmCache};
}

// Data management module (internal)
mod data;

#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "plugins")]
pub mod plugins;

#[cfg(feature = "lsp")]
pub mod lsp;

// Template system
pub mod templates;

// Shared trait abstractions
pub mod traits;

// Durable event streams (replay, resume, Redis/memory/mmap backends)
pub mod streams;

// Core types - re-export from agent module
#[cfg(feature = "tools-rag")]
pub use agent::builder::WorkspaceEmbeddingMode;
pub use agent::public::{
    Agent, AgentBuilder, AgentPresets, AgentService, AgentSession, AgentSettings, Clarification,
    InteractionMode, PlanDocument, PlanTask,
};
pub use agent::service::{AgentServiceConfig, ToolSelection};
pub use agent::service::{DEFAULT_PROJECT_CONTEXT_SHARE, ProjectContext};
pub use artifacts::{
    ArtifactMetadata, ArtifactRecord, ArtifactSelector, ArtifactStore, StoredArtifact,
};
#[cfg(feature = "tools-rag")]
pub use workspace::{PersistenceConfig, WorkspaceIndexerConfig};

// Re-export agent submodules for convenience
#[cfg(feature = "compaction")]
pub use agent::core::compaction::{
    CompactionConfig, CompactionResult, CompactionStrategy, CompactionType, CompositeCompaction,
    ImportanceScoringCompaction, LlmSummarizationCompaction, SlidingWindowCompaction,
};
#[cfg(feature = "compaction")]
pub use agent::core::compaction_jev::{
    JevAsker, JevCallAction, JevCallDecision, JevCompactOptions, JevCompactionConfig,
    JevHttpClient, JevQuestion, JevResponse,
};
pub use agent::memory::{self, MemoryBuilder};
#[cfg(feature = "sandbox")]
pub use agent::operations::SandboxOperationTransport;
pub use agent::operations::{
    LocalOperationManager, Operation, OperationExecution, OperationExecutionMode, OperationId,
    OperationKind, OperationManager, OperationStatus, RemoteOperationManager,
    RemoteOperationTransport, ToolTranslator, TranslatorRegistry,
};
pub use agent::public::hooks;
pub use agent::public::tools;

// Error context helpers
pub mod errors;
pub use agent::public::tools::{
    BuiltInTool, Tool, ToolCall, ToolCapability, ToolMeta, ToolResult, Toolkit,
};
pub use agent::tools::ToolRegistry;

// Re-exports for doctests and ergonomic root-level imports (also in prelude)
pub use config::Config;
pub use llm::{
    LlmClient, LlmClientBuilder, ModelCatalogEntry, ModelCatalogOptions, OpenAIProvider,
    ProviderCatalogEntry, ProviderType, compiled_provider_types, discover_provider_catalog,
    discover_provider_models, flatten_catalog_models,
};
pub use messages::{Message, ReasoningEffort, ToolChoice};

/// Presets module for `caliche::presets::AgentPresets` (doc examples).
pub mod presets {
    pub use crate::agent::presets::*;
}

#[cfg(feature = "chains")]
pub use chains::ChainBuilder;

#[cfg(feature = "tools-rag")]
pub use agent::tools::RagTool;

// Orchestration and kernel exports
pub use agent::kernel::{Handoff, SharedState, TurnResult};
pub use orchestration::{AgentCoordinator, Workflow, WorkflowBuilder, WorkflowEngine};

// Harness exports for agentic workflows
pub use agent::subagent::{SharedSubagentSelection, SubagentModelSelection};
#[cfg(feature = "session-tools")]
pub use harness::HarnessSessionSpawner;
pub use harness::{
    AgentHarness, AgentRuntime, ComplexTaskConfig, ComplexTaskOutcome, ControlMessage, ControlMode,
    HarnessConfig, InboxInput, InboxTurnOutcome, InputKind, SessionInbox, run_complex_task,
};
pub use messages::{CompactionChange, CompactionChangeKind, CompactionReport};

/// Convenient single-import prelude for the most commonly used types.
///
/// # Example
///
/// ```rust,no_run
/// use caliche::prelude::*;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let mut agent = AgentBuilder::new()
///         .provider(ProviderType::OpenAI)
///         .model("gpt-4o".to_string())
///         .build()
///         .await?;
///     let response = agent.chat("Hello!").await?;
///     println!("{}", response);
///     Ok(())
/// }
/// ```
pub mod prelude {
    // Core agent types
    pub use crate::{Agent, AgentBuilder, AgentService, AgentServiceConfig, ToolSelection};

    // Orchestration
    pub use crate::orchestration::ExecutionContext;
    pub use crate::{
        AgentCoordinator, Handoff, TurnResult, Workflow, WorkflowBuilder, WorkflowEngine,
    };

    // Harness
    pub use crate::{AgentHarness, HarnessConfig};

    // LLM clients and providers
    pub use crate::llm::{
        LlmClient, LlmClientBuilder, ModelCatalogEntry, ModelCatalogOptions, Provider,
        ProviderCatalogEntry, ProviderType, compiled_provider_types, discover_provider_catalog,
        discover_provider_models, flatten_catalog_models,
    };

    #[cfg(feature = "router")]
    pub use crate::llm::router::{
        CostOptimizedRouter, ModelCost, ProviderRouter, ProviderSelection, RoutingContext,
        StaticRouter, WeightedEntry, WeightedRouter,
    };

    // Message types
    pub use crate::messages::{
        ChatResponse, Message, MessageRole, TokenUsage, ToolChoice, ToolSchema,
    };

    // Tool call type (from agent::tools, not messages)
    pub use crate::agent::tools::ToolCall;

    // Tool system
    pub use crate::agent::public::tools::{Tool, ToolMeta, ToolResult, Toolkit};
    pub use crate::agent::tools::ToolRegistry;

    // Memory
    pub use crate::agent::memory::{MemoryBuilder, MemoryStrategy, MemoryType};

    // Session
    pub use crate::agent::session::AgentSession;

    // Hooks / lifecycle
    pub use crate::agent::hooks::{
        HookContext, HookMatcher, HookRegistry, HookResult, LifecycleHook,
        SteeringFollowUpProvider, permission::PermissionMode,
    };

    // Shared state
    pub use crate::SharedState;

    // Presets
    pub use crate::agent::presets::AgentPresets;

    // Error types
    pub use crate::errors::{AgentError, ErrorContext, ErrorContextExt, ToolError};

    // Trait abstractions
    pub use crate::traits::{
        CacheBackend, CacheStats, ConfigSource, CredentialProvider, Credentials, HealthStatus,
        InputValidator, ObservabilityHook, OutputValidator, Resource, ResourceManager, RetryError,
        RetryStrategy, RetryStrategyBuilder, StreamTransformer, ValidationResult,
    };

    #[cfg(feature = "chains")]
    pub use crate::chains::{Chain, ChainBuilder, ChainError, ChainResult, ComposableChain};

    #[cfg(feature = "tools-rag")]
    pub use crate::data::storage::vector::{VectorBackendConfig, chunking};

    #[cfg(feature = "reasoning-patterns")]
    pub use crate::reasoning::{
        ReasoningPattern, ReasoningResult, ReasoningStep, StepType,
        plan_execute::PlanExecutePattern, react::ReActPattern, reflexion::ReflexionPattern,
        self_ask::SelfAskPattern, tree_of_thoughts::TreeOfThoughtsPattern,
    };

    #[cfg(feature = "lsp")]
    pub use crate::lsp::{
        Diagnostic, DiagnosticFormatter, DiagnosticSeverity, LspConfig, LspManager,
        LspServerRegistry, Position, Range, SymbolInfo, SymbolKind,
    };
}
