use crate::agent::tools::credentials;
use crate::utils::beef_helpers::{deserialize_cow_str, serialize_cow_str};
use crate::utils::user_context::UserContext;
use async_trait::async_trait;
use beef::lean::Cow;
use serde::{Deserialize, Serialize};

/// Declares what kind of work a tool can do.
///
/// Returned by [`ToolMeta::capabilities`] so the execution engine can apply
/// the right middleware (sandboxing, audit logging, credential injection, etc.)
/// without hard-coding a list of tool names in every callsite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCapability {
    /// The tool has no observable write side-effects (safe to run in ask/read-only modes).
    ReadOnly,
    /// The tool executes arbitrary processes and should run inside an OS sandbox.
    SandboxExecution,
    /// The tool reaches out to external services and needs credential injection.
    RequiresCredentials,
    /// The tool reads file content that can be chunked and indexed for RAG.
    RagEnrichment,
    /// Every invocation of this tool is written to the audit log.
    Audited,
}

/// Synchronous metadata access for tools.
///
/// Implement this trait to describe a tool's identity and parameter schema without
/// requiring async infrastructure. All types that implement [`Tool`] automatically
/// satisfy `ToolMeta` (via the blanket impl below), so code that only inspects tool
/// metadata (e.g. schema listing, help generation) can use the lighter `ToolMeta`
/// bound instead of the full async `Tool` bound.
///
/// # Example
///
/// ```rust,no_run
/// use caliche::agent::tools::ToolMeta;
///
/// fn list_tool_names<'a>(tools: &'a [&'a dyn ToolMeta]) -> Vec<&'a str> {
///     tools.iter().map(|t| t.name()).collect()
/// }
/// ```
pub trait ToolMeta: Send + Sync + std::fmt::Debug {
    /// Unique machine-readable name used when calling this tool.
    fn name(&self) -> &str;

    /// Human-readable description shown to the LLM.
    fn description(&self) -> &str;

    /// JSON schema for the tool's parameters (`{"type":"object","properties":{...}}`).
    fn parameters(&self) -> serde_json::Value;

    /// Return `name` as a [`Cow`] for zero-copy schema building.
    ///
    /// Override with `Cow::borrowed("literal")` when the name is a `'static` string.
    fn name_as_cow(&self) -> Cow<'static, str> {
        Cow::owned(self.name().to_string())
    }

    /// Return `description` as a [`Cow`] for zero-copy schema building.
    ///
    /// Override with `Cow::borrowed("literal")` when the description is a `'static` string.
    fn description_as_cow(&self) -> Cow<'static, str> {
        Cow::owned(self.description().to_string())
    }

    /// Returns `true` if the tool has no observable side effects.
    ///
    /// Default implementation uses the built-in write-tool blocklist. Override to
    /// provide a more precise answer for custom tools.
    fn is_readonly(&self) -> bool {
        !crate::agent::modes::is_write_tool(self.name())
    }

    /// Declares capabilities used by the execution engine middleware.
    fn capabilities(&self) -> Vec<ToolCapability> {
        let mut out = Vec::new();
        if self.is_readonly() {
            out.push(ToolCapability::ReadOnly);
        }
        match self.name() {
            "fs" => {
                out.push(ToolCapability::SandboxExecution);
                out.push(ToolCapability::RagEnrichment);
                out.push(ToolCapability::Audited);
            }
            "bash" | "execute_python" => out.push(ToolCapability::SandboxExecution),
            "http" | "sql_execution" => out.push(ToolCapability::RequiresCredentials),
            _ => {}
        }
        out
    }
}

/// A request from the LLM to invoke a specific tool.
///
/// `name` is a [`Cow<'static, str>`] so callers that have a `'static` string
/// literal (the common case for built-in tools) can borrow it without
/// allocating, while runtime-generated names can still be stored as owned
/// `String`s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// The tool's machine-readable name (matches [`ToolMeta::name`]).
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub name: Cow<'static, str>,
    /// JSON arguments as supplied by the LLM.
    pub arguments: serde_json::Value,
}

/// The outcome of a tool invocation, returned to the agent's chat loop.
///
/// Construct via the infallible helpers [`ToolResult::success`],
/// [`ToolResult::success_with_usage`], [`ToolResult::success_with_warnings`],
/// or [`ToolResult::error`] rather than building the struct directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// `true` when the tool completed its work without a hard error.
    pub success: bool,
    /// Human-readable output presented to the LLM (the "tool output" in the conversation).
    pub content: String,
    /// Optional structured payload that the framework can inspect without parsing `content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typed_payload: Option<serde_json::Value>,
    /// Error description when `success == false`; `None` on success.
    pub error: Option<String>,
    /// Token usage charged by any sub-LLM calls made inside the tool (e.g. an embedded RAG call).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<crate::messages::TokenUsage>,
    /// Non-fatal advisory messages (e.g. robot-exclusion notices, deprecation warnings).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<Vec<String>>,
}

impl ToolResult {
    /// Create a successful tool result with the given content.
    ///
    /// Accepts any type that implements `Into<String>`, so callers can pass
    /// `&str`, `String`, `Cow<str>`, or `format!(…)` directly.
    #[must_use]
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            success: true,
            content: content.into(),
            typed_payload: None,
            error: None,
            usage: None,
            warnings: None,
        }
    }

    /// Create a successful tool result with token usage metadata.
    #[must_use]
    pub fn success_with_usage(
        content: impl Into<String>,
        usage: crate::messages::TokenUsage,
    ) -> Self {
        Self {
            success: true,
            content: content.into(),
            typed_payload: None,
            error: None,
            usage: Some(usage),
            warnings: None,
        }
    }

    /// Create a successful tool result with warning annotations.
    #[must_use]
    pub fn success_with_warnings(content: impl Into<String>, warnings: Vec<String>) -> Self {
        Self {
            success: true,
            content: content.into(),
            typed_payload: None,
            error: None,
            usage: None,
            warnings: Some(warnings),
        }
    }

    /// Create a successful tool result with structured metadata.
    #[must_use]
    pub fn success_with_payload(
        content: impl Into<String>,
        typed_payload: serde_json::Value,
    ) -> Self {
        Self {
            success: true,
            content: content.into(),
            typed_payload: Some(typed_payload),
            error: None,
            usage: None,
            warnings: None,
        }
    }

    /// Create a successful tool result with both token usage and warnings.
    #[must_use]
    pub fn success_with_usage_and_warnings(
        content: impl Into<String>,
        usage: crate::messages::TokenUsage,
        warnings: Vec<String>,
    ) -> Self {
        Self {
            success: true,
            content: content.into(),
            typed_payload: None,
            error: None,
            usage: Some(usage),
            warnings: Some(warnings),
        }
    }

    /// Create a failed tool result with the given error description.
    ///
    /// Accepts any type that implements [`std::fmt::Display`], including [`anyhow::Error`].
    #[must_use]
    pub fn error(error: impl std::fmt::Display) -> Self {
        Self {
            success: false,
            content: String::new(),
            typed_payload: None,
            error: Some(error.to_string()),
            usage: None,
            warnings: None,
        }
    }
}

/// The core tool interface used by the agent.
///
/// Implement [`ToolMeta`] for the synchronous metadata (name, description, parameters),
/// then implement this trait for async execution. The two-trait split lets code that
/// only inspects tool schemas use the lighter `ToolMeta` bound.
///
/// # Example
///
/// ```rust,no_run
/// use caliche::{Tool, ToolResult};
/// use caliche::agent::tools::ToolMeta;
/// use async_trait::async_trait;
///
/// #[derive(Debug)]
/// struct GreetTool;
///
/// impl ToolMeta for GreetTool {
///     fn name(&self) -> &str { "greet" }
///     fn description(&self) -> &str { "Greet the user by name" }
///     fn parameters(&self) -> serde_json::Value {
///         serde_json::json!({"type":"object","properties":{"name":{"type":"string"}}})
///     }
/// }
///
/// #[async_trait]
/// impl Tool for GreetTool {
///     async fn execute(&self, args: serde_json::Value) -> ToolResult {
///         let name = args["name"].as_str().unwrap_or("world");
///         ToolResult::success(format!("Hello, {name}!"))
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: ToolMeta {
    /// Execute the tool with the provided JSON arguments.
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult;

    /// Execute the tool with optional user context.
    ///
    /// The default implementation ignores the context and delegates to [`execute`](Tool::execute).
    /// Override this when the tool needs to perform audit logging, permission checks,
    /// or result filtering based on the caller's identity.
    async fn execute_with_context(
        &self,
        arguments: serde_json::Value,
        _context: Option<&UserContext>,
    ) -> ToolResult {
        self.execute(arguments).await
    }

    /// Execute the tool inside a sandbox when one is provided; otherwise delegates to
    /// [`execute`](Tool::execute).
    ///
    /// Override in bash/filesystem tools to run operations inside the sandbox boundary.
    #[cfg(feature = "sandbox")]
    async fn execute_sandboxed(
        &self,
        arguments: serde_json::Value,
        _sandbox: Option<&dyn caliche_sandbox::SandboxHandleOps>,
    ) -> ToolResult {
        self.execute(arguments).await
    }
}

/// Returns the shared credential parameter schema block.
///
/// Merge the returned JSON object into a tool's own `parameters()` object to
/// give the LLM a consistent set of fields (`credential_id`, `api_key`, etc.)
/// for supplying runtime credentials.
#[must_use]
pub fn credential_parameters() -> serde_json::Value {
    credentials::credential_parameters()
}
