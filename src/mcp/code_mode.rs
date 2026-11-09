//! Code-mode MCP server support.

use crate::agent::tools::{Tool, ToolMeta, ToolRegistry, ToolResult};
use crate::mcp::server::{McpServer, McpServerBuilder};
use crate::templates::TemplateRegistry;
use anyhow::Result;
use async_trait::async_trait;
use beef::lean::Cow;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use caliche_sandbox::{
    CodeModeExecuteContext, CodeModeSandboxConfig as SandboxExecConfig, NamedPolicy,
    SandboxBackendType, SandboxConfig, run_execute, run_execute_with_api, run_search,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeModeSchema {
    pub operations: HashMap<String, OperationDef>,
}

impl CodeModeSchema {
    pub fn as_json(&self) -> Value {
        json!({ "operations": self.operations })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationDef {
    pub description: String,
    pub parameters: Value,
    pub returns: String,
}

#[derive(Debug, Clone)]
pub struct CodeModeAuth {
    pub scheme: String,
    pub token: String,
}

#[derive(Debug, Clone)]
pub struct CodeModeSandboxConfig {
    pub sandbox: SandboxConfig,
    pub workspace: PathBuf,
    pub runtime_alias: String,
    pub timeout: Duration,
}

impl CodeModeSandboxConfig {
    /// Default sandbox timeout applied when callers don't override it.
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

    /// Build a sandbox config explicitly scoped to `workspace`.
    ///
    /// This is the preferred way to construct a [`CodeModeSandboxConfig`]
    /// because it avoids the hidden `std::env::current_dir()` probe that
    /// [`Default`] falls back to (which can fail or point somewhere
    /// surprising, e.g. in tests or daemonized contexts).
    pub fn for_workspace(workspace: impl Into<PathBuf>) -> Self {
        let workspace = workspace.into();
        let mut sandbox = caliche_sandbox::policy_to_config(
            NamedPolicy::WorkspaceWrite,
            &workspace,
            None,
            SandboxBackendType::Noop,
        );
        sandbox.backend = SandboxBackendType::WasmWasmtime {
            policy: NamedPolicy::WorkspaceWrite,
            modules: HashMap::new(),
        };
        Self {
            sandbox,
            workspace,
            runtime_alias: "js".to_string(),
            timeout: Self::DEFAULT_TIMEOUT,
        }
    }

    fn as_sandbox_exec_config(&self) -> SandboxExecConfig {
        SandboxExecConfig {
            sandbox: self.sandbox.clone(),
            workspace: self.workspace.clone(),
            runtime_alias: self.runtime_alias.clone(),
            timeout: self.timeout,
        }
    }
}

impl Default for CodeModeSandboxConfig {
    /// Falls back to the current working directory when no workspace is
    /// specified. Callers that need deterministic behaviour (tests,
    /// background services) should use [`CodeModeSandboxConfig::for_workspace`]
    /// instead so the workspace path is explicit.
    fn default() -> Self {
        let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self::for_workspace(workspace)
    }
}

#[derive(Debug, Clone)]
pub struct CodeModeOptions {
    pub schema: CodeModeSchema,
    pub api_base_url: String,
    pub api_auth: Option<CodeModeAuth>,
    pub sandbox_config: CodeModeSandboxConfig,
}

pub struct CodeModeMcpServerBuilder {
    name: Cow<'static, str>,
    version: Option<Cow<'static, str>>,
    schema: Option<CodeModeSchema>,
    api_base_url: Option<String>,
    api_auth: Option<CodeModeAuth>,
    sandbox_config: CodeModeSandboxConfig,
    template_registry: Option<TemplateRegistry>,
    enable_prompts: bool,
    enable_resources: bool,
}

impl Default for CodeModeMcpServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeModeMcpServerBuilder {
    pub fn new() -> Self {
        Self {
            name: Cow::borrowed("caliche"),
            version: Some(Cow::borrowed(env!("CARGO_PKG_VERSION"))),
            schema: None,
            api_base_url: None,
            api_auth: None,
            sandbox_config: CodeModeSandboxConfig::default(),
            template_registry: None,
            enable_prompts: false,
            enable_resources: false,
        }
    }

    pub fn name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.name = name.into();
        self
    }

    pub fn version(mut self, version: impl Into<Cow<'static, str>>) -> Self {
        self.version = Some(version.into());
        self
    }

    pub fn schema(mut self, schema: CodeModeSchema) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn api_base_url(mut self, api_base_url: impl Into<String>) -> Self {
        self.api_base_url = Some(api_base_url.into());
        self
    }

    pub fn api_auth(mut self, api_auth: Option<CodeModeAuth>) -> Self {
        self.api_auth = api_auth;
        self
    }

    pub fn sandbox_config(mut self, config: CodeModeSandboxConfig) -> Self {
        self.sandbox_config = config;
        self
    }

    pub fn template_registry(mut self, registry: TemplateRegistry) -> Self {
        self.template_registry = Some(registry);
        self
    }

    pub fn enable_prompts(mut self, enabled: bool) -> Self {
        self.enable_prompts = enabled;
        self
    }

    pub fn enable_resources(mut self, enabled: bool) -> Self {
        self.enable_resources = enabled;
        self
    }

    pub fn build(self) -> Result<McpServer> {
        let schema = self.schema.unwrap_or_else(|| CodeModeSchema {
            operations: HashMap::new(),
        });
        let api_base_url = self
            .api_base_url
            .unwrap_or_else(|| "http://127.0.0.1".to_string());

        let registry = Arc::new(ToolRegistry::new());
        let sandbox_exec_config = self.sandbox_config.as_sandbox_exec_config();
        let schema_json = schema.as_json();
        let api_descriptor = json!({
            "base_url": api_base_url,
            "operations": schema.operations,
            "auth": self.api_auth.as_ref().map(|auth| json!({"scheme": auth.scheme, "token": auth.token}))
        });

        registry.register(Arc::new(CodeModeSearchTool {
            schema: schema_json,
            sandbox_config: sandbox_exec_config.clone(),
        }));
        registry.register(Arc::new(CodeModeExecuteTool {
            api_descriptor,
            sandbox_config: sandbox_exec_config,
        }));

        let mut builder = McpServerBuilder::new()
            .name(self.name)
            .enable_tools(true)
            .enable_prompts(self.enable_prompts)
            .enable_resources(self.enable_resources)
            .tool_registry(registry);
        if let Some(version) = self.version {
            builder = builder.version(version);
        }
        if let Some(template_registry) = self.template_registry {
            builder = builder.template_registry(template_registry);
        }
        builder.build()
    }
}

#[derive(Debug)]
struct CodeModeSearchTool {
    schema: Value,
    sandbox_config: SandboxExecConfig,
}

impl ToolMeta for CodeModeSearchTool {
    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Run JavaScript discovery code against the code-mode operations schema."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "JavaScript async code body to execute against the schema object."
                }
            },
            "required": ["code"]
        })
    }
}

#[async_trait]
impl Tool for CodeModeSearchTool {
    async fn execute(&self, arguments: Value) -> ToolResult {
        let code = arguments
            .get("code")
            .and_then(Value::as_str)
            .map(ToString::to_string)
            .unwrap_or_default();
        if code.trim().is_empty() {
            return ToolResult::error("missing required argument: code".to_string());
        }

        match run_search(&self.sandbox_config, &code, &self.schema).await {
            Ok(result) => match serde_json::to_string_pretty(&result) {
                Ok(serialized) => ToolResult::success(serialized),
                Err(err) => ToolResult::error(format!("serialize search result: {err}")),
            },
            Err(err) => ToolResult::error(format!("code-mode search failed: {err}")),
        }
    }
}

#[derive(Debug)]
struct CodeModeExecuteTool {
    api_descriptor: Value,
    sandbox_config: SandboxExecConfig,
}

impl ToolMeta for CodeModeExecuteTool {
    fn name(&self) -> &str {
        "execute"
    }

    fn description(&self) -> &str {
        "Run JavaScript operation code against an authenticated API client descriptor."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "JavaScript async code body to execute against the api object."
                }
            },
            "required": ["code"]
        })
    }
}

#[async_trait]
impl Tool for CodeModeExecuteTool {
    async fn execute(&self, arguments: Value) -> ToolResult {
        let code = arguments
            .get("code")
            .and_then(Value::as_str)
            .map(ToString::to_string)
            .unwrap_or_default();
        if code.trim().is_empty() {
            return ToolResult::error("missing required argument: code".to_string());
        }

        let execution = match (
            self.api_descriptor
                .get("base_url")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            self.api_descriptor
                .get("auth")
                .and_then(|value| value.get("token"))
                .and_then(Value::as_str)
                .map(ToString::to_string),
        ) {
            (Some(api_base_url), Some(api_key)) => {
                let context = CodeModeExecuteContext {
                    api_base_url,
                    api_key,
                };
                run_execute_with_api(&self.sandbox_config, &code, &self.api_descriptor, &context)
                    .await
            }
            _ => run_execute(&self.sandbox_config, &code, &self.api_descriptor).await,
        };

        match execution {
            Ok(result) => match serde_json::to_string_pretty(&result) {
                Ok(serialized) => ToolResult::success(serialized),
                Err(err) => ToolResult::error(format!("serialize execute result: {err}")),
            },
            Err(err) => ToolResult::error(format!("code-mode execute failed: {err}")),
        }
    }
}
