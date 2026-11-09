//! Python bindings for `AgentBuilder`

use crate::agent::Agent;
use crate::builtin_tools::PyBuiltinTool;
use crate::conversions::python_to_json_value;
use crate::external_provider::{create_external_embedding_provider, create_external_provider};
use crate::get_runtime;
use crate::hooks::{PyHookMatcher, PyLifecycleHook, PyPermissionMode};
use crate::inspection::PyInspectionHook;
use crate::memory_config::{
    PyBufferMemoryConfig, PyEntityMemoryConfig, PySummaryMemoryConfig, PyVectorMemoryConfig,
};
use crate::memory_persistence::PyMemoryPersistence;
use crate::mutation::PyMutationStyle;
use crate::providers::ProviderType;
use crate::tool::PyTool;
use caliche::AgentBuilder as RustAgentBuilder;
use caliche::agent::hooks::lifecycle::LifecycleHook;
use caliche::agent::hooks::permission::PermissionMode;
use caliche::agent::memory::MemoryPersistence;
use caliche::agent::service::AgentService;
#[cfg(feature = "tools-rag")]
use caliche::agent::tools::RagTool;
use caliche::agent::tools::Tool;
use caliche::llm::ProviderType as RustProviderType;
use caliche::messages::ToolChoice;
use caliche::streams::DurableStreamBackend;
#[cfg(feature = "vector-store-postgres")]
use caliche::vecstore::{HnswConfig, HnswGraphSpace};
use caliche::vecstore::{
    SearchStrategy as DripstoneSearchStrategy, TierConfig as DripstoneTierConfig,
    VectorBackendConfig, WriteStrategy as DripstoneWriteStrategy, tier_config_from_json_value,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;

/// RAG tool sharing is only meaningful with the `tools-rag` feature; keep the
/// field present on `AgentBuilder` regardless of features.
#[cfg(feature = "tools-rag")]
type SharedRagTool = Option<Arc<RagTool>>;
#[cfg(not(feature = "tools-rag"))]
type SharedRagTool = Option<()>;
type ExternalProviderConfig = (String, String, Py<PyAny>, Py<PyAny>, bool, bool);

fn backend_dict_to_config(dict: &Bound<'_, PyDict>) -> PyResult<VectorBackendConfig> {
    let value = python_to_json_value(dict.as_any())?;
    VectorBackendConfig::from_json_value(&value).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

fn tier_dict_to_config(dict: &Bound<'_, PyDict>) -> PyResult<DripstoneTierConfig> {
    let value = python_to_json_value(dict.as_any())?;
    tier_config_from_json_value(&value).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyclass]
pub struct AgentBuilder {
    inner: RustAgentBuilder,
    // Store LLM parameters that aren't yet supported in Rust builder
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    // Store custom Python tools to add during build
    custom_tools: Vec<Arc<dyn Tool>>,
    // Store memory persistence configuration
    memory_persistence: Option<(Arc<dyn MemoryPersistence>, String, bool)>,
    #[cfg_attr(
        not(feature = "tools-rag"),
        expect(
            dead_code,
            reason = "only the tools-rag build attaches the shared RAG tool"
        )
    )]
    shared_rag_tool: SharedRagTool,
    // Store external provider callbacks (for LangChain, etc.)
    external_provider: Option<ExternalProviderConfig>,
    external_embedding_provider: Option<(String, Py<PyAny>)>,
    // Track if all default tools should be used
    use_all_default_tools: bool,
    // Track extended default tool set (external-provider build path)
    use_extended_default_tools: bool,
    // Store hooks (matcher -> hook) - stored separately to pass to Rust builder
    hooks: Vec<(
        caliche::agent::hooks::matcher::HookMatcher,
        Arc<dyn LifecycleHook>,
    )>,
    // Store permission mode
    permission_mode: PermissionMode,
    // File mutation style (fs write/edit vs apply_patch)
    mutation_style: caliche::agent::tools::MutationStyle,
    // When set, install Goose-style tool inspection at build time
    enable_tool_inspection: bool,
    inspection_workspace_root: Option<std::path::PathBuf>,
    // Store allowed tools list
    allowed_tools: Option<Vec<String>>,
    // Store in-process MCP servers
    mcp_servers: HashMap<String, Arc<caliche::mcp::in_process_server::InProcessMcpServer>>,
    // Store response format for structured outputs
    response_format: Option<caliche::output::ResponseFormat>,
    // Optional durable stream backend for event replay/resume (e.g. API harness)
    durable_stream_backend: Option<Arc<dyn DurableStreamBackend>>,
}

impl AgentBuilder {
    /// Create an agent builder from a Rust `AgentBuilder`
    pub(crate) fn from_rust_builder(rust_builder: RustAgentBuilder) -> Self {
        Self {
            inner: rust_builder,
            temperature: None,
            max_tokens: None,
            top_p: None,
            custom_tools: Vec::new(),
            memory_persistence: None,
            shared_rag_tool: None,
            external_provider: None,
            external_embedding_provider: None,
            use_all_default_tools: false,
            use_extended_default_tools: false,
            hooks: Vec::new(),
            permission_mode: PermissionMode::Default,
            mutation_style: caliche::agent::tools::MutationStyle::EditWrite,
            enable_tool_inspection: false,
            inspection_workspace_root: None,
            allowed_tools: None,
            mcp_servers: HashMap::new(),
            response_format: None,
            durable_stream_backend: None,
        }
    }

    fn take_configured_builder(&mut self) -> RustAgentBuilder {
        let mut builder = std::mem::take(&mut self.inner);
        for tool in &self.custom_tools {
            builder = builder.add_tool(Arc::clone(tool));
        }
        if let Some((persistence, conversation_id, autosave)) = &self.memory_persistence {
            builder = builder.persist_memory_with(
                Arc::clone(persistence),
                conversation_id.clone(),
                *autosave,
            );
        }
        #[cfg(feature = "tools-rag")]
        if let Some(ref rag_tool) = self.shared_rag_tool {
            builder = builder.shared_rag_tool(Arc::clone(rag_tool));
        }
        for (matcher, hook) in &self.hooks {
            builder = builder.hook(matcher.clone(), Arc::clone(hook));
        }
        builder = builder
            .permission_mode(self.permission_mode)
            .mutation_style(self.mutation_style);
        if self.enable_tool_inspection {
            builder = builder.enable_tool_inspection(self.inspection_workspace_root.clone());
        }
        if let Some(ref tools) = self.allowed_tools {
            let tool_names: Vec<&str> = tools.iter().map(String::as_str).collect();
            builder = builder.allowed_tools(&tool_names);
        }
        for (server_name, server) in &self.mcp_servers {
            builder = builder.mcp_server(server_name.clone(), (**server).clone());
        }
        if let Some(ref backend) = self.durable_stream_backend {
            builder = builder.durable_stream_backend(Arc::clone(backend));
        }
        builder
    }
}

#[pymethods]
impl AgentBuilder {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RustAgentBuilder::new(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            custom_tools: Vec::new(),
            memory_persistence: None,
            shared_rag_tool: None,
            external_provider: None,
            external_embedding_provider: None,
            use_all_default_tools: false,
            use_extended_default_tools: false,
            hooks: Vec::new(),
            permission_mode: PermissionMode::Default,
            mutation_style: caliche::agent::tools::MutationStyle::EditWrite,
            enable_tool_inspection: false,
            inspection_workspace_root: None,
            allowed_tools: None,
            mcp_servers: HashMap::new(),
            response_format: None,
            durable_stream_backend: None,
        }
    }

    /// Set an external provider using Python callables (for `LangChain`, etc.)
    ///
    /// # Arguments
    /// * `name` - Provider name (e.g., "langchain-openai")
    /// * `model_name` - Model name
    /// * `complete_fn` - Python callable that accepts a `CompletionRequest` dict and returns a `CompletionResponse` dict
    /// * `stream_fn` - Python callable that accepts a `CompletionRequest` dict and returns an async iterator of `StreamEvents`
    /// * `supports_native_tools` - Whether the provider supports native tool calling
    /// * `supports_streaming` - Whether the provider supports streaming
    ///
    /// # Example
    /// ```python
    /// async def complete(request):
    ///     # Process request and return response
    ///     return {"content": "response", "finish_reason": "stop", "usage": None}
    ///
    /// async def stream(request):
    ///     # Yield stream events
    ///     yield {"type": "text_chunk", "content": "chunk"}
    ///
    /// builder = AgentBuilder() \\
    ///     .external_provider("my-provider", "my-model", complete, stream, True, True)
    /// ```
    pub fn external_provider(
        mut slf: PyRefMut<Self>,
        name: String,
        model_name: String,
        complete_fn: Py<PyAny>,
        stream_fn: Py<PyAny>,
        supports_native_tools: bool,
        supports_streaming: bool,
    ) -> PyRefMut<Self> {
        slf.external_provider = Some((
            name,
            model_name,
            complete_fn,
            stream_fn,
            supports_native_tools,
            supports_streaming,
        ));
        slf
    }

    /// Set an external embedding provider using a Python callable
    ///
    /// # Arguments
    /// * `name` - Provider name
    /// * `embed_fn` - Python callable that accepts an `EmbeddingRequest` dict and returns an `EmbeddingResponse` dict
    pub fn external_embedding_provider(
        mut slf: PyRefMut<Self>,
        name: String,
        embed_fn: Py<PyAny>,
    ) -> PyRefMut<Self> {
        slf.external_embedding_provider = Some((name, embed_fn));
        slf
    }

    /// Set the provider
    pub fn provider(mut slf: PyRefMut<Self>, provider: ProviderType) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.provider(provider.into());
        slf
    }

    /// Set the model
    pub fn model(mut slf: PyRefMut<Self>, model: String) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.model(model);
        slf
    }

    /// Set the system prompt
    pub fn system_prompt(mut slf: PyRefMut<Self>, prompt: String) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.system_prompt(prompt);
        slf
    }

    /// Control whether runtime context is injected into the system prompt.
    ///
    /// Benchmarks and thin wrappers typically disable this to match raw SDK
    /// request shapes (`include_context_in_system(false)`).
    pub fn include_context_in_system(mut slf: PyRefMut<Self>, value: bool) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.include_context_in_system(value);
        slf
    }

    /// Enable JSON object mode (any valid JSON response)
    pub fn json_mode(mut slf: PyRefMut<Self>) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.json_mode();
        slf.response_format = Some(caliche::output::ResponseFormat::JsonObject);
        slf
    }

    /// Enable JSON schema mode with validation
    #[pyo3(signature = (name, schema, strict=true))]
    pub fn json_schema<'a>(
        mut slf: PyRefMut<'a, Self>,
        name: String,
        schema: &Bound<'_, PyDict>,
        strict: bool,
    ) -> PyResult<PyRefMut<'a, Self>> {
        use pythonize::depythonize;
        let schema_json: serde_json::Value = depythonize(schema)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid JSON schema: {e}")))?;

        // Normalize schema to ensure OpenAI compatibility (adds additionalProperties: false if needed)
        let normalized_schema =
            caliche::output::JsonSchemaDefinition::normalize_schema(schema_json);

        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.json_schema(&name, normalized_schema.clone());
        slf.response_format = Some(caliche::output::ResponseFormat::JsonSchema {
            json_schema: caliche::output::JsonSchemaDefinition {
                name,
                description: None,
                schema: normalized_schema,
                strict,
            },
        });
        Ok(slf)
    }

    /// Enable a tool
    pub fn enable_tool<'py>(mut slf: PyRefMut<'py, Self>, tool_name: &str) -> PyRefMut<'py, Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.enable_tool(tool_name);
        slf
    }

    /// Enable all default tools
    pub fn enable_all_default_tools(mut slf: PyRefMut<Self>) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.enable_all_default_tools();
        slf.use_all_default_tools = true;
        slf
    }

    /// Enable the broader compatibility default tool set (fs, bash, http, workspace, `execute_python`).
    pub fn enable_extended_default_tools(mut slf: PyRefMut<Self>) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.enable_extended_default_tools();
        slf.use_all_default_tools = false;
        slf.use_extended_default_tools = true;
        slf
    }

    /// Set the permission mode for tool execution
    pub fn permission_mode(mut slf: PyRefMut<Self>, mode: PyPermissionMode) -> PyRefMut<Self> {
        slf.permission_mode = mode.into();
        slf
    }

    /// Select file mutation style (`EditWrite` XOR `ApplyPatch`).
    pub fn mutation_style(mut slf: PyRefMut<Self>, style: PyMutationStyle) -> PyRefMut<Self> {
        slf.mutation_style = style.into();
        slf
    }

    /// Enable the Goose-style tool inspection chain (path security + permission mode).
    #[pyo3(signature = (workspace_root=None))]
    pub fn enable_tool_inspection(
        mut slf: PyRefMut<Self>,
        workspace_root: Option<String>,
    ) -> PyRefMut<Self> {
        slf.enable_tool_inspection = true;
        slf.inspection_workspace_root = workspace_root.map(std::path::PathBuf::from);
        slf
    }

    /// Enable LSP diagnostic/hover/symbol tools (included in the Rust `app` feature set).
    #[cfg(feature = "lsp")]
    pub fn enable_lsp_tools(mut slf: PyRefMut<Self>) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.enable_lsp_tools();
        slf
    }

    /// Set the list of allowed tools (only these tools will be available)
    pub fn allowed_tools(mut slf: PyRefMut<Self>, tools: Vec<String>) -> PyRefMut<Self> {
        slf.allowed_tools = Some(tools);
        slf
    }

    /// Register a lifecycle hook with a matcher
    pub fn hook<'py>(
        mut slf: PyRefMut<'py, Self>,
        matcher: PyHookMatcher,
        hook_obj: &Bound<'_, PyAny>,
    ) -> PyRefMut<'py, Self> {
        let hook: Arc<dyn LifecycleHook> = {
            let fallback = hook_obj.clone().unbind();
            hook_obj
                .extract::<PyRef<'_, PyInspectionHook>>()
                .map_or_else(
                    |_| Arc::new(PyLifecycleHook::new(fallback)) as Arc<dyn LifecycleHook>,
                    |inspection| inspection.as_lifecycle_hook(),
                )
        };
        slf.hooks.push((matcher.into(), hook));
        slf
    }

    /// Register an in-process MCP server
    ///
    /// Accepts either a `PyInProcessMcpServer` (Rust `PyClass`) or a Python wrapper object
    pub fn mcp_server<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: String,
        server: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // Extract the inner server Arc
        let server_arc =
            (|| -> PyResult<Arc<caliche::mcp::in_process_server::InProcessMcpServer>> {
                // First try to get it as PyInProcessMcpServer directly
                if let Ok(py_server) =
                    server.extract::<PyRef<'_, crate::in_process_mcp::PyInProcessMcpServer>>()
                {
                    return Ok(py_server.get_inner());
                }

                // Try to get it from _server attribute (for Python wrapper)
                if let Ok(wrapper) = server.getattr("_server")
                    && let Ok(py_server) =
                        wrapper.extract::<PyRef<'_, crate::in_process_mcp::PyInProcessMcpServer>>()
                {
                    return Ok(py_server.get_inner());
                }

                Err(PyRuntimeError::new_err(
                    "server must be a PyInProcessMcpServer or InProcessMcpServer wrapper with _server attribute",
                ))
            })()?;

        slf.mcp_servers.insert(name, server_arc);
        Ok(slf)
    }

    /// Set temperature for LLM completion.
    pub fn temperature(mut slf: PyRefMut<Self>, temp: f32) -> PyRefMut<Self> {
        slf.temperature = Some(temp);
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.temperature(temp);
        slf
    }

    /// Set `max_tokens` for LLM completion.
    pub fn max_tokens(mut slf: PyRefMut<Self>, tokens: u32) -> PyRefMut<Self> {
        slf.max_tokens = Some(tokens);
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.max_tokens(tokens);
        slf
    }

    /// Set `top_p` (stored for API parity; core builder does not yet apply it).
    pub fn top_p(mut slf: PyRefMut<Self>, top_p: f32) -> PyRefMut<Self> {
        slf.top_p = Some(top_p);
        slf
    }

    /// Set the maximum number of tool-call iterations.
    pub fn max_iterations(mut slf: PyRefMut<Self>, max_iterations: usize) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.max_iterations(max_iterations);
        slf
    }

    /// Control how the LLM uses tools (`auto`, `none`, `required`, or a tool name).
    pub fn tool_choice(mut slf: PyRefMut<Self>, choice: String) -> PyRefMut<Self> {
        let rust_choice = match choice.to_lowercase().as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => ToolChoice::Specific(beef::lean::Cow::owned(choice)),
        };
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.tool_choice(Some(rust_choice));
        slf
    }

    /// Enable/disable filesystem MCP auto-discovery during build (default: disabled).
    pub fn auto_discover_mcp(mut slf: PyRefMut<Self>, enabled: bool) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.auto_discover_mcp(enabled);
        slf
    }

    /// Enable/disable on-disk skill auto-indexing during build (default: disabled).
    pub fn auto_discover_skills(mut slf: PyRefMut<Self>, enabled: bool) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.auto_discover_skills(enabled);
        slf
    }

    /// Set API key for authentication
    pub fn api_key(mut slf: PyRefMut<Self>, key: String) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.api_key(key);
        slf
    }

    /// Set AWS credentials for Bedrock provider
    pub fn aws_credentials(
        mut slf: PyRefMut<Self>,
        access_key: String,
        secret_key: String,
        region: String,
    ) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.aws_credentials(access_key, secret_key, region);
        slf
    }

    /// Set AWS credentials with session token
    pub fn aws_credentials_with_token(
        mut slf: PyRefMut<Self>,
        access_key: String,
        secret_key: String,
        session_token: String,
        region: String,
    ) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner =
            builder.aws_credentials_with_token(access_key, secret_key, session_token, region);
        slf
    }

    /// Set base URL for `LMStudio` provider
    pub fn base_url(mut slf: PyRefMut<Self>, url: String) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.base_url(url);
        slf
    }

    /// Set Azure `OpenAI` configuration
    ///
    /// # Arguments
    /// * `resource_name` - Azure resource name
    /// * `deployment_id` - Deployment ID/name
    /// * `api_key` - Optional API key (can also be set via `api_key()` method or environment variable)
    /// * `api_version` - Optional API version (defaults to "2024-10-21")
    ///
    /// # Example
    /// ```python
    /// builder = AgentBuilder() \
    ///     .provider(ProviderType.AzureOpenAI) \
    ///     .model("gpt-4o") \
    ///     .azure_openai_config("my-resource", "my-deployment", api_key="key", api_version="2024-10-21")
    /// ```
    #[pyo3(signature = (_resource_name, _deployment_id, api_key=None, _api_version=None))]
    pub fn azure_openai_config(
        mut slf: PyRefMut<Self>,
        _resource_name: String,
        _deployment_id: String,
        api_key: Option<String>,
        _api_version: Option<String>,
    ) -> PyRefMut<Self> {
        // Set API key if provided
        if let Some(key) = api_key {
            let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
            slf.inner = builder.api_key(key);
        }
        // Note: Azure-specific config (resource_name, deployment_id, api_version)
        // will need to be stored separately or passed through environment variables
        // This is a placeholder - full implementation would require extending AgentBuilder
        // For now, users should set AZURE_OPENAI_RESOURCE_NAME, AZURE_OPENAI_DEPLOYMENT_ID, etc.
        slf
    }

    /// Configure buffer memory
    pub fn buffer_memory(mut slf: PyRefMut<Self>, config: PyBufferMemoryConfig) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.buffer_memory(config.into());
        slf
    }

    /// Configure summary memory
    pub fn summary_memory(
        mut slf: PyRefMut<Self>,
        config: PySummaryMemoryConfig,
    ) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.summary_memory(config.into());
        slf
    }

    /// Configure entity memory
    pub fn entity_memory(mut slf: PyRefMut<Self>, config: PyEntityMemoryConfig) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.entity_memory(config.into());
        slf
    }

    /// Configure vector-based semantic memory
    pub fn vector_memory(mut slf: PyRefMut<Self>, config: PyVectorMemoryConfig) -> PyRefMut<Self> {
        let builder = std::mem::replace(&mut slf.inner, RustAgentBuilder::new());
        slf.inner = builder.vector_memory(config.into());
        slf
    }

    /// Configure enhanced memory (buffer base + budget, relevance, compression, working memory, knowledge).
    pub fn with_enhanced_memory(mut slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().with_enhanced_memory();
        slf
    }

    /// Add a custom tool
    ///
    /// The tool object must implement the Tool protocol with methods:
    /// - `name()` -> str
    /// - `description()` -> str
    /// - `parameters()` -> dict
    /// - execute(arguments: dict) -> `ToolResult` or dict or str
    pub fn add_tool<'py>(
        mut slf: PyRefMut<'py, Self>,
        tool_obj: &Bound<'_, PyAny>,
    ) -> PyRefMut<'py, Self> {
        let tool: Arc<dyn Tool> = {
            let fallback = tool_obj.clone().unbind();
            tool_obj.extract::<PyRef<'_, PyBuiltinTool>>().map_or_else(
                |_| Arc::new(PyTool::new(fallback)) as Arc<dyn Tool>,
                |builtin| Arc::clone(&builtin.inner),
            )
        };
        slf.custom_tools.push(tool);
        slf
    }

    /// Register a toolkit (collection of related tools)
    ///
    /// The toolkit object must implement the Toolkit protocol with methods:
    /// - `get_tools()` -> List[Tool]
    /// - `name()` -> str
    ///
    /// Note: Currently only `FileSystemToolkit` is available as a built-in toolkit.
    /// Custom toolkits can be created by implementing the Toolkit protocol.
    pub fn register_toolkit<'py>(
        mut slf: PyRefMut<'py, Self>,
        toolkit_obj: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        {
            // Try to get tools from the toolkit
            if let Ok(get_tools_method) = toolkit_obj.getattr("get_tools")
                && let Ok(tools_list) = get_tools_method.call0()
                && let Ok(tools) = tools_list.extract::<Vec<Py<PyAny>>>()
            {
                for tool_obj in tools {
                    let py_tool = Arc::new(PyTool::new(tool_obj)) as Arc<dyn Tool>;
                    slf.custom_tools.push(py_tool);
                }
                return Ok(slf);
            }

            Err(PyRuntimeError::new_err(
                "Toolkit must implement get_tools() method that returns a list of Tool objects",
            ))
        }
    }

    /// Persist memory using a custom persistence backend
    ///
    /// The persistence object must implement the `MemoryPersistence` protocol with methods:
    /// - `save_snapshot(conversation_id`: str, snapshot: dict) -> None or coroutine
    /// - `load_snapshot(conversation_id`: str) -> Optional[dict] or coroutine
    /// - `clear_snapshot(conversation_id`: str) -> None or coroutine
    pub fn persist_memory_with(
        mut slf: PyRefMut<Self>,
        persistence_obj: Py<PyAny>,
        conversation_id: String,
        autosave: bool,
    ) -> PyRefMut<Self> {
        let py_persistence =
            Arc::new(PyMemoryPersistence::new(persistence_obj)) as Arc<dyn MemoryPersistence>;
        slf.memory_persistence = Some((py_persistence, conversation_id, autosave));
        slf
    }

    #[pyo3(signature = (ttl_secs=None, max_entries=None))]
    pub fn set_vector_backend_memory(
        mut slf: PyRefMut<Self>,
        ttl_secs: Option<u64>,
        max_entries: Option<usize>,
    ) -> PyRefMut<Self> {
        let config = VectorBackendConfig::Memory {
            ttl_secs: ttl_secs.unwrap_or(86400),
            max_entries,
        };
        slf.inner = slf.inner.clone().vector_backend_config(config);
        slf
    }

    #[cfg(feature = "vector-store-redis")]
    pub fn set_vector_backend_redis(
        mut slf: PyRefMut<Self>,
        url: String,
        prefix: String,
    ) -> PyRefMut<Self> {
        let config = VectorBackendConfig::Redis { url, prefix };
        slf.inner = slf.inner.clone().vector_backend_config(config);
        slf
    }

    #[cfg(feature = "vector-store-postgres")]
    #[pyo3(signature = (connection_string, table_name, vector_dimension, max_connections, hnsw_config=None))]
    pub fn set_vector_backend_postgres<'py>(
        mut slf: PyRefMut<'py, Self>,
        connection_string: String,
        table_name: String,
        vector_dimension: Option<usize>,
        max_connections: Option<u32>,
        hnsw_config: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let hnsw_config_rust = if let Some(dict) = hnsw_config {
            let m = dict
                .get_item("m")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<usize>().ok())
                .unwrap_or(16);
            let ef_construction = dict
                .get_item("ef_construction")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<usize>().ok())
                .unwrap_or(200);
            let ef_search = dict
                .get_item("ef_search")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<usize>().ok())
                .unwrap_or(50);
            let max_nb_connection = dict
                .get_item("max_nb_connection")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<usize>().ok())
                .unwrap_or(16);

            let mut hnsw = HnswConfig::default();
            hnsw.m = m;
            hnsw.ef_construction = ef_construction;
            hnsw.ef_search = ef_search;
            hnsw.max_nb_connection = max_nb_connection;
            if let Some(v) = dict
                .get_item("rerank_overfetch")
                .ok()
                .flatten()
                .and_then(|x| x.extract::<usize>().ok())
            {
                hnsw.rerank_overfetch = v;
            }
            if let Some(s) = dict
                .get_item("graph_space")
                .ok()
                .flatten()
                .and_then(|x| x.extract::<String>().ok())
            {
                hnsw.graph_space = match s.to_lowercase().as_str() {
                    "cosine_compatible" | "cosine" => HnswGraphSpace::CosineCompatible,
                    _ => HnswGraphSpace::RawL2,
                };
            }
            Some(hnsw)
        } else {
            None
        };

        let config = VectorBackendConfig::Postgres {
            connection_string,
            table_name,
            vector_dimension,
            max_connections: max_connections.unwrap_or(10),
            hnsw_config: hnsw_config_rust,
        };
        slf.inner = slf.inner.clone().vector_backend_config(config);
        Ok(slf)
    }

    #[cfg(feature = "vector-store-qdrant")]
    #[pyo3(signature = (url, vector_dimension, collection_prefix=None, api_key=None, batch_size=None, timeout_secs=None))]
    pub fn set_vector_backend_qdrant(
        mut slf: PyRefMut<Self>,
        url: String,
        vector_dimension: usize,
        collection_prefix: Option<String>,
        api_key: Option<String>,
        batch_size: Option<usize>,
        timeout_secs: Option<u64>,
    ) -> PyRefMut<Self> {
        let config = VectorBackendConfig::Qdrant {
            url,
            collection_prefix,
            vector_dimension,
            api_key,
            batch_size,
            timeout: timeout_secs.map(std::time::Duration::from_secs),
        };
        slf.inner = slf.inner.clone().vector_backend_config(config);
        slf
    }

    #[cfg(feature = "vector-store-qdrant-grpc")]
    #[pyo3(signature = (grpc_url, vector_dimension, collection_prefix=None, api_key=None, batch_size=None))]
    pub fn set_vector_backend_qdrant_grpc(
        mut slf: PyRefMut<Self>,
        grpc_url: String,
        vector_dimension: usize,
        collection_prefix: Option<String>,
        api_key: Option<String>,
        batch_size: Option<usize>,
    ) -> PyRefMut<Self> {
        let config = VectorBackendConfig::QdrantGrpc {
            grpc_url,
            collection_prefix,
            vector_dimension,
            api_key,
            batch_size,
        };
        slf.inner = slf.inner.clone().vector_backend_config(config);
        slf
    }

    #[cfg(feature = "vector-store-pinecone")]
    #[pyo3(signature = (base_url, api_key, dimension, namespace=None))]
    pub fn set_vector_backend_pinecone(
        mut slf: PyRefMut<Self>,
        base_url: String,
        api_key: String,
        dimension: usize,
        namespace: Option<String>,
    ) -> PyRefMut<Self> {
        let _ = namespace;
        let config = VectorBackendConfig::Pinecone {
            base_url,
            api_key,
            dimension,
        };
        slf.inner = slf.inner.clone().vector_backend_config(config);
        slf
    }

    #[cfg(feature = "vector-store-chroma")]
    #[pyo3(signature = (url, collection_name, dimension))]
    pub fn set_vector_backend_chroma(
        mut slf: PyRefMut<Self>,
        url: String,
        collection_name: String,
        dimension: usize,
    ) -> PyRefMut<Self> {
        let config = VectorBackendConfig::Chroma {
            url,
            collection_name,
            dimension,
        };
        slf.inner = slf.inner.clone().vector_backend_config(config);
        slf
    }

    #[cfg(feature = "vector-store-s3-backend")]
    #[pyo3(signature = (bucket, prefix, dimension, region=None))]
    pub fn set_vector_backend_s3(
        mut slf: PyRefMut<Self>,
        bucket: String,
        prefix: String,
        dimension: usize,
        region: Option<String>,
    ) -> PyRefMut<Self> {
        let config = VectorBackendConfig::S3 {
            bucket,
            prefix,
            region,
            dimension,
        };
        slf.inner = slf.inner.clone().vector_backend_config(config);
        slf
    }

    #[cfg(feature = "vector-store-coldforge")]
    #[pyo3(signature = (bucket, prefix, dimension, region=None, compression=None))]
    pub fn set_vector_backend_coldforge(
        mut slf: PyRefMut<Self>,
        bucket: String,
        prefix: String,
        dimension: usize,
        region: Option<String>,
        compression: Option<bool>,
    ) -> PyRefMut<Self> {
        let config = VectorBackendConfig::ColdForge {
            bucket,
            prefix,
            region,
            dimension,
            compression: compression.unwrap_or(true),
        };
        slf.inner = slf.inner.clone().vector_backend_config(config);
        slf
    }

    #[cfg(feature = "vector-store-dynamodb")]
    #[pyo3(signature = (table_name, dimension, index_name=None, region=None, endpoint_url=None, ensure_table=None))]
    pub fn set_vector_backend_dynamodb(
        mut slf: PyRefMut<Self>,
        table_name: String,
        dimension: usize,
        index_name: Option<String>,
        region: Option<String>,
        endpoint_url: Option<String>,
        ensure_table: Option<bool>,
    ) -> PyRefMut<Self> {
        let config = VectorBackendConfig::DynamoDb {
            table_name,
            index_name: index_name.unwrap_or_else(|| "embeddings".to_string()),
            region,
            endpoint_url,
            dimension,
            ensure_table: ensure_table.unwrap_or(false),
        };
        slf.inner = slf.inner.clone().vector_backend_config(config);
        slf
    }

    /// Configure a tiered (N-tier) vector backend.
    ///
    /// Each tier is a (`backend_spec`, `tier_config`) tuple. `backend_spec` is a dict with "type" and
    /// type-specific keys (e.g. {"type": "memory", "`ttl_secs"`: 86400} or {"type": "redis", "url": "...", "prefix": "..."}).
    /// `tier_config` is a dict with "name" and optionally "`max_vectors`", "`max_bytes`", "`promotion_threshold`".
    #[pyo3(signature = (tiers, *, write_strategy=None, search_strategy=None, background_sync=false, sync_interval_secs=60))]
    pub fn set_vector_backend_tiered<'py>(
        mut slf: PyRefMut<'py, Self>,
        tiers: &Bound<'py, PyAny>,
        write_strategy: Option<&str>,
        search_strategy: Option<&str>,
        background_sync: bool,
        sync_interval_secs: u64,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let list = tiers.cast::<PyList>().map_err(|_| {
            PyRuntimeError::new_err("tiers must be a list of (backend_spec, tier_config) tuples")
        })?;
        let py = tiers.py();
        let mut rust_tiers: Vec<(VectorBackendConfig, DripstoneTierConfig)> = Vec::new();
        for item in list.iter() {
            let (backend_dict, tier_dict) =
                item.extract::<(Py<PyDict>, Py<PyDict>)>().map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "each tier must be a (backend_spec, tier_config) tuple: {e}"
                    ))
                })?;
            let backend_config = backend_dict_to_config(backend_dict.bind(py))?;
            let tier_config = tier_dict_to_config(tier_dict.bind(py))?;
            rust_tiers.push((backend_config, tier_config));
        }
        if rust_tiers.len() < 2 {
            return Err(PyRuntimeError::new_err(
                "tiered backend requires at least 2 tiers",
            ));
        }
        let write_strategy_rust = write_strategy.and_then(|s| match s.to_lowercase().as_str() {
            "write_through" | "writethrough" => Some(DripstoneWriteStrategy::WriteThrough),
            "write_to_source" => Some(DripstoneWriteStrategy::WriteToSource),
            "write_to_source_and_hot" => Some(DripstoneWriteStrategy::WriteToSourceAndHot),
            _ => None,
        });
        let search_strategy_rust = search_strategy.and_then(|s| match s.to_lowercase().as_str() {
            "search_all_tiers" | "searchalltiers" => Some(DripstoneSearchStrategy::SearchAllTiers),
            "hot_then_source_fallback" | "hotthensourcefallback" => {
                Some(DripstoneSearchStrategy::HotThenSourceFallback)
            }
            _ => None,
        });
        let config = VectorBackendConfig::Tiered {
            tiers: rust_tiers,
            write_strategy: write_strategy_rust,
            search_strategy: search_strategy_rust,
            background_sync,
            sync_interval_secs,
        };
        slf.inner = slf.inner.clone().vector_backend_config(config);
        Ok(slf)
    }

    pub fn vector_backend_from_env(mut slf: PyRefMut<Self>) -> PyResult<PyRefMut<Self>> {
        match VectorBackendConfig::from_env() {
            Ok(config) => {
                slf.inner = slf.inner.clone().vector_backend_config(config);
                Ok(slf)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to configure vector backend from environment: {e}"
            ))),
        }
    }

    /// Enable or disable metrics collection for vector operations
    ///
    /// @param enabled - Whether to enable metrics collection
    /// @returns This builder for method chaining
    pub fn with_metrics(mut slf: PyRefMut<Self>, enabled: bool) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().with_metrics(enabled);
        slf
    }

    /// Set a shared RAG tool to use the same backend across agents.
    ///
    /// This ensures that content indexed via RagService can be queried by the agent's RAG tool,
    /// and vice versa, as they share the same vector store backend.
    ///
    /// # Arguments
    /// * `rag_tool` - A PyRagTool instance to share
    ///
    /// # Returns
    /// This builder for method chaining
    #[cfg(feature = "tools-rag")]
    pub fn shared_rag_tool<'a>(
        mut slf: PyRefMut<'a, Self>,
        rag_tool: PyRef<'a, crate::rag_tool::PyRagTool>,
    ) -> PyRefMut<'a, Self> {
        slf.shared_rag_tool = Some(Arc::clone(&rag_tool.inner));
        slf
    }

    /// Set the durable stream backend for event persistence and replay (e.g. when using the API).
    ///
    /// Use :class:`DurableStreamBackend.memory` or :class:`DurableStreamBackend.redis` to create a backend.
    pub fn durable_stream_backend<'a>(
        mut slf: PyRefMut<'a, Self>,
        backend: &crate::durable_streams::DurableStreamBackendWrapper,
    ) -> PyRefMut<'a, Self> {
        slf.durable_stream_backend = Some(Arc::clone(&backend.inner));
        slf
    }

    /// Use SQLite for durable event persistence (single file). Requires the ``durable-streams-sqlite`` feature.
    #[cfg(feature = "durable-streams-sqlite")]
    pub fn durable_stream_sqlite(mut slf: PyRefMut<Self>, path: String) -> PyRefMut<Self> {
        let mut inner = std::mem::take(&mut slf.inner);
        inner = inner.durable_stream_sqlite(path);
        slf.inner = inner;
        slf
    }

    /// Use Postgres for durable event persistence. Requires the ``durable-streams-postgres`` feature.
    #[cfg(feature = "durable-streams-postgres")]
    pub fn durable_stream_postgres(
        mut slf: PyRefMut<Self>,
        connection_string: String,
    ) -> PyRefMut<Self> {
        let mut inner = std::mem::take(&mut slf.inner);
        inner = inner.durable_stream_postgres(connection_string);
        slf.inner = inner;
        slf
    }

    /// Use S3 for durable event persistence (bucket, prefix, optional region). Requires the ``durable-streams-s3`` feature.
    #[cfg(feature = "durable-streams-s3")]
    pub fn durable_stream_s3(
        mut slf: PyRefMut<Self>,
        bucket: String,
        prefix: String,
        region: Option<String>,
    ) -> PyRefMut<Self> {
        let mut inner = std::mem::take(&mut slf.inner);
        inner = inner.durable_stream_s3(bucket, prefix, region);
        slf.inner = inner;
        slf
    }

    /// Build the agent (blocking)
    pub fn build(&mut self) -> PyResult<Agent> {
        let rt = get_runtime()?;
        let builder = self.take_configured_builder();

        // Handle external provider if set
        let service = if let Some((
            name,
            model_name,
            complete_fn,
            stream_fn,
            supports_native_tools,
            supports_streaming,
        )) = &self.external_provider
        {
            // Create external provider
            // Clone PyObjects using Python GIL
            let complete_fn_clone = Python::attach(|py| complete_fn.clone_ref(py));
            let stream_fn_clone = Python::attach(|py| stream_fn.clone_ref(py));
            let provider = create_external_provider(
                name.clone(),
                model_name.clone(),
                complete_fn_clone,
                stream_fn_clone,
                *supports_native_tools,
                *supports_streaming,
            );

            // Create embedding provider if set
            let embedding_provider =
                self.external_embedding_provider
                    .as_ref()
                    .map(|(name, embed_fn)| {
                        let embed_fn_clone = Python::attach(|py| embed_fn.clone_ref(py));
                        create_external_embedding_provider(name.clone(), embed_fn_clone)
                    });

            // Build tool registry (preserve mutation style / extended defaults)
            let use_all_default_tools = self.use_all_default_tools;
            let use_extended_default_tools = self.use_extended_default_tools;
            let mutation_style = self.mutation_style;
            let custom_tools_clone = self.custom_tools.clone();
            let tool_names = self.allowed_tools.clone();

            rt.block_on(async move {
                use caliche::agent::service::ToolRegistryBuilder;

                use std::sync::Arc;

                // Build tool registry
                let mut tool_registry_builder =
                    ToolRegistryBuilder::new(RustProviderType::External)
                        .with_tool_names(tool_names)
                        .with_mutation_style(mutation_style)
                        .with_embedding_provider(embedding_provider.as_ref().map(Arc::clone));

                if use_all_default_tools {
                    tool_registry_builder = tool_registry_builder.use_all_default_tools();
                } else if use_extended_default_tools {
                    tool_registry_builder = tool_registry_builder.use_extended_default_tools();
                }

                let mut tool_registry = tool_registry_builder.build().await?;

                // Add custom tools directly to the registry
                // Use Arc::get_mut to get mutable access (only works if there's a single Arc reference)
                if let Some(registry) = Arc::get_mut(&mut tool_registry) {
                    for tool in &custom_tools_clone {
                        registry.register(Arc::clone(tool));
                    }
                } else {
                    // If we can't get mutable access, we need to clone and rebuild
                    // This is a fallback - in practice, the single Arc reference should work
                    return Err(anyhow::anyhow!(
                        "Cannot register custom tools: ToolRegistry is shared"
                    ));
                }

                // Create agent service from components
                AgentService::from_components(
                    provider,
                    tool_registry,
                    RustProviderType::External,
                    model_name.clone(),
                    embedding_provider,
                )
            })
            .map_err(|e: anyhow::Error| {
                PyRuntimeError::new_err(format!(
                    "Failed to build agent with external provider: {e}"
                ))
            })?
        } else {
            // Normal build path
            rt.block_on(async move { builder.build().await })
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to build agent: {e}")))?
        };

        Ok(Agent::from_service(service))
    }
}
