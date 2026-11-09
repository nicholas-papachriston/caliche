//! NAPI-RS bindings for Caliche JavaScript SDK
#![expect(
    clippy::needless_pass_by_value,
    reason = "NAPI must own values converted from JavaScript arguments at the ABI boundary"
)]

mod builtin_tools;
mod cache;
mod chains;
mod coordinator;
mod durable_streams;
mod errors;
pub mod evaluation;
mod external_provider;
mod hooks;
mod in_process_mcp;
pub mod inspection;
pub mod jsonl_session;
mod loaders;
#[cfg(feature = "local-inference")]
pub mod local_inference;
mod mcp;
mod mutation;
pub mod observability;
mod output_parsers;
#[cfg(feature = "panic-probe")]
pub mod panic_probe;
mod patterns;
mod prompts;
#[cfg(feature = "tools-rag")]
mod rag;
mod reasoning;
mod security;
mod session;
#[cfg(feature = "s3")]
mod storage;
mod templates;
mod tool_registry;
mod workflow;

use caliche::agent::hooks::permission::PermissionMode;
use caliche::agent::hooks::{HookMatcher, LifecycleHook};
use caliche::agent::memory::{
    BufferMemoryConfig, EnhancedMemoryConfig, EntityMemoryConfig, MemoryPersistence,
    MemorySnapshot, MemoryType, SummaryMemoryConfig, VectorMemoryConfig,
};
use caliche::agent::tools::{Tool, ToolMeta, ToolResult};
use caliche::messages::{
    ACTION_ID_FIELD, MessageRole, StreamEvent, TokenUsage, ToolChoice, ToolSchema,
};

// Stream will emit StreamEvent directly, errors are converted to a special Error event
// This avoids the Result type system issues with NAPI-RS

use anyhow::Result as AnyhowResult;
use async_trait::async_trait;
use caliche::agent::builder::AwsCredentials;
use caliche::llm::{
    LlmClient as RustLlmClient, LlmClientBuilder as RustLlmClientBuilder, ProviderType,
};
use caliche::orchestration::state::SharedState as RustSharedState;
use caliche::orchestration::{AgentFlow as RustAgentFlow, FlowBuilder as RustFlowBuilder};
use caliche::streams::DurableStreamBackend as DurableStreamBackendTrait;
#[cfg(feature = "vector-store-postgres")]
use caliche::vecstore::{HnswConfig, HnswGraphSpace};
use caliche::vecstore::{
    SearchStrategy as DripstoneSearchStrategy, TierConfig as DripstoneTierConfig,
    VectorBackendConfig, WriteStrategy as DripstoneWriteStrategy, tiered_tiers_from_json_str,
};
use caliche::{AgentBuilder as RustAgentBuilder, AgentService};
use futures_util::{Stream, StreamExt};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use tokio::runtime::Runtime;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

// ============================================================================
// Shared Runtime for Async Operations
// ============================================================================

/// Shared Tokio runtime for all async operations
///
/// This runtime is lazily initialized on first use and reused across all
/// NAPI bindings to avoid the overhead of creating new runtimes.
static SHARED_RUNTIME: OnceLock<Runtime> = OnceLock::new();
const TEXT_CHUNK_BATCH_SIZE: usize = 5;

fn init_runtime() -> napi::Result<Runtime> {
    Runtime::new().or_else(|primary| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|fallback| {
                napi::Error::from_reason(format!(
                    "Failed to create Tokio runtime (multi-thread: {primary}; current-thread fallback: {fallback})"
                ))
            })
    })
}

/// Get or initialize the shared runtime
///
/// This function is thread-safe and will create the runtime on first call.
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub(crate) fn get_runtime() -> napi::Result<&'static Runtime> {
    if SHARED_RUNTIME.get().is_none() {
        let runtime = init_runtime()?;
        let _ = SHARED_RUNTIME.set(runtime);
    }
    SHARED_RUNTIME
        .get()
        .ok_or_else(|| napi::Error::from_reason("Tokio runtime initialized but unavailable"))
}

// ============================================================================
// Helper Functions for RwLock Operations
// ============================================================================

/// Helper function to handle `RwLock` read operations with proper error handling
fn rwlock_read<T>(lock: &RwLock<T>) -> napi::Result<std::sync::RwLockReadGuard<'_, T>> {
    lock.read()
        .map_err(|e| napi::Error::from_reason(format!("Failed to acquire read lock: {e}")))
}

/// Helper function to handle `RwLock` write operations with proper error handling
fn rwlock_write<T>(lock: &RwLock<T>) -> napi::Result<std::sync::RwLockWriteGuard<'_, T>> {
    lock.write()
        .map_err(|e| napi::Error::from_reason(format!("Failed to acquire write lock: {e}")))
}

// ============================================================================
// Async Helper Functions (NAPI-RS)
// ============================================================================

/// Call a JavaScript function synchronously and bridge it to an async Rust Future
/// Uses NAPI-RS `ThreadsafeFunction` with blocking calls and oneshot channels
/// to avoid thread safety issues with non-Send NAPI pointers.
///
/// IMPORTANT: JavaScript callbacks MUST return JSON strings (e.g., JSON.stringify(result))
/// The conversion happens on the JavaScript thread, then the result is sent via channel.
pub(crate) async fn call_js_sync_bridge(
    func: &ThreadsafeFunction<Value>,
    arg: Value,
) -> std::result::Result<String, String> {
    enum JsReturn {
        Immediate(String),
        Deferred(Promise<String>),
    }

    let (tx, rx) = oneshot::channel();

    // Call the JS function synchronously in blocking mode
    // The callback runs on the JS thread and sends result via channel
    let call_result = func.call_with_return_value(
        Ok(arg),
        ThreadsafeFunctionCallMode::Blocking,
        move |result, _env: Env| {
            let response = match result {
                Ok(unknown) => {
                    // Convert on JS thread (safe here, no async boundary).
                    // Support both direct string returns and Promise<string>.
                    String::from_unknown(unknown.clone()).map_or_else(
                        |_| {
                            PromiseRaw::<String>::from_unknown(unknown).map_or_else(
                                |_| {
                                    Err("Conversion error: expected string or Promise<string>"
                                        .to_string())
                                },
                                |promise_raw| {
                                    promise_raw
                                        .into_sendable_promise()
                                        .map(JsReturn::Deferred)
                                        .map_err(|e| format!("Promise conversion error: {e}"))
                                },
                            )
                        },
                        |s| Ok(JsReturn::Immediate(s)),
                    )
                }
                Err(e) => Err(format!("JS error: {e:?}")),
            };
            let _ = tx.send(response);
            Ok(())
        },
    );

    if call_result != Status::Ok {
        return Err(format!(
            "Failed to call JavaScript function: {call_result:?}"
        ));
    }

    // Await the result from the channel (no NAPI pointers captured in this future)
    match rx
        .await
        .map_err(|_| "Channel closed".to_string())?
        .map_err(|e| format!("JavaScript callback error: {e}"))?
    {
        JsReturn::Immediate(s) => Ok(s),
        JsReturn::Deferred(promise) => promise
            .await
            .map_err(|e| format!("JavaScript callback promise error: {e}")),
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub(crate) fn parse_provider_type(provider: &str) -> napi::Result<ProviderType> {
    provider
        .parse::<ProviderType>()
        .map_err(napi::Error::from_reason)
}

fn provider_type_to_string(provider: ProviderType) -> String {
    provider.to_string()
}

// ============================================================================
// Rust Types
// ============================================================================

/// RAG tool sharing is only meaningful with the `tools-rag` feature; keep the
/// slot present in `AgentBuilderState` regardless of features so the large
/// state tuple threaded through `build()` doesn't need a second shape.
#[cfg(feature = "tools-rag")]
type SharedRagTool = Option<Arc<caliche::agent::tools::RagTool>>;
#[cfg(not(feature = "tools-rag"))]
type SharedRagTool = Option<()>;

/// Consolidated state for `AgentBuilder` to avoid multiple locks
struct AgentBuilderState {
    inner: RustAgentBuilder,
    // Store LLM parameters that aren't yet supported in Rust builder
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    // Store custom JavaScript tools
    custom_tools: Vec<Arc<dyn Tool>>,
    // Store memory persistence configuration
    memory_persistence: Option<(Arc<dyn MemoryPersistence>, String, bool)>,
    // Store shared RAG tool for backend sharing
    shared_rag_tool: SharedRagTool,
    // Store external provider callbacks (for LangChain, etc.)
    external_provider: Option<(
        String,
        String,
        ThreadsafeFunction<Value>,
        ThreadsafeFunction<Value>,
        bool,
        bool,
    )>,
    external_embedding_provider: Option<(String, ThreadsafeFunction<Value>)>,
    // Track if all default tools should be used
    use_all_default_tools: bool,
    // Store memory type configuration
    memory_type: Option<MemoryType>,
    // Store lifecycle hooks
    hooks: Vec<(HookMatcher, Arc<dyn LifecycleHook>)>,
    // Store permission mode
    permission_mode: PermissionMode,
    // File mutation style (edit/write XOR apply_patch)
    mutation_style: caliche::agent::tools::MutationStyle,
    // Goose-style tool inspection chain
    enable_tool_inspection: bool,
    inspection_workspace_root: Option<std::path::PathBuf>,
    // Store allowed tools filter
    allowed_tools: Option<Vec<String>>,
    // Store in-process MCP servers
    mcp_servers: std::collections::HashMap<String, Arc<caliche::mcp::InProcessMcpServer>>,
    // Store response format for structured outputs
    response_format: Option<caliche::output::ResponseFormat>,
    // Optional durable stream backend for event replay/resume (e.g. API harness)
    durable_stream_backend: Option<Arc<dyn DurableStreamBackendTrait>>,
    /// If set, create SQLite backend in build() (requires durable-streams-sqlite feature).
    #[cfg(feature = "durable-streams-sqlite")]
    durable_stream_sqlite_path: Option<String>,
    /// If set, create Postgres backend in build() (requires durable-streams-postgres feature).
    #[cfg(feature = "durable-streams-postgres")]
    durable_stream_postgres_url: Option<String>,
    /// If set, create S3 backend in build() (bucket, prefix, region) (requires durable-streams-s3 feature).
    #[cfg(feature = "durable-streams-s3")]
    durable_stream_s3_config: Option<(String, String, Option<String>)>,
}

type ExternalProviderConfig = (
    String,
    String,
    ThreadsafeFunction<Value>,
    ThreadsafeFunction<Value>,
    bool,
    bool,
);

struct AgentBuildInputs {
    rust_builder: RustAgentBuilder,
    configuration: AgentConfigurationInputs,
    durable_stream_sqlite_path: Option<String>,
    durable_stream_postgres_url: Option<String>,
    durable_stream_s3_config: Option<(String, String, Option<String>)>,
}

struct AgentConfigurationInputs {
    custom_tools: Vec<Arc<dyn Tool>>,
    memory_persistence: Option<(Arc<dyn MemoryPersistence>, String, bool)>,
    external_provider: Option<ExternalProviderConfig>,
    external_embedding_provider: Option<(String, ThreadsafeFunction<Value>)>,
    shared_rag_tool: SharedRagTool,
    memory_type: Option<MemoryType>,
    hooks: Vec<(HookMatcher, Arc<dyn LifecycleHook>)>,
    permission_mode: PermissionMode,
    mutation_style: caliche::agent::tools::MutationStyle,
    enable_tool_inspection: bool,
    inspection_workspace_root: Option<std::path::PathBuf>,
    allowed_tools: Option<Vec<String>>,
    mcp_servers: HashMap<String, Arc<caliche::mcp::InProcessMcpServer>>,
    durable_stream_backend: Option<Arc<dyn DurableStreamBackendTrait>>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

fn take_agent_build_inputs(state: &Mutex<AgentBuilderState>) -> AgentBuildInputs {
    let mut state = state.lock();
    let inputs = AgentBuildInputs {
        rust_builder: state.inner.clone(),
        configuration: AgentConfigurationInputs {
            custom_tools: state.custom_tools.clone(),
            memory_persistence: state.memory_persistence.clone(),
            external_provider: state.external_provider.take(),
            external_embedding_provider: state.external_embedding_provider.take(),
            #[cfg(feature = "tools-rag")]
            shared_rag_tool: state.shared_rag_tool.as_ref().map(Arc::clone),
            #[cfg(not(feature = "tools-rag"))]
            shared_rag_tool: state.shared_rag_tool,
            memory_type: state.memory_type.clone(),
            hooks: state.hooks.clone(),
            permission_mode: state.permission_mode,
            mutation_style: state.mutation_style,
            enable_tool_inspection: state.enable_tool_inspection,
            inspection_workspace_root: state.inspection_workspace_root.clone(),
            allowed_tools: state.allowed_tools.clone(),
            mcp_servers: state.mcp_servers.clone(),
            durable_stream_backend: state.durable_stream_backend.as_ref().map(Arc::clone),
            temperature: state.temperature,
            max_tokens: state.max_tokens,
        },
        #[cfg(feature = "durable-streams-sqlite")]
        durable_stream_sqlite_path: state.durable_stream_sqlite_path.take(),
        #[cfg(not(feature = "durable-streams-sqlite"))]
        durable_stream_sqlite_path: None,
        #[cfg(feature = "durable-streams-postgres")]
        durable_stream_postgres_url: state.durable_stream_postgres_url.take(),
        #[cfg(not(feature = "durable-streams-postgres"))]
        durable_stream_postgres_url: None,
        #[cfg(feature = "durable-streams-s3")]
        durable_stream_s3_config: state.durable_stream_s3_config.take(),
        #[cfg(not(feature = "durable-streams-s3"))]
        durable_stream_s3_config: None,
    };
    drop(state);
    inputs
}

/// Agent builder for configuring and creating agents
#[napi]
pub struct AgentBuilder {
    state: Mutex<AgentBuilderState>,
}

#[napi(object)]
pub struct BufferMemoryConfigInput {
    pub max_tokens: Option<u32>,
    pub min_messages: Option<u32>,
    pub preserve_system: Option<bool>,
    pub summary_placeholder: Option<bool>,
}

fn checked_f64_to_f32(value: f64, field: &str) -> napi::Result<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(napi::Error::from_reason(format!(
            "{field} is outside the finite 32-bit float range"
        )));
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the finite f32 range is checked immediately above"
    )]
    Ok(value as f32)
}

#[allow(unused_variables, unused_mut)]
fn apply_agent_configuration(
    mut builder: RustAgentBuilder,
    inputs: AgentConfigurationInputs,
) -> RustAgentBuilder {
    if let Some(memory_type) = inputs.memory_type {
        builder = builder.memory(memory_type);
    }
    for tool in inputs.custom_tools {
        builder = builder.add_tool(tool);
    }
    if let Some((persistence, conversation_id, autosave)) = inputs.memory_persistence {
        builder = builder.persist_memory_with(persistence, conversation_id, autosave);
    }
    #[cfg(feature = "tools-rag")]
    if let Some(rag_tool) = inputs.shared_rag_tool {
        builder = builder.shared_rag_tool(rag_tool);
    }
    #[cfg(not(feature = "tools-rag"))]
    let _ = inputs.shared_rag_tool;
    for (matcher, hook) in inputs.hooks {
        builder = builder.hook(matcher, hook);
    }
    builder = builder
        .permission_mode(inputs.permission_mode)
        .mutation_style(inputs.mutation_style);
    if inputs.enable_tool_inspection {
        builder = builder.enable_tool_inspection(inputs.inspection_workspace_root);
    }
    if let Some(tools) = &inputs.allowed_tools {
        let tool_refs: Vec<&str> = tools.iter().map(String::as_str).collect();
        builder = builder.allowed_tools(&tool_refs);
    }
    if let Some(max_tokens) = inputs.max_tokens {
        builder = builder.max_tokens(max_tokens);
    }
    if let Some(temperature) = inputs.temperature {
        builder = builder.temperature(temperature);
    }
    for (name, server) in inputs.mcp_servers {
        builder = builder.mcp_server(name, (*server).clone());
    }
    if let Some(backend) = inputs.durable_stream_backend {
        builder = builder.durable_stream_backend(backend);
    }
    if let Some((
        name,
        model,
        complete_callback,
        stream_callback,
        supports_native_tools,
        supports_streaming,
    )) = inputs.external_provider
    {
        let provider = external_provider::create_external_provider(
            name,
            complete_callback,
            stream_callback,
            supports_native_tools,
            supports_streaming,
        );
        builder = builder
            .provider(ProviderType::External)
            .model(model)
            .provider_instance(provider);
    }
    if let Some((name, callback)) = inputs.external_embedding_provider {
        let provider = external_provider::create_external_embedding_provider(name, callback);
        builder = builder.embedding_provider_instance(provider);
    }
    builder
}

#[allow(unused_variables, unused_mut)]
async fn build_agent_internal(inputs: AgentBuildInputs) -> napi::Result<Agent> {
    let AgentBuildInputs {
        rust_builder,
        configuration,
        durable_stream_sqlite_path,
        durable_stream_postgres_url,
        durable_stream_s3_config,
    } = inputs;

    #[cfg(feature = "durable-streams-sqlite")]
    let mut rust_builder = if let Some(path) = durable_stream_sqlite_path {
        let backend = caliche::streams::SqliteDurableStreamBackend::new(&path)
            .await
            .map_err(|e| napi::Error::from_reason(format!("SQLite durable stream: {}", e)))?;
        rust_builder.durable_stream_backend(Arc::new(backend))
    } else {
        rust_builder
    };
    #[cfg(not(feature = "durable-streams-sqlite"))]
    let mut rust_builder = rust_builder;

    #[cfg(feature = "durable-streams-postgres")]
    if let Some(url) = durable_stream_postgres_url {
        let backend = caliche::streams::PostgresDurableStreamBackend::new(&url)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Postgres durable stream: {}", e)))?;
        rust_builder = rust_builder.durable_stream_backend(Arc::new(backend));
    }

    #[cfg(feature = "durable-streams-s3")]
    if let Some((bucket, prefix, region)) = durable_stream_s3_config {
        let config = caliche::streams::S3DurableStreamConfig {
            bucket,
            prefix,
            region,
        };
        let backend = caliche::streams::S3DurableStreamBackend::new(config)
            .await
            .map_err(|e| napi::Error::from_reason(format!("S3 durable stream: {}", e)))?;
        rust_builder = rust_builder.durable_stream_backend(Arc::new(backend));
    }

    let rust_builder = apply_agent_configuration(rust_builder, configuration);

    let service = rust_builder
        .build()
        .await
        .map_err(|e| napi::Error::from_reason(format!("Failed to build agent: {e}")))?;

    // Cache provider and model at build time to avoid blocking calls later
    let provider_cache = provider_type_to_string(service.provider());
    let model_cache = service.model().to_string();

    Ok(Agent {
        inner: Arc::new(tokio::sync::RwLock::new(service)),
        provider_cache,
        model_cache,
    })
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "NAPI converts and transfers ownership of JavaScript arguments at this ABI boundary"
)]
#[napi]
impl AgentBuilder {
    /// Create a new agent builder
    #[napi(constructor, catch_unwind)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(AgentBuilderState {
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
                memory_type: None,
                hooks: Vec::new(),
                permission_mode: PermissionMode::Default,
                mutation_style: caliche::agent::tools::MutationStyle::EditWrite,
                enable_tool_inspection: false,
                inspection_workspace_root: None,
                allowed_tools: None,
                mcp_servers: std::collections::HashMap::new(),
                response_format: None,
                durable_stream_backend: None,
                #[cfg(feature = "durable-streams-sqlite")]
                durable_stream_sqlite_path: None,
                #[cfg(feature = "durable-streams-postgres")]
                durable_stream_postgres_url: None,
                #[cfg(feature = "durable-streams-s3")]
                durable_stream_s3_config: None,
            }),
        }
    }

    /// Configure buffer memory
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn buffer_memory(
        &self,
        max_tokens: u32,
        min_messages: u32,
        preserve_system: bool,
        summary_placeholder: bool,
    ) -> napi::Result<()> {
        let config = BufferMemoryConfig {
            max_tokens: max_tokens as usize,
            min_messages: min_messages as usize,
            preserve_system,
            summary_placeholder,
        };
        let mut state = self.state.lock();
        state.memory_type = Some(MemoryType::Buffer(config));
        drop(state);
        Ok(())
    }

    /// Configure buffer memory using an object-based config.
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn buffer_memory_config(&self, config: BufferMemoryConfigInput) -> napi::Result<()> {
        self.buffer_memory(
            config.max_tokens.unwrap_or(4096),
            config.min_messages.unwrap_or(4),
            config.preserve_system.unwrap_or(true),
            config.summary_placeholder.unwrap_or(true),
        )
    }

    /// Configure summary memory
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn summary_memory(
        &self,
        max_tokens: u32,
        summarize_threshold_pct: f64,
        chunk_size: u32,
        max_summaries: u32,
    ) -> napi::Result<()> {
        let config = SummaryMemoryConfig {
            max_tokens: max_tokens as usize,
            max_summary_tokens: None,
            summarize_threshold_pct: checked_f64_to_f32(
                summarize_threshold_pct,
                "summary threshold",
            )?,
            chunk_size: chunk_size as usize,
            max_summaries: max_summaries as usize,
        };
        let mut state = self.state.lock();
        state.memory_type = Some(MemoryType::Summary(config));
        drop(state);
        Ok(())
    }

    /// Configure entity memory
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn entity_memory(
        &self,
        max_tokens: u32,
        max_entities: u32,
        extraction_interval: u32,
        similarity_threshold: f64,
    ) -> napi::Result<()> {
        let config = EntityMemoryConfig {
            max_tokens: max_tokens as usize,
            max_entities: max_entities as usize,
            extraction_interval: extraction_interval as usize,
            similarity_threshold: checked_f64_to_f32(
                similarity_threshold,
                "entity similarity threshold",
            )?,
        };
        let mut state = self.state.lock();
        state.memory_type = Some(MemoryType::Entity(config));
        drop(state);
        Ok(())
    }

    /// Configure vector memory for semantic retrieval.
    #[napi(js_name = "vectorMemory", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn vector_memory(
        &self,
        max_tokens: u32,
        max_memories: u32,
        top_k: u32,
        similarity_threshold: f64,
        buffer_tokens: u32,
    ) -> napi::Result<()> {
        let config = VectorMemoryConfig {
            max_tokens: max_tokens as usize,
            max_memories: max_memories as usize,
            top_k: top_k as usize,
            similarity_threshold: checked_f64_to_f32(
                similarity_threshold,
                "vector similarity threshold",
            )?,
            buffer_tokens: buffer_tokens as usize,
            namespace: None,
        };
        let mut state = self.state.lock();
        state.memory_type = Some(MemoryType::Vector(config));
        drop(state);
        Ok(())
    }

    /// Configure enhanced memory (buffer base + budget, relevance, compression, working memory, knowledge).
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn with_enhanced_memory(&self) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.memory_type = Some(MemoryType::Enhanced(EnhancedMemoryConfig::default()));
        drop(state);
        Ok(())
    }

    /// Persist memory with a custom persistence backend
    /// Note: This method stores the configuration for later use during build.
    /// The actual `MemoryPersistence` is created from JavaScript callbacks.
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn persist_memory_with(
        &self,
        save_fn: ThreadsafeFunction<Value>,
        load_fn: ThreadsafeFunction<Value>,
        clear_fn: ThreadsafeFunction<Value>,
        conversation_id: String,
        autosave: bool,
    ) -> napi::Result<()> {
        let persistence = Arc::new(JsMemoryPersistence {
            save: save_fn,
            load: load_fn,
            clear: clear_fn,
        });
        let mut state = self.state.lock();
        state.memory_persistence = Some((persistence, conversation_id, autosave));
        drop(state);
        Ok(())
    }

    // Note: builder_persist_memory_with is handled in TypeScript by storing the persistence
    // object and converting it during build. The actual implementation would require
    // converting JavaScript functions to ThreadsafeFunction, which is complex.
    // For now, persistMemoryWith stores the object and handles it in the build method.

    /// Set the LLM provider
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_provider(&self, provider: String) -> napi::Result<()> {
        let provider_type = parse_provider_type(&provider)?;
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).provider(provider_type);
        drop(state);
        Ok(())
    }

    /// Set the model name
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_model(&self, model: String) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).model(model);
        drop(state);
        Ok(())
    }

    /// Enable JSON object mode (any valid JSON response)
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn json_mode(&self) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).json_mode();
        state.response_format = Some(caliche::output::ResponseFormat::JsonObject);
        drop(state);
        Ok(())
    }

    /// Enable JSON schema mode with validation
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn json_schema(&self, name: String, schema: String) -> napi::Result<()> {
        let schema_value: serde_json::Value = serde_json::from_str(&schema)
            .map_err(|e| napi::Error::from_reason(format!("Invalid JSON schema: {e}")))?;

        // Normalize schema to ensure OpenAI compatibility (adds additionalProperties: false if needed)
        let normalized_schema =
            caliche::output::JsonSchemaDefinition::normalize_schema(schema_value);

        let mut state = self.state.lock();
        state.inner =
            std::mem::take(&mut state.inner).json_schema(&name, normalized_schema.clone());
        state.response_format = Some(caliche::output::ResponseFormat::JsonSchema {
            json_schema: caliche::output::JsonSchemaDefinition {
                name,
                description: None,
                schema: normalized_schema,
                strict: true,
            },
        });
        drop(state);
        Ok(())
    }

    /// Set the system prompt
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_system_prompt(&self, prompt: String) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).system_prompt(prompt);
        drop(state);
        Ok(())
    }

    /// Control whether runtime context is injected into the system prompt.
    ///
    /// Benchmarks and thin wrappers typically disable this to match raw SDK
    /// request shapes (`includeContextInSystem(false)`).
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_include_context_in_system(&self, value: bool) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).include_context_in_system(value);
        drop(state);
        Ok(())
    }

    /// Enable a single tool
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn enable_tool(&self, tool: String) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).enable_tool(&tool);
        drop(state);
        Ok(())
    }

    /// Enable multiple tools
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn enable_tools(&self, tools: Vec<String>) -> napi::Result<()> {
        let mut state = self.state.lock();
        for tool in tools {
            state.inner = std::mem::take(&mut state.inner).enable_tool(&tool);
        }
        drop(state);
        Ok(())
    }

    /// Enable all default tools
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn enable_all_default_tools(&self) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).enable_all_default_tools();
        state.use_all_default_tools = true;
        drop(state);
        Ok(())
    }

    /// Enable the broader default tool set (grep/git/etc. beyond coding defaults).
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn enable_extended_default_tools(&self) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).enable_extended_default_tools();
        drop(state);
        Ok(())
    }

    /// Select file mutation style (`EditWrite` XOR `ApplyPatch`).
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn mutation_style(&self, style: &mutation::JsMutationStyle) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.mutation_style = (*style).into();
        drop(state);
        Ok(())
    }

    /// Enable Goose-style tool inspection (path security + permission mode).
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn enable_tool_inspection(&self, workspace_root: Option<String>) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.enable_tool_inspection = true;
        state.inspection_workspace_root = workspace_root.map(std::path::PathBuf::from);
        drop(state);
        Ok(())
    }

    /// Add a native inspection hook adapter.
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_inspection_hook(
        &self,
        matcher: &hooks::JsHookMatcher,
        hook: &inspection::JsInspectionHook,
    ) -> napi::Result<()> {
        let mut state = self.state.lock();
        state
            .hooks
            .push((HookMatcher::from(matcher.clone()), hook.as_lifecycle_hook()));
        drop(state);
        Ok(())
    }

    /// Set temperature
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_temperature(&self, temp: f64) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.temperature = Some(checked_f64_to_f32(temp, "temperature")?);
        drop(state);
        Ok(())
    }

    /// Set max tokens
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_max_tokens(&self, tokens: u32) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.max_tokens = Some(tokens);
        drop(state);
        Ok(())
    }

    /// Set `top_p`
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_top_p(&self, top_p: f64) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.top_p = Some(checked_f64_to_f32(top_p, "top_p")?);
        drop(state);
        Ok(())
    }

    /// Set API key
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_api_key(&self, key: String) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).api_key(key);
        drop(state);
        Ok(())
    }

    /// Set API key scoped to a provider/tool target.
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_api_key_for(&self, _target: String, key: String) -> napi::Result<()> {
        // Native core currently stores a single active API key.
        // Scoped routing is handled in the TypeScript wrapper/environment.
        self.set_api_key(key)
    }

    /// Set AWS credentials
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_aws_credentials(
        &self,
        access_key: String,
        secret_key: String,
        region: String,
    ) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner =
            std::mem::take(&mut state.inner).aws_credentials(access_key, secret_key, region);
        drop(state);
        Ok(())
    }

    /// Set base URL
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_base_url(&self, url: String) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).base_url(url);
        drop(state);
        Ok(())
    }

    /// Set max iterations
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_max_iterations(&self, max_iterations: u32) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).max_iterations(max_iterations as usize);
        drop(state);
        Ok(())
    }

    /// Set tool choice for agent chat turns (`auto`, `none`, `required`, or a tool name).
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_tool_choice(&self, choice: String) -> napi::Result<()> {
        use caliche::messages::ToolChoice;
        let mut state = self.state.lock();
        let tool_choice = match choice.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            name => ToolChoice::Specific(beef::lean::Cow::owned(name.to_string())),
        };
        state.inner = std::mem::take(&mut state.inner).tool_choice(Some(tool_choice));
        drop(state);
        Ok(())
    }

    /// Enable/disable filesystem MCP auto-discovery during build (default: disabled).
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_auto_discover_mcp(&self, enabled: bool) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).auto_discover_mcp(enabled);
        drop(state);
        Ok(())
    }

    /// Enable/disable on-disk skill auto-indexing during build (default: disabled).
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_auto_discover_skills(&self, enabled: bool) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).auto_discover_skills(enabled);
        drop(state);
        Ok(())
    }

    /// Configure a tiered (N-tier) vector backend.
    /// `tiers_json`: JSON array of { backend: { type, ... }, tierConfig: { name, `max_vectors`?, `max_bytes`?, `promotion_threshold`? } }
    #[napi(js_name = "setVectorBackendTiered", catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_vector_backend_tiered(
        &self,
        tiers_json: String,
        write_strategy: Option<String>,
        background_sync: Option<bool>,
        sync_interval_secs: Option<u32>,
        search_strategy: Option<String>,
    ) -> napi::Result<()> {
        let rust_tiers = parse_tiered_tiers_from_json(&tiers_json)
            .map_err(|e| napi::Error::from_reason(format!("tiered backend config: {e}")))?;
        if rust_tiers.len() < 2 {
            return Err(napi::Error::from_reason(
                "tiered backend requires at least 2 tiers",
            ));
        }
        let write_strategy_rust =
            write_strategy
                .as_deref()
                .and_then(|s| match s.to_lowercase().as_str() {
                    "write_through" | "writethrough" => Some(DripstoneWriteStrategy::WriteThrough),
                    "write_to_source" => Some(DripstoneWriteStrategy::WriteToSource),
                    "write_to_source_and_hot" => Some(DripstoneWriteStrategy::WriteToSourceAndHot),
                    _ => None,
                });
        let search_strategy_rust =
            search_strategy
                .as_deref()
                .and_then(|s| match s.to_lowercase().as_str() {
                    "search_all_tiers" | "searchalltiers" => {
                        Some(DripstoneSearchStrategy::SearchAllTiers)
                    }
                    "hot_then_source_fallback" | "hotthensourcefallback" => {
                        Some(DripstoneSearchStrategy::HotThenSourceFallback)
                    }
                    _ => None,
                });
        let config = VectorBackendConfig::Tiered {
            tiers: rust_tiers,
            write_strategy: write_strategy_rust,
            search_strategy: search_strategy_rust,
            background_sync: background_sync.unwrap_or(false),
            sync_interval_secs: u64::from(sync_interval_secs.unwrap_or(60)),
        };
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).vector_backend_config(config);
        drop(state);
        Ok(())
    }

    /// Configure in-memory vector backend
    #[napi(js_name = "setVectorBackendMemory", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_vector_backend_memory(
        &self,
        ttl_secs: Option<u32>,
        max_entries: Option<u32>,
    ) -> napi::Result<()> {
        let config = VectorBackendConfig::Memory {
            ttl_secs: u64::from(ttl_secs.unwrap_or(86400)),
            max_entries: max_entries.map(|n| n as usize),
        };
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).vector_backend_config(config);
        drop(state);
        Ok(())
    }

    /// Auto-configure vector backend from environment variables
    #[napi(js_name = "vectorBackendFromEnv", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn vector_backend_from_env(&self) -> napi::Result<()> {
        let config = VectorBackendConfig::from_env().map_err(|e| {
            napi::Error::from_reason(format!(
                "Failed to configure vector backend from environment: {e}"
            ))
        })?;
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).vector_backend_config(config);
        drop(state);
        Ok(())
    }

    /// Enable or disable metrics collection for vector operations
    #[napi(js_name = "withMetrics", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn with_metrics(&self, enabled: bool) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).with_metrics(enabled);
        drop(state);
        Ok(())
    }

    /// Set external provider using `ThreadsafeFunctions`
    /// JavaScript callbacks must return JSON strings (e.g., JSON.stringify(result))
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_external_provider(
        &self,
        name: String,
        model_name: String,
        complete_fn: ThreadsafeFunction<Value>,
        stream_fn: ThreadsafeFunction<Value>,
        supports_native_tools: bool,
        supports_streaming: bool,
    ) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.external_provider = Some((
            name,
            model_name,
            complete_fn,
            stream_fn,
            supports_native_tools,
            supports_streaming,
        ));
        drop(state);
        Ok(())
    }

    /// Set external embedding provider using `ThreadsafeFunction`
    /// JavaScript callback must return JSON strings (e.g., JSON.stringify(result))
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_external_embedding_provider(
        &self,
        name: String,
        embed_fn: ThreadsafeFunction<Value>,
    ) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.external_embedding_provider = Some((name, embed_fn));
        drop(state);
        Ok(())
    }

    /// Set the permission mode for tool execution
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_permission_mode(&self, mode: &hooks::JsPermissionMode) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.permission_mode = PermissionMode::from(*mode);
        drop(state);
        Ok(())
    }

    /// Set the list of allowed tools (if None, all tools are allowed)
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_allowed_tools(&self, tools: Vec<String>) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.allowed_tools = Some(tools);
        drop(state);
        Ok(())
    }

    /// Add a custom JavaScript tool to the builder
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn add_tool(
        &self,
        name: String,
        description: String,
        parameters: Value,
        execute_fn: ThreadsafeFunction<Value>,
    ) -> napi::Result<()> {
        let tool = Arc::new(JsTool {
            name: Arc::from(name.as_str()),
            description: Arc::from(description.as_str()),
            parameters,
            execute_fn,
        });
        let mut state = self.state.lock();
        state.custom_tools.push(tool);
        drop(state);
        Ok(())
    }

    /// Add a lifecycle hook with a matcher
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_hook(
        &self,
        matcher: &hooks::JsHookMatcher,
        pre_tool_use_fn: ThreadsafeFunction<Value>,
        post_tool_use_fn: Option<ThreadsafeFunction<Value>>,
    ) -> napi::Result<()> {
        let hook = hooks::create_js_lifecycle_hook(pre_tool_use_fn, post_tool_use_fn);
        let mut state = self.state.lock();
        state.hooks.push((HookMatcher::from(matcher.clone()), hook));
        drop(state);
        Ok(())
    }

    /// Add an in-process MCP server
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_mcp_server(
        &self,
        name: String,
        server: &in_process_mcp::InProcessMcpServer,
    ) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.mcp_servers.insert(name, server.get_inner());
        drop(state);
        Ok(())
    }

    /// Set the durable stream backend for event persistence and replay (e.g. when using the API).
    #[napi(js_name = "setDurableStreamBackend", catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_durable_stream_backend(
        &self,
        backend: &durable_streams::DurableStreamBackendJs,
    ) -> napi::Result<()> {
        let mut state = self.state.lock();
        state.durable_stream_backend = Some(backend.get_inner());
        drop(state);
        Ok(())
    }

    /// Use `SQLite` for durable event persistence (single file). Requires the durable-streams-sqlite feature.
    #[napi(js_name = "setDurableStreamSqlite", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_durable_stream_sqlite(&self, db_path: String) -> napi::Result<()> {
        #[cfg(feature = "durable-streams-sqlite")]
        {
            let mut state = self.state.lock();
            state.durable_stream_sqlite_path = Some(db_path);
            drop(state);
        }
        #[cfg(not(feature = "durable-streams-sqlite"))]
        let _ = db_path;
        Ok(())
    }

    /// Use Postgres for durable event persistence. Requires the durable-streams-postgres feature.
    #[napi(js_name = "setDurableStreamPostgres", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_durable_stream_postgres(&self, connection_string: String) -> napi::Result<()> {
        #[cfg(feature = "durable-streams-postgres")]
        {
            let mut state = self.state.lock();
            state.durable_stream_postgres_url = Some(connection_string);
            drop(state);
        }
        #[cfg(not(feature = "durable-streams-postgres"))]
        let _ = connection_string;
        Ok(())
    }

    /// Use S3 for durable event persistence (bucket, prefix, optional region). Requires the durable-streams-s3 feature.
    #[napi(js_name = "setDurableStreamS3", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_durable_stream_s3(
        &self,
        bucket: String,
        prefix: String,
        region: Option<String>,
    ) -> napi::Result<()> {
        #[cfg(feature = "durable-streams-s3")]
        {
            let mut state = self.state.lock();
            state.durable_stream_s3_config = Some((bucket, prefix, region));
            drop(state);
        }
        #[cfg(not(feature = "durable-streams-s3"))]
        let _ = (bucket, prefix, region);
        Ok(())
    }

    /// Build the agent
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn build(&self) -> napi::Result<Agent> {
        build_agent_internal(take_agent_build_inputs(&self.state)).await
    }
}

/// Helper function to add a tool to a builder (for use from TypeScript)
/// This is a wrapper that matches the native interface signature
#[napi(js_name = "builderAddTool", catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn builder_add_tool(
    builder: &AgentBuilder,
    name: String,
    description: String,
    parameters: Value,
    execute_fn: ThreadsafeFunction<Value>,
) -> napi::Result<()> {
    builder.add_tool(name, description, parameters, execute_fn)
}

/// JavaScript memory persistence wrapper
// Note: Cannot derive Debug because ThreadsafeFunction doesn't implement Debug
struct JsMemoryPersistence {
    save: ThreadsafeFunction<Value>,
    load: ThreadsafeFunction<Value>,
    clear: ThreadsafeFunction<Value>,
}

impl std::fmt::Debug for JsMemoryPersistence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsMemoryPersistence")
            .field("save", &"<ThreadsafeFunction>")
            .field("load", &"<ThreadsafeFunction>")
            .field("clear", &"<ThreadsafeFunction>")
            .finish()
    }
}

#[async_trait]
impl MemoryPersistence for JsMemoryPersistence {
    async fn save_snapshot(
        &self,
        conversation_id: &str,
        snapshot: &MemorySnapshot,
    ) -> anyhow::Result<()> {
        // Serialize snapshot to JSON Value for passing to JavaScript
        let snapshot_value = serde_json::to_value(snapshot)
            .map_err(|e| anyhow::anyhow!("Failed to serialize snapshot: {e}"))?;

        // Call the JS function asynchronously
        let args = serde_json::json!([conversation_id, snapshot_value]);
        let result = call_js_sync_bridge(&self.save, args)
            .await
            .map_err(|e| anyhow::anyhow!("JavaScript save_snapshot error: {e}"))?;
        validate_memory_persistence_result(&result)
    }

    async fn load_snapshot(&self, conversation_id: &str) -> anyhow::Result<Option<MemorySnapshot>> {
        // Call the JS function asynchronously
        let args = serde_json::json!([conversation_id]);
        let result_str = call_js_sync_bridge(&self.load, args)
            .await
            .map_err(|e| anyhow::anyhow!("JavaScript load_snapshot error: {e}"))?;

        // Parse result - if null or empty, return None
        if result_str == "null" || result_str.is_empty() {
            return Ok(None);
        }

        // Try to deserialize as MemorySnapshot
        let snapshot_value: Value = serde_json::from_str(&result_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse snapshot JSON: {e}"))?;

        let snapshot: MemorySnapshot = serde_json::from_value(snapshot_value)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize snapshot: {e}"))?;

        Ok(Some(snapshot))
    }

    async fn clear_snapshot(&self, conversation_id: &str) -> anyhow::Result<()> {
        // Call the JS function asynchronously
        let args = serde_json::json!([conversation_id]);
        let result = call_js_sync_bridge(&self.clear, args)
            .await
            .map_err(|e| anyhow::anyhow!("JavaScript clear_snapshot error: {e}"))?;
        validate_memory_persistence_result(&result)
    }
}

fn validate_memory_persistence_result(result: &str) -> anyhow::Result<()> {
    if result.is_empty() || result == "null" || result == "undefined" {
        return Ok(());
    }
    if let Ok(error_object) = serde_json::from_str::<serde_json::Map<String, Value>>(result)
        && let Some(message) = error_object.get("error").and_then(Value::as_str)
    {
        return Err(anyhow::anyhow!("JavaScript error: {message}"));
    }
    Ok(())
}

/// JavaScript tool wrapper that implements the Rust Tool trait
// Note: Cannot derive Debug because ThreadsafeFunction doesn't implement Debug
struct JsTool {
    name: Arc<str>,
    description: Arc<str>,
    parameters: Value,
    execute_fn: ThreadsafeFunction<Value>,
}

impl std::fmt::Debug for JsTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("execute_fn", &"<ThreadsafeFunction>")
            .finish()
    }
}

impl ToolMeta for JsTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }
}

fn parse_js_tool_result(result: String) -> ToolResult {
    serde_json::from_str::<ToolResult>(&result).map_or_else(
        |_| {
            serde_json::from_str::<serde_json::Map<String, Value>>(&result).map_or_else(
                |_| ToolResult::success(result),
                |object| ToolResult {
                    success: object
                        .get("success")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                    content: object
                        .get("content")
                        .and_then(Value::as_str)
                        .map_or_else(String::new, ToString::to_string),
                    typed_payload: object.get("payload").cloned(),
                    error: object
                        .get("error")
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                    usage: None,
                    warnings: None,
                },
            )
        },
        |tool_result| tool_result,
    )
}

#[async_trait]
impl Tool for JsTool {
    async fn execute(&self, arguments: Value) -> ToolResult {
        let result = call_js_sync_bridge(&self.execute_fn, arguments).await;

        match result {
            Ok(result) => parse_js_tool_result(result),
            Err(e) => ToolResult::error(format!("JavaScript execution error: {e}")),
        }
    }
}

// Remote vector-store backend setters live in their own per-feature `impl`
// blocks: napi-rs's codegen for an `impl` block scans every method's tokens
// (including those under an inner `#[cfg]`) to build the class registration
// table, so a method gated behind a feature that is off inside a shared
// `#[napi] impl` block still leaves a dangling reference to its generated
// callback. Gating the whole `impl` block instead avoids that.

/// Configure Redis vector backend
#[cfg(feature = "vector-store-redis")]
#[napi]
impl AgentBuilder {
    #[napi(js_name = "setVectorBackendRedis", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_vector_backend_redis(&self, url: String, prefix: String) -> napi::Result<()> {
        let config = VectorBackendConfig::Redis { url, prefix };
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).vector_backend_config(config);
        drop(state);
        Ok(())
    }
}

/// Configure PostgreSQL vector backend
#[cfg(feature = "vector-store-postgres")]
#[napi]
impl AgentBuilder {
    #[napi(js_name = "setVectorBackendPostgres", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_vector_backend_postgres(
        &self,
        connection_string: String,
        table_name: String,
        vector_dimension: Option<u32>,
        max_connections: Option<u32>,
        hnsw_config: Option<Value>,
    ) -> napi::Result<()> {
        let config = VectorBackendConfig::Postgres {
            connection_string,
            table_name,
            vector_dimension: vector_dimension.map(|n| n as usize),
            max_connections: max_connections.unwrap_or(10),
            hnsw_config: hnsw_config
                .as_ref()
                .map(json_to_hnsw_config)
                .transpose()
                .map_err(|e| napi::Error::from_reason(e.to_string()))?,
        };
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).vector_backend_config(config);
        drop(state);
        Ok(())
    }
}

/// Configure Qdrant HTTP vector backend
#[cfg(feature = "vector-store-qdrant")]
#[napi]
impl AgentBuilder {
    #[napi(js_name = "setVectorBackendQdrant", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_vector_backend_qdrant(
        &self,
        url: String,
        vector_dimension: u32,
        collection_prefix: Option<String>,
        api_key: Option<String>,
    ) -> napi::Result<()> {
        let config = VectorBackendConfig::Qdrant {
            url,
            collection_prefix,
            vector_dimension: vector_dimension as usize,
            api_key,
            batch_size: None,
            timeout: None,
        };
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).vector_backend_config(config);
        drop(state);
        Ok(())
    }
}

/// Configure Qdrant gRPC vector backend
#[cfg(feature = "vector-store-qdrant-grpc")]
#[napi]
impl AgentBuilder {
    #[napi(js_name = "setVectorBackendQdrantGrpc", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_vector_backend_qdrant_grpc(
        &self,
        grpc_url: String,
        vector_dimension: u32,
        collection_prefix: Option<String>,
        api_key: Option<String>,
        batch_size: Option<u32>,
    ) -> napi::Result<()> {
        let config = VectorBackendConfig::QdrantGrpc {
            grpc_url,
            collection_prefix,
            vector_dimension: vector_dimension as usize,
            api_key,
            batch_size: batch_size.map(|n| n as usize),
        };
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).vector_backend_config(config);
        drop(state);
        Ok(())
    }
}

/// Configure Pinecone vector backend
#[cfg(feature = "vector-store-pinecone")]
#[napi]
impl AgentBuilder {
    #[napi(js_name = "setVectorBackendPinecone", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_vector_backend_pinecone(
        &self,
        base_url: String,
        api_key: String,
        dimension: u32,
        _timeout_secs: Option<u32>,
    ) -> napi::Result<()> {
        let _ = _timeout_secs;
        let config = VectorBackendConfig::Pinecone {
            base_url,
            api_key,
            dimension: dimension as usize,
        };
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).vector_backend_config(config);
        drop(state);
        Ok(())
    }
}

/// Configure Chroma vector backend
#[cfg(feature = "vector-store-chroma")]
#[napi]
impl AgentBuilder {
    #[napi(js_name = "setVectorBackendChroma", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_vector_backend_chroma(
        &self,
        base_url: String,
        collection_prefix: Option<String>,
        expected_dimension: Option<u32>,
        _timeout_secs: Option<u32>,
    ) -> napi::Result<()> {
        let _ = _timeout_secs;
        let config = VectorBackendConfig::Chroma {
            url: base_url,
            collection_name: collection_prefix.unwrap_or_else(|| "embeddings".to_string()),
            dimension: expected_dimension.unwrap_or(1536) as usize,
        };
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).vector_backend_config(config);
        drop(state);
        Ok(())
    }
}

/// Configure S3 vector backend
#[cfg(feature = "vector-store-s3-backend")]
#[napi]
impl AgentBuilder {
    #[napi(js_name = "setVectorBackendS3", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_vector_backend_s3(
        &self,
        bucket: String,
        key_prefix: String,
        _enable_compression: Option<bool>,
        _batch_size: Option<u32>,
        _cache_indices: Option<bool>,
    ) -> napi::Result<()> {
        let _ = (_enable_compression, _batch_size, _cache_indices);
        let config = VectorBackendConfig::S3 {
            bucket,
            prefix: key_prefix,
            region: None,
            dimension: 1536,
        };
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).vector_backend_config(config);
        drop(state);
        Ok(())
    }
}

/// Configure ColdForge vector backend
#[cfg(feature = "vector-store-coldforge")]
#[napi]
impl AgentBuilder {
    #[napi(js_name = "setVectorBackendColdforge", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_vector_backend_coldforge(
        &self,
        bucket: String,
        key_prefix: String,
        region: String,
        vector_dimension: u32,
        _cache_size_mb: Option<u32>,
        _num_centroids: Option<u32>,
    ) -> napi::Result<()> {
        let _ = (_cache_size_mb, _num_centroids);
        let config = VectorBackendConfig::ColdForge {
            bucket,
            prefix: key_prefix,
            region: Some(region),
            dimension: vector_dimension as usize,
            compression: true,
        };
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).vector_backend_config(config);
        drop(state);
        Ok(())
    }
}

/// Configure DynamoDB native vector backend
#[cfg(feature = "vector-store-dynamodb")]
#[napi]
impl AgentBuilder {
    #[napi(js_name = "setVectorBackendDynamodb", catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_vector_backend_dynamodb(
        &self,
        table_name: String,
        index_name: String,
        region: String,
        vector_dimension: u32,
        ensure_table: Option<bool>,
    ) -> napi::Result<()> {
        let config = VectorBackendConfig::DynamoDb {
            table_name,
            index_name,
            region: Some(region),
            endpoint_url: None,
            dimension: vector_dimension as usize,
            ensure_table: ensure_table.unwrap_or(false),
        };
        let mut state = self.state.lock();
        state.inner = std::mem::take(&mut state.inner).vector_backend_config(config);
        drop(state);
        Ok(())
    }
}

/// Agent instance for chat interactions.
/// Uses `RwLock` so read-only operations (`list_tools`) don't block `chat()`.
#[napi]
pub struct Agent {
    pub(crate) inner: Arc<tokio::sync::RwLock<AgentService>>,
    provider_cache: String,
    model_cache: String,
}

#[napi]
impl Agent {
    /// Create an isolated request-scoped agent that shares immutable provider,
    /// tool registry, and configuration state without rebuilding connection pools.
    ///
    /// # Errors
    ///
    /// Returns a NAPI error if the request-scoped wrapper cannot be created.
    #[napi]
    pub async fn clone_for_request(&self) -> napi::Result<Self> {
        let service = {
            let agent = self.inner.read().await;
            agent.clone_for_request()
        };
        Ok(Self {
            inner: Arc::new(tokio::sync::RwLock::new(service)),
            provider_cache: self.provider_cache.clone(),
            model_cache: self.model_cache.clone(),
        })
    }

    /// Send a chat message and get a response
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn chat(&self, message: String) -> napi::Result<ChatResponse> {
        let mut agent = self.inner.write().await;
        let response = agent
            .chat(&message)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Chat failed: {e}")))?;
        drop(agent);

        Ok(ChatResponse {
            response: response.text().to_string(),
            usage: response.usage().clone().ensure_total_tokens().into(),
            metrics: response
                .metrics
                .clone()
                .map(TryInto::try_into)
                .transpose()?,
        })
    }

    /// Get the agent's provider
    #[napi(catch_unwind)]
    #[must_use]
    pub fn get_provider(&self) -> String {
        // Return cached value - no lock needed
        self.provider_cache.clone()
    }

    /// Get the agent's model
    #[napi(catch_unwind)]
    #[must_use]
    pub fn get_model(&self) -> String {
        // Return cached value - no lock needed
        self.model_cache.clone()
    }

    /// List available tools (uses read lock; does not block chat)
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn list_tools(&self) -> napi::Result<Vec<ToolInfo>> {
        let agent = self.inner.read().await;
        let tools = agent
            .list_tools()
            .into_iter()
            .map(|(name, description, parameters)| ToolInfo {
                name,
                description,
                parameters,
            })
            .collect();
        drop(agent);
        // Clone all data before dropping the lock
        Ok(tools)
    }

    /// Access the agent's tool registry (active/deferred controls).
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn tool_registry(&self) -> napi::Result<tool_registry::JsToolRegistry> {
        let agent = self.inner.read().await;
        let registry = Arc::clone(agent.tool_registry());
        drop(agent);
        Ok(tool_registry::JsToolRegistry::from_inner(registry))
    }

    /// Chat with conversation history
    #[napi]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub async fn chat_with_history(
        &self,
        message: String,
        history: Vec<MessageJs>,
    ) -> napi::Result<ChatResponse> {
        use caliche::messages::Message;
        let mut agent = self.inner.write().await;

        // Convert JS messages to Rust messages (preserve multimodal content_parts)
        let rust_messages: Vec<Message> = history.into_iter().map(message_js_to_rust).collect();

        let response = agent
            .chat_with_history(&message, rust_messages)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Chat with history failed: {e}")))?;
        drop(agent);

        Ok(ChatResponse {
            response: response.text().to_string(),
            usage: response.usage().clone().ensure_total_tokens().into(),
            metrics: response
                .metrics
                .clone()
                .map(TryInto::try_into)
                .transpose()?,
        })
    }

    /// Add a custom tool to the agent
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn add_tool(
        &self,
        name: String,
        description: String,
        parameters: Value,
        execute_fn: ThreadsafeFunction<Value>,
    ) -> napi::Result<()> {
        let tool = Arc::new(JsTool {
            name: Arc::from(name.as_str()),
            description: Arc::from(description.as_str()),
            parameters,
            execute_fn,
        });
        let mut agent = self.inner.write().await;
        agent.add_tool(tool);
        drop(agent);
        Ok(())
    }

    /// Set user context for the agent
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn set_user_context(
        &self,
        user_id: Option<String>,
        metadata: Option<Value>,
    ) -> napi::Result<()> {
        use caliche::UserContext;
        let context = user_id.map(|uid| {
            let mut ctx = UserContext {
                user_id: uid,
                tenant_id: None,
                email: None,
                roles: Vec::new(),
                custom_claims: std::collections::HashMap::new(),
            };
            if let Some(meta) = metadata
                && let Some(map) = meta.as_object()
            {
                for (k, v) in map {
                    ctx.custom_claims.insert(k.clone(), v.clone());
                }
            }
            ctx
        });
        let mut agent = self.inner.write().await;
        agent.set_user_context(context);
        drop(agent);
        Ok(())
    }

    /// Get user context
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn get_user_context(&self) -> napi::Result<Option<UserContextJs>> {
        let agent = self.inner.read().await;
        let context = agent.user_context().cloned().map(|ctx| UserContextJs {
            user_id: ctx.user_id.clone(),
            metadata: serde_json::json!(ctx.custom_claims),
        });
        drop(agent);
        Ok(context)
    }

    /// Get conversation messages from memory
    #[napi]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub async fn get_messages(&self) -> napi::Result<Vec<MessageJs>> {
        let agent = self.inner.read().await;
        let memory = agent.memory();
        drop(agent);
        if let Some(memory) = memory {
            let memory_guard = memory.read().await;
            let context = memory_guard.get_context().await.map_err(|e| {
                napi::Error::from_reason(format!("Failed to get memory context: {e}"))
            })?;
            drop(memory_guard);
            Ok(context
                .into_iter()
                .map(|msg| MessageJs {
                    role: match msg.role {
                        MessageRole::User => "user".to_string(),
                        MessageRole::Assistant => "assistant".to_string(),
                        MessageRole::System => "system".to_string(),
                        MessageRole::Tool => "tool".to_string(),
                    },
                    content: msg.extract_text(),
                    content_parts: None,
                })
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Execute a single step (for advanced reasoning patterns)
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn step(&self, messages: Vec<MessageJs>) -> napi::Result<StepResultJs> {
        use caliche::messages::Message;
        use std::sync::Arc;

        // Convert JS messages to Rust messages
        let rust_messages: Vec<Message> = messages.into_iter().map(message_js_to_rust).collect();

        let agent = self.inner.write().await;
        let step_result = agent
            .step(Arc::new(rust_messages))
            .await
            .map_err(|e| napi::Error::from_reason(format!("Step execution failed: {e}")))?;
        drop(agent);

        // Convert tool calls
        let tool_calls: Vec<ToolCallJs> = step_result
            .tool_calls
            .iter()
            .map(|tc| ToolCallJs {
                id: String::new(), // ToolCall doesn't have id, generate one or leave empty
                name: tc.name.to_string(),
                arguments: tc.arguments.clone(),
            })
            .collect();

        // Convert updated messages
        let updated_messages: Vec<MessageJs> = step_result
            .updated_history
            .iter()
            .map(|msg| MessageJs {
                role: match msg.role {
                    MessageRole::User => "user".to_string(),
                    MessageRole::Assistant => "assistant".to_string(),
                    MessageRole::System => "system".to_string(),
                    MessageRole::Tool => "tool".to_string(),
                },
                content: msg.extract_text(),
                content_parts: None,
            })
            .collect();

        Ok(StepResultJs {
            response: step_result.response_text.to_string(),
            tool_calls,
            updated_messages,
            usage: step_result.usage.clone().ensure_total_tokens().into(),
        })
    }

    /// Stream chat responses
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn stream_chat(&self, message: String) -> napi::Result<StreamIterator> {
        self.stream_chat_with_options(message, None)
    }

    /// Stream chat responses with explicit conversation history.
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn stream_chat_with_history(
        &self,
        message: String,
        history: Vec<MessageJs>,
    ) -> napi::Result<StreamIterator> {
        self.stream_chat_with_history_and_options(message, history, None)
    }

    /// Stream chat with optional resume from event ID (durable streams).
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn stream_chat_with_options(
        &self,
        message: String,
        options: Option<StreamChatOptions>,
    ) -> napi::Result<StreamIterator> {
        self.stream_chat_with_history_and_options(message, Vec::new(), options)
    }

    /// Stream chat with conversation history and optional stream options.
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn stream_chat_with_history_and_options(
        &self,
        message: String,
        history: Vec<MessageJs>,
        options: Option<StreamChatOptions>,
    ) -> napi::Result<StreamIterator> {
        // resume_from would be used when agent has durable stream for StreamEvent; ignored for now
        let _ = options;
        let (tx, stream_rx) = mpsc::unbounded_channel::<AnyhowResult<StreamEvent>>();
        let agent_arc = Arc::clone(&self.inner);
        let message_clone = message;
        let cancellation_flag = Arc::new(AtomicBool::new(false));
        let cancellation_for_task = Arc::clone(&cancellation_flag);
        // Preserve multimodal content_parts in history conversion.
        let rust_history: Vec<caliche::messages::Message> =
            history.into_iter().map(message_js_to_rust).collect();
        let rt = get_runtime()?;
        rt.spawn(async move {
            let agent = agent_arc.read().await;
            let mut stream = if rust_history.is_empty() {
                agent.stream_chat_with_cancel_flag(
                    &message_clone,
                    Some(Arc::clone(&cancellation_for_task)),
                )
            } else {
                agent.stream_chat_with_history(&message_clone, rust_history)
            };
            let mut text_buffer = String::new();
            let mut buffer_count = 0usize;
            while let Some(item) = stream.next().await {
                if cancellation_for_task.load(Ordering::Relaxed) {
                    break;
                }
                match item {
                    Ok(StreamEvent::TextChunk { content }) => {
                        text_buffer.push_str(&content);
                        buffer_count += 1;
                        if buffer_count >= TEXT_CHUNK_BATCH_SIZE {
                            if !text_buffer.is_empty()
                                && tx
                                    .send(Ok(StreamEvent::TextChunk {
                                        content: std::mem::take(&mut text_buffer),
                                    }))
                                    .is_err()
                            {
                                break;
                            }
                            buffer_count = 0;
                        }
                    }
                    Ok(other) => {
                        if !text_buffer.is_empty() {
                            let _ = tx.send(Ok(StreamEvent::TextChunk {
                                content: std::mem::take(&mut text_buffer),
                            }));
                            buffer_count = 0;
                        }
                        if tx.send(Ok(other)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        if !text_buffer.is_empty() {
                            let _ = tx.send(Ok(StreamEvent::TextChunk {
                                content: std::mem::take(&mut text_buffer),
                            }));
                        }
                        if tx.send(Err(e)).is_err() {
                            break;
                        }
                    }
                }
            }
            if !text_buffer.is_empty() {
                let _ = tx.send(Ok(StreamEvent::TextChunk {
                    content: text_buffer,
                }));
            }
        });

        let boxed_stream: Pin<Box<dyn Stream<Item = AnyhowResult<StreamEvent>> + Send>> =
            Box::pin(UnboundedReceiverStream::new(stream_rx));

        Ok(StreamIterator {
            stream: Arc::new(tokio::sync::Mutex::new(boxed_stream)),
            pending_tool_call_ids: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            cancellation_flag: Some(cancellation_flag),
        })
    }
}

#[napi(object)]
pub struct StepMetricsJs {
    pub step: u32,
    #[napi(js_name = "llm_duration_ms")]
    pub llm_duration_ms: u32,
    #[napi(js_name = "tool_duration_ms")]
    pub tool_duration_ms: u32,
    #[napi(js_name = "tools_executed")]
    pub tools_executed: Vec<String>,
}

#[napi(object)]
pub struct PerformanceMetricsJs {
    #[napi(js_name = "total_duration_ms")]
    pub total_duration_ms: Option<u32>,
    #[napi(js_name = "llm_duration_ms")]
    pub llm_duration_ms: Option<u32>,
    #[napi(js_name = "tool_duration_ms")]
    pub tool_duration_ms: Option<u32>,
    #[napi(js_name = "overhead_ms")]
    pub overhead_ms: Option<u32>,
    #[napi(js_name = "step_count")]
    pub step_count: Option<u32>,
    #[napi(js_name = "time_to_first_token_ms")]
    pub time_to_first_token_ms: Option<u32>,
    pub steps: Option<Vec<StepMetricsJs>>,
}

#[napi(object)]
pub struct HarnessDisciplineMetricsJs {
    #[napi(js_name = "tokens_per_turn")]
    pub tokens_per_turn: Option<f64>,
    #[napi(js_name = "tool_calls")]
    pub tool_calls: u32,
    #[napi(js_name = "tool_calls_per_turn")]
    pub tool_calls_per_turn: Option<f64>,
    pub compactions: u32,
    #[napi(js_name = "readonly_filter_hits")]
    pub readonly_filter_hits: u32,
    #[napi(js_name = "estimated_cost_usd")]
    pub estimated_cost_usd: Option<f64>,
    pub summary: Option<String>,
}

#[napi(object)]
pub struct ChatResponse {
    pub response: String,
    pub usage: TokenUsageJs,
    pub metrics: Option<PerformanceMetricsJs>,
}

#[napi(object)]
pub struct TokenUsageJs {
    #[napi(js_name = "prompt_tokens")]
    pub prompt_tokens: Option<u32>,
    #[napi(js_name = "completion_tokens")]
    pub completion_tokens: Option<u32>,
    #[napi(js_name = "total_tokens")]
    pub total_tokens: Option<u32>,
    #[napi(js_name = "embedding_tokens")]
    pub embedding_tokens: Option<u32>,
    #[napi(js_name = "cache_creation_tokens")]
    pub cache_creation_tokens: Option<u32>,
    #[napi(js_name = "cache_read_tokens")]
    pub cache_read_tokens: Option<u32>,
}

#[napi(object)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[napi(object)]
pub struct MessageJs {
    pub role: String,
    pub content: String,
    #[napi(js_name = "content_parts")]
    pub content_parts: Option<Vec<ContentPartJs>>,
}

#[napi(object)]
pub struct ContentPartJs {
    #[napi(js_name = "type")]
    pub part_type: String,
    pub text: Option<String>,
    pub data: Option<String>,
    #[napi(js_name = "media_type")]
    pub media_type: Option<String>,
    pub url: Option<String>,
    pub filename: Option<String>,
}

#[napi(object)]
pub struct UserContextJs {
    #[napi(js_name = "user_id")]
    pub user_id: String,
    pub metadata: Value,
}

#[napi(object)]
pub struct ToolCallJs {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[napi(object)]
pub struct StepResultJs {
    pub response: String,
    #[napi(js_name = "tool_calls")]
    pub tool_calls: Vec<ToolCallJs>,
    #[napi(js_name = "updated_messages")]
    pub updated_messages: Vec<MessageJs>,
    pub usage: TokenUsageJs,
}

impl From<TokenUsage> for TokenUsageJs {
    fn from(usage: TokenUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            embedding_tokens: usage.embedding_tokens,
            cache_creation_tokens: usage.cache_creation_tokens,
            cache_read_tokens: usage.cache_read_tokens,
        }
    }
}

fn u64_to_u32(value: u64, field: &str) -> napi::Result<u32> {
    u32::try_from(value)
        .map_err(|_| napi::Error::from_reason(format!("{field} exceeds the JavaScript API limit")))
}

impl TryFrom<caliche::messages::StepMetrics> for StepMetricsJs {
    type Error = napi::Error;

    fn try_from(m: caliche::messages::StepMetrics) -> napi::Result<Self> {
        Ok(Self {
            step: m.step,
            llm_duration_ms: u64_to_u32(m.llm_duration_ms, "LLM duration")?,
            tool_duration_ms: u64_to_u32(m.tool_duration_ms, "tool duration")?,
            tools_executed: m.tools_executed,
        })
    }
}

impl TryFrom<caliche::messages::PerformanceMetrics> for PerformanceMetricsJs {
    type Error = napi::Error;

    fn try_from(m: caliche::messages::PerformanceMetrics) -> napi::Result<Self> {
        Ok(Self {
            total_duration_ms: m
                .total_duration_ms
                .map(|value| u64_to_u32(value, "total duration"))
                .transpose()?,
            llm_duration_ms: m
                .llm_duration_ms
                .map(|value| u64_to_u32(value, "LLM duration"))
                .transpose()?,
            tool_duration_ms: m
                .tool_duration_ms
                .map(|value| u64_to_u32(value, "tool duration"))
                .transpose()?,
            overhead_ms: m
                .overhead_ms
                .map(|value| u64_to_u32(value, "overhead duration"))
                .transpose()?,
            step_count: m.step_count,
            time_to_first_token_ms: m
                .time_to_first_token_ms
                .map(|value| u64_to_u32(value, "time to first token"))
                .transpose()?,
            steps: m
                .steps
                .map(|steps| {
                    steps
                        .into_iter()
                        .map(TryInto::try_into)
                        .collect::<napi::Result<Vec<_>>>()
                })
                .transpose()?,
        })
    }
}

impl TryFrom<caliche::harness::HarnessDisciplineMetrics> for HarnessDisciplineMetricsJs {
    type Error = napi::Error;

    fn try_from(metrics: caliche::harness::HarnessDisciplineMetrics) -> napi::Result<Self> {
        Ok(Self {
            tokens_per_turn: metrics.tokens_per_turn,
            tool_calls: u32::try_from(metrics.tool_calls)
                .map_err(|_| napi::Error::from_reason("tool call count exceeds the API limit"))?,
            tool_calls_per_turn: metrics.tool_calls_per_turn,
            compactions: u32::try_from(metrics.compactions)
                .map_err(|_| napi::Error::from_reason("compaction count exceeds the API limit"))?,
            readonly_filter_hits: u32::try_from(metrics.readonly_filter_hits).map_err(|_| {
                napi::Error::from_reason("readonly filter hit count exceeds the API limit")
            })?,
            estimated_cost_usd: metrics.estimated_cost_usd,
            summary: Some(metrics.summary_line()),
        })
    }
}

fn performance_metrics_to_json(
    metrics: &caliche::messages::PerformanceMetrics,
) -> napi::Result<serde_json::Value> {
    let metrics = PerformanceMetricsJs::try_from(metrics.clone())?;
    let steps = metrics.steps.map(|steps| {
        steps
            .into_iter()
            .map(|step| {
                serde_json::json!({
                    "step": step.step,
                    "llm_duration_ms": step.llm_duration_ms,
                    "tool_duration_ms": step.tool_duration_ms,
                    "tools_executed": step.tools_executed,
                })
            })
            .collect::<Vec<_>>()
    });
    Ok(serde_json::json!({
        "total_duration_ms": metrics.total_duration_ms,
        "llm_duration_ms": metrics.llm_duration_ms,
        "tool_duration_ms": metrics.tool_duration_ms,
        "overhead_ms": metrics.overhead_ms,
        "step_count": metrics.step_count,
        "time_to_first_token_ms": metrics.time_to_first_token_ms,
        "steps": steps,
    }))
}

/// Stream iterator for streaming chat responses
/// Options for stream chat (e.g. resume from durable stream).
#[napi(object)]
pub struct StreamChatOptions {
    /// Resume from this event ID when using a durable stream backend.
    pub resume_from: Option<String>,
}

/// Errors are converted to `StreamEvent::Metadata` events with error information
#[napi]
pub struct StreamIterator {
    stream:
        Arc<tokio::sync::Mutex<Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send>>>>,
    pending_tool_call_ids: Arc<tokio::sync::Mutex<HashMap<String, Vec<String>>>>,
    cancellation_flag: Option<Arc<AtomicBool>>,
}

impl StreamIterator {
    /// Internal method to get the next event from the stream
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn next_event(&self) -> napi::Result<Option<Value>> {
        let mut stream_guard = self.stream.lock().await;
        let event = stream_guard.next().await;
        drop(stream_guard);
        match event {
            Some(Ok(event)) => {
                let mut pending = self.pending_tool_call_ids.lock().await;
                let json_value = stream_event_to_js_value(&event, &mut pending, None)?;
                drop(pending);
                Ok(Some(json_value))
            }
            Some(Err(e)) => Ok(Some(serde_json::json!({
                "type": "error",
                "message": format!("{}", e),
                "recoverable": false,
                "event_id": serde_json::Value::Null
            }))),
            None => Ok(None),
        }
    }
}

fn tool_call_event_to_json(
    event: &StreamEvent,
    pending_tool_call_ids: &mut HashMap<String, Vec<String>>,
) -> Option<serde_json::Value> {
    match event {
        StreamEvent::ToolCallStart {
            id,
            name,
            arguments,
            thought_signature,
            action_id,
        } => {
            pending_tool_call_ids
                .entry(name.to_string())
                .or_default()
                .push(id.to_string());
            let mut value = serde_json::json!({
                "type": "tool_call_start",
                "id": id,
                "name": name,
                "arguments": arguments
            });
            insert_optional_string(&mut value, "thought_signature", thought_signature);
            insert_optional_string(&mut value, ACTION_ID_FIELD, action_id);
            Some(value)
        }
        StreamEvent::ToolCallResult {
            name,
            result,
            typed_payload,
            id: result_id,
            action_id,
        } => {
            let queue = pending_tool_call_ids.entry(name.to_string()).or_default();
            let queued_id = if queue.is_empty() {
                None
            } else {
                Some(queue.remove(0))
            };
            if queue.is_empty() {
                pending_tool_call_ids.remove(name.as_ref());
            }
            let id = match (result_id.as_deref(), queued_id) {
                (Some(id), _) => id.to_string(),
                (None, Some(id)) => id,
                (None, None) => name.to_string(),
            };
            let payload = typed_payload.clone().unwrap_or_else(|| {
                serde_json::from_str(result)
                    .unwrap_or_else(|_| serde_json::Value::String(result.clone()))
            });
            let mut value = serde_json::json!({
                "type": "tool_call_result",
                "id": id,
                "name": name,
                "result": {"success": true, "content": result, "payload": payload}
            });
            insert_optional_string(&mut value, ACTION_ID_FIELD, action_id);
            Some(value)
        }
        _ => None,
    }
}

fn insert_optional_string(value: &mut serde_json::Value, key: &str, field: &Option<String>) {
    let Some(text) = field else {
        return;
    };
    let Some(map) = value.as_object_mut() else {
        return;
    };
    map.insert(key.to_string(), serde_json::Value::String(text.clone()));
}

fn insert_stream_event_id(
    mut value: serde_json::Value,
    event_id: Option<&str>,
) -> serde_json::Value {
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "event_id".to_string(),
            event_id.map_or(serde_json::Value::Null, |id| {
                serde_json::Value::String(id.to_string())
            }),
        );
    }
    value
}

fn metadata_stream_event_to_json(
    usage: Option<&TokenUsage>,
    finish_reason: Option<&str>,
    metrics: Option<&caliche::messages::PerformanceMetrics>,
) -> napi::Result<serde_json::Value> {
    let metrics = metrics.map(performance_metrics_to_json).transpose()?;
    Ok(serde_json::json!({
        "type": "metadata",
        "usage": usage,
        "finish_reason": finish_reason,
        "metrics": metrics,
        "metadata": {}
    }))
}

/// Serialize `StreamEvent` to JS Value, optionally with `event_id` for resume.
fn stream_event_to_js_value(
    event: &StreamEvent,
    pending_tool_call_ids: &mut HashMap<String, Vec<String>>,
    event_id: Option<&str>,
) -> napi::Result<serde_json::Value> {
    let value = match event {
        StreamEvent::ToolCallStart { .. } | StreamEvent::ToolCallResult { .. } => {
            tool_call_event_to_json(event, pending_tool_call_ids).ok_or_else(|| {
                napi::Error::from_reason("failed to convert tool-call stream event")
            })?
        }
        StreamEvent::Metadata {
            usage,
            finish_reason,
            metrics,
        } => metadata_stream_event_to_json(
            usage.as_ref(),
            finish_reason.as_deref(),
            metrics.as_ref(),
        )?,
        StreamEvent::TextChunk { content } => serde_json::json!({
            "type": "text_chunk",
            "content": content
        }),
        StreamEvent::ThinkingChunk { content, action_id } => {
            let mut value = serde_json::json!({
                "type": "thinking_chunk",
                "content": content
            });
            insert_optional_string(&mut value, ACTION_ID_FIELD, action_id);
            value
        }
        StreamEvent::AudioChunk { data, format } => serde_json::json!({
            "type": "audio_chunk",
            "data": data,
            "format": format
        }),
        StreamEvent::ImageGenerated {
            data,
            media_type,
            revised_prompt,
        } => serde_json::json!({
            "type": "image_generated",
            "data": data,
            "media_type": media_type,
            "revised_prompt": revised_prompt
        }),
        StreamEvent::AttachmentProcessed {
            filename,
            attachment_type,
            summary,
        } => serde_json::json!({
            "type": "attachment_processed",
            "filename": filename,
            "attachment_type": attachment_type,
            "summary": summary
        }),
        StreamEvent::LimitReached {
            max_iterations,
            usage,
        } => serde_json::json!({
            "type": "limit_reached",
            "max_iterations": max_iterations,
            "usage": usage
        }),
        StreamEvent::Compaction {
            tokens_before,
            tokens_after,
            messages_removed,
            summary,
            report,
        } => serde_json::json!({
            "type": "compaction",
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "messages_removed": messages_removed,
            "summary": summary,
            "report": report
        }),
        StreamEvent::PhaseTransition {
            phase_index,
            phase_total,
            tool_names,
        } => serde_json::json!({
            "type": "phase_transition",
            "phase_index": phase_index,
            "phase_total": phase_total,
            "tool_names": tool_names
        }),
        StreamEvent::Status { message } => serde_json::json!({
            "type": "status",
            "message": message
        }),
        StreamEvent::Done => serde_json::json!({
            "type": "done"
        }),
        StreamEvent::AgentStart => serde_json::json!({
            "type": "agent_start"
        }),
        StreamEvent::AgentEnd => serde_json::json!({
            "type": "agent_end"
        }),
        StreamEvent::TurnStart => serde_json::json!({
            "type": "turn_start"
        }),
        StreamEvent::TurnEnd => serde_json::json!({
            "type": "turn_end"
        }),
    };
    Ok(insert_stream_event_id(value, event_id))
}

/// Standalone function for getting next stream event
#[napi(js_name = "streamNext")]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub async fn stream_next(iter: &StreamIterator) -> napi::Result<Option<Value>> {
    let mut stream_guard = iter.stream.lock().await;
    let event = stream_guard.next().await;
    drop(stream_guard);
    match event {
        Some(Ok(event)) => {
            let mut pending = iter.pending_tool_call_ids.lock().await;
            let json_value = stream_event_to_js_value(&event, &mut pending, None)?;
            drop(pending);
            Ok(Some(json_value))
        }
        Some(Err(e)) => Ok(Some(serde_json::json!({
            "type": "error",
            "message": format!("{}", e),
            "recoverable": false,
            "event_id": serde_json::Value::Null
        }))),
        None => Ok(None),
    }
}

#[napi(js_name = "streamCancel", catch_unwind)]
pub fn stream_cancel(iter: &StreamIterator) {
    if let Some(flag) = &iter.cancellation_flag {
        flag.store(true, Ordering::Relaxed);
    }
}

/// Shared state container for agents
#[napi]
pub struct SharedState {
    inner: Arc<RustSharedState>,
}

#[napi]
impl SharedState {
    /// Create a new shared state
    #[napi(constructor, catch_unwind)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RustSharedState::new()),
        }
    }

    /// Get a value from the state using a JSON pointer path
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn get(&self, path: String) -> napi::Result<Option<String>> {
        let result = self.inner.get(&path).await.map_err(|e| {
            napi::Error::from_reason(format!("Failed to get value from shared state: {e}"))
        })?;
        match result {
            Some(value) => Ok(Some(serde_json::to_string(&value).map_err(|e| {
                napi::Error::from_reason(format!("Failed to serialize value: {e}"))
            })?)),
            None => Ok(None),
        }
    }

    /// Set a value at a JSON pointer path
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn set(&self, path: String, value_json: String) -> napi::Result<()> {
        let value: Value = serde_json::from_str(&value_json)
            .map_err(|e| napi::Error::from_reason(format!("Failed to parse value JSON: {e}")))?;
        self.inner.set(&path, value).await.map_err(|e| {
            napi::Error::from_reason(format!("Failed to set value in shared state: {e}"))
        })?;
        Ok(())
    }

    /// Get the entire state as JSON string
    #[napi]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub async fn get_all(&self) -> napi::Result<String> {
        let value = self.inner.get_all().await;
        serde_json::to_string(&value)
            .map_err(|e| napi::Error::from_reason(format!("Failed to serialize state: {e}")))
    }
}

/// Agent flow builder
#[napi]
pub struct FlowBuilder {
    inner: RwLock<RustFlowBuilder>,
}

#[napi]
impl FlowBuilder {
    /// Create a new flow builder
    #[napi(constructor, catch_unwind)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(RustFlowBuilder::new()),
        }
    }

    /// Add an agent node to the flow
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn add_node(&self, name: String, agent: &Agent) -> napi::Result<()> {
        let agent_clone = {
            let agent_guard = agent.inner.read().await;
            agent_guard.clone_agent()
        };
        let mut builder_guard = rwlock_write(&self.inner)?;
        let builder = std::mem::take(&mut *builder_guard);
        *builder_guard = builder.add_node(name, agent_clone);
        drop(builder_guard);
        Ok(())
    }

    /// Set the entry point for the flow
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_entry_point(&self, name: String) -> napi::Result<()> {
        let mut builder_guard = rwlock_write(&self.inner)?;
        let builder = std::mem::take(&mut *builder_guard);
        *builder_guard = builder.set_entry_point(name);
        drop(builder_guard);
        Ok(())
    }

    /// Build the agent flow
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn build(&self) -> napi::Result<AgentFlow> {
        let mut builder_guard = rwlock_write(&self.inner)?;
        let builder = std::mem::take(&mut *builder_guard);
        drop(builder_guard);
        let flow = builder
            .build()
            .map_err(|e| napi::Error::from_reason(format!("Failed to build flow: {e}")))?;
        Ok(AgentFlow {
            inner: Arc::new(flow),
        })
    }
}

/// Agent flow for executing multi-agent workflows
#[napi]
#[expect(
    dead_code,
    reason = "the handle owns the flow, and no exported method reads it yet"
)]
pub struct AgentFlow {
    inner: Arc<RustAgentFlow>,
}

/// LLM client builder for constructing non-agentic LLM clients
#[napi]
pub struct LlmClientBuilder {
    inner: RwLock<RustLlmClientBuilder>,
}

#[napi]
impl LlmClientBuilder {
    /// Create a new `LlmClientBuilder`
    #[napi(constructor, catch_unwind)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(RustLlmClientBuilder::new()),
        }
    }

    /// Set the provider type
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_provider(&self, provider: String) -> napi::Result<()> {
        let provider_type = parse_provider_type(&provider)?;
        let mut inner = rwlock_write(&self.inner)?;
        *inner = (*inner).clone().provider(provider_type);
        drop(inner);
        Ok(())
    }

    /// Set the model name
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_model(&self, model: String) -> napi::Result<()> {
        let mut inner = rwlock_write(&self.inner)?;
        *inner = (*inner).clone().model(model);
        drop(inner);
        Ok(())
    }

    /// Set the API key for authentication
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_api_key(&self, key: String) -> napi::Result<()> {
        let mut inner = rwlock_write(&self.inner)?;
        *inner = (*inner).clone().api_key(key);
        drop(inner);
        Ok(())
    }

    /// Set API key scoped to a specific provider.
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_api_key_for(&self, target: String, key: String) -> napi::Result<()> {
        let provider = parse_provider_type(&target)?;
        let mut inner = rwlock_write(&self.inner)?;
        *inner = (*inner).clone().api_key_for(provider, key);
        drop(inner);
        Ok(())
    }

    /// Set AWS credentials for Bedrock provider
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_aws_credentials(
        &self,
        access_key: String,
        secret_key: String,
        region: String,
        session_token: Option<String>,
    ) -> napi::Result<()> {
        let mut inner = rwlock_write(&self.inner)?;
        let credentials = AwsCredentials {
            access_key_id: Some(access_key),
            secret_access_key: Some(secret_key),
            session_token,
            region: Some(region),
            bearer_token: None,
        };
        *inner = (*inner).clone().aws_credentials(credentials);
        drop(inner);
        Ok(())
    }

    /// Set base URL (for `LMStudio` or custom endpoints)
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_base_url(&self, url: String) -> napi::Result<()> {
        let mut inner = rwlock_write(&self.inner)?;
        *inner = (*inner).clone().base_url(url);
        drop(inner);
        Ok(())
    }

    /// Set temperature
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_temperature(&self, temp: f64) -> napi::Result<()> {
        let mut inner = rwlock_write(&self.inner)?;
        *inner = (*inner)
            .clone()
            .temperature(checked_f64_to_f32(temp, "temperature")?);
        drop(inner);
        Ok(())
    }

    /// Set max tokens
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_max_tokens(&self, tokens: u32) -> napi::Result<()> {
        let mut inner = rwlock_write(&self.inner)?;
        *inner = (*inner).clone().max_tokens(tokens);
        drop(inner);
        Ok(())
    }

    /// Set timeout in seconds
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_timeout(&self, seconds: u32) -> napi::Result<()> {
        let mut inner = rwlock_write(&self.inner)?;
        *inner = (*inner)
            .clone()
            .timeout(std::time::Duration::from_secs(u64::from(seconds)));
        drop(inner);
        Ok(())
    }

    /// Set max retries
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_max_retries(&self, retries: u32) -> napi::Result<()> {
        let mut inner = rwlock_write(&self.inner)?;
        *inner = (*inner).clone().max_retries(retries as usize);
        drop(inner);
        Ok(())
    }

    /// Set tools
    #[napi(catch_unwind)]
    /// # Errors
    ///
    /// Returns a NAPI error if validation, synchronization, serialization,
    /// or the delegated runtime operation fails.
    pub fn set_tools(&self, tools: Vec<serde_json::Value>) -> napi::Result<()> {
        let mut inner = rwlock_write(&self.inner)?;
        let mut tool_schemas = Vec::new();
        for mut v in tools {
            // Normalize tool schema: ensure parameters.required exists (even if empty)
            if let Some(params) = v.get_mut("parameters").and_then(|p| p.as_object_mut())
                && !params.contains_key("required")
            {
                params.insert("required".to_string(), serde_json::json!([]));
            }
            let schema: ToolSchema = serde_json::from_value(v).map_err(|e| {
                napi::Error::from_reason(format!("Failed to parse tool schema: {e}"))
            })?;
            tool_schemas.push(schema);
        }
        *inner = (*inner).clone().tools(tool_schemas);
        drop(inner);
        Ok(())
    }

    /// Set tool choice
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_tool_choice(&self, choice: String) -> napi::Result<()> {
        let mut inner = rwlock_write(&self.inner)?;
        let tool_choice = match choice.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            name => ToolChoice::Specific(beef::lean::Cow::owned(name.to_string())),
        };
        *inner = (*inner).clone().tool_choice(tool_choice);
        drop(inner);
        Ok(())
    }

    /// Build the LLM client
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn build(&self) -> napi::Result<LlmClient> {
        let builder = {
            let inner = rwlock_read(&self.inner)?;
            let builder = inner.clone();
            drop(inner);
            builder
        };
        let client = builder
            .build()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to build LLM client: {e}")))?;
        Ok(LlmClient {
            inner: std::sync::Arc::new(client),
        })
    }
}

/// LLM client for non-agentic completions
#[napi]
pub struct LlmClient {
    inner: Arc<RustLlmClient>,
}

fn parse_tiered_tiers_from_json(
    json: &str,
) -> anyhow::Result<Vec<(VectorBackendConfig, DripstoneTierConfig)>> {
    tiered_tiers_from_json_str(json)
}

#[cfg(feature = "vector-store-postgres")]
fn json_to_hnsw_config(v: &Value) -> anyhow::Result<HnswConfig> {
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow!("hnsw_config must be object"))?;
    let mut hnsw = HnswConfig::default();
    if let Some(m) = obj.get("m").and_then(|v| v.as_u64()) {
        hnsw.m = m as usize;
    }
    if let Some(ef) = obj
        .get("ef_construction")
        .or(obj.get("efConstruction"))
        .and_then(|v| v.as_u64())
    {
        hnsw.ef_construction = ef as usize;
    }
    if let Some(ef) = obj
        .get("ef_search")
        .or(obj.get("efSearch"))
        .and_then(|v| v.as_u64())
    {
        hnsw.ef_search = ef as usize;
    }
    if let Some(max) = obj
        .get("max_nb_connection")
        .or(obj.get("maxNbConnection"))
        .and_then(|v| v.as_u64())
    {
        hnsw.max_nb_connection = max as usize;
    }
    if let Some(v) = obj
        .get("rerank_overfetch")
        .or(obj.get("rerankOverfetch"))
        .and_then(|v| v.as_u64())
    {
        hnsw.rerank_overfetch = v as usize;
    }
    if let Some(s) = obj
        .get("graph_space")
        .or(obj.get("graphSpace"))
        .and_then(|v| v.as_str())
    {
        hnsw.graph_space = match s.to_lowercase().as_str() {
            "cosine_compatible" | "cosine" => HnswGraphSpace::CosineCompatible,
            _ => HnswGraphSpace::RawL2,
        };
    }
    Ok(hnsw)
}

/// Configure buffer memory on a builder
#[napi(js_name = "builderBufferMemory", catch_unwind)]
pub fn builder_buffer_memory(
    builder: &AgentBuilder,
    max_tokens: u32,
    min_messages: u32,
    preserve_system: bool,
    summary_placeholder: bool,
) {
    let _ = builder.buffer_memory(
        max_tokens,
        min_messages,
        preserve_system,
        summary_placeholder,
    );
}

/// Configure summary memory on a builder
#[napi(js_name = "builderSummaryMemory", catch_unwind)]
pub fn builder_summary_memory(
    builder: &AgentBuilder,
    max_tokens: u32,
    summarize_threshold_pct: f64,
    chunk_size: u32,
    max_summaries: u32,
) {
    let _ = builder.summary_memory(
        max_tokens,
        summarize_threshold_pct,
        chunk_size,
        max_summaries,
    );
}

/// Configure entity memory on a builder  
#[napi(js_name = "builderEntityMemory", catch_unwind)]
pub fn builder_entity_memory(
    builder: &AgentBuilder,
    max_tokens: u32,
    max_entities: u32,
    extraction_interval: u32,
    similarity_threshold: f64,
) {
    let _ = builder.entity_memory(
        max_tokens,
        max_entities,
        extraction_interval,
        similarity_threshold,
    );
}

/// Configure in-memory vector backend on a builder
#[napi(js_name = "builderSetVectorBackendMemory", catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn builder_set_vector_backend_memory(
    builder: &AgentBuilder,
    ttl_secs: Option<u32>,
    max_entries: Option<u32>,
) -> napi::Result<()> {
    builder.set_vector_backend_memory(ttl_secs, max_entries)
}

/// Configure Redis vector backend on a builder
#[cfg(feature = "vector-store-redis")]
#[napi(js_name = "builderSetVectorBackendRedis", catch_unwind)]
pub fn builder_set_vector_backend_redis(
    builder: &AgentBuilder,
    url: String,
    prefix: String,
) -> napi::Result<()> {
    builder.set_vector_backend_redis(url, prefix)
}

/// Configure PostgreSQL vector backend on a builder
#[cfg(feature = "vector-store-postgres")]
#[napi(js_name = "builderSetVectorBackendPostgres", catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn builder_set_vector_backend_postgres(
    builder: &AgentBuilder,
    connection_string: String,
    table_name: String,
    vector_dimension: Option<u32>,
    max_connections: Option<u32>,
    hnsw_config: Option<Value>,
) -> napi::Result<()> {
    builder.set_vector_backend_postgres(
        connection_string,
        table_name,
        vector_dimension,
        max_connections,
        hnsw_config,
    )
}

/// Configure Qdrant HTTP vector backend on a builder
#[cfg(feature = "vector-store-qdrant")]
#[napi(js_name = "builderSetVectorBackendQdrant", catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn builder_set_vector_backend_qdrant(
    builder: &AgentBuilder,
    url: String,
    vector_dimension: u32,
    collection_prefix: Option<String>,
    api_key: Option<String>,
) -> napi::Result<()> {
    builder.set_vector_backend_qdrant(url, vector_dimension, collection_prefix, api_key)
}

/// Configure Qdrant gRPC vector backend on a builder
#[cfg(feature = "vector-store-qdrant-grpc")]
#[napi(js_name = "builderSetVectorBackendQdrantGrpc", catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn builder_set_vector_backend_qdrant_grpc(
    builder: &AgentBuilder,
    grpc_url: String,
    vector_dimension: u32,
    collection_prefix: Option<String>,
    api_key: Option<String>,
    batch_size: Option<u32>,
) -> napi::Result<()> {
    builder.set_vector_backend_qdrant_grpc(
        grpc_url,
        vector_dimension,
        collection_prefix,
        api_key,
        batch_size,
    )
}

/// Configure Pinecone vector backend on a builder
#[cfg(feature = "vector-store-pinecone")]
#[napi(js_name = "builderSetVectorBackendPinecone", catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn builder_set_vector_backend_pinecone(
    builder: &AgentBuilder,
    base_url: String,
    api_key: String,
    dimension: u32,
    timeout_secs: Option<u32>,
) -> napi::Result<()> {
    builder.set_vector_backend_pinecone(base_url, api_key, dimension, timeout_secs)
}

/// Configure Chroma vector backend on a builder
#[cfg(feature = "vector-store-chroma")]
#[napi(js_name = "builderSetVectorBackendChroma", catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn builder_set_vector_backend_chroma(
    builder: &AgentBuilder,
    base_url: String,
    collection_prefix: Option<String>,
    expected_dimension: Option<u32>,
    timeout_secs: Option<u32>,
) -> napi::Result<()> {
    builder.set_vector_backend_chroma(
        base_url,
        collection_prefix,
        expected_dimension,
        timeout_secs,
    )
}

/// Configure S3 vector backend on a builder
#[cfg(feature = "vector-store-s3-backend")]
#[napi(js_name = "builderSetVectorBackendS3", catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn builder_set_vector_backend_s3(
    builder: &AgentBuilder,
    bucket: String,
    key_prefix: String,
    enable_compression: Option<bool>,
    batch_size: Option<u32>,
    cache_indices: Option<bool>,
) -> napi::Result<()> {
    builder.set_vector_backend_s3(
        bucket,
        key_prefix,
        enable_compression,
        batch_size,
        cache_indices,
    )
}

/// Configure ColdForge vector backend on a builder
#[cfg(feature = "vector-store-coldforge")]
#[napi(js_name = "builderSetVectorBackendColdforge", catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn builder_set_vector_backend_coldforge(
    builder: &AgentBuilder,
    bucket: String,
    key_prefix: String,
    region: String,
    vector_dimension: u32,
    cache_size_mb: Option<u32>,
    num_centroids: Option<u32>,
) -> napi::Result<()> {
    builder.set_vector_backend_coldforge(
        bucket,
        key_prefix,
        region,
        vector_dimension,
        cache_size_mb,
        num_centroids,
    )
}

/// Configure DynamoDB native vector backend on a builder
#[cfg(feature = "vector-store-dynamodb")]
#[napi(js_name = "builderSetVectorBackendDynamodb", catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn builder_set_vector_backend_dynamodb(
    builder: &AgentBuilder,
    table_name: String,
    index_name: String,
    region: String,
    vector_dimension: u32,
    ensure_table: Option<bool>,
) -> napi::Result<()> {
    builder.set_vector_backend_dynamodb(
        table_name,
        index_name,
        region,
        vector_dimension,
        ensure_table,
    )
}

/// Auto-configure vector backend from environment variables
#[napi(js_name = "builderVectorBackendFromEnv", catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn builder_vector_backend_from_env(builder: &AgentBuilder) -> napi::Result<()> {
    builder.vector_backend_from_env()
}

/// Enable or disable vector metrics collection on a builder
#[napi(js_name = "builderWithMetrics", catch_unwind)]
/// # Errors
///
/// Returns a NAPI error if validation, synchronization, serialization,
/// or the delegated runtime operation fails.
pub fn builder_with_metrics(builder: &AgentBuilder, enabled: bool) -> napi::Result<()> {
    builder.with_metrics(enabled)
}

/// Stream chat from an agent
#[napi(js_name = "agentStreamChat", catch_unwind)]
/// # Errors
///
/// Returns a NAPI error if validation, synchronization, serialization,
/// or the delegated runtime operation fails.
pub fn agent_stream_chat(agent: &Agent, message: String) -> napi::Result<StreamIterator> {
    agent.stream_chat(message)
}

/// Create a `SequentialChain` (wrapper for TypeScript compatibility)
#[napi(js_name = "createSequentialChain", catch_unwind)]
#[must_use]
pub fn create_sequential_chain() -> chains::SequentialChain {
    chains::SequentialChain::new()
}

/// Create an `LLMChain` from an Agent instance (wrapper for TypeScript compatibility)
#[napi(js_name = "LlmChainFromAgent")]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub async fn llm_chain_from_agent(
    agent: &Agent,
    prompt_template: String,
) -> napi::Result<chains::LLMChain> {
    use std::sync::Arc;

    let agent_service = {
        let agent_guard = agent.inner.read().await;
        let provider = agent_guard.provider();
        let model = agent_guard.model().to_string();
        drop(agent_guard);
        caliche::AgentService::from_config(
            caliche::AgentServiceConfig::new().with_provider(Some(provider), Some(model)),
        )
        .await
    }
    .map_err(|e| napi::Error::from_reason(format!("Failed to create agent service: {e}")))?;

    Ok(chains::LLMChain::from_agent_internal(
        Arc::new(tokio::sync::Mutex::new(agent_service)),
        &prompt_template,
    ))
}

/// Create a `MapReduceChain` from Agent instances (wrapper for TypeScript compatibility)
#[napi(js_name = "MapReduceChainFromAgents")]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub async fn map_reduce_chain_from_agents(
    map_agent: &Agent,
    map_prompt: String,
    reduce_agent: &Agent,
    reduce_prompt: String,
) -> napi::Result<chains::MapReduceChain> {
    use caliche::chains::{Chain, LLMChain as RustLLMChain};
    use caliche::prompts::PromptTemplate;
    use std::sync::Arc;

    let map_agent_service = {
        let agent_guard = map_agent.inner.read().await;
        let provider = agent_guard.provider();
        let model = agent_guard.model().to_string();
        drop(agent_guard);
        caliche::AgentService::from_config(
            caliche::AgentServiceConfig::new().with_provider(Some(provider), Some(model)),
        )
        .await
    }
    .map_err(|e| napi::Error::from_reason(format!("Failed to create map agent service: {e}")))?;

    let reduce_agent_service = {
        let agent_guard = reduce_agent.inner.read().await;
        let provider = agent_guard.provider();
        let model = agent_guard.model().to_string();
        drop(agent_guard);
        caliche::AgentService::from_config(
            caliche::AgentServiceConfig::new().with_provider(Some(provider), Some(model)),
        )
        .await
    }
    .map_err(|e| napi::Error::from_reason(format!("Failed to create reduce agent service: {e}")))?;

    let map_prompt_template = PromptTemplate::new(&map_prompt);
    let reduce_prompt_template = PromptTemplate::new(&reduce_prompt);

    let map_chain = Arc::new(RustLLMChain::new(
        Arc::new(tokio::sync::Mutex::new(map_agent_service)),
        map_prompt_template,
    )) as Arc<dyn Chain>;
    let reduce_chain = Arc::new(RustLLMChain::new(
        Arc::new(tokio::sync::Mutex::new(reduce_agent_service)),
        reduce_prompt_template,
    )) as Arc<dyn Chain>;

    Ok(chains::MapReduceChain::from_agents_internal(
        map_chain,
        reduce_chain,
    ))
}

/// Create a `WorkflowBuilder` (wrapper for TypeScript compatibility)
#[napi(js_name = "createWorkflowBuilder", catch_unwind)]
#[must_use]
pub fn create_workflow_builder(name: String) -> workflow::JsWorkflowBuilder {
    workflow::JsWorkflowBuilder::new(name)
}

/// Create a `WorkflowEngine` (wrapper for TypeScript compatibility)
#[napi(js_name = "createWorkflowEngine", catch_unwind)]
#[must_use]
pub fn create_workflow_engine() -> workflow::JsWorkflowEngine {
    workflow::JsWorkflowEngine::new()
}

/// Create an `ExactMatchEvaluator` (wrapper for TypeScript compatibility)
#[napi(js_name = "createExactMatchEvaluator", catch_unwind)]
#[must_use]
pub fn create_exact_match_evaluator(case_sensitive: bool) -> evaluation::ExactMatchEvaluator {
    evaluation::ExactMatchEvaluator::new(Some(case_sensitive))
}

/// Create a `ContainsEvaluator` (wrapper for TypeScript compatibility)
#[napi(js_name = "createContainsEvaluator", catch_unwind)]
#[must_use]
pub fn create_contains_evaluator(case_sensitive: bool) -> evaluation::ContainsEvaluator {
    evaluation::ContainsEvaluator::new(Some(case_sensitive))
}

/// Create an `LLMEvaluator` (wrapper for TypeScript compatibility)
///
/// # Errors
///
/// Returns a NAPI error when `provider` is not a known provider name.
#[napi(js_name = "createLLMEvaluator", catch_unwind)]
pub fn create_llm_evaluator(
    criteria: String,
    provider: Option<String>,
    model: Option<String>,
) -> napi::Result<evaluation::LLMEvaluator> {
    evaluation::LLMEvaluator::new(criteria, provider, model)
}

fn message_js_to_rust(msg: MessageJs) -> caliche::messages::Message {
    use caliche::messages::{ContentPart, MessageContent, MessageRole};
    let role = match msg.role.as_str() {
        "assistant" => MessageRole::Assistant,
        "system" => MessageRole::System,
        "tool" => MessageRole::Tool,
        _ => MessageRole::User,
    };
    if let Some(parts) = msg.content_parts {
        let converted_parts: Vec<ContentPart> = parts
            .into_iter()
            .filter_map(|part| match part.part_type.as_str() {
                "text" => part.text.map(|text| ContentPart::Text {
                    text: beef::lean::Cow::owned(text),
                }),
                "image" => Some(ContentPart::Image {
                    data: beef::lean::Cow::owned(part.data.unwrap_or_default()),
                    media_type: beef::lean::Cow::owned(
                        part.media_type.unwrap_or_else(|| "image/jpeg".to_string()),
                    ),
                }),
                "image_url" => part.url.map(|url| ContentPart::ImageUrl {
                    url: beef::lean::Cow::owned(url),
                }),
                "video" => Some(ContentPart::Video {
                    data: beef::lean::Cow::owned(part.data.unwrap_or_default()),
                    media_type: beef::lean::Cow::owned(
                        part.media_type.unwrap_or_else(|| "video/mp4".to_string()),
                    ),
                }),
                "video_url" => part.url.map(|url| ContentPart::VideoUrl {
                    url: beef::lean::Cow::owned(url),
                }),
                "audio" => Some(ContentPart::Audio {
                    data: beef::lean::Cow::owned(part.data.unwrap_or_default()),
                    media_type: beef::lean::Cow::owned(
                        part.media_type.unwrap_or_else(|| "audio/wav".to_string()),
                    ),
                }),
                "audio_url" => part.url.map(|url| ContentPart::AudioUrl {
                    url: beef::lean::Cow::owned(url),
                }),
                "document" => Some(ContentPart::Document {
                    data: beef::lean::Cow::owned(part.data.unwrap_or_default()),
                    media_type: beef::lean::Cow::owned(
                        part.media_type
                            .unwrap_or_else(|| "application/octet-stream".to_string()),
                    ),
                    filename: part.filename.map(beef::lean::Cow::owned),
                }),
                "document_url" => part.url.map(|url| ContentPart::DocumentUrl {
                    url: beef::lean::Cow::owned(url),
                    filename: part.filename.map(beef::lean::Cow::owned),
                }),
                _ => None,
            })
            .collect();
        if !converted_parts.is_empty() {
            return caliche::messages::Message {
                role,
                content: MessageContent::Multimodal(converted_parts),
                metadata: None,
            };
        }
    }
    caliche::messages::Message::new(role, msg.content)
}

/// Complete a conversation with messages (non-streaming)
#[napi(js_name = "llmClientComplete")]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub async fn llm_client_complete(
    client: &LlmClient,
    messages: Vec<MessageJs>,
) -> napi::Result<Value> {
    let rust_messages = messages.into_iter().map(message_js_to_rust).collect();
    let response = client
        .inner
        .complete(rust_messages)
        .await
        .map_err(|e| napi::Error::from_reason(format!("Completion failed: {e}")))?;

    serde_json::to_value(&response)
        .map_err(|e| napi::Error::from_reason(format!("Failed to serialize response: {e}")))
}

/// Forward stream events while coalescing adjacent text chunks across NAPI.
async fn forward_batched_text_chunks<S>(
    mut stream: S,
    tx: tokio::sync::mpsc::UnboundedSender<anyhow::Result<StreamEvent>>,
) where
    S: Stream<Item = anyhow::Result<StreamEvent>> + Unpin,
{
    const TEXT_CHUNK_BATCH_SIZE: usize = 5;
    let mut text_buffer = String::new();
    let mut buffer_count = 0usize;

    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::TextChunk { content }) => {
                text_buffer.push_str(&content);
                buffer_count += 1;
                if buffer_count >= TEXT_CHUNK_BATCH_SIZE {
                    if tx
                        .send(Ok(StreamEvent::TextChunk {
                            content: std::mem::take(&mut text_buffer),
                        }))
                        .is_err()
                    {
                        return;
                    }
                    buffer_count = 0;
                }
            }
            Ok(other) => {
                if !text_buffer.is_empty()
                    && tx
                        .send(Ok(StreamEvent::TextChunk {
                            content: std::mem::take(&mut text_buffer),
                        }))
                        .is_err()
                {
                    return;
                }
                buffer_count = 0;
                if tx.send(Ok(other)).is_err() {
                    return;
                }
            }
            Err(error) => {
                if !text_buffer.is_empty() {
                    let _ = tx.send(Ok(StreamEvent::TextChunk {
                        content: std::mem::take(&mut text_buffer),
                    }));
                }
                if tx.send(Err(error)).is_err() {
                    return;
                }
            }
        }
    }

    if !text_buffer.is_empty() {
        let _ = tx.send(Ok(StreamEvent::TextChunk {
            content: text_buffer,
        }));
    }
}

/// Stream a conversation with messages
#[napi(js_name = "llmClientStream", catch_unwind)]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn llm_client_stream(
    client: &LlmClient,
    messages: Vec<MessageJs>,
) -> napi::Result<StreamIterator> {
    use anyhow::Result as AnyhowResult;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::UnboundedReceiverStream;

    let rust_messages: Vec<caliche::messages::Message> =
        messages.into_iter().map(message_js_to_rust).collect();

    let (tx, stream_rx) = mpsc::unbounded_channel::<AnyhowResult<StreamEvent>>();
    let client_arc = Arc::clone(&client.inner);

    let rt = get_runtime()?;
    rt.spawn(async move {
        forward_batched_text_chunks(client_arc.stream(rust_messages), tx).await;
    });

    let boxed_stream: Pin<Box<dyn Stream<Item = AnyhowResult<StreamEvent>> + Send>> =
        Box::pin(UnboundedReceiverStream::new(stream_rx));
    Ok(StreamIterator {
        stream: Arc::new(tokio::sync::Mutex::new(boxed_stream)),
        pending_tool_call_ids: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        cancellation_flag: None,
    })
}

/// Complete a single text prompt (convenience method)
#[napi(js_name = "llmClientCompleteText")]
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub async fn llm_client_complete_text(client: &LlmClient, prompt: String) -> napi::Result<String> {
    client
        .inner
        .complete_text(prompt.as_str())
        .await
        .map_err(|e| napi::Error::from_reason(format!("Completion failed: {e}")))
}

/// Stream a single text prompt (convenience method)
#[napi(js_name = "llmClientStreamText", catch_unwind)]
/// # Errors
///
/// Returns a NAPI error if validation, synchronization, serialization,
/// or the delegated runtime operation fails.
pub fn llm_client_stream_text(client: &LlmClient, prompt: String) -> napi::Result<StreamIterator> {
    use anyhow::Result as AnyhowResult;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::UnboundedReceiverStream;

    let (tx, stream_rx) = mpsc::unbounded_channel::<AnyhowResult<StreamEvent>>();
    let client_arc = Arc::clone(&client.inner);

    let rt = get_runtime()?;
    rt.spawn(async move {
        forward_batched_text_chunks(client_arc.stream_text(prompt.as_str()), tx).await;
    });

    let boxed_stream: Pin<Box<dyn Stream<Item = AnyhowResult<StreamEvent>> + Send>> =
        Box::pin(UnboundedReceiverStream::new(stream_rx));
    Ok(StreamIterator {
        stream: Arc::new(tokio::sync::Mutex::new(boxed_stream)),
        pending_tool_call_ids: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        cancellation_flag: None,
    })
}

/// Get provider name from `LlmClient` (wrapper for TypeScript compatibility)
#[napi(js_name = "llmClientGetProviderName", catch_unwind)]
#[must_use]
pub fn llm_client_get_provider_name(client: &LlmClient) -> String {
    client.inner.provider_name().to_string()
}

/// Get model name from `LlmClient` (wrapper for TypeScript compatibility)
#[napi(js_name = "llmClientGetModel", catch_unwind)]
#[must_use]
pub fn llm_client_get_model(client: &LlmClient) -> String {
    client.inner.model().to_string()
}

// NAPI-RS automatically exports all #[napi] functions, so no manual registration is needed

#[cfg(test)]
mod tests {
    use super::{StreamEvent, forward_batched_text_chunks};
    use futures_util::stream;

    #[tokio::test]
    async fn batches_adjacent_llm_text_chunks_and_flushes_before_other_events() {
        let mut events = Vec::new();
        for index in 0..7 {
            events.push(Ok(StreamEvent::TextChunk {
                content: index.to_string(),
            }));
        }
        events.push(Ok(StreamEvent::TurnEnd));

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        forward_batched_text_chunks(stream::iter(events), tx).await;

        let mut received = Vec::new();
        while let Some(item) = rx.recv().await {
            if let Ok(event) = item {
                received.push(event);
            }
        }

        assert_eq!(received.len(), 3);
        assert!(matches!(
            &received[0],
            StreamEvent::TextChunk { content } if content == "01234"
        ));
        assert!(matches!(
            &received[1],
            StreamEvent::TextChunk { content } if content == "56"
        ));
        assert!(matches!(received[2], StreamEvent::TurnEnd));
    }
}
