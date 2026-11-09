//! Python bindings for Caliche

use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::sync::OnceLock;
use tokio::runtime::Runtime;

mod agent;
mod builder;
mod builtin_tools;
mod cache;
mod chains;
mod conversions;
mod coordinator;
mod durable_streams;
mod evaluation;
mod external_provider;
mod flow;
pub mod hooks;
mod in_process_mcp;
mod inspection;
mod jsonl_session;
mod llm_client;
mod loaders;
#[cfg(feature = "local-inference")]
mod local_inference;
mod mcp;
mod memory_config;
mod memory_persistence;
mod mutation;
mod observability;
mod output_parsers;
mod patterns;
#[cfg(feature = "s3")]
mod prompts_storage;
pub mod providers;
mod python_async;
#[cfg(feature = "tools-rag")]
mod rag;
#[cfg(feature = "tools-rag")]
mod rag_tool;
mod reasoning;
mod security;
mod session;
mod shared_state;
#[cfg(feature = "s3")]
mod storage;
mod streaming;
mod templates;
pub(crate) mod tool;
mod tool_registry;
mod tools;
mod workflow;

/// Shared Tokio runtime for all async operations
///
/// This runtime is lazily initialized on first use and reused across all
/// Python bindings to avoid the overhead of creating new runtimes.
static SHARED_RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();

fn init_shared_runtime() -> Result<Runtime, String> {
    match Runtime::new() {
        Ok(runtime) => Ok(runtime),
        Err(err) => {
            #[expect(
                clippy::print_stderr,
                reason = "surface Tokio runtime fallback to the Python process stderr"
            )]
            {
                eprintln!("failed to create multi-thread Tokio runtime: {err}");
            }
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|fallback_err| {
                    format!(
                        "failed to create Tokio runtime (multi-thread: {err}; current-thread fallback: {fallback_err})"
                    )
                })
        }
    }
}

/// Get or initialize the shared runtime
///
/// This function is thread-safe and will create the runtime on first call.
pub(crate) fn get_runtime() -> PyResult<&'static Runtime> {
    let runtime = SHARED_RUNTIME
        .get_or_init(init_shared_runtime)
        .as_ref()
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.clone()))?;
    // Share the same Tokio runtime with pyo3-async-runtimes (idempotent after first call).
    let _ = pyo3_async_runtimes::tokio::init_with_runtime(runtime);
    Ok(runtime)
}

/// Run a future on the shared Tokio runtime while releasing the Python GIL.
pub(crate) fn block_on_detached<F, T>(py: Python<'_>, fut: F) -> PyResult<T>
where
    F: std::future::Future<Output = T> + Send,
    T: Send,
{
    let rt = get_runtime()?;
    Ok(py.detach(|| rt.block_on(fut)))
}

fn add_runtime_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<agent::Agent>()?;
    m.add_class::<builder::AgentBuilder>()?;
    m.add_class::<durable_streams::DurableStreamBackendWrapper>()?;
    m.add_class::<llm_client::LlmClient>()?;
    m.add_class::<llm_client::LlmClientBuilder>()?;
    m.add_class::<llm_client::PyMessage>()?;
    m.add_class::<llm_client::PyAttachment>()?;
    m.add_class::<providers::ProviderType>()?;
    m.add_class::<tools::ToolResult>()?;
    m.add_class::<streaming::StreamIterator>()?;
    m.add_class::<memory_config::PyBufferMemoryConfig>()?;
    m.add_class::<memory_config::PySummaryMemoryConfig>()?;
    m.add_class::<memory_config::PyEntityMemoryConfig>()?;
    m.add_class::<memory_config::PyVectorMemoryConfig>()?;
    m.add_class::<shared_state::SharedState>()?;
    m.add_class::<hooks::PyPermissionMode>()?;
    m.add_class::<hooks::PyHookMatcher>()?;
    m.add_class::<hooks::PyHookContext>()?;
    m.add_class::<hooks::PyHookResult>()?;
    m.add_class::<mutation::PyMutationStyle>()?;
    m.add_class::<inspection::PyInspectionAction>()?;
    m.add_class::<inspection::PyInspectionResult>()?;
    m.add_class::<inspection::PyPathSecurityInspector>()?;
    m.add_class::<inspection::PyPermissionModeInspector>()?;
    m.add_class::<inspection::PyToolInspectionManager>()?;
    m.add_class::<inspection::PyInspectionHook>()?;
    m.add_class::<tool_registry::PyToolRegistry>()?;
    m.add_class::<jsonl_session::PyJsonlSessionHeader>()?;
    m.add_class::<jsonl_session::PyJsonlSessionEntry>()?;
    m.add_class::<jsonl_session::PyJsonlSessionTree>()?;
    m.add_class::<jsonl_session::PyJsonlSessionStore>()?;
    m.add_class::<session::PyAgentSession>()?;
    m.add_class::<flow::FlowBuilder>()?;
    m.add_class::<flow::AgentFlow>()?;
    Ok(())
}

fn add_orchestration_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<chains::PyLLMChain>()?;
    m.add_class::<chains::PySequentialChain>()?;
    m.add_class::<chains::PyMapReduceChain>()?;
    m.add_class::<chains::PyParallelChain>()?;
    m.add_class::<chains::PyChainBuilder>()?;
    m.add_class::<reasoning::PyReActPattern>()?;
    m.add_class::<reasoning::PyPlanExecutePattern>()?;
    m.add_class::<reasoning::PyReflexionPattern>()?;
    m.add_class::<reasoning::PySelfAskPattern>()?;
    m.add_class::<reasoning::PyTreeOfThoughtsPattern>()?;
    m.add_class::<workflow::PyWorkflowBuilder>()?;
    m.add_class::<workflow::PyWorkflow>()?;
    m.add_class::<workflow::PyWorkflowEngine>()?;
    m.add_class::<coordinator::PyAgentCoordinator>()?;
    m.add_class::<coordinator::PyMessageBus>()?;
    m.add_class::<evaluation::PyExactMatchEvaluator>()?;
    m.add_class::<evaluation::PyContainsEvaluator>()?;
    m.add_class::<evaluation::PyLLMEvaluator>()?;
    m.add_class::<evaluation::PyEvaluationDataset>()?;
    m.add_class::<templates::PyTemplateRegistry>()?;
    m.add_class::<templates::PyAgentTemplate>()?;
    m.add_class::<patterns::PyDebatePattern>()?;
    m.add_class::<patterns::PySupervisorPattern>()?;
    m.add_class::<patterns::PyHierarchicalPattern>()?;
    m.add_class::<patterns::PyHierarchicalPatternBuilder>()?;
    m.add_class::<patterns::PyPipelinePattern>()?;
    m.add_class::<patterns::PyPipelinePatternBuilder>()?;
    Ok(())
}

fn add_utility_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<loaders::PyDocument>()?;
    m.add_class::<loaders::PyTextSplitter>()?;
    m.add_class::<loaders::PyCsvLoader>()?;
    m.add_class::<loaders::PyJsonLoader>()?;
    m.add_class::<loaders::PyMarkdownLoader>()?;
    m.add_class::<loaders::PyHtmlLoader>()?;
    m.add_class::<output_parsers::PyJsonOutputParser>()?;
    m.add_class::<output_parsers::PyListOutputParser>()?;
    m.add_class::<output_parsers::PyEnumOutputParser>()?;
    m.add_class::<output_parsers::PyStructuredOutputParser>()?;
    m.add_class::<security::PyAuthContext>()?;
    m.add_class::<security::PyApiKeyValidator>()?;
    m.add_class::<security::PyJwtValidator>()?;
    m.add_class::<security::PyRateLimitConfig>()?;
    m.add_class::<security::PyRateLimiter>()?;
    m.add_class::<security::PyPermission>()?;
    m.add_class::<security::PyRole>()?;
    m.add_class::<security::PyRbacEnforcer>()?;
    m.add_class::<security::PyAuditEvent>()?;
    m.add_class::<security::PyAuditLogger>()?;
    m.add_class::<cache::PyCachePolicy>()?;
    m.add_class::<cache::PyCachedResponse>()?;
    m.add_class::<cache::PyCacheStats>()?;
    m.add_class::<cache::PyLlmCache>()?;
    m.add_class::<observability::PyToolStats>()?;
    m.add_class::<observability::PyToolMetrics>()?;
    m.add_class::<observability::PyMetricsRecorder>()?;
    #[cfg(feature = "s3")]
    m.add_class::<storage::PyStorageConfig>()?;
    #[cfg(feature = "s3")]
    m.add_class::<prompts_storage::PyPromptTemplateStore>()?;
    #[cfg(feature = "tools-rag")]
    m.add_class::<rag::PyRagService>()?;
    #[cfg(feature = "tools-rag")]
    m.add_class::<rag_tool::PyRagTool>()?;
    Ok(())
}

fn add_module_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(observability::py_init_logging, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        observability::py_set_log_callback,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        observability::py_get_logging_config,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(observability::py_init_telemetry, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        observability::py_register_metrics,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        observability::py_categorize_error,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        builtin_tools::create_calculator_tool,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(builtin_tools::create_fs_tool, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        builtin_tools::create_apply_patch_tool,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(builtin_tools::create_http_tool, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        builtin_tools::create_code_execution_tool,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        inspection::py_default_inspection_manager,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        jsonl_session::py_parse_jsonl_session,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(jsonl_session::py_new_entry_id, m)?)?;
    #[cfg(feature = "local-inference")]
    {
        m.add_function(pyo3::wrap_pyfunction!(
            local_inference::probe_local_inference_host,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            local_inference::fit_local_inference_catalog,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            local_inference::fit_local_inference_model,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            local_inference::safe_local_inference_models,
            m
        )?)?;
    }
    m.add_class::<builtin_tools::PyBuiltinTool>()?;
    m.add_class::<mcp::PyMcpServerBuilder>()?;
    m.add_class::<mcp::PyMcpServer>()?;
    m.add_class::<in_process_mcp::PyInProcessMcpServerBuilder>()?;
    m.add_class::<in_process_mcp::PyInProcessMcpServer>()?;
    m.add_function(pyo3::wrap_pyfunction!(mcp::get_latest_protocol_version, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        mcp::get_supported_protocol_versions,
        m
    )?)?;
    Ok(())
}

/// Test-only entry point that panics. The tests use it to check that a Rust
/// panic reaches Python as `PanicException` and does not end the process.
#[cfg(feature = "panic-probe")]
#[pyfunction]
#[expect(clippy::panic, reason = "the probe exists to panic")]
fn _panic_probe() {
    panic!("caliche panic probe");
}

/// Python module for Caliche.
#[pymodule]
fn _caliche_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    add_runtime_classes(m)?;
    add_orchestration_classes(m)?;
    add_utility_classes(m)?;
    add_module_functions(m)?;
    #[cfg(feature = "panic-probe")]
    m.add_function(pyo3::wrap_pyfunction!(_panic_probe, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
