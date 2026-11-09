use crate::agent::core::ToolIntelligenceState;
use crate::agent::tools::ToolExecutionRecord;
use crate::agent::tools::common::set_tool_workspace_root;
use crate::agent::tools::registry::ToolRegistry;
#[cfg(any(feature = "sandbox", feature = "tools-rag"))]
use crate::agent::tools::tool::ToolCapability;
use crate::agent::tools::tool::{ToolCall, ToolResult};
use crate::observability::ToolMetrics;
use crate::utils::user_context::UserContext;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::sync::{Mutex, Semaphore};

use super::middleware::{ToolExecutionContext, ToolExecutionMiddleware};

const TOOL_EXECUTION_TIMEOUT_DEFAULT_SECS: u64 = 900;
/// Default cap on the number of tool calls executed concurrently within a single
/// `execute_parallel` batch. Without a bound, a model turn that emits many tool calls
/// (e.g. dozens of greps/HTTP fetches/sandboxed shells) would spawn unbounded concurrent
/// work, risking fd/process exhaustion or provider rate-limit storms.
///
/// Default is cores-minus-one, clamped to 4, matching background index ingest so a
/// laptop REPL stays interactive. `CALICHE_TOOL_MAX_CONCURRENCY` overrides this
/// without a cap so operators can raise it on dedicated hosts.
const TOOL_MAX_CONCURRENCY_DEFAULT_CAP: usize = 4;

#[derive(Debug, Default)]
pub struct ToolExecutionResult {
    pub(crate) executed: Vec<(usize, String, String, serde_json::Value, u64, ToolResult)>,
    pub(crate) denied: Vec<(usize, String, String, beef::lean::Cow<'static, str>)>,
}

pub struct ToolExecutionRequest<'a> {
    pub(crate) valid_tool_calls: Vec<(usize, String, Arc<ToolCall>)>,
    pub(crate) tools: &'a ToolRegistry,
    pub(crate) user_context: Option<&'a UserContext>,
    pub(crate) metrics: Option<&'a Arc<ToolMetrics>>,
    pub(crate) tool_intelligence: &'a Arc<Mutex<ToolIntelligenceState>>,
    pub(crate) cancellation_flag: Option<&'a Arc<std::sync::atomic::AtomicBool>>,
    /// When set, relative paths (e.g. ".") in fs/grep and other path-using tools resolve against this instead of `env::current_dir()`.
    pub(crate) workspace_root: Option<PathBuf>,
    #[cfg(feature = "agentfs")]
    pub(crate) agentfs_tool_logger: Option<&'a crate::agent::tools::AgentFsToolLogger>,
    #[cfg(feature = "sandbox")]
    pub(crate) sandbox: Option<&'a dyn caliche_sandbox::SandboxHandleOps>,
    #[cfg(feature = "tools-rag")]
    pub(crate) workspace_rag: Option<&'a crate::rag::RagService>,
    #[cfg(feature = "tools-rag")]
    pub(crate) workspace_rag_top_k: usize,
    #[cfg(feature = "tools-rag")]
    pub(crate) workspace_rag_min_similarity: f32,
}

/// Clears the thread-local workspace root when dropped so later code does not see a stale value.
///
/// One guard covers a whole `execute_parallel` batch. A per-call guard would clear the root
/// as soon as the first tool call finished, and the concurrent calls still in flight would
/// then resolve relative paths against the process working directory.
struct WorkspaceRootGuard;

impl Drop for WorkspaceRootGuard {
    fn drop(&mut self) {
        set_tool_workspace_root(None);
    }
}

type ParallelToolOk = (usize, String, String, serde_json::Value, u64, ToolResult);
type ParallelToolErr = (usize, String, String, beef::lean::Cow<'static, str>);

struct ExecuteOneParallelArgs<'a> {
    idx: usize,
    id: String,
    tool_call: Arc<ToolCall>,
    tools: &'a ToolRegistry,
    user_context: Option<&'a UserContext>,
    metrics: Option<&'a Arc<ToolMetrics>>,
    cancellation_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    middlewares: &'a [&'a dyn ToolExecutionMiddleware],
    semaphore: Arc<Semaphore>,
    retry_attempts: usize,
    tool_execution_timeout: Duration,
    #[cfg(feature = "sandbox")]
    sandbox: Option<&'a dyn caliche_sandbox::SandboxHandleOps>,
    #[cfg(feature = "tools-rag")]
    workspace_rag: Option<&'a crate::rag::RagService>,
    #[cfg(feature = "tools-rag")]
    workspace_rag_top_k: usize,
    #[cfg(feature = "tools-rag")]
    workspace_rag_min_similarity: f32,
}

async fn execute_one_parallel_call(
    args: ExecuteOneParallelArgs<'_>,
) -> Result<ParallelToolOk, ParallelToolErr> {
    let tool_name = args.tool_call.name.as_ref().to_string();
    let Ok(_permit) = args.semaphore.acquire().await else {
        return Err((
            args.idx,
            args.id,
            tool_name,
            beef::lean::Cow::borrowed("tool execution semaphore closed"),
        ));
    };

    let tool_args = args.tool_call.arguments.clone();
    let ctx = ToolExecutionContext {
        tool_name: &tool_name,
        arguments: &tool_args,
        call_id: &args.id,
    };

    if let Some(blocked) =
        crate::agent::tools::failure_breaker::blocked_result(&tool_name, &args.id, &tool_args)
    {
        return Ok((args.idx, args.id, tool_name, tool_args, 0, blocked));
    }

    if let Err(reason) = run_before_middleware(args.middlewares, &ctx).await {
        let denied = ToolResult::error(reason.as_ref());
        crate::agent::tools::failure_breaker::log_tool_end(
            &tool_name, &args.id, &tool_args, &denied,
        );
        return Err((args.idx, args.id, tool_name, reason));
    }

    if let Err(reason) = admit_middlewares(args.middlewares, &ctx).await {
        let denied = ToolResult::error(reason.as_ref());
        crate::agent::tools::failure_breaker::log_tool_end(
            &tool_name, &args.id, &tool_args, &denied,
        );
        return Err((args.idx, args.id, tool_name, reason));
    }

    crate::agent::tools::failure_breaker::log_tool_start(&tool_name, &args.id, &tool_args);
    let start_time = std::time::Instant::now();
    let (mut result, retry_count) = execute_with_retries(&args, &tool_name, &tool_args).await;

    for middleware in args.middlewares {
        middleware.after_execute(&ctx, &mut result).await;
    }

    #[cfg(feature = "tools-rag")]
    maybe_enrich_with_rag(&args, &tool_name, &tool_args, &mut result).await;

    let duration_ms = start_time.elapsed().as_micros() as u64 / 1000;
    tracing::debug!(
        target: "caliche::tools",
        tool = %tool_name,
        call_id = %args.id,
        duration_ms,
        success = result.success,
        "tool.execute.complete"
    );
    if retry_count > 0 {
        result.content.push_str(&format!(
            "\n\n[tool retry]: attempted {retry_count} additional time(s)"
        ));
    }

    crate::agent::tools::failure_breaker::observe(&tool_name, &tool_args, &result);
    crate::agent::tools::failure_breaker::log_tool_end(&tool_name, &args.id, &tool_args, &result);

    Ok((args.idx, args.id, tool_name, tool_args, duration_ms, result))
}

async fn run_before_middleware(
    middlewares: &[&dyn ToolExecutionMiddleware],
    ctx: &ToolExecutionContext<'_>,
) -> Result<(), beef::lean::Cow<'static, str>> {
    for middleware in middlewares {
        middleware.before_execute(ctx).await?;
    }
    Ok(())
}

async fn admit_middlewares(
    middlewares: &[&dyn ToolExecutionMiddleware],
    ctx: &ToolExecutionContext<'_>,
) -> Result<(), beef::lean::Cow<'static, str>> {
    for middleware in middlewares {
        middleware.admit_execution(ctx).await?;
    }
    Ok(())
}

async fn execute_with_retries(
    args: &ExecuteOneParallelArgs<'_>,
    tool_name: &str,
    tool_args: &serde_json::Value,
) -> (ToolResult, usize) {
    let mut retry_count = 0usize;
    loop {
        if args
            .cancellation_flag
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return (
                ToolResult::error("Operation cancelled by user"),
                retry_count,
            );
        }

        let current_result = execute_one_attempt(args, tool_name, tool_args).await;

        if retry_count < args.retry_attempts && should_retry_tool_result(&current_result) {
            retry_count += 1;
            let backoff_ms = 100_u64.saturating_mul(retry_count as u64);
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            continue;
        }
        return (current_result, retry_count);
    }
}

async fn execute_one_attempt(
    args: &ExecuteOneParallelArgs<'_>,
    tool_name: &str,
    tool_args: &serde_json::Value,
) -> ToolResult {
    #[cfg(feature = "sandbox")]
    if should_execute_sandboxed(args, tool_name) {
        return unwrap_tool_timeout(
            tokio::time::timeout(
                args.tool_execution_timeout,
                args.tools.execute_sandboxed(
                    tool_name,
                    tool_args.clone(),
                    args.sandbox,
                    args.user_context,
                    args.metrics,
                ),
            )
            .await,
            tool_name,
            args.tool_execution_timeout,
        );
    }

    unwrap_tool_timeout(
        tokio::time::timeout(
            args.tool_execution_timeout,
            args.tools.execute_with_context(
                tool_name,
                tool_args.clone(),
                args.user_context,
                args.metrics,
            ),
        )
        .await,
        tool_name,
        args.tool_execution_timeout,
    )
}

#[cfg(feature = "sandbox")]
fn should_execute_sandboxed(args: &ExecuteOneParallelArgs<'_>, tool_name: &str) -> bool {
    args.sandbox.is_some()
        && args
            .tools
            .get(tool_name)
            .map(|t| t.capabilities())
            .unwrap_or_default()
            .contains(&ToolCapability::SandboxExecution)
}

fn unwrap_tool_timeout(
    result: Result<ToolResult, tokio::time::error::Elapsed>,
    tool_name: &str,
    timeout: Duration,
) -> ToolResult {
    match result {
        Ok(result) => result,
        Err(_elapsed) => ToolResult::error(format!(
            "Tool '{tool_name}' timed out after {} seconds",
            timeout.as_secs()
        )),
    }
}

#[cfg(feature = "tools-rag")]
async fn maybe_enrich_with_rag(
    args: &ExecuteOneParallelArgs<'_>,
    tool_name: &str,
    tool_args: &serde_json::Value,
    result: &mut ToolResult,
) {
    if !should_run_rag_enrichment(args, tool_name, tool_args, result) {
        return;
    }
    let Some(rag) = args.workspace_rag else {
        return;
    };
    let Some(rag_query) = rag_query_for_tool(tool_name, tool_args) else {
        return;
    };
    let rag_call = tokio::time::timeout(
        tool_rag_enrichment_timeout(),
        rag.query(
            &rag_query,
            args.workspace_rag_top_k.min(3),
            args.workspace_rag_min_similarity,
        ),
    )
    .await;
    if let Ok(Ok((rag_results, _, _))) = rag_call {
        append_rag_enrichment(result, &rag_results);
    }
}

#[cfg(feature = "tools-rag")]
fn should_run_rag_enrichment(
    args: &ExecuteOneParallelArgs<'_>,
    tool_name: &str,
    tool_args: &serde_json::Value,
    result: &ToolResult,
) -> bool {
    let capabilities = args
        .tools
        .get(tool_name)
        .map(|t| t.capabilities())
        .unwrap_or_default();
    tool_rag_enrichment_enabled()
        && capabilities.contains(&ToolCapability::RagEnrichment)
        && should_enrich_with_rag(tool_name, tool_args, result)
}

async fn record_execution_history(
    tool_intelligence: &Arc<Mutex<ToolIntelligenceState>>,
    executed: &[ParallelToolOk],
) {
    let mut state = tool_intelligence.lock().await;
    for (_idx, _id, tool_name, tool_args, duration_ms, tool_result) in executed {
        state
            .execution_history
            .save_execution(execution_record_from_result(
                tool_name,
                tool_args,
                *duration_ms,
                tool_result,
            ));
    }
}

fn execution_record_from_result(
    tool_name: &str,
    tool_args: &serde_json::Value,
    duration_ms: u64,
    tool_result: &ToolResult,
) -> ToolExecutionRecord {
    let mut record = ToolExecutionRecord::new(
        tool_name.to_string(),
        tool_args.clone(),
        tool_result.success,
    )
    .with_timing(duration_ms)
    .with_output_size(tool_result.content.len());
    if !tool_result.success
        && let Some(err) = tool_result.error.as_ref()
    {
        record = record.with_error(err.clone());
    }
    record
}

#[cfg(feature = "agentfs")]
async fn record_agentfs_tool_logs(
    logger: Option<&crate::agent::tools::AgentFsToolLogger>,
    executed: &[ParallelToolOk],
) {
    let Some(logger) = logger else {
        return;
    };
    for (_idx, _id, tool_name, tool_args, duration_ms, tool_result) in executed {
        let completed_at = chrono::Utc::now();
        let started_at =
            completed_at - chrono::Duration::milliseconds(i64::try_from(*duration_ms).unwrap_or(0));
        if let Err(e) = logger
            .record(crate::agent::tools::ToolLogRecord {
                name: tool_name,
                started_at,
                completed_at,
                parameters: tool_args.clone(),
                success: tool_result.success,
                result_content: &tool_result.content,
                error: tool_result.error.as_deref(),
            })
            .await
        {
            tracing::warn!("AgentFS tool record failed: {e}");
        }
    }
}

#[derive(Default)]
pub struct ToolExecutionEngine;

impl ToolExecutionEngine {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    pub async fn execute_parallel(
        &self,
        request: ToolExecutionRequest<'_>,
        middlewares: &[&dyn ToolExecutionMiddleware],
    ) -> ToolExecutionResult {
        let retry_attempts = configured_tool_retry_attempts();
        let tool_execution_timeout = configured_tool_execution_timeout();
        let cancellation_flag = request.cancellation_flag.cloned();
        let semaphore = Arc::new(Semaphore::new(configured_tool_max_concurrency()));

        // Every call in a batch shares one workspace root, so set it once for the batch.
        let _workspace_root_guard = request.workspace_root.clone().map(|root| {
            set_tool_workspace_root(Some(root));
            WorkspaceRootGuard
        });

        let tool_futures: Vec<_> = request
            .valid_tool_calls
            .into_iter()
            .map(|(idx, id, tool_call)| {
                execute_one_parallel_call(ExecuteOneParallelArgs {
                    idx,
                    id,
                    tool_call,
                    tools: request.tools,
                    user_context: request.user_context,
                    metrics: request.metrics,
                    cancellation_flag: cancellation_flag.clone(),
                    middlewares,
                    semaphore: Arc::clone(&semaphore),
                    retry_attempts,
                    tool_execution_timeout,
                    #[cfg(feature = "sandbox")]
                    sandbox: request.sandbox,
                    #[cfg(feature = "tools-rag")]
                    workspace_rag: request.workspace_rag,
                    #[cfg(feature = "tools-rag")]
                    workspace_rag_top_k: request.workspace_rag_top_k,
                    #[cfg(feature = "tools-rag")]
                    workspace_rag_min_similarity: request.workspace_rag_min_similarity,
                })
            })
            .collect();

        let mut executed = Vec::new();
        let mut denied = Vec::new();
        for outcome in futures_util::future::join_all(tool_futures).await {
            match outcome {
                Ok(item) => executed.push(item),
                Err((idx, id, name, reason)) => denied.push((idx, id, name, reason)),
            }
        }
        executed.sort_by_key(|(idx, _, _, _, _, _)| *idx);

        record_execution_history(request.tool_intelligence, &executed).await;

        #[cfg(feature = "agentfs")]
        record_agentfs_tool_logs(request.agentfs_tool_logger, &executed).await;

        ToolExecutionResult { executed, denied }
    }
}

#[cfg(feature = "tools-rag")]
fn tool_rag_enrichment_enabled() -> bool {
    std::env::var("CALICHE_TOOL_RAG_ENRICHMENT")
        .ok()
        .as_deref()
        .is_some_and(|value| matches!(value, "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
}

#[cfg(feature = "tools-rag")]
fn tool_rag_enrichment_timeout() -> Duration {
    const DEFAULT_RAG_ENRICHMENT_TIMEOUT_MS: u64 = 75;
    std::env::var("CALICHE_TOOL_RAG_ENRICHMENT_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .map_or(
            Duration::from_millis(DEFAULT_RAG_ENRICHMENT_TIMEOUT_MS),
            Duration::from_millis,
        )
}

/// Maximum number of tool calls allowed to run concurrently in a single `execute_parallel`
/// batch. Configurable via `CALICHE_TOOL_MAX_CONCURRENCY`.
/// `0` or an unparsable value falls back to the default; the value is otherwise unclamped so
/// operators can intentionally raise it for high-throughput sandboxed environments.
fn configured_tool_max_concurrency() -> usize {
    parse_tool_max_concurrency(
        std::env::var("CALICHE_TOOL_MAX_CONCURRENCY")
            .ok()
            .as_deref(),
        std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(2),
    )
}

fn default_tool_max_concurrency(available_parallelism: usize) -> usize {
    available_parallelism
        .saturating_sub(1)
        .clamp(1, TOOL_MAX_CONCURRENCY_DEFAULT_CAP)
}

fn parse_tool_max_concurrency(raw: Option<&str>, available_parallelism: usize) -> usize {
    raw.and_then(|value| value.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or_else(|| default_tool_max_concurrency(available_parallelism))
}

fn configured_tool_retry_attempts() -> usize {
    std::env::var("CALICHE_TOOL_RETRY_ATTEMPTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0)
        .min(3)
}

fn configured_tool_execution_timeout() -> Duration {
    let secs = std::env::var("CALICHE_TOOL_EXECUTION_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(TOOL_EXECUTION_TIMEOUT_DEFAULT_SECS);
    Duration::from_secs(secs)
}

fn should_retry_tool_result(result: &ToolResult) -> bool {
    if result.success {
        return false;
    }
    let err = result
        .error
        .as_deref()
        .unwrap_or(result.content.as_str())
        .to_lowercase();
    err.contains("timed out")
        || err.contains("timeout")
        || err.contains("temporarily unavailable")
        || err.contains("rate limit")
        || err.contains("connection reset")
        || err.contains("network")
        || err.contains("broken pipe")
}

#[cfg(feature = "tools-rag")]
fn should_enrich_with_rag(tool_name: &str, args: &serde_json::Value, result: &ToolResult) -> bool {
    if !result.success || tool_name != "fs" {
        return false;
    }
    matches!(
        args.get("op").and_then(|v| v.as_str()),
        Some("read" | "list")
    )
}

#[cfg(feature = "tools-rag")]
fn rag_query_for_tool(tool_name: &str, args: &serde_json::Value) -> Option<String> {
    if tool_name != "fs" {
        return None;
    }
    let op = args.get("op").and_then(|v| v.as_str())?;
    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
    match op {
        "read" => {
            if let Some(query) = args.get("query").and_then(|v| v.as_str())
                && !query.trim().is_empty()
            {
                return Some(query.to_string());
            }
            Some(format!("file {path} implementation and related context"))
        }
        "list" => Some(format!(
            "directory {path} project structure and relevant files"
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{ToolExecutionEngine, ToolExecutionRequest};
    use crate::agent::core::ToolIntelligenceState;
    use crate::agent::tools::registry::ToolRegistry;
    use crate::agent::tools::tool::{Tool, ToolCall, ToolMeta, ToolResult};
    use async_trait::async_trait;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// Reports the workspace root the tool sees, after an optional delay that lets the
    /// sibling call in the same batch finish first.
    #[derive(Debug)]
    struct WorkspaceRootProbe {
        name: &'static str,
        delay_ms: u64,
    }

    impl ToolMeta for WorkspaceRootProbe {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            "reports the resolved workspace root"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
    }

    #[async_trait]
    impl Tool for WorkspaceRootProbe {
        async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
            if self.delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            }
            match crate::agent::tools::common::path::get_current_dir() {
                Ok(dir) => ToolResult::success(dir.display().to_string()),
                Err(e) => ToolResult::error(e.to_string()),
            }
        }
    }

    fn probe_call(name: &'static str) -> Arc<ToolCall> {
        Arc::new(ToolCall {
            name: beef::lean::Cow::borrowed(name),
            arguments: serde_json::json!({}),
        })
    }

    #[tokio::test]
    async fn workspace_root_stays_set_until_the_whole_batch_finishes() {
        let tools = ToolRegistry::new();
        tools.register(Arc::new(WorkspaceRootProbe {
            name: "fast_probe",
            delay_ms: 0,
        }));
        tools.register(Arc::new(WorkspaceRootProbe {
            name: "slow_probe",
            delay_ms: 30,
        }));

        let workspace_root = PathBuf::from("/tmp/caliche-workspace-root-probe");
        let intelligence = Arc::new(Mutex::new(ToolIntelligenceState::default()));
        let result = ToolExecutionEngine::new()
            .execute_parallel(
                ToolExecutionRequest {
                    valid_tool_calls: vec![
                        (0, "call_0".to_string(), probe_call("slow_probe")),
                        (1, "call_1".to_string(), probe_call("fast_probe")),
                    ],
                    tools: &tools,
                    user_context: None,
                    metrics: None,
                    tool_intelligence: &intelligence,
                    cancellation_flag: None,
                    workspace_root: Some(workspace_root.clone()),
                    #[cfg(feature = "agentfs")]
                    agentfs_tool_logger: None,
                    #[cfg(feature = "sandbox")]
                    sandbox: None,
                    #[cfg(feature = "tools-rag")]
                    workspace_rag: None,
                    #[cfg(feature = "tools-rag")]
                    workspace_rag_top_k: 0,
                    #[cfg(feature = "tools-rag")]
                    workspace_rag_min_similarity: 0.0,
                },
                &[],
            )
            .await;

        assert!(result.denied.is_empty(), "no call should be denied");
        assert_eq!(result.executed.len(), 2);
        let expected = workspace_root.display().to_string();
        for (_idx, _id, name, _args, _dur, tool_result) in &result.executed {
            assert_eq!(
                tool_result.content, expected,
                "{name} must resolve relative paths against the batch workspace root"
            );
        }
    }

    #[test]
    fn default_tool_concurrency_leaves_a_core_and_caps() {
        assert_eq!(super::default_tool_max_concurrency(1), 1);
        assert_eq!(super::default_tool_max_concurrency(2), 1);
        assert_eq!(super::default_tool_max_concurrency(8), 4);
        assert_eq!(super::default_tool_max_concurrency(32), 4);
    }

    /// Byte 600 of the related context falls inside "é"; the preview keeps the 599
    /// ASCII bytes before it.
    #[cfg(feature = "tools-rag")]
    #[test]
    fn rag_preview_cuts_before_a_multibyte_character() {
        let mut result = ToolResult::success("file body");
        let item = format!("{}é more text", "x".repeat(599));

        super::append_rag_enrichment(&mut result, &[item]);

        assert_eq!(
            result.content,
            format!(
                "file body\n\nRelated context (workspace index):\n- {}…\n",
                "x".repeat(599)
            )
        );
    }

    #[test]
    fn parse_tool_concurrency_honors_env_override() {
        assert_eq!(super::parse_tool_max_concurrency(None, 8), 4);
        assert_eq!(super::parse_tool_max_concurrency(Some("0"), 8), 4);
        assert_eq!(super::parse_tool_max_concurrency(Some("nope"), 2), 1);
        assert_eq!(super::parse_tool_max_concurrency(Some("16"), 2), 16);
    }
}

#[cfg(feature = "tools-rag")]
fn append_rag_enrichment(result: &mut ToolResult, rag_results: &[String]) {
    if rag_results.is_empty() {
        return;
    }
    let mut section = String::from("\n\nRelated context (workspace index):\n");
    for item in rag_results.iter().take(3) {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        let preview = if trimmed.len() > 600 {
            format!("{}…", caliche_contracts::text::truncate_utf8(trimmed, 600))
        } else {
            trimmed.to_string()
        };
        section.push_str("- ");
        section.push_str(&preview);
        section.push('\n');
    }
    result.content.push_str(&section);
}
