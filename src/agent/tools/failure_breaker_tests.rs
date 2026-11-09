use async_trait::async_trait;

use super::*;
use crate::agent::core::ToolIntelligenceState;
use crate::agent::hooks::{
    ApprovalOutcome, ApprovalResolver, HookContext, HookMatcher, HookResult, LifecycleHook,
};
use crate::agent::tools::ToolRegistry;
use crate::agent::tools::execution::middleware::HookExecutionMiddleware;
use crate::agent::tools::execution::{
    ToolExecutionEngine, ToolExecutionMiddleware, ToolExecutionRequest,
};
use crate::agent::tools::tool::{Tool, ToolCall, ToolMeta, ToolResult};

#[test]
fn executor_logs_a_failed_tool_and_does_not_rerun_it() {
    let (lines, _guard) = capture_outcomes();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let action_id = "01936b8e-7c3a-7000-8000-0000000000aa";
    let action_ids = std::sync::Arc::new(Mutex::new(HashMap::from([(
        "call-1".to_string(),
        action_id.to_string(),
    )])));
    let breaker = std::sync::Arc::new(Mutex::new(IdenticalFailureBreaker::with_enabled(true)));
    let hits_for_run = std::sync::Arc::clone(&hits);
    runtime.block_on(with_turn_scope(
        breaker,
        action_ids,
        run_failure_sequence(hits_for_run),
    ));

    assert_eq!(
        hits.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "the repeat must be skipped and the post-edit retry must run"
    );
    let logged = lines.lock().expect("log lines").clone();
    let joined = logged.join("\n");
    assert!(
        joined.contains(action_id),
        "failed-tool log dropped the action id: {joined}"
    );
    assert!(
        joined.contains("\"ok\":false"),
        "failed-tool log missing ok=false: {joined}"
    );
    assert!(
        joined.contains("\"retry_blocked\":true"),
        "blocked retry was not logged: {joined}"
    );
    assert!(
        joined.contains(RETRY_BLOCKED_CODE),
        "blocked retry omitted {RETRY_BLOCKED_CODE}: {joined}"
    );
    let blocked_lines: Vec<_> = logged
        .iter()
        .filter(|line| line.contains("\"retry_blocked\":true"))
        .collect();
    assert!(
        blocked_lines
            .iter()
            .all(|line| !line.contains("Cannot find name")),
        "blocked log repeated stderr: {blocked_lines:?}"
    );
}

#[test]
fn executor_logs_an_approval_pause() {
    let (lines, _guard) = capture_outcomes();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let action_id = "01936b8e-7c3a-7000-8000-0000000000bb";
    let action_ids = std::sync::Arc::new(Mutex::new(HashMap::from([(
        "call-hitl".to_string(),
        action_id.to_string(),
    )])));
    let breaker = std::sync::Arc::new(Mutex::new(IdenticalFailureBreaker::with_enabled(true)));
    runtime.block_on(with_turn_scope(breaker, action_ids, run_approval_pause()));
    let joined = lines.lock().expect("log lines").join("\n");
    assert!(
        joined.contains("\"event\":\"tool.awaiting_confirmation\""),
        "approval pause was not logged: {joined}"
    );
    assert!(
        joined.contains("\"awaiting_confirmation\":true"),
        "approval pause omitted the flag: {joined}"
    );
    assert!(
        joined.contains(action_id),
        "approval pause dropped the action id: {joined}"
    );
}

#[test]
fn outcome_log_redacts_secrets_and_keeps_claimed_fields() {
    let (lines, _guard) = capture_outcomes();
    let sk = format!("s{}-{}", "k", "abcdefghijklmnopqrstuvwxyz");
    let secret = format!("OPENAI_API_KEY={sk} Bearer abc.def-ghi_jkl");
    let body = format!("{secret} {}", "y".repeat(400));
    let mut result = ToolResult::error(body.clone());
    result.typed_payload = Some(serde_json::json!({"exit_code": 7}));
    log_tool_end(
        TOOL_BASH,
        "call-log",
        &serde_json::json!({"command": "bun run typecheck", "working_directory": "/workspace"}),
        &result,
    );
    let joined = lines.lock().expect("log lines").join("\n");
    let line: serde_json::Value = serde_json::from_str(
        joined
            .lines()
            .find(|line| line.contains("\"event\":\"tool.end\""))
            .expect("tool.end line"),
    )
    .expect("outcome json");
    assert_eq!(line["exit_code"], 7, "{joined}");
    assert_eq!(line["cwd"], "/workspace", "{joined}");
    assert_eq!(line["stderr_hash"], stderr_hash(&body), "{joined}");
    let stderr = line["stderr"].as_str().expect("stderr");
    assert!(stderr.contains("[REDACTED]"), "{stderr}");
    assert!(!stderr.contains(&sk), "{stderr}");
    assert!(!stderr.contains("abc.def-ghi_jkl"), "{stderr}");
    assert!(stderr.ends_with("..."), "{stderr}");
    assert_eq!(
        stderr.trim_end_matches("...").chars().count(),
        STDERR_LOG_CHARS,
        "{stderr}"
    );
}

#[test]
fn blocked_retry_does_not_ask_for_approval() {
    let (lines, _guard) = capture_outcomes();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let action_ids = std::sync::Arc::new(Mutex::new(HashMap::new()));
    let breaker = std::sync::Arc::new(Mutex::new(IdenticalFailureBreaker::with_enabled(true)));
    let hits_for_run = std::sync::Arc::clone(&hits);
    runtime.block_on(with_turn_scope(
        breaker,
        action_ids,
        run_blocked_retry_with_pause(hits_for_run),
    ));
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 1);
    let joined = lines.lock().expect("log lines").join("\n");
    let pauses = joined
        .lines()
        .filter(|line| line.contains("\"event\":\"tool.awaiting_confirmation\""))
        .count();
    assert_eq!(
        pauses, 1,
        "blocked retry asked for approval again: {joined}"
    );
}

async fn run_failure_sequence(hits: std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    #[derive(Debug)]
    struct CountingBash {
        hits: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ToolMeta for CountingBash {
        fn name(&self) -> &str {
            "bash"
        }
        fn description(&self) -> &str {
            "counting bash"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
    }

    #[async_trait]
    impl Tool for CountingBash {
        async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut result = ToolResult::error(
                "Command failed with exit code 2:\nsrc/main.ts(1,1): Cannot find name 'x'.",
            );
            result.typed_payload = Some(serde_json::json!({"exit_code": 2}));
            result.content = result.error.clone().unwrap_or_default();
            result
        }
    }

    #[derive(Debug)]
    struct TouchFile;

    impl ToolMeta for TouchFile {
        fn name(&self) -> &str {
            "fs"
        }
        fn description(&self) -> &str {
            "touch"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
    }

    #[async_trait]
    impl Tool for TouchFile {
        async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
            ToolResult::success("edited")
        }
    }

    let tools = ToolRegistry::new();
    tools.register(std::sync::Arc::new(CountingBash { hits }));
    tools.register(std::sync::Arc::new(TouchFile));
    let intelligence =
        std::sync::Arc::new(tokio::sync::Mutex::new(ToolIntelligenceState::default()));
    let engine = ToolExecutionEngine::new();
    let bash = serde_json::json!({
        "command": "bun run typecheck",
        "working_directory": "/workspace"
    });
    let _ = engine
        .execute_parallel(
            execution_request(&tools, &intelligence, "bash", "call-1", bash.clone()),
            &[],
        )
        .await;
    let second = engine
        .execute_parallel(
            execution_request(&tools, &intelligence, "bash", "call-1", bash.clone()),
            &[],
        )
        .await;
    let blocked = &second.executed[0].5;
    assert!(
        blocked
            .error
            .as_deref()
            .unwrap_or("")
            .contains(RETRY_BLOCKED_CODE),
        "second execution was not blocked: {blocked:?}"
    );
    let _ = engine
        .execute_parallel(
            execution_request(
                &tools,
                &intelligence,
                "fs",
                "call-edit",
                serde_json::json!({"op": "edit", "path": "src/main.ts"}),
            ),
            &[],
        )
        .await;
    let _ = engine
        .execute_parallel(
            execution_request(&tools, &intelligence, "bash", "call-1", bash),
            &[],
        )
        .await;
}

async fn run_approval_pause() {
    #[derive(Debug)]
    struct OkTool;

    impl ToolMeta for OkTool {
        fn name(&self) -> &str {
            "bash"
        }
        fn description(&self) -> &str {
            "ok"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
    }

    #[async_trait]
    impl Tool for OkTool {
        async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
            ToolResult::success("ok")
        }
    }

    #[derive(Debug)]
    struct PauseHook;

    #[async_trait]
    impl LifecycleHook for PauseHook {
        async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
            HookResult::require_approval("needs review", None)
        }
    }

    struct AllowOnce;

    #[async_trait]
    impl ApprovalResolver for AllowOnce {
        fn resolve_approval(
            &self,
            _tool_name: &str,
            _tool_input: &serde_json::Value,
            _reason: &str,
            _proposed_amendment: Option<&[String]>,
        ) -> futures_util::future::BoxFuture<'_, anyhow::Result<ApprovalOutcome>> {
            Box::pin(async { Ok(ApprovalOutcome::Allow) })
        }
    }

    let tools = ToolRegistry::new();
    tools.register(std::sync::Arc::new(OkTool));
    let mut hooks = crate::agent::hooks::HookRegistry::new();
    hooks.register(HookMatcher::tool("bash"), std::sync::Arc::new(PauseHook));
    let middleware = HookExecutionMiddleware::new(hooks)
        .with_approval_resolver(Some(std::sync::Arc::new(AllowOnce)));
    let middlewares: [&dyn ToolExecutionMiddleware; 1] = [&middleware];
    let intelligence =
        std::sync::Arc::new(tokio::sync::Mutex::new(ToolIntelligenceState::default()));
    let _ = ToolExecutionEngine::new()
        .execute_parallel(
            execution_request(
                &tools,
                &intelligence,
                "bash",
                "call-hitl",
                serde_json::json!({"command": "bun test"}),
            ),
            &middlewares,
        )
        .await;
}

async fn run_blocked_retry_with_pause(hits: std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    #[derive(Debug)]
    struct CountingBash {
        hits: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ToolMeta for CountingBash {
        fn name(&self) -> &str {
            TOOL_BASH
        }
        fn description(&self) -> &str {
            "counting bash"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
    }

    #[async_trait]
    impl Tool for CountingBash {
        async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut result = ToolResult::error("Command failed with exit code 2:\nnope");
            result.typed_payload = Some(serde_json::json!({"exit_code": 2}));
            result
        }
    }

    #[derive(Debug)]
    struct PauseHook;

    #[async_trait]
    impl LifecycleHook for PauseHook {
        async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
            HookResult::require_approval("needs review", None)
        }
    }

    struct AllowOnce;

    #[async_trait]
    impl ApprovalResolver for AllowOnce {
        fn resolve_approval(
            &self,
            _tool_name: &str,
            _tool_input: &serde_json::Value,
            _reason: &str,
            _proposed_amendment: Option<&[String]>,
        ) -> futures_util::future::BoxFuture<'_, anyhow::Result<ApprovalOutcome>> {
            Box::pin(async { Ok(ApprovalOutcome::Allow) })
        }
    }

    let tools = ToolRegistry::new();
    tools.register(std::sync::Arc::new(CountingBash { hits }));
    let mut hooks = crate::agent::hooks::HookRegistry::new();
    hooks.register(HookMatcher::tool(TOOL_BASH), std::sync::Arc::new(PauseHook));
    let middleware = HookExecutionMiddleware::new(hooks)
        .with_approval_resolver(Some(std::sync::Arc::new(AllowOnce)));
    let middlewares: [&dyn ToolExecutionMiddleware; 1] = [&middleware];
    let intelligence =
        std::sync::Arc::new(tokio::sync::Mutex::new(ToolIntelligenceState::default()));
    let bash =
        serde_json::json!({"command": "bun run typecheck", "working_directory": "/workspace"});
    let engine = ToolExecutionEngine::new();
    let _ = engine
        .execute_parallel(
            execution_request(&tools, &intelligence, TOOL_BASH, "call-1", bash.clone()),
            &middlewares,
        )
        .await;
    let _ = engine
        .execute_parallel(
            execution_request(&tools, &intelligence, TOOL_BASH, "call-2", bash),
            &middlewares,
        )
        .await;
}

fn execution_request<'a>(
    tools: &'a ToolRegistry,
    intelligence: &'a std::sync::Arc<tokio::sync::Mutex<ToolIntelligenceState>>,
    name: &'static str,
    id: &str,
    arguments: serde_json::Value,
) -> ToolExecutionRequest<'a> {
    ToolExecutionRequest {
        valid_tool_calls: vec![(
            0,
            id.to_string(),
            std::sync::Arc::new(ToolCall {
                name: beef::lean::Cow::borrowed(name),
                arguments,
            }),
        )],
        tools,
        user_context: None,
        metrics: None,
        tool_intelligence: intelligence,
        cancellation_flag: None,
        workspace_root: None,
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
    }
}

/// Capture outcome log lines on this thread until the guard drops.
///
/// tracing-core caches each callsite's interest. While at most one
/// dispatcher is registered, a callsite first hit on another test thread
/// takes its interest from that thread's default (none) and caches `never`,
/// so this thread's capture misses the event. A second dispatcher that lives
/// for the whole process keeps every registration on the shared list, which
/// includes the capture. The rebuild clears a `never` cached before the
/// capture was registered.
fn capture_outcomes() -> (
    std::sync::Arc<Mutex<Vec<String>>>,
    tracing::subscriber::DefaultGuard,
) {
    static ANCHOR: OnceLock<tracing::Dispatch> = OnceLock::new();
    ANCHOR.get_or_init(|| tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default()));
    let lines = std::sync::Arc::new(Mutex::new(Vec::new()));
    let guard = tracing::subscriber::set_default(Capture(std::sync::Arc::clone(&lines)));
    tracing::callsite::rebuild_interest_cache();
    (lines, guard)
}

struct Capture(std::sync::Arc<Mutex<Vec<String>>>);

impl tracing::Subscriber for Capture {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == OUTCOME_TARGET
    }

    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        let mut message = String::new();
        event.record(&mut MessageVisit(&mut message));
        if !message.is_empty() {
            self.0.lock().expect("log lines").push(message);
        }
    }

    fn enter(&self, _span: &tracing::span::Id) {}

    fn exit(&self, _span: &tracing::span::Id) {}
}

struct MessageVisit<'a>(&'a mut String);

impl tracing::field::Visit for MessageVisit<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            *self.0 = format!("{value:?}");
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            *self.0 = value.to_string();
        }
    }
}
