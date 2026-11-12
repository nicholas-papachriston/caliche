use async_trait::async_trait;
use beef::lean::Cow;
use std::sync::{Arc, Mutex};

use crate::agent::hooks::{
    AmendmentPersister, ApprovalOutcome, ApprovalResolver, HookContext, HookRegistry,
};
use crate::agent::tools::tool::ToolResult;
use crate::policy::{
    ApprovalAnswer, ApprovalBroker, ApprovalRequest, ApprovalWaiter, BrokerError, Effect, FsRoots,
    HumanChannel, ModeOverlay, PendingId, Principal, evaluate_tool_gate,
};
use std::collections::HashMap;

pub struct ToolExecutionContext<'a> {
    pub tool_name: &'a str,
    pub arguments: &'a serde_json::Value,
    pub call_id: &'a str,
}

impl<'a> ToolExecutionContext<'a> {
    /// This call under the name the registry runs it as, so the gate and
    /// hooks see the tool that runs (" fs" is `fs`, `memory` is `workspace`).
    #[must_use]
    pub fn as_run(&self) -> ToolExecutionContext<'a> {
        ToolExecutionContext {
            tool_name: crate::agent::tools::registry::ToolRegistry::canonical_tool_name(
                self.tool_name,
            ),
            arguments: self.arguments,
            call_id: self.call_id,
        }
    }
}

#[async_trait]
pub trait ToolExecutionMiddleware: Send + Sync {
    async fn before_execute(
        &self,
        _ctx: &ToolExecutionContext<'_>,
    ) -> Result<(), Cow<'static, str>> {
        Ok(())
    }

    /// Re-check a granted approval immediately before the tool runs.
    ///
    /// The default does nothing. [`HookExecutionMiddleware`] hashes the
    /// arguments about to run and drops a stale grant.
    async fn admit_execution(
        &self,
        _ctx: &ToolExecutionContext<'_>,
    ) -> Result<(), Cow<'static, str>> {
        Ok(())
    }

    async fn after_execute(&self, _ctx: &ToolExecutionContext<'_>, _result: &mut ToolResult) {}
}

#[derive(Clone)]
pub struct HookExecutionMiddleware {
    hooks: HookRegistry,
    approval_resolver: Option<Arc<dyn ApprovalResolver>>,
    amendment_persister: Option<Arc<dyn AmendmentPersister>>,
    approvals: Arc<ApprovalBroker>,
    approver: Principal,
    overlay: ModeOverlay,
    workspace: Option<String>,
    grants: Arc<Mutex<HashMap<String, HeldGrant>>>,
}

impl HookExecutionMiddleware {
    #[must_use]
    pub fn new(hooks: HookRegistry) -> Self {
        Self {
            hooks,
            approval_resolver: None,
            amendment_persister: None,
            approvals: Arc::new(ApprovalBroker::new()),
            approver: Principal::human("operator", HumanChannel::CliTty),
            overlay: ModeOverlay::Default,
            workspace: None,
            grants: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Workspace root for the filesystem resolver.
    ///
    /// Without a root, or with a blank one, every filesystem tool denies. A
    /// rejected path denies. A path outside this root asks before the tool
    /// runs. Accept-all cannot lower that ask for a write.
    #[must_use]
    pub fn with_workspace(mut self, workspace: impl Into<String>) -> Self {
        self.workspace = Some(workspace.into());
        self
    }

    /// Session overlay for the live policy gate.
    ///
    /// [`ModeOverlay::Default`] keeps the trusted deployment default.
    /// [`ModeOverlay::Interactive`] asks before a side effect, including HTTP
    /// GET and an MCP tool. The floor still denies under every overlay.
    #[must_use]
    pub fn with_overlay(mut self, overlay: ModeOverlay) -> Self {
        self.overlay = overlay;
        self
    }

    /// Principal recorded when a door returns an answer.
    ///
    /// The default is a human at the CLI. An unattended principal makes every
    /// answer fail closed, even when the resolver returns allow.
    #[must_use]
    pub fn with_approver(mut self, approver: Principal) -> Self {
        self.approver = approver;
        self
    }

    #[must_use]
    pub fn with_approval_resolver(mut self, resolver: Option<Arc<dyn ApprovalResolver>>) -> Self {
        self.approval_resolver = resolver;
        self
    }

    #[must_use]
    pub fn with_amendment_persister(
        mut self,
        persister: Option<Arc<dyn AmendmentPersister>>,
    ) -> Self {
        self.amendment_persister = persister;
        self
    }

    async fn evaluate_pre_tool_hooks(
        &self,
        ctx: &ToolExecutionContext<'_>,
    ) -> Result<(), Cow<'static, str>> {
        self.apply_policy_gate(ctx).await?;
        let matching_hooks = self.hooks.get_matching(ctx.tool_name);
        let hook_ctx = HookContext::new(ctx.tool_name, ctx.arguments, ctx.call_id);
        for hook in &matching_hooks {
            let hook_result = hook.on_pre_tool_use(&hook_ctx).await;
            if hook_result.is_denied() {
                let reason = hook_result
                    .permission_decision
                    .denial_reason()
                    .unwrap_or("Hook denied tool execution")
                    .to_string();
                return Err(Cow::owned(reason));
            }
            if hook_result.is_approval_required() {
                let reason = hook_result
                    .permission_decision
                    .approval_reason()
                    .unwrap_or("Approval required")
                    .to_string();
                let proposed_amendment = hook_result
                    .permission_decision
                    .proposed_amendment()
                    .map(<[std::string::String]>::to_vec);
                self.settle_required_approval(ctx, reason, proposed_amendment)
                    .await?;
            }
        }
        Ok(())
    }

    async fn apply_policy_gate(
        &self,
        ctx: &ToolExecutionContext<'_>,
    ) -> Result<(), Cow<'static, str>> {
        let waiter = if self.approval_resolver.is_some() && self.approver.is_human() {
            ApprovalWaiter::Present(HumanChannel::CliTty)
        } else {
            ApprovalWaiter::Absent
        };
        let cwd = approval_cwd();
        let decision = evaluate_tool_gate(
            ctx.tool_name,
            ctx.arguments,
            self.overlay,
            waiter,
            FsRoots::new(&cwd, self.workspace.as_deref().unwrap_or_default()),
        );
        match decision.effect {
            Effect::Allow => Ok(()),
            Effect::Deny => Err(Cow::owned(decision.reason.message().to_string())),
            Effect::Ask => {
                self.settle_required_approval(ctx, decision.reason.message().to_string(), None)
                    .await
            }
        }
    }

    async fn settle_required_approval(
        &self,
        ctx: &ToolExecutionContext<'_>,
        reason: String,
        proposed_amendment: Option<Vec<String>>,
    ) -> Result<(), Cow<'static, str>> {
        crate::agent::tools::failure_breaker::log_awaiting_confirmation(
            ctx.tool_name,
            ctx.call_id,
            ctx.arguments,
        );
        let Some(resolver) = &self.approval_resolver else {
            tracing::debug!(
                "RequireApproval from hook but no approval_resolver configured; denying"
            );
            return Err(Cow::owned(reason));
        };
        let cwd = approval_cwd();
        let dialect = tool_dialect(ctx.tool_name);
        let opened = self.approvals.open(
            ApprovalRequest {
                tool: ctx.tool_name,
                args: ctx.arguments,
                cwd: &cwd,
                dialect,
                policy_version: 0,
                reason: &reason,
            },
            std::time::Instant::now(),
        );
        let outcome = resolver
            .resolve_approval_for_call(
                ctx.tool_name,
                ctx.arguments,
                &reason,
                proposed_amendment.as_deref(),
                ctx.call_id,
            )
            .await;
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(err) => {
                self.approvals.dismiss(&opened.id);
                return Err(Cow::owned(format!("Approval resolution failed: {err}")));
            }
        };
        let answer = match outcome {
            ApprovalOutcome::Allow | ApprovalOutcome::AllowAndPersist { .. } => {
                ApprovalAnswer::Once
            }
            ApprovalOutcome::Deny => ApprovalAnswer::Deny,
        };
        if let Err(err) = self.approvals.answer(
            &opened.id,
            &self.approver,
            answer,
            std::time::Instant::now(),
        ) {
            self.approvals.dismiss(&opened.id);
            return Err(Cow::owned(err.to_string()));
        }
        if matches!(outcome, ApprovalOutcome::Deny) {
            self.approvals.dismiss(&opened.id);
            let text = reason
                .strip_prefix("Requires user approval: ")
                .unwrap_or(&reason)
                .to_string();
            return Err(Cow::owned(text));
        }
        let rule_prefix = if let ApprovalOutcome::AllowAndPersist { rule_prefix } = &outcome {
            Some(rule_prefix.clone())
        } else {
            None
        };
        self.hold_grant(ctx.call_id, opened.id, rule_prefix);
        Ok(())
    }

    fn hold_grant(&self, call_id: &str, id: PendingId, rule_prefix: Option<Vec<String>>) {
        let mut grants = lock_grants(&self.grants);
        if let Some(previous) = grants.insert(call_id.to_owned(), HeldGrant { id, rule_prefix }) {
            self.approvals.dismiss(&previous.id);
        }
    }

    async fn persist_rule(&self, rule_prefix: &[String]) {
        if let Some(ref persister) = self.amendment_persister
            && let Err(err) = persister.persist_allow_rule(rule_prefix).await
        {
            tracing::warn!(
                "Failed to persist allow rule: {err}. Rule will not apply next session."
            );
        }
    }
}

struct HeldGrant {
    id: PendingId,
    rule_prefix: Option<Vec<String>>,
}

fn lock_grants(
    grants: &Mutex<HashMap<String, HeldGrant>>,
) -> std::sync::MutexGuard<'_, HashMap<String, HeldGrant>> {
    grants.lock().unwrap_or_else(|err| err.into_inner())
}

fn release_message(err: BrokerError) -> String {
    match err {
        BrokerError::StaleBinding => "approval binding is stale and will not be reused".to_owned(),
        other => other.to_string(),
    }
}

/// The directory relative tool paths resolve against: the tool workspace
/// root when the engine set one, as the tools themselves resolve them.
fn approval_cwd() -> String {
    match crate::agent::tools::common::path::get_current_dir() {
        Ok(path) => path.to_string_lossy().into_owned(),
        Err(_) => ".".to_owned(),
    }
}

fn tool_dialect(tool: &str) -> Option<&'static str> {
    match tool {
        "bash" | "shell" => Some("posix"),
        "execute_python" => Some("python"),
        _ => None,
    }
}

#[async_trait]
impl ToolExecutionMiddleware for HookExecutionMiddleware {
    async fn before_execute(
        &self,
        ctx: &ToolExecutionContext<'_>,
    ) -> Result<(), Cow<'static, str>> {
        let ctx = &ctx.as_run();
        let outcome = self.evaluate_pre_tool_hooks(ctx).await;
        crate::agent::hooks::policy_shadow::observe_live_outcome(
            ctx.tool_name,
            ctx.arguments,
            &outcome,
        );
        outcome
    }

    async fn admit_execution(
        &self,
        ctx: &ToolExecutionContext<'_>,
    ) -> Result<(), Cow<'static, str>> {
        let ctx = &ctx.as_run();
        let Some(held) = lock_grants(&self.grants).remove(ctx.call_id) else {
            return Ok(());
        };
        let cwd = approval_cwd();
        let actual = crate::policy::ActionBinding::compute(
            ctx.tool_name,
            ctx.arguments,
            &cwd,
            tool_dialect(ctx.tool_name),
            0,
        );
        if let Err(err) = self.approvals.release(&held.id, actual) {
            return Err(Cow::owned(release_message(err)));
        }
        if let Some(rule_prefix) = held.rule_prefix {
            self.persist_rule(&rule_prefix).await;
        }
        Ok(())
    }

    async fn after_execute(&self, ctx: &ToolExecutionContext<'_>, result: &mut ToolResult) {
        let ctx = &ctx.as_run();
        let matching_hooks = self.hooks.get_matching(ctx.tool_name);
        let hook_ctx = HookContext::new(ctx.tool_name, ctx.arguments, ctx.call_id);
        for hook in &matching_hooks {
            let hook_result = hook.on_post_tool_use(&hook_ctx, result).await;
            let Some(output) = hook_result.hook_specific_output else {
                continue;
            };
            result
                .content
                .push_str("\n\n[Post-edit validation failed]: ");
            result.content.push_str(&output.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::hooks::{HookContext, HookMatcher, HookResult, LifecycleHook};
    use async_trait::async_trait;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Debug)]
    struct DenyOnceHook {
        called: AtomicBool,
    }

    #[async_trait]
    impl LifecycleHook for DenyOnceHook {
        async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
            self.called.store(true, Ordering::SeqCst);
            HookResult::deny("blocked by policy")
        }
    }

    #[derive(Debug)]
    struct CountingHook {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl LifecycleHook for CountingHook {
        async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
            self.calls.fetch_add(1, Ordering::SeqCst);
            HookResult::allow()
        }
    }

    #[derive(Debug)]
    struct ApprovalHook;

    #[async_trait]
    impl LifecycleHook for ApprovalHook {
        async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
            HookResult::require_approval("needs review", None)
        }
    }

    #[tokio::test]
    async fn hook_execution_middleware_denies_before_execution_and_stops_later_hooks() {
        let deny = Arc::new(DenyOnceHook {
            called: AtomicBool::new(false),
        });
        let later = Arc::new(CountingHook {
            calls: AtomicUsize::new(0),
        });
        let mut hooks = HookRegistry::new();
        hooks.register(
            HookMatcher::tool("bash"),
            Arc::clone(&deny) as Arc<dyn LifecycleHook>,
        );
        hooks.register(
            HookMatcher::tool("bash"),
            Arc::clone(&later) as Arc<dyn LifecycleHook>,
        );

        let middleware = HookExecutionMiddleware::new(hooks);
        let args = serde_json::json!({});
        let ctx = ToolExecutionContext {
            tool_name: "bash",
            arguments: &args,
            call_id: "c1",
        };
        let err = middleware
            .before_execute(&ctx)
            .await
            .expect_err("deny should block");
        assert!(err.contains("blocked by policy"));
        assert!(deny.called.load(Ordering::SeqCst));
        assert_eq!(later.calls.load(Ordering::SeqCst), 0);
    }

    struct RecordingResolver {
        call_id: Mutex<Option<String>>,
    }

    #[async_trait]
    impl ApprovalResolver for RecordingResolver {
        fn resolve_approval(
            &self,
            _tool_name: &str,
            _tool_input: &serde_json::Value,
            _reason: &str,
            _proposed_amendment: Option<&[String]>,
        ) -> futures_util::future::BoxFuture<'_, anyhow::Result<ApprovalOutcome>> {
            Box::pin(async { Ok(ApprovalOutcome::Deny) })
        }

        fn resolve_approval_for_call<'a>(
            &'a self,
            _tool_name: &'a str,
            _tool_input: &'a serde_json::Value,
            _reason: &'a str,
            _proposed_amendment: Option<&'a [String]>,
            call_id: &'a str,
        ) -> futures_util::future::BoxFuture<'a, anyhow::Result<ApprovalOutcome>> {
            *self.call_id.lock().unwrap_or_else(|err| err.into_inner()) = Some(call_id.to_owned());
            Box::pin(async { Ok(ApprovalOutcome::Allow) })
        }
    }

    struct DenyResolver;

    #[async_trait]
    impl ApprovalResolver for DenyResolver {
        fn resolve_approval(
            &self,
            _tool_name: &str,
            _tool_input: &serde_json::Value,
            _reason: &str,
            _proposed_amendment: Option<&[String]>,
        ) -> futures_util::future::BoxFuture<'_, anyhow::Result<ApprovalOutcome>> {
            Box::pin(async { Ok(ApprovalOutcome::Deny) })
        }
    }

    struct PanicResolver;

    #[async_trait]
    impl ApprovalResolver for PanicResolver {
        fn resolve_approval(
            &self,
            _tool_name: &str,
            _tool_input: &serde_json::Value,
            _reason: &str,
            _proposed_amendment: Option<&[String]>,
        ) -> futures_util::future::BoxFuture<'_, anyhow::Result<ApprovalOutcome>> {
            panic!("resolver must not run when the hook allows");
        }
    }

    #[tokio::test]
    async fn require_approval_forwards_the_tool_call_id() {
        let seen = Arc::new(RecordingResolver {
            call_id: Mutex::new(None),
        });
        let mut hooks = HookRegistry::new();
        hooks.register(HookMatcher::tool("bash"), Arc::new(ApprovalHook));
        let middleware = HookExecutionMiddleware::new(hooks)
            .with_approval_resolver(Some(Arc::clone(&seen) as Arc<dyn ApprovalResolver>));
        let args = serde_json::json!({"command": "echo hi"});
        let ctx = ToolExecutionContext {
            tool_name: "bash",
            arguments: &args,
            call_id: "call-7",
        };
        middleware
            .before_execute(&ctx)
            .await
            .expect("allow once the resolver returns allow");
        assert_eq!(
            seen.call_id
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .as_deref(),
            Some("call-7")
        );
    }

    #[tokio::test]
    async fn allow_does_not_call_the_resolver() {
        let hook = Arc::new(CountingHook {
            calls: AtomicUsize::new(0),
        });
        let mut hooks = HookRegistry::new();
        hooks.register(
            HookMatcher::tool("bash"),
            Arc::clone(&hook) as Arc<dyn LifecycleHook>,
        );
        let middleware = HookExecutionMiddleware::new(hooks)
            .with_approval_resolver(Some(Arc::new(PanicResolver)));
        let args = serde_json::json!({"command": "echo hi"});
        let ctx = ToolExecutionContext {
            tool_name: "bash",
            arguments: &args,
            call_id: "call-8",
        };
        middleware
            .before_execute(&ctx)
            .await
            .expect("allow skips the resolver");
        assert_eq!(hook.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn require_approval_without_resolver_fails_closed_with_hook_reason() {
        let mut hooks = HookRegistry::new();
        hooks.register(HookMatcher::tool("fs"), Arc::new(ApprovalHook));
        let cwd = std::env::current_dir().expect("cwd");
        let middleware = HookExecutionMiddleware::new(hooks).with_workspace(cwd.to_string_lossy());
        let args = serde_json::json!({"op": "write"});
        let ctx = ToolExecutionContext {
            tool_name: "fs",
            arguments: &args,
            call_id: "c2",
        };
        let err = middleware
            .before_execute(&ctx)
            .await
            .expect_err("approval without resolver must deny");
        assert!(err.contains("needs review"));
    }

    /// A relative path resolves against the tool workspace root, as the tools
    /// resolve it, not the process directory. `.git/config` in a workspace
    /// outside the process directory is git metadata, not an outside path.
    #[tokio::test(flavor = "current_thread")]
    async fn relative_paths_resolve_against_the_tool_workspace_root() {
        let workspace = tempfile::tempdir().expect("workspace");
        let root = workspace.path().canonicalize().expect("canonical");
        crate::agent::tools::common::set_tool_workspace_root(Some(root.clone()));
        let middleware = HookExecutionMiddleware::new(HookRegistry::new())
            .with_workspace(root.to_string_lossy());
        let args = serde_json::json!({"op": "write", "path": ".git/config"});
        let ctx = ToolExecutionContext {
            tool_name: "fs",
            arguments: &args,
            call_id: "c-root",
        };

        let err = middleware.before_execute(&ctx).await.expect_err("denied");
        crate::agent::tools::common::set_tool_workspace_root(None);

        assert_eq!(err, "writes to the repository's .git directory are denied");
    }

    #[tokio::test]
    async fn bot_approver_cannot_allow_a_tool() {
        let mut hooks = HookRegistry::new();
        hooks.register(HookMatcher::tool("bash"), Arc::new(ApprovalHook));
        let middleware = HookExecutionMiddleware::new(hooks)
            .with_approval_resolver(Some(Arc::new(RecordingResolver {
                call_id: Mutex::new(None),
            })))
            .with_approver(crate::policy::Principal::bot(
                "helper",
                crate::policy::BotOrigin::UnattendedPost,
            ));
        let args = serde_json::json!({"command": "echo hi"});
        let ctx = ToolExecutionContext {
            tool_name: "bash",
            arguments: &args,
            call_id: "call-bot",
        };
        let err = middleware
            .before_execute(&ctx)
            .await
            .expect_err("an unattended answer must not run the tool");
        assert!(err.contains("human"));
    }

    #[tokio::test]
    async fn human_deny_does_not_run_the_tool() {
        let mut hooks = HookRegistry::new();
        hooks.register(HookMatcher::tool("fs"), Arc::new(ApprovalHook));
        let cwd = std::env::current_dir().expect("cwd");
        let middleware = HookExecutionMiddleware::new(hooks)
            .with_workspace(cwd.to_string_lossy())
            .with_approval_resolver(Some(Arc::new(DenyResolver)));
        let args = serde_json::json!({"op": "write", "path": "src/lib.rs"});
        let ctx = ToolExecutionContext {
            tool_name: "fs",
            arguments: &args,
            call_id: "call-deny",
        };
        let err = middleware
            .before_execute(&ctx)
            .await
            .expect_err("a human deny must block the tool");
        assert!(err.contains("needs review"));
    }

    #[tokio::test]
    async fn admit_accepts_the_approved_arguments() {
        let mut hooks = HookRegistry::new();
        hooks.register(HookMatcher::tool("bash"), Arc::new(ApprovalHook));
        let middleware = HookExecutionMiddleware::new(hooks).with_approval_resolver(Some(
            Arc::new(RecordingResolver {
                call_id: Mutex::new(None),
            }),
        ));
        let args = serde_json::json!({"command": "git status"});
        let ctx = ToolExecutionContext {
            tool_name: "bash",
            arguments: &args,
            call_id: "call-admit",
        };
        middleware
            .before_execute(&ctx)
            .await
            .expect("human allow records a grant");
        middleware
            .admit_execution(&ctx)
            .await
            .expect("the same arguments are fresh at spawn");
    }

    #[tokio::test]
    async fn admit_rejects_changed_arguments() {
        let mut hooks = HookRegistry::new();
        hooks.register(HookMatcher::tool("bash"), Arc::new(ApprovalHook));
        let middleware = HookExecutionMiddleware::new(hooks).with_approval_resolver(Some(
            Arc::new(RecordingResolver {
                call_id: Mutex::new(None),
            }),
        ));
        let args = serde_json::json!({"command": "git status"});
        let ctx = ToolExecutionContext {
            tool_name: "bash",
            arguments: &args,
            call_id: "call-stale",
        };
        middleware
            .before_execute(&ctx)
            .await
            .expect("human allow records a grant");
        let changed = serde_json::json!({"command": "rm -rf /"});
        let spawn = ToolExecutionContext {
            tool_name: "bash",
            arguments: &changed,
            call_id: "call-stale",
        };
        let err = middleware
            .admit_execution(&spawn)
            .await
            .expect_err("changed arguments must not reuse the grant");
        assert!(err.contains("stale"));
    }

    #[tokio::test]
    async fn interactive_http_get_asks_then_admit_accepts() {
        let middleware = HookExecutionMiddleware::new(HookRegistry::new())
            .with_overlay(ModeOverlay::Interactive)
            .with_approval_resolver(Some(Arc::new(RecordingResolver {
                call_id: Mutex::new(None),
            })));
        let args = serde_json::json!({"method": "GET", "url": "https://example.com/health"});
        let ctx = ToolExecutionContext {
            tool_name: "http",
            arguments: &args,
            call_id: "call-http",
        };
        middleware
            .before_execute(&ctx)
            .await
            .expect("interactive http get asks and the human allows");
        middleware
            .admit_execution(&ctx)
            .await
            .expect("the same http arguments stay fresh");
    }

    #[tokio::test]
    async fn interactive_http_get_without_a_resolver_denies() {
        let middleware = HookExecutionMiddleware::new(HookRegistry::new())
            .with_overlay(ModeOverlay::Interactive);
        let args = serde_json::json!({"method": "GET", "url": "https://example.com/health"});
        let ctx = ToolExecutionContext {
            tool_name: "http",
            arguments: &args,
            call_id: "call-http-closed",
        };
        let err = middleware
            .before_execute(&ctx)
            .await
            .expect_err("an ask with no waiter denies");
        assert!(err.contains("waiter"));
    }

    #[tokio::test]
    async fn default_mode_allows_public_http_get() {
        let middleware = HookExecutionMiddleware::new(HookRegistry::new());
        let args = serde_json::json!({"method": "GET", "url": "https://example.com/health"});
        let ctx = ToolExecutionContext {
            tool_name: "http",
            arguments: &args,
            call_id: "call-http-default",
        };
        middleware
            .before_execute(&ctx)
            .await
            .expect("default overlay allows a public http get");
    }

    #[tokio::test]
    async fn interactive_mcp_tool_asks() {
        let middleware = HookExecutionMiddleware::new(HookRegistry::new())
            .with_overlay(ModeOverlay::Interactive)
            .with_approval_resolver(Some(Arc::new(RecordingResolver {
                call_id: Mutex::new(None),
            })));
        let args = serde_json::json!({"title": "bug"});
        let ctx = ToolExecutionContext {
            tool_name: "github__create_issue",
            arguments: &args,
            call_id: "call-mcp",
        };
        middleware
            .before_execute(&ctx)
            .await
            .expect("interactive mcp asks and the human allows");
    }

    #[tokio::test]
    async fn floor_denies_vendor_auth_read() {
        let middleware = HookExecutionMiddleware::new(HookRegistry::new());
        let args = serde_json::json!({"op": "read", "path": "/tmp/.codex/auth.json"});
        let ctx = ToolExecutionContext {
            tool_name: "fs",
            arguments: &args,
            call_id: "call-auth",
        };
        let err = middleware
            .before_execute(&ctx)
            .await
            .expect_err("the floor denies a vendor auth read");
        assert!(err.contains("vendor auth"));
    }

    /// The registry runs " fs" as `fs`, `memory` as `workspace` and `web` as
    /// `http`, so the gate decides on those tools: the vendor auth floor, the
    /// outside-workspace check and the SSRF floor all apply.
    #[tokio::test]
    async fn a_tool_name_is_gated_as_the_tool_it_runs() {
        let cwd = std::env::current_dir().expect("cwd");
        let middleware =
            HookExecutionMiddleware::new(HookRegistry::new()).with_workspace(cwd.to_string_lossy());
        for (name, args) in [
            (
                " fs",
                serde_json::json!({"op": "read", "path": "/tmp/.codex/auth.json"}),
            ),
            (
                "fs ",
                serde_json::json!({"op": "read", "path": "/etc/passwd"}),
            ),
            (
                " fs",
                serde_json::json!({"op": "read", "path": "/etc/passwd"}),
            ),
            (
                "memory",
                serde_json::json!({"action": "config", "path": "/etc"}),
            ),
            (
                " workspace",
                serde_json::json!({"action": "structure", "path": "/etc"}),
            ),
            (
                "web",
                serde_json::json!({"method": "GET", "url": "http://169.254.169.254/latest"}),
            ),
        ] {
            let ctx = ToolExecutionContext {
                tool_name: name,
                arguments: &args,
                call_id: "call-alias",
            };
            let denied = middleware.before_execute(&ctx).await;
            assert!(denied.is_err(), "{name:?} {args} was allowed");
        }
    }

    #[tokio::test]
    async fn path_inside_the_workspace_is_allowed() {
        let cwd = std::env::current_dir().expect("cwd");
        let middleware =
            HookExecutionMiddleware::new(HookRegistry::new()).with_workspace(cwd.to_string_lossy());
        let args = serde_json::json!({"op": "read", "path": "."});
        let ctx = ToolExecutionContext {
            tool_name: "fs",
            arguments: &args,
            call_id: "call-inside",
        };
        middleware
            .before_execute(&ctx)
            .await
            .expect("a path inside the workspace is allowed");
    }

    #[tokio::test]
    async fn path_outside_the_workspace_asks() {
        let middleware = HookExecutionMiddleware::new(HookRegistry::new())
            .with_workspace("/caliche-missing-workspace")
            .with_approval_resolver(Some(Arc::new(RecordingResolver {
                call_id: Mutex::new(None),
            })));
        let args = serde_json::json!({"op": "read", "path": "/etc/passwd"});
        let ctx = ToolExecutionContext {
            tool_name: "fs",
            arguments: &args,
            call_id: "call-outside",
        };
        middleware
            .before_execute(&ctx)
            .await
            .expect("an outside path asks and the human allows");
        middleware
            .admit_execution(&ctx)
            .await
            .expect("the same outside path stays fresh");
    }

    #[tokio::test]
    async fn path_outside_without_a_resolver_denies() {
        let middleware = HookExecutionMiddleware::new(HookRegistry::new())
            .with_workspace("/caliche-missing-workspace");
        let args = serde_json::json!({"op": "read", "path": "/etc/passwd"});
        let ctx = ToolExecutionContext {
            tool_name: "fs",
            arguments: &args,
            call_id: "call-outside-closed",
        };
        let err = middleware
            .before_execute(&ctx)
            .await
            .expect_err("an outside path with no waiter denies");
        assert!(err.contains("waiter"));
    }

    #[tokio::test]
    async fn rejected_path_denies_before_the_tool() {
        let middleware = HookExecutionMiddleware::new(HookRegistry::new())
            .with_workspace("/caliche-missing-workspace");
        let args = serde_json::json!({"op": "read", "path": "notes.txt:Zone.Identifier"});
        let ctx = ToolExecutionContext {
            tool_name: "fs",
            arguments: &args,
            call_id: "call-ads",
        };
        let err = middleware
            .before_execute(&ctx)
            .await
            .expect_err("an alternate data stream denies");
        assert!(err.contains("rejected"));
    }

    fn fs_write(path: &str) -> serde_json::Value {
        serde_json::json!({"op": "write", "path": path, "content": "x"})
    }

    #[tokio::test]
    async fn accept_all_outside_write_still_asks_the_human() {
        let seen = Arc::new(RecordingResolver {
            call_id: Mutex::new(None),
        });
        let middleware = HookExecutionMiddleware::new(HookRegistry::new())
            .with_workspace("/caliche-missing-workspace")
            .with_overlay(ModeOverlay::AcceptAll)
            .with_approval_resolver(Some(Arc::clone(&seen) as Arc<dyn ApprovalResolver>));
        let args = fs_write("/tmp/caliche-outside.txt");
        let ctx = ToolExecutionContext {
            tool_name: "fs",
            arguments: &args,
            call_id: "call-accept-all-outside",
        };
        middleware
            .before_execute(&ctx)
            .await
            .expect("the human allows the outside write");
        assert_eq!(
            seen.call_id
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .as_deref(),
            Some("call-accept-all-outside"),
            "accept-all must not skip the outside-write ask"
        );
    }

    #[tokio::test]
    async fn accept_all_outside_write_without_a_resolver_denies() {
        let middleware = HookExecutionMiddleware::new(HookRegistry::new())
            .with_workspace("/caliche-missing-workspace")
            .with_overlay(ModeOverlay::AcceptAll);
        for (tool, args) in [
            ("fs", fs_write("/tmp/caliche-outside.txt")),
            (
                "apply_patch",
                serde_json::json!({"patch": "*** Add File: /tmp/caliche-outside.txt\n+x\n"}),
            ),
        ] {
            let ctx = ToolExecutionContext {
                tool_name: tool,
                arguments: &args,
                call_id: "call-accept-all-closed",
            };
            let err = middleware
                .before_execute(&ctx)
                .await
                .expect_err("an outside write with no waiter denies under accept-all");
            assert!(err.contains("waiter"), "{tool}: {err}");
        }
    }

    #[tokio::test]
    async fn accept_all_inside_write_does_not_ask() {
        let cwd = std::env::current_dir().expect("cwd");
        let middleware = HookExecutionMiddleware::new(HookRegistry::new())
            .with_workspace(cwd.to_string_lossy())
            .with_overlay(ModeOverlay::AcceptAll)
            .with_approval_resolver(Some(Arc::new(PanicResolver)));
        let args = fs_write("caliche-inside.txt");
        let ctx = ToolExecutionContext {
            tool_name: "fs",
            arguments: &args,
            call_id: "call-accept-all-inside",
        };
        middleware
            .before_execute(&ctx)
            .await
            .expect("accept-all allows a write inside the workspace");
    }

    #[tokio::test]
    async fn missing_or_blank_workspace_denies_filesystem_tools() {
        let cwd = std::env::current_dir().expect("cwd");
        let unset = HookExecutionMiddleware::new(HookRegistry::new());
        let blank = HookExecutionMiddleware::new(HookRegistry::new()).with_workspace("  ");
        for middleware in [unset, blank] {
            let middleware = middleware
                .with_overlay(ModeOverlay::AcceptAll)
                .with_approval_resolver(Some(Arc::new(PanicResolver)));
            for (tool, args) in [
                (
                    "fs",
                    serde_json::json!({"op": "read", "path": "Cargo.toml"}),
                ),
                (
                    "fs",
                    fs_write(&cwd.join("caliche-inside.txt").to_string_lossy()),
                ),
                ("fs", serde_json::json!({"op": "list"})),
                (
                    "apply_patch",
                    serde_json::json!({"patch": "*** Update File: Cargo.toml\n"}),
                ),
            ] {
                let ctx = ToolExecutionContext {
                    tool_name: tool,
                    arguments: &args,
                    call_id: "call-no-workspace",
                };
                let err = middleware
                    .before_execute(&ctx)
                    .await
                    .expect_err("no workspace root fails closed");
                assert!(err.contains("no workspace root"), "{tool}: {err}");
            }
        }
    }
}
