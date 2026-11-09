//! `WorkflowEngine` method implementations.

use super::*;

impl WorkflowEngine {
    pub fn new() -> Self {
        Self::with_executor(NoopWorkflowNodeExecutor::new())
    }

    pub fn with_executor(executor: Arc<dyn WorkflowNodeExecutor>) -> Self {
        Self {
            workflows: Arc::new(DashMap::new()),
            executions: Arc::new(DashMap::new()),
            cancellation_flags: Arc::new(DashMap::new()),
            executor,
            node_error_policy: NodeErrorPolicy::Halt,
            parallel_branch_semaphore: Arc::new(Semaphore::new(
                configured_parallel_branch_concurrency(),
            )),
            idempotency_cache: Arc::new(DashMap::new()),
            node_retry_attempts: DEFAULT_NODE_RETRY_ATTEMPTS,
            node_retry_backoff: DEFAULT_NODE_RETRY_BACKOFF,
            max_nodes: None,
            max_tokens: None,
            run_usage: Arc::new(DashMap::new()),
        }
    }

    pub fn with_node_error_policy(mut self, policy: NodeErrorPolicy) -> Self {
        self.node_error_policy = policy;
        self
    }

    /// Set the retry attempts and base backoff for agent nodes.
    pub fn with_node_retries(mut self, attempts: u32, backoff: Duration) -> Self {
        self.node_retry_attempts = attempts;
        self.node_retry_backoff = backoff;
        self
    }

    /// Cap how many agent nodes one run may start and how many tokens it may spend.
    pub fn with_run_budgets(mut self, max_nodes: Option<usize>, max_tokens: Option<u64>) -> Self {
        self.max_nodes = max_nodes.filter(|limit| *limit > 0);
        self.max_tokens = max_tokens.filter(|limit| *limit > 0);
        self
    }

    /// Apply the retry, concurrency, and budget limits from [`OrchestrationConfig`].
    pub fn with_orchestration_config(mut self, config: &OrchestrationConfig) -> Self {
        self.node_retry_attempts = config.node_retry_attempts_or_default();
        self.node_retry_backoff = config.node_retry_backoff();
        self.parallel_branch_semaphore =
            Arc::new(Semaphore::new(config.pattern_concurrency_or_default()));
        self.with_run_budgets(config.pattern_max_nodes, config.pattern_max_tokens)
    }

    /// A failed node halts the run unless the policy explicitly continues.
    ///
    /// [`NodeErrorPolicy::Retry`] retries inside the node first (see
    /// [`agent_node`]); once the attempts are spent the run halts, so an exhausted
    /// retry is not silently treated as success.
    pub(super) fn should_halt_on_node_error(&self, result: &NodeExecutionResult) -> bool {
        !result.success
            && matches!(
                self.node_error_policy,
                NodeErrorPolicy::Halt | NodeErrorPolicy::Retry
            )
    }

    pub(super) fn mark_failed_on_node_error(
        &self,
        context: &mut ExecutionContext,
        result: &NodeExecutionResult,
    ) {
        context.state = WorkflowState::Failed;
        if let Some(err) = &result.error {
            context.set_variable(VAR_ERROR, serde_json::json!(err));
        }
        context.end_time = Some(Utc::now());
    }

    /// Register a workflow
    #[expect(
        clippy::unused_async,
        reason = "public API consumed as async across caliche_server/js/py bindings and integration tests; changing the signature is a cross-repo break out of scope here"
    )]
    pub async fn register_workflow(&self, workflow: Arc<dyn Workflow>) -> Result<()> {
        // Validate workflow
        workflow.validate()?;

        let workflow_id = workflow.id().to_string();
        // DashMap insert is lock-free - no await needed
        self.workflows.insert(workflow_id.clone(), workflow);

        tracing::info!(workflow_id = %workflow_id, "Registered workflow");
        Ok(())
    }

    /// Get a workflow by ID
    pub fn get_workflow(&self, workflow_id: &str) -> Option<Arc<dyn Workflow>> {
        // DashMap get is lock-free - no await needed
        self.workflows
            .get(workflow_id)
            .map(|entry| Arc::clone(entry.value()))
    }

    /// Spawn background task that runs workflow execution and updates context on completion/failure.
    pub(super) fn spawn_workflow_task(
        engine: Self,
        workflow: Arc<dyn Workflow>,
        context_arc: Arc<RwLock<ExecutionContext>>,
    ) {
        tokio::spawn(async move {
            let workflow_id = workflow.id().to_string();
            let execution_id = {
                let ctx = context_arc.read().await;
                ctx.execution_id.clone()
            };
            let cancel_flag = engine
                .cancellation_flags
                .get(&execution_id)
                .map(|entry| Arc::clone(entry.value()));

            let execute_result = {
                let mut guard = context_arc.write().await;
                if let Some(flag) = cancel_flag.as_ref() {
                    guard.cooperative_cancel = Some(Arc::clone(flag));
                }
                workflow.execute(&engine, &mut guard).await
            };

            let mut guard = context_arc.write().await;
            match execute_result {
                Ok(result) => {
                    guard.set_variable(VAR_RESULT, result);
                    if guard.is_cancelled() {
                        guard.state = WorkflowState::Cancelled;
                        guard.end_time = Some(Utc::now());
                    } else if guard.state != WorkflowState::Paused
                        && guard.state != WorkflowState::Failed
                    {
                        guard.state = WorkflowState::Completed;
                        guard.end_time = Some(Utc::now());
                    }
                    tracing::info!(
                        workflow_id = %workflow_id,
                        execution_id = %execution_id,
                        state = ?guard.state,
                        "Workflow execution finished"
                    );
                }
                Err(e) => {
                    guard.set_variable(VAR_ERROR, serde_json::json!(e.to_string()));
                    guard.state = WorkflowState::Failed;
                    guard.end_time = Some(Utc::now());
                    tracing::error!(
                        workflow_id = %workflow_id,
                        execution_id = %execution_id,
                        error = %e,
                        "Workflow execution failed"
                    );
                }
            }
        });
    }

    /// Start workflow execution
    #[expect(
        clippy::unused_async,
        reason = "public API consumed as async across caliche_py bindings; changing the signature is a cross-repo break out of scope here"
    )]
    pub async fn start_execution(&self, workflow_id: &str) -> Result<String> {
        let workflow = self
            .get_workflow(workflow_id)
            .context("Workflow not found")?;

        let mut context = ExecutionContext::new(workflow_id);
        context.state = WorkflowState::Running;
        let execution_id = context.execution_id.clone();

        let context_arc = Arc::new(RwLock::new(context));
        self.executions
            .insert(execution_id.clone(), Arc::clone(&context_arc));
        self.cancellation_flags
            .insert(execution_id.clone(), Arc::new(AtomicBool::new(false)));

        Self::spawn_workflow_task(self.clone(), Arc::clone(&workflow), context_arc);
        Ok(execution_id)
    }

    /// Start workflow execution with a pre-seeded execution context.
    ///
    /// This is useful for callers that need to inject initial variables (e.g., API input payloads).
    #[expect(
        clippy::unused_async,
        reason = "public API consumed as async by caliche_server; changing the signature is a cross-repo break out of scope here"
    )]
    pub async fn start_execution_with_context(
        &self,
        workflow_id: &str,
        mut context: ExecutionContext,
    ) -> Result<String> {
        let workflow = self
            .get_workflow(workflow_id)
            .context("Workflow not found")?;

        context.workflow_id = workflow_id.to_string();
        context.state = WorkflowState::Running;
        let execution_id = context.execution_id.clone();

        let context_arc = Arc::new(RwLock::new(context));
        self.executions
            .insert(execution_id.clone(), Arc::clone(&context_arc));
        self.cancellation_flags
            .insert(execution_id.clone(), Arc::new(AtomicBool::new(false)));

        Self::spawn_workflow_task(self.clone(), Arc::clone(&workflow), context_arc);
        Ok(execution_id)
    }

    /// Request cooperative cancellation for a running execution.
    pub fn cancel_execution(&self, execution_id: &str) -> bool {
        if let Some(flag) = self.cancellation_flags.get(execution_id) {
            flag.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Remove finished executions older than `max_age_secs` (retention policy skeleton).
    pub fn prune_executions(&self, max_age_secs: u64) -> usize {
        let cutoff = Utc::now() - chrono::Duration::seconds(max_age_secs as i64);
        let mut removed = 0usize;
        let stale: Vec<String> = self
            .executions
            .iter()
            .filter_map(|entry| {
                let ctx = entry.value().try_read().ok()?;
                let ended = ctx.end_time?;
                if ended < cutoff {
                    Some(entry.key().clone())
                } else {
                    None
                }
            })
            .collect();
        for execution_id in stale {
            self.executions.remove(&execution_id);
            self.cancellation_flags.remove(&execution_id);
            self.forget_run_usage(&execution_id);
            removed += 1;
        }
        removed
    }

    /// Get execution context
    #[expect(
        clippy::unused_async,
        reason = "public API consumed as async across caliche_server/py bindings; changing the signature is a cross-repo break out of scope here"
    )]
    pub async fn get_execution(&self, execution_id: &str) -> Option<Arc<RwLock<ExecutionContext>>> {
        // DashMap get is lock-free - no await needed
        self.executions
            .get(execution_id)
            .map(|entry| Arc::clone(entry.value()))
    }

    /// Check if execution is complete
    pub async fn is_execution_complete(&self, execution_id: &str) -> bool {
        if let Some(context_arc) = self.get_execution(execution_id).await {
            let context = context_arc.read().await;
            context.end_time.is_some()
        } else {
            false
        }
    }

    /// Get execution result
    pub async fn get_execution_result(&self, execution_id: &str) -> Result<serde_json::Value> {
        let context_arc = self
            .get_execution(execution_id)
            .await
            .context("Execution not found")?;

        let context = context_arc.read().await;

        if context.end_time.is_none() {
            anyhow::bail!("Execution still running");
        }

        if let Some(error) = context.get_variable(VAR_ERROR) {
            anyhow::bail!("Execution failed: {}", error);
        }

        context
            .get_variable(VAR_RESULT)
            .cloned()
            .context("No result available")
    }

    /// Evaluate a condition expression against context.
    ///
    /// Supports comparisons such as `status == 'approved'`, `count < 3`, and
    /// `$.result.success == true`.
    pub fn evaluate_condition(
        condition: &str,
        context: &serde_json::Value,
        node_output: Option<&serde_json::Value>,
    ) -> bool {
        crate::orchestration::workflow_conditions::evaluate_condition(
            condition,
            context,
            node_output,
        )
    }

    /// List all registered workflows
    pub fn list_workflows(&self) -> Vec<String> {
        // DashMap iteration is lock-free
        self.workflows
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// List all executions
    pub fn list_executions(&self) -> Vec<String> {
        // DashMap iteration is lock-free
        self.executions
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// List all pending interrupts (workflows waiting for human approval)
    pub async fn list_pending_interrupts(&self) -> Vec<PendingInterrupt> {
        let mut interrupts = Vec::new();

        // DashMap iteration is lock-free, but we still need RwLock for ExecutionContext
        for entry in self.executions.iter() {
            let context_arc = entry.value();
            let context = context_arc.read().await;
            if let Some(ref interrupt) = context.pending_interrupt {
                interrupts.push(interrupt.clone());
            }
        }

        interrupts
    }

    /// Get a pending interrupt by interrupt ID
    pub async fn get_interrupt(&self, interrupt_id: &str) -> Option<PendingInterrupt> {
        // DashMap iteration is lock-free, but we still need RwLock for ExecutionContext
        for entry in self.executions.iter() {
            let context_arc = entry.value();
            let context = context_arc.read().await;
            if let Some(ref interrupt) = context.pending_interrupt
                && interrupt.interrupt_id == interrupt_id
            {
                return Some(interrupt.clone());
            }
        }

        None
    }

    /// Resume a paused workflow execution with human feedback.
    ///
    /// Continues from the first selected outgoing edge of the paused node via
    /// [`Self::execute_state_machine`], for both entry-point workflows and DAG definitions.
    /// Complex DAG fan-out may require a single continuation path; parallel branches are not
    /// re-scheduled separately on resume.
    pub async fn resume_execution(
        &self,
        interrupt_id: &str,
        feedback: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        // Find the execution with this interrupt - DashMap iteration is lock-free
        let mut found_context: Option<Arc<RwLock<ExecutionContext>>> = None;
        for entry in self.executions.iter() {
            let context_arc = entry.value();
            let context = context_arc.read().await;
            if let Some(ref interrupt) = context.pending_interrupt
                && interrupt.interrupt_id == interrupt_id
            {
                found_context = Some(Arc::clone(context_arc));
                break;
            }
        }

        let context_arc = found_context.context("Interrupt not found")?;

        // Validate required fields if specified
        let mut ctx = context_arc.write().await;
        if let Some(ref interrupt) = ctx.pending_interrupt
            && let Some(ref required_fields) = interrupt.required_fields
        {
            for field in required_fields {
                if !feedback.contains_key(field) {
                    anyhow::bail!("Missing required field: {}", field);
                }
            }
        }

        // Add feedback to context variables
        for (key, value) in feedback {
            ctx.set_variable(format!("interrupt_{}", key), value);
        }

        // Clear interrupt and resume execution
        ctx.pending_interrupt = None;
        ctx.state = WorkflowState::Running;

        // Get workflow and continue execution
        let workflow = self
            .get_workflow(&ctx.workflow_id)
            .context("Workflow not found")?;

        let paused_node_id = ctx.current_node_id.clone();

        // Next node after the interrupt (state machine and DAG); avoids re-entering the paused node.
        let outgoing_edges = build_outgoing_edges(workflow.edges());
        let resume_from = if let Some(ref paused_id) = paused_node_id {
            Self::select_next_node_id(paused_id.as_str(), &outgoing_edges, &ctx, None)
        } else {
            tracing::warn!(
                execution_id = %ctx.execution_id,
                workflow_id = %ctx.workflow_id,
                "resume_execution: missing current_node_id; cannot continue from a specific node"
            );
            None
        };

        if let Some(ref next_id) = resume_from {
            ctx.current_node_id = Some(next_id.clone());
        }
        drop(ctx); // Release write lock

        // Continue execution in background
        let engine = self.clone();
        let workflow_clone = Arc::clone(&workflow);
        let context_clone = Arc::clone(&context_arc);

        tokio::spawn(async move {
            let mut ctx = context_clone.write().await;

            let Some(start_node_id) = resume_from else {
                // No next node after interrupt -> treat as completed
                ctx.state = WorkflowState::Completed;
                ctx.end_time = Some(Utc::now());
                return;
            };

            match engine
                .execute_state_machine(
                    workflow_clone.nodes(),
                    workflow_clone.edges(),
                    &start_node_id,
                    &mut ctx,
                )
                .await
            {
                Ok(_) => {
                    if ctx.state != WorkflowState::Paused {
                        ctx.state = WorkflowState::Completed;
                        ctx.end_time = Some(Utc::now());
                    }
                }
                Err(e) => {
                    ctx.state = WorkflowState::Failed;
                    ctx.set_variable(VAR_ERROR, serde_json::json!(e.to_string()));
                    ctx.end_time = Some(Utc::now());
                }
            }
        });

        Ok(())
    }
}

impl Clone for WorkflowEngine {
    fn clone(&self) -> Self {
        Self {
            workflows: Arc::clone(&self.workflows),
            executions: Arc::clone(&self.executions),
            cancellation_flags: Arc::clone(&self.cancellation_flags),
            executor: Arc::clone(&self.executor),
            node_error_policy: self.node_error_policy,
            parallel_branch_semaphore: Arc::clone(&self.parallel_branch_semaphore),
            idempotency_cache: Arc::clone(&self.idempotency_cache),
            node_retry_attempts: self.node_retry_attempts,
            node_retry_backoff: self.node_retry_backoff,
            max_nodes: self.max_nodes,
            max_tokens: self.max_tokens,
            run_usage: Arc::clone(&self.run_usage),
        }
    }
}

impl Default for WorkflowEngine {
    fn default() -> Self {
        Self::new()
    }
}
