use crate::agent::hooks::SteeringFollowUpProvider;
use crate::agent::hooks::approval::{AmendmentPersister, ApprovalResolver};
use crate::agent::hooks::permission::PermissionMode;
use crate::agent::hooks::{HookMatcher, HookRegistry, LifecycleHook};
use crate::agent::kernel::SharedState;
use crate::agent::memory::TokenCounter;
use crate::llm::EmbeddingProvider;
use crate::utils::user_context::UserContext;
use std::collections::HashSet;
use std::sync::Arc;

impl super::Agent {
    // =====================================================================
    // Builder-style (consuming) methods
    // =====================================================================

    /// Set the maximum number of tool call iterations (default: 50).
    /// Use `usize::MAX` for unlimited iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = system_prompt.into();
        self
    }

    /// Set the tool choice preference (default: Auto).
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: Option<crate::messages::ToolChoice>) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Set the hooks registry.
    #[must_use]
    pub fn with_hooks(mut self, hooks: HookRegistry) -> Self {
        self.hooks = hooks;
        self
    }

    /// Set the permission mode.
    #[must_use]
    pub const fn with_permission_mode(mut self, mode: PermissionMode) -> Self {
        self.permission_mode = mode;
        self
    }

    /// Set allowed tools (if None, all tools are allowed).
    /// Converts `Vec` to `HashSet` for O(1) lookup performance.
    #[must_use]
    pub fn with_allowed_tools(mut self, tools: Option<Vec<String>>) -> Self {
        self.allowed_tools = tools.map(|v| Arc::new(v.into_iter().collect::<HashSet<_>>()));
        self
    }

    /// Set the maximum context tokens before aggressive pruning (default: 128K).
    #[must_use]
    pub const fn with_max_context_tokens(mut self, max_tokens: usize) -> Self {
        self.max_context_tokens = max_tokens;
        self
    }

    // =====================================================================
    // Mutable setters (in-place, avoids cloning the whole Agent)
    // =====================================================================

    pub const fn set_max_iterations(&mut self, max_iterations: usize) {
        self.max_iterations = max_iterations;
    }

    pub fn set_system_prompt(&mut self, system_prompt: impl Into<String>) {
        let system_prompt = system_prompt.into();
        if system_prompt == self.system_prompt {
            return;
        }
        self.system_prompt = system_prompt;
        // The cached system messages hold the old prompt.
        *self.cached_system_messages.write() = None;
        #[cfg(feature = "tools-rag")]
        {
            *self.cached_system_messages_with_workspace.write() = None;
        }
    }

    pub fn set_tool_choice(&mut self, tool_choice: Option<crate::messages::ToolChoice>) {
        self.tool_choice = tool_choice;
    }

    /// Compute the tool choice to use for this iteration.
    ///
    /// On the first iteration this respects explicit `Specific` / `Required` choices.
    /// On subsequent iterations those forcing modes are relaxed to `Auto` so the model
    /// can emit a final answer instead of being trapped in tool-only turns.
    pub(crate) fn tool_choice_for_iteration(
        &self,
        iteration: usize,
        use_native: bool,
    ) -> Option<crate::messages::ToolChoice> {
        resolve_tool_choice_for_iteration(self.tool_choice.as_ref(), iteration, use_native)
    }

    pub fn set_hooks(&mut self, hooks: HookRegistry) {
        self.hooks = hooks;
    }

    /// Attach a shared reasoning-effort slot (used by the reasoning scheduler).
    pub fn set_reasoning_effort_slot(
        &mut self,
        slot: Option<Arc<crate::agent::hooks::ReasoningEffortSlot>>,
    ) {
        self.reasoning_effort_slot = slot;
    }

    pub const fn set_permission_mode(&mut self, mode: PermissionMode) {
        self.permission_mode = mode;
    }

    /// Permission mode currently applied to this agent.
    #[must_use]
    pub const fn permission_mode(&self) -> PermissionMode {
        self.permission_mode
    }

    pub fn set_allowed_tools(&mut self, tools: Option<Vec<String>>) {
        self.allowed_tools = tools.map(|v| Arc::new(v.into_iter().collect::<HashSet<_>>()));
    }

    /// Return the current allowed tool names. `None` means all registered tools are allowed.
    #[must_use]
    pub fn allowed_tools(&self) -> Option<Vec<String>> {
        self.allowed_tools
            .as_ref()
            .map(|s| s.iter().cloned().collect())
    }

    /// Shared counter handle for stream paths that cannot borrow `self`.
    pub(crate) fn readonly_filter_hits_counter(&self) -> Arc<std::sync::atomic::AtomicUsize> {
        Arc::clone(&self.readonly_filter_hits)
    }

    /// Record tool denials from the allowed-tools / readonly filter.
    pub(crate) fn record_readonly_filter_hits(&self, hits: usize) {
        if hits == 0 {
            return;
        }
        self.readonly_filter_hits
            .fetch_add(hits, std::sync::atomic::Ordering::Relaxed);
    }

    /// Drain readonly-filter denial count since the last take.
    #[must_use]
    pub fn take_readonly_filter_hits(&self) -> usize {
        self.readonly_filter_hits
            .swap(0, std::sync::atomic::Ordering::Relaxed)
    }

    /// Current readonly-filter denial count without draining.
    #[must_use]
    pub fn readonly_filter_hits(&self) -> usize {
        self.readonly_filter_hits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub const fn set_max_context_tokens(&mut self, max_tokens: usize) {
        self.max_context_tokens = max_tokens;
    }

    pub fn set_session_id(&mut self, session_id: Option<String>) {
        self.session_id = session_id;
    }

    #[must_use]
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Set cached provider context window (computed at construction for streaming path).
    pub const fn set_cached_context_window(&mut self, window: Option<usize>) {
        self.cached_context_window = window;
    }

    /// Set compaction strategy for context window guard.
    #[cfg(feature = "compaction")]
    pub fn set_compaction_strategy(
        &mut self,
        strategy: Arc<dyn super::compaction::CompactionStrategy>,
    ) {
        self.compaction_guard = Some(Arc::new(
            super::context_window_guard::ContextWindowGuard::new(strategy),
        ));
    }

    /// Set the context window guard directly (e.g. with custom thresholds).
    #[cfg(feature = "compaction")]
    pub fn set_compaction_guard(
        &mut self,
        guard: Arc<super::context_window_guard::ContextWindowGuard>,
    ) {
        self.compaction_guard = Some(guard);
    }

    pub const fn set_max_tokens(&mut self, max_tokens: Option<u32>) {
        self.max_tokens = max_tokens;
    }

    #[must_use]
    pub const fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    pub const fn set_temperature(&mut self, temperature: Option<f32>) {
        self.temperature = temperature;
    }

    #[must_use]
    pub const fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Set token counter (e.g., model-specific tokenizer when available).
    pub fn set_token_counter(&mut self, token_counter: Arc<dyn TokenCounter>) {
        self.token_counter = token_counter;
    }

    /// Append additional text to the system prompt (useful for skills).
    pub fn append_to_system_prompt(&mut self, additional: &str) {
        self.system_prompt.push_str("\n\n");
        self.system_prompt.push_str(additional);
    }

    /// Set whether to include "Current date and time" and "Current working directory" in the
    /// system message when tools or workspace are present. Default `true`. Set `false` for
    /// minimal prompt (benchmark parity, low latency).
    pub const fn set_include_context_in_system(&mut self, value: bool) {
        self.include_context_in_system = value;
    }

    pub const fn set_tool_execution_driver(&mut self, driver: super::ToolExecutionDriver) {
        self.tool_execution_driver = driver;
    }

    #[must_use]
    pub const fn tool_execution_driver(&self) -> super::ToolExecutionDriver {
        self.tool_execution_driver
    }

    /// Set optional embedding provider for tool schema ordering by query similarity.
    pub fn set_tool_ordering_embedding_provider(
        &mut self,
        provider: Option<Arc<dyn EmbeddingProvider>>,
    ) {
        self.tool_ordering_embedding_provider = provider;
    }

    /// Set max characters for tool output kept in conversation history. Default `12_000`.
    pub const fn set_max_tool_output_in_history(&mut self, max_chars: usize) {
        self.max_tool_output_in_history = max_chars;
    }

    /// Set whether to add "[truncated for context]" when tool output is summarised for history.
    pub const fn set_annotate_tool_output_truncation(&mut self, value: bool) {
        self.annotate_tool_output_truncation = value;
    }

    /// Set the AgentFS tool logger for persistent audit trail.
    #[cfg(feature = "agentfs")]
    pub fn set_agentfs_tool_logger(
        &mut self,
        logger: Option<Arc<crate::agent::tools::AgentFsToolLogger>>,
    ) {
        self.agentfs_tool_logger = logger;
    }

    /// Set the sandbox handle for tool execution (bash, fs).
    #[cfg(feature = "sandbox")]
    pub fn set_sandbox(&mut self, handle: Arc<caliche_sandbox::SandboxHandle>) {
        self.sandbox = Some(handle);
    }

    /// Opt into the translate → commit → execute operation path.
    pub fn set_operation_manager(
        &mut self,
        manager: Arc<dyn crate::agent::operations::OperationManager>,
    ) {
        self.operation_manager = Some(manager);
        if self.operation_translators.is_none() {
            self.operation_translators =
                Some(crate::agent::operations::shared_builtin_translators());
        }
    }

    /// True when an [`crate::agent::operations::OperationManager`] is installed.
    #[must_use]
    pub const fn operation_dispatch_enabled(&self) -> bool {
        self.operation_manager.is_some()
    }

    /// Apply an inbox settings reasoning-effort hint (creates the slot if needed).
    pub fn apply_reasoning_effort(&mut self, effort: Option<crate::messages::ReasoningEffort>) {
        let slot = self
            .reasoning_effort_slot
            .get_or_insert_with(|| Arc::new(crate::agent::hooks::ReasoningEffortSlot::new()));
        slot.set(effort);
    }

    /// Current inbox / scheduler reasoning-effort hint.
    #[must_use]
    pub fn current_reasoning_effort(&self) -> Option<crate::messages::ReasoningEffort> {
        self.reasoning_effort_slot
            .as_ref()
            .and_then(|slot| slot.current())
    }

    /// Record the latest compaction report (survives until the next compaction).
    pub fn store_compaction_report(&self, report: crate::messages::CompactionReport) {
        *self.last_compaction_report.write() = Some(report);
    }

    /// Clone the latest compaction report, if any.
    #[must_use]
    pub fn last_compaction_report(&self) -> Option<crate::messages::CompactionReport> {
        self.last_compaction_report.read().clone()
    }

    /// Set the approval resolver for `RequireApproval` from hooks.
    pub fn set_approval_resolver(&mut self, resolver: Option<Arc<dyn ApprovalResolver>>) {
        let _ = self.replace_approval_resolver(resolver);
    }

    /// Replace the approval resolver and return the previous one.
    pub fn replace_approval_resolver(
        &mut self,
        resolver: Option<Arc<dyn ApprovalResolver>>,
    ) -> Option<Arc<dyn ApprovalResolver>> {
        std::mem::replace(&mut self.approval_resolver, resolver)
    }

    /// Whether a `RequireApproval` resolver is installed.
    #[must_use]
    pub fn has_approval_resolver(&self) -> bool {
        self.approval_resolver.is_some()
    }

    /// Set the amendment persister for `AllowAndPersist` outcomes.
    pub fn set_amendment_persister(&mut self, persister: Option<Arc<dyn AmendmentPersister>>) {
        self.amendment_persister = persister;
    }

    /// Set the steering/follow-up provider. When set, the chat loop checks for follow-up messages when the agent would stop.
    pub fn set_steering_followup_provider(
        &mut self,
        provider: Option<Arc<dyn SteeringFollowUpProvider>>,
    ) {
        self.steering_followup_provider = provider;
    }

    // =====================================================================
    // RAG configuration (tools-rag feature)
    // =====================================================================

    #[cfg(feature = "tools-rag")]
    pub(crate) fn set_workspace_rag(
        &mut self,
        rag: Option<Arc<crate::rag::RagService>>,
        indexing_ready: Option<Arc<crate::agent::workspace_indexing::WorkspaceIndexingReady>>,
    ) {
        self.workspace_rag = rag;
        self.workspace_indexing_ready = indexing_ready;
    }

    #[cfg(feature = "tools-rag")]
    pub const fn set_workspace_rag_top_k(&mut self, top_k: usize) {
        self.workspace_rag_top_k = top_k;
    }

    #[cfg(feature = "tools-rag")]
    pub const fn set_workspace_rag_min_similarity(&mut self, min_similarity: f32) {
        self.workspace_rag_min_similarity = min_similarity;
    }

    #[cfg(feature = "tools-rag")]
    pub fn set_workspace_rag_retrieval(
        &mut self,
        line_level_bm25: bool,
        search_options: Option<crate::rag::RagSearchOptions>,
    ) {
        self.workspace_rag_line_level_bm25 = line_level_bm25;
        self.workspace_rag_search_options = search_options;
    }

    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub fn workspace_context_query(&self) -> crate::rag::WorkspaceContextQuery {
        crate::rag::WorkspaceContextQuery {
            top_k: self.workspace_rag_top_k,
            min_similarity: self.workspace_rag_min_similarity,
            line_level_bm25: self.workspace_rag_line_level_bm25,
            search_options: self.workspace_rag_search_options.clone(),
        }
    }

    #[cfg(feature = "tools-rag")]
    pub fn set_workspace_rag_context_cache(
        &mut self,
        cache: Option<Arc<dyn crate::traits::cache::CacheBackend>>,
        ttl_secs: u64,
    ) {
        self.workspace_rag_context_cache = cache;
        self.workspace_rag_context_cache_ttl_secs = ttl_secs;
    }

    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub const fn workspace_rag_top_k(&self) -> usize {
        self.workspace_rag_top_k
    }

    #[cfg(feature = "tools-rag")]
    #[must_use]
    pub const fn workspace_rag_min_similarity(&self) -> f32 {
        self.workspace_rag_min_similarity
    }

    // =====================================================================
    // Observability & runtime state
    // =====================================================================

    /// Set metrics tracker for tool execution observability.
    pub fn set_metrics(&mut self, metrics: Arc<crate::observability::ToolMetrics>) {
        self.metrics = Some(metrics);
    }

    /// Register a lifecycle hook with a matcher.
    pub fn register_hook(&mut self, matcher: HookMatcher, hook: Arc<dyn LifecycleHook>) {
        self.hooks.register(matcher, hook);
    }

    /// Register a completion hook (called on `on_pre_completion` when LLM signals intent to stop).
    pub fn register_completion_hook(&mut self, hook: Arc<dyn LifecycleHook>) {
        self.hooks.register_completion_hook(hook);
    }

    /// Set user context for tool execution.
    pub fn set_user_context(&mut self, context: Option<UserContext>) {
        self.user_context = context;
    }

    // =====================================================================
    // Shared state access
    // =====================================================================

    #[must_use]
    pub const fn get_shared_state(&self) -> &SharedState {
        &self.shared_state
    }

    pub fn set_shared_state(&mut self, state: SharedState) {
        self.shared_state = state;
    }
}

/// Pure policy for per-iteration tool choice (unit-tested independently of Agent construction).
pub fn resolve_tool_choice_for_iteration(
    configured: Option<&crate::messages::ToolChoice>,
    iteration: usize,
    use_native: bool,
) -> Option<crate::messages::ToolChoice> {
    use crate::messages::ToolChoice;

    if !use_native {
        return None;
    }

    if iteration <= 1 {
        return configured.cloned().or(Some(ToolChoice::Auto));
    }

    match configured {
        Some(ToolChoice::Specific(_) | ToolChoice::Required) => Some(ToolChoice::Auto),
        other => other.cloned().or(Some(ToolChoice::Auto)),
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_tool_choice_for_iteration;
    use crate::messages::ToolChoice;
    use beef::lean::Cow;

    #[test]
    fn first_iteration_preserves_required_and_specific() {
        assert!(matches!(
            resolve_tool_choice_for_iteration(Some(&ToolChoice::Required), 1, true),
            Some(ToolChoice::Required)
        ));
        assert!(matches!(
            resolve_tool_choice_for_iteration(
                Some(&ToolChoice::Specific(Cow::borrowed("calculator"))),
                1,
                true
            ),
            Some(ToolChoice::Specific(name)) if name.as_ref() == "calculator"
        ));
    }

    #[test]
    fn later_iterations_relax_required_and_specific_to_auto() {
        assert!(matches!(
            resolve_tool_choice_for_iteration(Some(&ToolChoice::Required), 2, true),
            Some(ToolChoice::Auto)
        ));
        assert!(matches!(
            resolve_tool_choice_for_iteration(
                Some(&ToolChoice::Specific(Cow::borrowed("calculator"))),
                2,
                true
            ),
            Some(ToolChoice::Auto)
        ));
    }

    #[test]
    fn none_and_auto_are_stable_across_iterations() {
        assert!(matches!(
            resolve_tool_choice_for_iteration(Some(&ToolChoice::None), 2, true),
            Some(ToolChoice::None)
        ));
        assert!(matches!(
            resolve_tool_choice_for_iteration(Some(&ToolChoice::Auto), 2, true),
            Some(ToolChoice::Auto)
        ));
        assert!(matches!(
            resolve_tool_choice_for_iteration(None, 2, true),
            Some(ToolChoice::Auto)
        ));
    }

    #[test]
    fn non_native_tool_loop_disables_tool_choice() {
        assert!(resolve_tool_choice_for_iteration(Some(&ToolChoice::Required), 1, false).is_none());
    }
}
