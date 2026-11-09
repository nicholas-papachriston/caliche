use super::{
    AgentBuilder, AmendmentPersister, ApprovalResolver, Arc, EnvironmentOnboardingConfig,
    EnvironmentOnboardingMiddleware, HookMatcher, LifecycleHook, LoopDetectionConfig,
    LoopDetectionMiddleware, PermissionMode, PreCompletionChecklistConfig,
    PreCompletionChecklistMiddleware, ReasoningEffortSlot, ReasoningScheduleConfig,
    ReasoningScheduler, SteeringFollowUpProvider,
};

impl AgentBuilder {
    /// Register a lifecycle hook with a matcher
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    /// use caliche::agent::hooks::{HookMatcher, LifecycleHook, HookContext, HookResult};
    /// use std::sync::Arc;
    ///
    /// #[derive(Debug)]
    /// struct MyHook;
    ///
    /// #[async_trait::async_trait]
    /// impl LifecycleHook for MyHook {
    ///     async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
    ///         HookResult::allow()
    ///     }
    /// }
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::Bedrock)
    ///     .hook(HookMatcher::tool("Bash"), Arc::new(MyHook))
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn hook(mut self, matcher: HookMatcher, hook: Arc<dyn LifecycleHook>) -> Self {
        self.hooks.register(matcher, hook);
        self
    }

    /// Add pre-completion checklist middleware (injects verification prompt when no test command run recently)
    #[must_use]
    pub fn with_pre_completion_checklist(mut self, config: PreCompletionChecklistConfig) -> Self {
        let hook: Arc<dyn LifecycleHook> = Arc::new(PreCompletionChecklistMiddleware::new(config));
        self.hooks.register(HookMatcher::All, Arc::clone(&hook));
        self.hooks.register_completion_hook(hook);
        self
    }

    /// Add loop detection middleware (warns when same file edited too many times)
    #[must_use]
    pub fn with_loop_detection(mut self, config: LoopDetectionConfig) -> Self {
        let hook = Arc::new(LoopDetectionMiddleware::new(config));
        self.hooks.register(HookMatcher::All, hook);
        self
    }

    /// Add environment onboarding middleware (prepends environment discovery at session start)
    #[must_use]
    pub fn with_environment_onboarding(mut self, config: EnvironmentOnboardingConfig) -> Self {
        let hook = Arc::new(EnvironmentOnboardingMiddleware::new(config));
        self.hooks.register(HookMatcher::All, hook);
        self
    }

    /// Enable the three high-value harness middleware hooks with defaults.
    #[must_use]
    pub fn with_default_middleware(self) -> Self {
        self.with_pre_completion_checklist(PreCompletionChecklistConfig::default())
            .with_loop_detection(LoopDetectionConfig::default())
            .with_environment_onboarding(EnvironmentOnboardingConfig::default())
    }

    /// Schedule per-turn reasoning effort via a shared slot on the agent.
    #[must_use]
    pub fn with_reasoning_schedule(mut self, config: ReasoningScheduleConfig) -> Self {
        let slot = Arc::new(ReasoningEffortSlot::new());
        self.reasoning_effort_slot = Some(Arc::clone(&slot));
        let max_iterations = self.max_iterations.unwrap_or(50);
        let hook = Arc::new(ReasoningScheduler::new(config, slot, max_iterations));
        self.hooks.register(HookMatcher::All, hook);
        self
    }

    /// Set the permission mode for tool execution
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    /// use caliche::agent::hooks::permission::PermissionMode;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::Bedrock)
    ///     .permission_mode(PermissionMode::AcceptEdits)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn permission_mode(mut self, mode: PermissionMode) -> Self {
        self.permission_mode = mode;
        self
    }

    /// Set the approval resolver for `RequireApproval` from hooks.
    #[must_use]
    pub fn approval_resolver(mut self, resolver: Option<Arc<dyn ApprovalResolver>>) -> Self {
        self.approval_resolver = resolver;
        self
    }

    /// Set the amendment persister for `AllowAndPersist` outcomes.
    #[must_use]
    pub fn amendment_persister(mut self, persister: Option<Arc<dyn AmendmentPersister>>) -> Self {
        self.amendment_persister = persister;
        self
    }

    /// Set the steering/follow-up provider. When set, the chat loop checks for follow-up messages when the agent would stop; useful for CLI queues or WebSocket input.
    #[must_use]
    pub fn steering_followup_provider(
        mut self,
        provider: Option<Arc<dyn SteeringFollowUpProvider>>,
    ) -> Self {
        self.steering_followup_provider = provider;
        self
    }

    /// Enable the autonomous learning loop. The agent will evaluate recent
    /// tool-call trajectories at each checkpoint and can create or patch
    /// skill plugins via the configured plugin store. Requires
    /// [`Self::with_plugins_store`] to be set for durable persistence; when
    /// no store is configured the loop operates in-memory and
    /// `skill_manage` actions will fail with a clear error.
    #[cfg(feature = "learning-loop")]
    pub fn with_learning_loop(mut self, config: crate::agent::learning::CheckpointConfig) -> Self {
        self.learning_config = Some(config);
        self
    }

    /// Enable schema-based plugin authoring: store plugins at the given path and register the plugin tool plus any persisted tool plugins.
    #[cfg(feature = "plugins")]
    pub fn with_plugins_store(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.plugins_store_path = Some(path.into());
        self
    }

    /// Control how plugin load failures are handled during [`Self::build`].
    ///
    /// Defaults to [`crate::plugins::PluginLoadPolicy::Warn`] or the value of
    /// `CALICHE_PLUGIN_LOAD_POLICY` (`warn` | `fail`).
    #[cfg(feature = "plugins")]
    pub fn plugin_load_policy(mut self, policy: crate::plugins::PluginLoadPolicy) -> Self {
        self.plugin_load_policy = policy;
        self
    }
}
