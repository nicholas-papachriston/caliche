//! Split McCabe CCN for `AgentBuilder::build`.

#[cfg(feature = "vector-memory")]
use super::Backend;
#[cfg(any(
    feature = "agentfs",
    feature = "vector-memory",
    feature = "durable-streams-sqlite",
    feature = "durable-streams-postgres",
    feature = "durable-streams-s3"
))]
use super::Context;
#[cfg(feature = "tools-rag")]
use super::InMemoryCacheBackend;
#[cfg(feature = "vector-memory")]
use super::MemoryType;
use super::{
    AgentBuilder, AgentService, Arc, CredentialProvider, EnvironmentCredentialProvider,
    ProviderType, ResearchTool, ToolExecutionDriver,
};

impl AgentBuilder {
    #[cfg_attr(
        not(any(
            feature = "durable-streams-sqlite",
            feature = "durable-streams-postgres",
            feature = "durable-streams-s3"
        )),
        expect(
            clippy::unused_async,
            reason = "durable stream backends await only when those features are on"
        )
    )]
    pub(super) async fn init_durable_stream_backends(&mut self) -> anyhow::Result<()> {
        #[cfg(feature = "session-tools")]
        if let Some(c) = self.session_tools_config.take() {
            self.session_spawner = c.spawner;
        }

        #[cfg(feature = "durable-streams-sqlite")]
        if let Some(path) = self.durable_stream_sqlite_path.take() {
            let backend = crate::streams::SqliteDurableStreamBackend::new(&path)
                .await
                .context("Failed to create SQLite durable stream backend")?;
            self.durable_stream_backend = Some(Arc::new(backend));
        }

        #[cfg(feature = "durable-streams-postgres")]
        if let Some(url) = self.durable_stream_postgres_url.take() {
            let backend = crate::streams::PostgresDurableStreamBackend::new(&url)
                .await
                .context("Failed to create Postgres durable stream backend")?;
            self.durable_stream_backend = Some(Arc::new(backend));
        }

        #[cfg(feature = "durable-streams-s3")]
        if let Some((bucket, prefix, region)) = self.durable_stream_s3_config.take() {
            let config = crate::streams::S3DurableStreamConfig {
                bucket,
                prefix,
                region,
            };
            let backend = crate::streams::S3DurableStreamBackend::new(config)
                .await
                .context("Failed to create S3 durable stream backend")?;
            self.durable_stream_backend = Some(Arc::new(backend));
        }
        Ok(())
    }

    pub(super) fn apply_provider_model_from_env(&mut self) {
        if self.provider.is_none()
            && let Ok(provider_str) = std::env::var("AGENT_PROVIDER")
        {
            if let Ok(provider_type) = provider_str.parse::<ProviderType>() {
                self.provider = Some(provider_type);
            } else {
                tracing::warn!(
                    "Invalid AGENT_PROVIDER value: {}. Supported: bedrock, openai, anthropic, openrouter, perplexity, lmstudio, google",
                    provider_str
                );
            }
        }

        if self.model.is_none()
            && let Ok(model) = std::env::var("AGENT_MODEL")
        {
            self.model = Some(model);
        }
    }

    #[cfg(feature = "vector-memory")]
    fn rag_tools_requested(&self) -> bool {
        self.use_all_default_tools
            || self
                .tools
                .iter()
                .any(|t| (*t).eq_ignore_ascii_case("semantic_search"))
            || self
                .tools
                .iter()
                .any(|t| (*t).eq_ignore_ascii_case("sql_generation"))
            || self
                .tools
                .iter()
                .any(|t| (*t).eq_ignore_ascii_case("sql_execution"))
    }

    #[cfg(feature = "vector-memory")]
    pub(super) async fn prebuild_vector_backend(
        &self,
    ) -> anyhow::Result<Option<Arc<dyn Backend + Send + Sync>>> {
        let rag_requested = self.rag_tools_requested();
        let vector_memory_requested =
            matches!(self.memory_type.as_ref(), Some(MemoryType::Vector(_)));
        if !(rag_requested || vector_memory_requested) {
            return Ok(None);
        }
        let Some(ref backend_config) = self.vector_backend_config else {
            return Ok(None);
        };
        backend_config
            .build_backend()
            .await
            .context("Failed to build configured vector backend")
            .map(Some)
    }

    pub(super) async fn construct_service(&mut self) -> anyhow::Result<AgentService> {
        #[cfg(feature = "local-inference-base")]
        let local_inference_config = self.local_inference_config_for_builder();

        let tool_names = if self.use_all_default_tools {
            None
        } else {
            Some(
                std::mem::take(&mut self.tools)
                    .into_iter()
                    .map(|c| c.as_ref().to_string())
                    .collect(),
            )
        };
        let built_in_tools = std::mem::take(&mut self.built_in_tools);

        if let Some(runner) = self.research_runner.take() {
            self.custom_tools.push(Arc::new(ResearchTool::new(runner)));
        }

        let provider_override = self.provider_instance.take();
        let mut custom_tools = std::mem::take(&mut self.custom_tools);

        if let Some(provider_instance) = provider_override {
            self.construct_from_provider_instance(
                provider_instance,
                tool_names,
                built_in_tools,
                &mut custom_tools,
            )
            .await
        } else if !custom_tools.is_empty()
            && !self.use_all_default_tools
            && !self.defer_missing_credentials
        {
            AgentService::with_provider_and_tools(
                self.provider.take(),
                self.model.take(),
                custom_tools,
                tool_names,
                self.llm_cache.take(),
            )
            .await
        } else {
            self.construct_from_config(
                tool_names,
                built_in_tools,
                custom_tools,
                #[cfg(feature = "local-inference-base")]
                local_inference_config,
            )
            .await
        }
    }

    async fn construct_from_provider_instance(
        &mut self,
        provider_instance: Arc<dyn crate::llm::Provider>,
        tool_names: Option<Vec<String>>,
        built_in_tools: Vec<crate::agent::tools::BuiltInTool>,
        custom_tools: &mut Vec<Arc<dyn crate::agent::tools::Tool>>,
    ) -> anyhow::Result<AgentService> {
        let provider_type = self.provider.unwrap_or(ProviderType::External);
        let model = self.model.take().unwrap_or_else(|| "external".to_string());

        let mut tr_builder = crate::agent::service::ToolRegistryBuilder::new(provider_type)
            .with_embedding_provider(self.embedding_provider_instance.as_ref().map(Arc::clone))
            .with_image_generation(
                self.image_generation_provider_instance
                    .as_ref()
                    .map(Arc::clone),
                self.artifact_store.as_ref().map(Arc::clone),
            )
            .with_built_in_tools(built_in_tools)
            .with_mutation_style(self.mutation_style);

        if self.use_all_default_tools {
            tr_builder = tr_builder.use_all_default_tools();
        } else {
            tr_builder = tr_builder.with_tool_names(tool_names);
        }

        #[cfg(feature = "tools-rag")]
        {
            let _ = self.vector_metrics_enabled;
        }

        #[cfg(feature = "tools-graph")]
        {
            tr_builder = tr_builder
                .with_graph_backend(self.graph_backend.as_ref().map(Arc::clone))
                .with_frontier_prioritizer(self.frontier_prioritizer.clone());
        }

        #[cfg(feature = "session-tools")]
        {
            tr_builder = tr_builder.session_spawner(self.session_spawner.take());
        }

        #[cfg(feature = "lsp")]
        {
            let lsp_manager = self.lsp_manager.as_ref().map(Arc::clone).or_else(|| {
                self.lsp_config
                    .clone()
                    .map(crate::lsp::LspManager::new)
                    .map(Arc::new)
            });
            tr_builder = tr_builder.with_lsp_manager(lsp_manager);
        }

        let registry_arc = tr_builder.build().await?;
        let registry = Arc::try_unwrap(registry_arc).map_err(|_| {
            anyhow::anyhow!("Failed to unwrap ToolRegistry Arc for custom tool registration")
        })?;
        for tool in custom_tools.drain(..) {
            registry.register(tool);
        }
        let registry_arc = Arc::new(registry);

        AgentService::from_components(
            provider_instance,
            registry_arc,
            provider_type,
            model,
            self.embedding_provider_instance.as_ref().map(Arc::clone),
        )
    }

    async fn construct_from_config(
        &mut self,
        tool_names: Option<Vec<String>>,
        built_in_tools: Vec<crate::agent::tools::BuiltInTool>,
        custom_tools: Vec<Arc<dyn crate::agent::tools::Tool>>,
        #[cfg(feature = "local-inference-base")] local_inference_config: Option<
            crate::llm::LocalInferenceConfig,
        >,
    ) -> anyhow::Result<AgentService> {
        let cred_provider = self.credential_provider.take().unwrap_or_else(|| {
            Arc::new(EnvironmentCredentialProvider::new()) as Arc<dyn CredentialProvider>
        });

        let mut tool_selection = if self.use_all_default_tools {
            crate::agent::service::ToolSelection::all_defaults()
        } else {
            crate::agent::service::ToolSelection::explicit_names(tool_names.unwrap_or_default())
        };
        tool_selection.builtins = built_in_tools;

        let mut config = crate::agent::service::AgentServiceConfig::new()
            .with_provider(self.provider.take(), self.model.take())
            .with_tool_selection(tool_selection)
            .with_mutation_style(self.mutation_style)
            .with_credential_provider(cred_provider)
            .with_defer_missing_credentials(self.defer_missing_credentials)
            .with_base_url(self.base_url.take())
            .with_context_window(self.context_window)
            .with_image_generation_provider(
                self.image_generation_provider_instance
                    .as_ref()
                    .map(Arc::clone),
            )
            .with_artifact_store(self.artifact_store.as_ref().map(Arc::clone))
            .with_llm_cache(self.llm_cache.take());
        #[cfg(feature = "local-inference-base")]
        {
            config.local_inference_config = local_inference_config;
        }
        #[cfg(feature = "tools-rag")]
        {
            let _ = self.vector_metrics_enabled;
            config.shared_rag_tool = self.shared_rag_tool.take();
        }
        #[cfg(feature = "tools-graph")]
        {
            config.graph_backend = self.graph_backend.as_ref().map(Arc::clone);
            config.frontier_prioritizer = self.frontier_prioritizer.clone();
        }
        #[cfg(feature = "reqwest")]
        {
            config.shared_http_client = self.shared_http_client.take();
        }
        #[cfg(feature = "session-tools")]
        {
            config.session_spawner = self.session_spawner.take();
        }
        #[cfg(feature = "lsp")]
        {
            config.lsp_config = self.lsp_config.take();
            config.lsp_manager = self.lsp_manager.take();
        }
        let mut svc = AgentService::from_config(config).await?;
        for tool in custom_tools {
            svc.add_tool(tool);
        }
        Ok(svc)
    }

    pub(super) fn apply_prompt_and_limits(
        &mut self,
        service: &mut AgentService,
    ) -> anyhow::Result<()> {
        if let Some(prompt) = self.system_prompt.take() {
            service.agent_mut().set_system_prompt(prompt);
        }

        if let Some(inc) = self.include_context_in_system {
            service.agent_mut().set_include_context_in_system(inc);
        }

        if let Some(context_window) = self.context_window {
            service.set_context_window(context_window)?;
        }

        if let Some(max_iterations) = self.max_iterations {
            service.agent_mut().set_max_iterations(max_iterations);
        }

        if let Some(max_tokens) = self.max_tokens {
            service.agent_mut().set_max_tokens(Some(max_tokens));
        }
        if let Some(temperature) = self.temperature {
            service.agent_mut().set_temperature(Some(temperature));
        }

        if let Some(tool_choice) = self.tool_choice.take() {
            service.agent_mut().set_tool_choice(Some(tool_choice));
        }
        Ok(())
    }

    pub(super) fn apply_hooks_and_permissions(&mut self, service: &mut AgentService) {
        if self.enable_tool_inspection {
            let manager = std::sync::Arc::new(crate::agent::hooks::default_inspection_manager(
                self.permission_mode,
                self.inspection_workspace_root.clone(),
            ));
            self.hooks.register(
                crate::agent::hooks::HookMatcher::All,
                std::sync::Arc::new(crate::agent::hooks::InspectionHookAdapter::new(manager)),
            );
        }

        if !self.hooks.is_empty() {
            service
                .agent_mut()
                .set_hooks(std::mem::take(&mut self.hooks));
        }

        if let Some(slot) = self.reasoning_effort_slot.take() {
            service.agent_mut().set_reasoning_effort_slot(Some(slot));
        }

        service
            .agent_mut()
            .set_approval_resolver(self.approval_resolver.take());
        service
            .agent_mut()
            .set_amendment_persister(self.amendment_persister.take());
        service
            .agent_mut()
            .set_steering_followup_provider(self.steering_followup_provider.take());

        service
            .agent_mut()
            .set_permission_mode(self.permission_mode);

        service.set_retry_attempts(self.agent_retry_attempts);
    }

    #[cfg_attr(
        not(feature = "agentfs"),
        expect(
            clippy::unused_async,
            reason = "AgentFS open awaits only when the agentfs feature is on"
        )
    )]
    pub(super) async fn apply_session_store(
        &mut self,
        service: &mut AgentService,
    ) -> anyhow::Result<()> {
        #[cfg(feature = "agentfs")]
        {
            let use_agentfs = self.agentfs_id.is_some() || self.agentfs_path.is_some();
            if use_agentfs {
                let options = match (&self.agentfs_path, &self.agentfs_id) {
                    (Some(path), _) => {
                        agentfs_sdk::AgentFSOptions::with_path(path.to_string_lossy().into_owned())
                    }
                    (_, Some(id)) => agentfs_sdk::AgentFSOptions::with_id(id),
                    _ => {
                        return Err(anyhow::anyhow!(
                            "AgentFS requires agentfs_id or agentfs_path"
                        ));
                    }
                };
                let agentfs = agentfs_sdk::AgentFS::open(options)
                    .await
                    .context("Failed to open AgentFS")?;
                service.enable_agentfs(Arc::new(agentfs))?;
            } else if let Some(path) = self.session_autosave_path.take() {
                service.enable_session_autosave(path)?;
            }
        }
        #[cfg(not(feature = "agentfs"))]
        if let Some(path) = self.session_autosave_path.take() {
            service.enable_session_autosave(path)?;
        }
        Ok(())
    }

    pub(super) fn apply_driver_and_skills(&mut self, service: &mut AgentService) {
        if let Some(allowed_tools) = self.allowed_tools.take() {
            service.agent_mut().set_allowed_tools(Some(allowed_tools));
        }

        service.response_format = self.response_format.take();

        let driver = self.tool_execution_driver.take().or_else(|| {
            std::env::var("CALICHE_TOOL_EXECUTION_DRIVER")
                .ok()
                .and_then(|v| ToolExecutionDriver::parse(&v))
        });
        if let Some(driver) = driver {
            service.agent_mut().set_tool_execution_driver(driver);
        }

        for skill in &self.skills {
            service.agent_mut().apply_skill(skill);
        }
        if self.auto_discover_skills {
            let discovered_skills = crate::agent::skills::discovered_skills();
            let discovered_for_index: Vec<_> = discovered_skills
                .iter()
                .filter(|skill| !self.skills.iter().any(|s| s.name == skill.name))
                .cloned()
                .collect();
            service
                .agent_mut()
                .register_discovered_skills_index(&discovered_for_index);
        }
    }

    #[cfg(feature = "tools-rag")]
    pub(super) fn apply_workspace_rag(
        &mut self,
        service: &mut AgentService,
        used_workspace_embedding: bool,
    ) {
        if used_workspace_embedding {
            service.set_workspace_rag_for_context(
                self.workspace_indexing_ready.take(),
                self.workspace_config.take(),
                self.workspace_manifest.take(),
            );
        }
        if let Some(top_k) = self.workspace_rag_top_k {
            service.agent_mut().set_workspace_rag_top_k(top_k);
        }
        if let Some(min_sim) = self.workspace_rag_min_similarity {
            service
                .agent_mut()
                .set_workspace_rag_min_similarity(min_sim);
        }
        if let Some(cache) = self.workspace_rag_context_cache.take() {
            service.agent_mut().set_workspace_rag_context_cache(
                Some(cache),
                self.workspace_rag_context_cache_ttl_secs,
            );
        } else if used_workspace_embedding {
            let default_cache = Arc::new(InMemoryCacheBackend::new(
                500,
                std::time::Duration::from_secs(60),
            ));
            service
                .agent_mut()
                .set_workspace_rag_context_cache(Some(default_cache), 60);
        }
    }

    pub(super) fn apply_output_and_compaction(
        &mut self,
        service: &mut AgentService,
    ) -> anyhow::Result<()> {
        if self.use_embedding_tool_ordering
            && let Some(emb) = service.embedding_provider()
        {
            service
                .agent_mut()
                .set_tool_ordering_embedding_provider(Some(emb));
        }
        if let Some(max_chars) = self.max_tool_output_in_history {
            service
                .agent_mut()
                .set_max_tool_output_in_history(max_chars);
        }
        if let Some(annotate) = self.annotate_tool_output_truncation {
            service
                .agent_mut()
                .set_annotate_tool_output_truncation(annotate);
        }
        #[cfg(feature = "compaction")]
        let jev_config_from_builder = match (self.jev_asker.take(), self.jev_options.take()) {
            (Some(asker), opts) => Some(crate::agent::core::compaction_jev::JevCompactionConfig {
                asker,
                options: opts.unwrap_or_default(),
            }),
            (None, _) => crate::agent::core::compaction_jev::JevCompactionConfig::from_env(),
        };

        #[cfg(feature = "compaction")]
        if let Some(config) = self.compaction_config.take() {
            let guard =
                crate::agent::core::context_window_guard::ContextWindowGuard::new(config.strategy)
                    .trigger_threshold(config.trigger_threshold)
                    .target_threshold(config.target_threshold)
                    .max_compaction_attempts(config.max_attempts);
            service.agent_mut().set_compaction_guard(Arc::new(guard));
        } else if let Some(strategy) = self.compaction_strategy.take() {
            let mut guard =
                crate::agent::core::context_window_guard::ContextWindowGuard::new(strategy);
            if let Some(t) = self.compaction_trigger_threshold.take() {
                guard = guard.trigger_threshold(t);
            }
            if let Some(t) = self.compaction_target_threshold.take() {
                guard = guard.target_threshold(t);
            }
            if let Some(n) = self.compaction_max_attempts.take() {
                guard = guard.max_compaction_attempts(n);
            }
            service.agent_mut().set_compaction_guard(Arc::new(guard));
        } else if let Some(jev_cfg) = jev_config_from_builder {
            let summarizer = crate::agent::core::compaction::LlmSummarizationCompaction::builder()
                .provider(Arc::clone(service.agent().llm_provider()))
                .jev_asker(jev_cfg.asker)
                .jev_options(jev_cfg.options)
                .build();
            if let Ok(strat) = summarizer {
                let guard = crate::agent::core::context_window_guard::ContextWindowGuard::new(
                    Arc::new(strat),
                )
                .trigger_threshold(0.90)
                .target_threshold(0.70)
                .max_compaction_attempts(2);
                service.agent_mut().set_compaction_guard(Arc::new(guard));
            }
        }

        self.attach_operation_manager(service)?;
        Ok(())
    }

    pub(super) fn attach_operation_manager(
        &mut self,
        service: &mut AgentService,
    ) -> anyhow::Result<()> {
        if let Some(manager) = self.operation_manager.take() {
            install_operation_manager(service, manager)?;
            return Ok(());
        }
        let env_enabled = std::env::var("CALICHE_OPERATION_DISPATCH")
            .ok()
            .is_some_and(|raw| {
                matches!(
                    raw.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            });
        if self.enable_operation_dispatch || env_enabled {
            let manager = crate::agent::operations::LocalOperationManager::new();
            install_operation_manager(service, Arc::new(manager))?;
        }
        Ok(())
    }
}

pub(crate) fn install_operation_manager(
    service: &mut AgentService,
    manager: Arc<dyn crate::agent::operations::OperationManager>,
) -> anyhow::Result<()> {
    #[cfg(feature = "sandbox")]
    {
        use crate::agent::operations::OperationExecutionMode;
        if service.agent().sandbox.is_some()
            && manager.execution_mode() == OperationExecutionMode::Remote
        {
            if !manager.isolates_in_sandbox() {
                anyhow::bail!(
                    "sandbox is configured; RemoteOperationManager must use \
                     SandboxOperationTransport (LocalOperationManager keeps \
                     ToolExecutionEngine isolation)"
                );
            }
            let configured = service
                .agent()
                .sandbox
                .as_ref()
                .map(|handle| handle.as_ref() as &dyn caliche_sandbox::SandboxHandleOps);
            if !manager.binds_configured_sandbox(configured) {
                anyhow::bail!(
                    "sandbox is configured; isolating RemoteOperationTransport must bind \
                     the agent's configured sandbox handle"
                );
            }
        }
    }
    service.agent_mut().set_operation_manager(manager);
    Ok(())
}
