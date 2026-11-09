//! MCP, plugin, learning-loop, and memory phases for `AgentBuilder::build`.

#[cfg(any(
    feature = "plugins",
    feature = "vector-memory",
    all(feature = "mcp", feature = "mcp-client")
))]
use super::Arc;
#[cfg(feature = "vector-memory")]
use super::Backend;
use super::{
    AgentBuilder, AgentService, BufferMemoryConfig, Context, EntityMemoryConfig, MemoryType,
    PersistentMemory, SummaryMemoryConfig, TokenCounterFactory,
};

#[cfg(all(feature = "plugins", feature = "tools-rag", feature = "tools-sql"))]
type PluginSqlExecutor = Option<Arc<dyn crate::agent::tools::SqlExecutor>>;
#[cfg(all(
    feature = "plugins",
    not(all(feature = "tools-rag", feature = "tools-sql"))
))]
type PluginSqlExecutor = Option<()>;

impl AgentBuilder {
    #[cfg(feature = "mcp")]
    pub(super) fn attach_in_process_mcp(&mut self, service: &mut AgentService) {
        for (_server_name, server) in std::mem::take(&mut self.mcp_servers) {
            let tool_registry = server.tool_registry();
            for (tool_name, _desc, _params) in tool_registry.list() {
                if let Some(tool) = tool_registry.get(&tool_name) {
                    service.agent_mut().add_tool(tool);
                }
            }
        }
    }

    #[cfg(not(feature = "mcp"))]
    pub(super) fn attach_in_process_mcp(&mut self, _service: &mut AgentService) {}

    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    pub(super) async fn attach_remote_mcp(
        &mut self,
        service: &mut AgentService,
    ) -> anyhow::Result<()> {
        self.discover_remote_mcp_servers();
        let mut resource_clients = Vec::new();
        for remote_cfg in std::mem::take(&mut self.remote_mcp_servers) {
            if let Some(client) = Self::connect_remote_mcp_server(remote_cfg).await? {
                let server_name = client.name().to_string();
                Self::register_remote_mcp_tools(service, &client, &server_name).await;
                resource_clients.push(client);
            }
        }
        service.set_mcp_clients(resource_clients);
        Ok(())
    }

    #[cfg(not(all(feature = "mcp", feature = "mcp-client")))]
    #[expect(
        clippy::unused_async,
        reason = "remote MCP attach is compiled out without mcp-client"
    )]
    pub(super) async fn attach_remote_mcp(
        &mut self,
        _service: &mut AgentService,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    fn discover_remote_mcp_servers(&mut self) {
        if !self.auto_discover_mcp {
            return;
        }
        let snapshot =
            crate::discovery::auto::discover_all(&crate::discovery::auto::DiscoveryOptions {
                rules: false,
                skills: false,
                commands: false,
                subagents: false,
                mcp: true,
            });
        if snapshot.mcp_servers.is_empty() {
            return;
        }
        let existing: std::collections::HashSet<String> = self
            .remote_mcp_servers
            .iter()
            .map(|cfg| cfg.name.clone())
            .collect();
        for server in snapshot.mcp_servers {
            if existing.contains(&server.name) {
                continue;
            }
            match server.to_mcp_client_config() {
                Ok(cfg) => self.remote_mcp_servers.push(cfg),
                Err(e) => {
                    tracing::warn!(
                        server = %server.name,
                        source = %server.source_path.display(),
                        error = %e,
                        "Skipping invalid discovered MCP server"
                    );
                }
            }
        }
    }

    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    async fn connect_remote_mcp_server(
        remote_cfg: crate::mcp::McpClientConfig,
    ) -> anyhow::Result<Option<Arc<crate::mcp::McpClient>>> {
        if !remote_cfg.skip_stdio_allowlist
            && let crate::mcp::McpClientTransportConfig::Stdio { command, .. } =
                &remote_cfg.transport
        {
            crate::mcp::validate_stdio_command_for_library(command)
                .with_context(|| format!("MCP stdio server '{}'", remote_cfg.name))?;
        }
        let client = Arc::new(crate::mcp::McpClient::new(remote_cfg));
        let server_name = client.name().to_string();
        match tokio::time::timeout(std::time::Duration::from_secs(10), client.connect()).await {
            Ok(Ok(())) => Ok(Some(client)),
            Ok(Err(err)) => {
                tracing::warn!(
                    server = %server_name,
                    error = %err,
                    "Failed to connect to remote MCP server"
                );
                Ok(None)
            }
            Err(_) => {
                tracing::warn!(
                    server = %server_name,
                    "Timed out connecting to remote MCP server"
                );
                Ok(None)
            }
        }
    }

    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    async fn register_remote_mcp_tools(
        service: &mut AgentService,
        client: &Arc<crate::mcp::McpClient>,
        server_name: &str,
    ) {
        match tokio::time::timeout(std::time::Duration::from_secs(10), client.list_tools()).await {
            Ok(Ok(tools)) => {
                for mcp_tool in tools {
                    let wrapped_name = format!("{}__{}", server_name, mcp_tool.name);
                    service.agent_mut().add_tool(Arc::new(
                        crate::agent::tools::RemoteMcpTool::new(
                            wrapped_name,
                            mcp_tool.description.to_string(),
                            mcp_tool.input_schema,
                            mcp_tool.name.to_string(),
                            Arc::clone(client),
                        ),
                    ));
                }
            }
            Ok(Err(e)) => {
                tracing::warn!(
                    server = %server_name,
                    error = %e,
                    "Failed to list tools from remote MCP server"
                );
            }
            Err(_) => {
                tracing::warn!(
                    server = %server_name,
                    "Timed out listing tools from remote MCP server"
                );
            }
        }
    }

    pub(super) async fn attach_sql_and_plugins(
        &mut self,
        service: &mut AgentService,
        #[cfg(feature = "learning-loop")] plugin_store_shared: &mut Option<
            Arc<crate::plugins::persistence::PluginStore>,
        >,
    ) -> anyhow::Result<()> {
        let sql_executor_for_plugins = if let Some(sql_config) = self.sql_tools.take() {
            Self::setup_sql_tools(service, sql_config).await?
        } else {
            None
        };
        #[cfg(not(all(feature = "plugins", feature = "tools-sql")))]
        let _ = &sql_executor_for_plugins;
        #[cfg(feature = "plugins")]
        if let Some(path) = self.plugins_store_path.take() {
            self.load_and_register_plugins(
                service,
                path,
                sql_executor_for_plugins,
                #[cfg(feature = "learning-loop")]
                plugin_store_shared,
            )
            .await?;
        }
        Ok(())
    }

    #[cfg(feature = "plugins")]
    async fn load_and_register_plugins(
        &mut self,
        service: &mut AgentService,
        path: std::path::PathBuf,
        sql_executor_for_plugins: PluginSqlExecutor,
        #[cfg(feature = "learning-loop")] plugin_store_shared: &mut Option<
            Arc<crate::plugins::persistence::PluginStore>,
        >,
    ) -> anyhow::Result<()> {
        use crate::plugins::adapters::skill_adapter::skill_from_plugin;
        use crate::plugins::tool::PluginTool;
        use crate::plugins::validation::SandboxPolicy;
        use crate::traits::credentials::EnvironmentCredentialProvider;

        // Plugins reach the SQL executor only with `tools-sql`.
        #[cfg(not(feature = "tools-sql"))]
        let _ = sql_executor_for_plugins;

        let store = crate::plugins::persistence::PluginStore::from_path(&path)
            .await
            .context("plugins store")?;
        let store_arc = Arc::new(store);
        #[cfg(feature = "learning-loop")]
        {
            *plugin_store_shared = Some(Arc::clone(&store_arc));
        }
        let registry = Arc::new(crate::plugins::registry::PluginRegistry::new(Arc::clone(
            &store_arc,
        )));
        if let Err(e) = registry.load_all().await {
            match self.plugin_load_policy {
                crate::plugins::PluginLoadPolicy::Warn => {
                    tracing::warn!(
                        error = %e,
                        "Failed to load plugins from configured plugin store; continuing without loaded plugins"
                    );
                }
                crate::plugins::PluginLoadPolicy::Fail => {
                    return Err(e).context("Failed to load plugins from configured plugin store");
                }
            }
        }
        let http_client = Arc::new(reqwest::Client::new());
        let plugin_http_allowlist = crate::plugins::http_policy::HttpPolicy::from_env()?
            .domain_allowlist()
            .map(<[String]>::to_vec);
        #[cfg(feature = "tools-sql")]
        let sql_execution_policy =
            Arc::new(crate::agent::tools::sql_execution::SqlExecutionPolicy::from_env(1000));
        let runtime = Arc::new(crate::plugins::runtime::StepRuntime::new(
            Arc::clone(&http_client),
            SandboxPolicy::ReadOnly,
            plugin_http_allowlist.clone(),
            #[cfg(feature = "tools-sql")]
            Arc::clone(&sql_execution_policy),
        ));
        let capabilities = Arc::new(crate::plugins::runtime::AgentCapabilities {
            llm_provider: service.agent().llm_provider().clone(),
            http_client,
            credential_provider: Arc::new(EnvironmentCredentialProvider::new()),
            #[cfg(feature = "tools-sql")]
            sql_executor: sql_executor_for_plugins.clone(),
            #[cfg(feature = "vector-memory")]
            vector_backend: None,
            embedding_provider: None,
            sandbox_policy: SandboxPolicy::ReadOnly,
            domain_allowlist: plugin_http_allowlist,
            #[cfg(feature = "sandbox")]
            sandbox: service.agent().sandbox.clone(),
        });
        let plugin_tool = PluginTool::new(
            Arc::clone(&registry),
            Arc::clone(&runtime),
            service.agent().tools.clone(),
            Arc::clone(&capabilities),
        );
        service.agent_mut().add_tool(Arc::new(plugin_tool));
        Self::register_plugin_tools(service, &registry, &runtime, &capabilities);
        let plugin_skills: Vec<_> = registry
            .list()
            .into_iter()
            .filter_map(|def| skill_from_plugin(def.as_ref()))
            .collect();
        service.set_plugin_skills(plugin_skills.clone());
        for skill in &plugin_skills {
            service.agent_mut().apply_skill(skill);
        }
        Self::register_plugin_hooks(service, &registry, &runtime, &capabilities);
        Self::register_plugin_embedding(service, &registry, &capabilities);
        if self.durable_stream_backend.is_none() {
            Self::register_plugin_stream_backend(
                &mut self.durable_stream_backend,
                &registry,
                &capabilities,
            );
        }
        Ok(())
    }

    #[cfg(feature = "plugins")]
    fn register_plugin_tools(
        service: &mut AgentService,
        registry: &Arc<crate::plugins::registry::PluginRegistry>,
        runtime: &Arc<crate::plugins::runtime::StepRuntime>,
        capabilities: &Arc<crate::plugins::runtime::AgentCapabilities>,
    ) {
        use crate::plugins::adapters::tool_adapter::SchemaToolAdapter;
        for def in registry.list() {
            if let crate::plugins::schema::PluginSpec::Tool(_) = &def.spec
                && let Some(adapter) = SchemaToolAdapter::new(
                    Arc::clone(&def),
                    Arc::clone(runtime),
                    Arc::clone(capabilities),
                )
            {
                service.agent_mut().add_tool(Arc::new(adapter));
            }
        }
    }

    #[cfg(feature = "plugins")]
    fn register_plugin_hooks(
        service: &mut AgentService,
        registry: &Arc<crate::plugins::registry::PluginRegistry>,
        runtime: &Arc<crate::plugins::runtime::StepRuntime>,
        capabilities: &Arc<crate::plugins::runtime::AgentCapabilities>,
    ) {
        use crate::plugins::adapters::hook_adapter::SchemaHookAdapter;
        for def in registry.list() {
            if let crate::plugins::schema::PluginSpec::Hook(spec) = &def.spec
                && let Some(adapter) = SchemaHookAdapter::new(
                    Arc::clone(&def),
                    Arc::clone(runtime),
                    Arc::clone(capabilities),
                )
            {
                let matcher = if spec.matcher.eq_ignore_ascii_case("all") {
                    crate::agent::hooks::HookMatcher::All
                } else {
                    crate::agent::hooks::HookMatcher::tool(spec.matcher.clone())
                };
                service
                    .agent_mut()
                    .register_hook(matcher, Arc::new(adapter));
            }
        }
    }

    #[cfg(feature = "plugins")]
    fn register_plugin_embedding(
        service: &mut AgentService,
        registry: &Arc<crate::plugins::registry::PluginRegistry>,
        capabilities: &Arc<crate::plugins::runtime::AgentCapabilities>,
    ) {
        use crate::plugins::adapters::embedding_adapter::SchemaEmbeddingAdapter;
        for def in registry.list() {
            if let crate::plugins::schema::PluginSpec::Embedding(_) = &def.spec
                && let Some(adapter) = SchemaEmbeddingAdapter::new(
                    Arc::clone(&def),
                    Arc::clone(&capabilities.http_client),
                )
            {
                service.set_embedding_provider(Some(Arc::new(adapter)));
                break;
            }
        }
    }

    #[cfg(feature = "plugins")]
    fn register_plugin_stream_backend(
        durable_stream_backend: &mut Option<Arc<dyn crate::streams::DurableStreamBackend>>,
        registry: &Arc<crate::plugins::registry::PluginRegistry>,
        capabilities: &Arc<crate::plugins::runtime::AgentCapabilities>,
    ) {
        use crate::plugins::adapters::stream_backend_adapter::SchemaStreamBackendAdapter;
        use crate::plugins::schema::ConnectionSpec;
        for def in registry.list() {
            if let crate::plugins::schema::PluginSpec::StreamBackend(ref spec) = def.spec
                && matches!(&spec.connection, ConnectionSpec::Inline { .. })
                && let Some(adapter) = SchemaStreamBackendAdapter::new(
                    Arc::clone(&def),
                    Arc::clone(&capabilities.http_client),
                )
            {
                *durable_stream_backend = Some(Arc::new(adapter));
                break;
            }
        }
    }

    #[cfg(feature = "learning-loop")]
    pub(super) fn apply_learning_loop(
        &mut self,
        service: &mut AgentService,
        plugin_store_shared: Option<Arc<crate::plugins::persistence::PluginStore>>,
    ) {
        let Some(learning_config) = self.learning_config.take() else {
            return;
        };
        let mut learning = crate::agent::learning::LearningLoop::new(learning_config);
        if let Some(ref store) = plugin_store_shared {
            learning = learning.with_plugin_store(Arc::clone(store));
        } else {
            tracing::warn!(
                "with_learning_loop is enabled without a plugin store; skill create/patch/delete will operate in-memory only"
            );
        }
        let learning_arc = Arc::new(learning);
        service.agent_mut().learning_loop = Some(Arc::clone(&learning_arc));
        service.agent_mut().add_tool(Arc::new(
            crate::agent::learning::skill_manage::SkillManageTool::new(Arc::clone(&learning_arc)),
        ));
    }

    #[cfg(not(feature = "learning-loop"))]
    pub(super) fn apply_learning_loop(&mut self, _service: &mut AgentService) {}

    pub(super) async fn apply_memory(
        &mut self,
        service: &mut AgentService,
        #[cfg(feature = "vector-memory")] prebuilt_vector_backend: Option<
            Arc<dyn Backend + Send + Sync>,
        >,
    ) -> anyhow::Result<()> {
        let persistence_cfg = self.memory_persistence.take();
        #[cfg(feature = "vector-memory")]
        let vector_backend_config = self.vector_backend_config.take();

        let memory_choice = self.memory_type.take().unwrap_or_else(|| {
            MemoryType::Buffer(BufferMemoryConfig {
                max_tokens: 8192,
                min_messages: 2,
                ..BufferMemoryConfig::default()
            })
        });

        let token_counter_factory =
            TokenCounterFactory::with_model(Some(service.model().to_string()));
        let token_counter = token_counter_factory.build();

        let agent_context_budget = service.agent().max_context_tokens;
        let default_buffer_tokens = BufferMemoryConfig::default().max_tokens;
        let default_summary_tokens = SummaryMemoryConfig::default().max_tokens;
        let default_entity_tokens = EntityMemoryConfig::default().max_tokens;

        let memory_strategy = Self::build_one_memory_strategy(
            memory_choice,
            service,
            token_counter,
            agent_context_budget,
            default_buffer_tokens,
            default_summary_tokens,
            default_entity_tokens,
            #[cfg(feature = "vector-memory")]
            prebuilt_vector_backend,
            #[cfg(feature = "vector-memory")]
            vector_backend_config,
            persistence_cfg.as_ref(),
        )
        .await?;

        if let Some(config) = persistence_cfg {
            let mut persisted =
                PersistentMemory::new(memory_strategy, config.persistence, config.conversation_id)
                    .with_autosave(config.autosave);

            persisted
                .hydrate()
                .await
                .context("Failed to load persisted memory state")?;

            service.set_memory_strategy(Some(Box::new(persisted)));
        } else {
            service.set_memory_strategy(Some(memory_strategy));
        }
        Ok(())
    }
}
