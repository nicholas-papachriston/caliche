#[cfg(all(feature = "tools-rag", feature = "tools-sql"))]
use super::Backend;
use super::{AgentBuilder, AgentService, Arc, Cow, SqlToolsConfig, Tool, anyhow};
#[cfg(feature = "tools-rag")]
use super::{BufferMemoryConfig, MemoryType};
#[cfg(all(feature = "tools-rag", feature = "tools-sql"))]
use super::{
    DatabaseIntrospector, PostgresSchemaIntrospector, PostgresSqlExecutor, SchemaEmbeddingPipeline,
    SqlExecutionTool, SqlExecutor, SqlGenerationTool,
};
#[cfg(feature = "tools-rag")]
use super::{
    PersistenceConfig, RagHashlineOptions, RagTool, TOOL_FILESYSTEM, WorkspaceEmbeddingMode,
    WorkspaceIndexer, WorkspaceIndexerConfig, WorkspaceIndexingReady,
    discover_best_embedding_provider,
};
#[cfg(all(feature = "tools-rag", feature = "tools-sql"))]
use anyhow::Context;

impl AgentBuilder {
    #[must_use]
    pub fn enable_tool(mut self, tool_name: &str) -> Self {
        self.tools.push(Cow::owned(tool_name.to_string()));
        self
    }

    #[must_use]
    pub fn enable_builtin_tool(mut self, tool: crate::agent::tools::BuiltInTool) -> Self {
        self.built_in_tools.push(tool);
        self
    }

    /// Enable multiple built-in tools at once using a slice of tool names.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::AgentBuilder;
    ///
    /// let builder = AgentBuilder::new()
    ///     .enable_tools(&["calculator", "fs", "http"]);
    /// ```
    #[must_use]
    pub fn enable_tools(mut self, tool_names: &[&str]) -> Self {
        self.tools
            .extend(tool_names.iter().map(|s| Cow::owned((*s).to_string())));
        self
    }

    /// Apply settings from a named agent profile (Markdown + YAML frontmatter).
    ///
    /// # Errors
    ///
    /// Returns an error when `permission.edit` is not a known permission mode.
    /// Missing, empty, and `default` select the default mode.
    #[cfg(feature = "profiles")]
    pub fn from_profile(profile: &crate::agent::profile::AgentProfile) -> Result<Self, String> {
        let mut b = Self::new();
        if let Some(ref m) = profile.model {
            b = b.model(m.clone());
        }
        if let Some(t) = profile.temperature {
            b = b.temperature(t);
        }
        if let Some(s) = profile.steps {
            b = b.max_iterations(s as usize);
        }
        if let Some(body) = profile.system_prompt() {
            b = b.system_prompt(body.to_string());
        }
        if let Some(names) = profile.enabled_tools() {
            b.use_all_default_tools = false;
            b.tools = names.into_iter().map(Cow::owned).collect();
        }
        if let Some(permission) = profile.permission.as_ref() {
            if let Some(edit) = permission.edit.as_deref() {
                let config = crate::agent::hooks::parse_permission_mode_str(Some(edit))?;
                b = b.permission_mode(crate::agent::hooks::permission_mode_config_to_runtime(
                    config,
                ));
            }
            if let Some(rules) = permission.bash.as_ref()
                && !rules.is_empty()
            {
                let policy =
                    std::sync::Arc::new(crate::agent::hooks::ExecPolicy::new(rules.clone()));
                let hook: std::sync::Arc<dyn crate::agent::hooks::LifecycleHook> =
                    std::sync::Arc::new(crate::agent::hooks::ExecPolicyHook::new(policy));
                b.hooks
                    .register(crate::agent::hooks::HookMatcher::tool("bash"), hook);
            }
        }
        Ok(b)
    }

    /// Enable multiple built-in tools using the `BuiltInTool` enum for compile-time safety.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::AgentBuilder;
    ///
    /// let builder = AgentBuilder::new()
    ///     .enable_tools(&["calculator", "fs"]);
    /// ```
    #[must_use]
    pub fn enable_builtin_tools(mut self, tools: &[crate::agent::tools::BuiltInTool]) -> Self {
        self.built_in_tools.extend_from_slice(tools);
        self
    }

    /// Set the queryable graph backend for `graph_query` and `graph_explorer` tools.
    #[cfg(feature = "tools-graph")]
    pub fn with_graph_backend(
        mut self,
        backend: Arc<dyn dripstone::graph::QueryableGraphBackend>,
    ) -> Self {
        self.graph_backend = Some(backend);
        self
    }

    /// Set a custom frontier prioritizer for the graph explorer tool.
    #[cfg(feature = "tools-graph")]
    #[must_use]
    pub fn with_frontier_prioritizer(
        mut self,
        prioritizer: crate::agent::tools::graph::FrontierPrioritizer,
    ) -> Self {
        self.frontier_prioritizer = Some(prioritizer);
        self
    }

    /// Enable `graph_query` and `graph_explorer` tools. Requires a graph backend via `with_graph_backend()`.
    #[cfg(feature = "tools-graph")]
    #[must_use]
    pub fn enable_graph_tools(mut self) -> Self {
        self.tools.push(Cow::borrowed("graph_query"));
        self.tools.push(Cow::borrowed("graph_explorer"));
        self
    }

    #[cfg(feature = "lsp")]
    #[must_use]
    pub fn with_lsp_config(mut self, config: crate::lsp::LspConfig) -> Self {
        self.lsp_config = Some(config);
        self
    }

    #[cfg(feature = "lsp")]
    #[must_use]
    pub fn with_lsp_manager(mut self, manager: Arc<crate::lsp::LspManager>) -> Self {
        self.lsp_manager = Some(manager);
        self
    }

    #[cfg(feature = "lsp")]
    #[must_use]
    pub fn enable_lsp_tools(mut self) -> Self {
        self.tools.push(Cow::borrowed("lsp_diagnostics"));
        self.tools.push(Cow::borrowed("lsp_hover"));
        self.tools.push(Cow::borrowed("lsp_symbols"));
        self
    }

    /// Add a custom tool without requiring manual `Arc::new` wrapping.
    ///
    /// The tool will be automatically wrapped in an `Arc` internally.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, Tool, ToolResult};
    /// use async_trait::async_trait;
    ///
    /// #[derive(Debug)]
    /// struct MyTool;
    ///
    /// impl caliche::agent::tools::ToolMeta for MyTool {
    ///     fn name(&self) -> &str { "my_tool" }
    ///     fn description(&self) -> &str { "My custom tool" }
    ///     fn parameters(&self) -> serde_json::Value { serde_json::json!({}) }
    /// }
    ///
    /// #[async_trait]
    /// impl Tool for MyTool {
    ///     async fn execute(&self, _args: serde_json::Value) -> ToolResult {
    ///         ToolResult::success("Tool executed".to_string())
    ///     }
    /// }
    ///
    /// let builder = AgentBuilder::new()
    ///     .with_tool(MyTool);
    /// ```
    pub fn with_tool(mut self, tool: impl Tool + 'static) -> Self {
        self.custom_tools.push(Arc::new(tool));
        self
    }

    /// Add multiple custom tools at once without requiring manual `Arc::new` wrapping.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, Tool, ToolResult};
    /// use std::sync::Arc;
    /// use async_trait::async_trait;
    ///
    /// #[derive(Debug)]
    /// struct Tool1;
    /// #[derive(Debug)]
    /// struct Tool2;
    ///
    /// # impl caliche::agent::tools::ToolMeta for Tool1 {
    /// #     fn name(&self) -> &str { "tool1" }
    /// #     fn description(&self) -> &str { "" }
    /// #     fn parameters(&self) -> serde_json::Value { serde_json::json!({}) }
    /// # }
    /// # #[async_trait]
    /// # impl Tool for Tool1 {
    /// #     async fn execute(&self, _args: serde_json::Value) -> ToolResult {
    /// #         ToolResult::success("".to_string())
    /// #     }
    /// # }
    /// # impl caliche::agent::tools::ToolMeta for Tool2 {
    /// #     fn name(&self) -> &str { "tool2" }
    /// #     fn description(&self) -> &str { "" }
    /// #     fn parameters(&self) -> serde_json::Value { serde_json::json!({}) }
    /// # }
    /// # #[async_trait]
    /// # impl Tool for Tool2 {
    /// #     async fn execute(&self, _args: serde_json::Value) -> ToolResult {
    /// #         ToolResult::success("".to_string())
    /// #     }
    /// # }
    ///
    /// let builder = AgentBuilder::new()
    ///     .with_tool(Tool1)
    ///     .with_tool(Tool2);
    /// ```
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<impl Tool + 'static>) -> Self {
        self.custom_tools
            .extend(tools.into_iter().map(|t| Arc::new(t) as Arc<dyn Tool>));
        self
    }

    /// Register media understanding providers (audio/image/video preprocessing).
    #[cfg(feature = "tools-media")]
    pub fn with_media_providers(
        mut self,
        providers: Vec<
            Arc<dyn crate::agent::tools::media_understanding::MediaUnderstandingProvider>,
        >,
    ) -> Self {
        self.media_providers = providers;
        self
    }

    /// Set the document processor used for attachment/document ingestion paths.
    #[cfg(feature = "document-loaders")]
    #[must_use]
    pub const fn with_document_processor(
        mut self,
        processor: crate::data::loaders::DocumentProcessor,
    ) -> Self {
        self.document_processor = Some(processor);
        self
    }

    /// Enable the canonical default tools (`fs`, `bash`, and `web_fetch`).
    ///
    /// `web_fetch` reads public URLs. It is not a browser. For the broader
    /// extended set (http / workspace / `execute_python`, which also includes
    /// `web_fetch`), use [`Self::enable_extended_default_tools`].
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::AgentBuilder;
    ///
    /// let builder = AgentBuilder::new()
    ///     .enable_all_default_tools();
    /// ```
    #[must_use]
    pub const fn enable_all_default_tools(mut self) -> Self {
        self.use_all_default_tools = true;
        self
    }

    /// Enable the broader extended default tool set.
    ///
    /// Includes `fs`, `bash`, `web_fetch`, `http`, `workspace`, and `execute_python`.
    /// Semantic search, Perplexity, `web_search`, image generation, and graph tools remain opt-in.
    #[must_use]
    pub fn enable_extended_default_tools(mut self) -> Self {
        self.use_all_default_tools = false;
        for name in crate::agent::tools::names::EXTENDED_DEFAULT_TOOLS {
            if !self
                .tools
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(name))
            {
                self.tools.push(beef::lean::Cow::borrowed(*name));
            }
        }
        self
    }

    /// Select file mutation style. `ApplyPatch` is mutually exclusive with `fs` write/edit.
    #[must_use]
    pub const fn mutation_style(mut self, style: crate::agent::tools::MutationStyle) -> Self {
        self.mutation_style = style;
        self
    }

    /// Enable the Goose-style tool inspection chain (path security + permission mode).
    #[must_use]
    pub fn enable_tool_inspection(mut self, workspace_root: Option<std::path::PathBuf>) -> Self {
        self.enable_tool_inspection = true;
        self.inspection_workspace_root = workspace_root;
        self
    }

    /// Returns true if the agent will have fs write/edit capability (default tools or explicit).
    #[cfg(feature = "tools-rag")]
    fn agent_has_edit_tool(&self) -> bool {
        self.use_all_default_tools || self.tools.iter().any(|t| t.eq_ignore_ascii_case("fs"))
    }

    /// Set up workspace embedding when enabled. Returns true if workspace embedding was configured.
    #[cfg(feature = "tools-rag")]
    pub(crate) async fn setup_workspace_embedding(&mut self) -> bool {
        let Some(mode) = self.workspace_embedding.as_ref().cloned() else {
            return false;
        };
        // Use pre-discovered provider when available (skips redundant network probes).
        // Otherwise fall back to concurrent discovery (all providers probed in parallel).
        // The discovery future is ~22 KiB, so box it instead of inlining it here.
        let discovered = match self.pre_discovered_embedding.take() {
            Some(pre) => {
                tracing::debug!(
                    "Using pre-discovered embedding provider: {:?}",
                    pre.provider_type
                );
                Some(pre)
            }
            None => Box::pin(discover_best_embedding_provider()).await,
        };
        let Some(discovery) = discovered else {
            tracing::warn!(
                "Workspace embedding enabled but no embedding provider found. \
                 Falling back to buffer memory. To enable embeddings: set OPENAI_API_KEY \
                 for OpenAI, configure Ollama locally, or run /doctor for diagnostics."
            );
            if self.memory_type.is_none() {
                self.memory_type = Some(MemoryType::Buffer(BufferMemoryConfig::default()));
            }
            return false;
        };
        let mut config = match &mode {
            WorkspaceEmbeddingMode::Auto => {
                let root = std::env::current_dir()
                    .map_err(|e| anyhow::anyhow!("Current dir unavailable: {e}"));
                let root = match root {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!("Workspace embedding: {}", e);
                        return false;
                    }
                };
                WorkspaceIndexerConfig::new(root).with_env_overrides()
            }
            WorkspaceEmbeddingMode::Custom(c) => c.clone(),
        };

        let has_edit_tools = self.agent_has_edit_tool();
        let mut hashline_options =
            RagHashlineOptions::for_workspace(has_edit_tools, config.line_level_bm25);
        if let Some(enabled) = config.workspace_hashline_enabled {
            hashline_options.enabled = enabled;
        }
        if let Some(format_results) = config.workspace_hashline_format {
            hashline_options.format_results = format_results;
        }
        if config.index_fingerprint.is_none() {
            config.index_fingerprint = Some(format!(
                "provider={:?};model={};dimension={};hashline={};format={};line_bm25={};chunking={:?}",
                discovery.provider_type,
                discovery.embedding_model,
                discovery
                    .embedding_dimension
                    .map_or_else(|| "unknown".to_string(), |dimension| dimension.to_string()),
                hashline_options.enabled,
                hashline_options.format_results,
                hashline_options.line_level_bm25,
                hashline_options.chunking_strategy
            ));
        }

        let manifest = match &config.persistence {
            PersistenceConfig::Disk { path } => {
                match crate::workspace::manifest::IndexManifest::load_with_fingerprint(
                    path,
                    config.index_fingerprint.as_deref(),
                )
                .await
                {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!("Failed to load index manifest: {}", e);
                        return false;
                    }
                }
            }
            PersistenceConfig::InMemory => crate::workspace::manifest::IndexManifest::new(),
        };
        let manifest = Arc::new(tokio::sync::Mutex::new(manifest));

        #[cfg(feature = "tools-rag")]
        fn apply_rag_search_options(
            tool: RagTool,
            config: &crate::workspace::WorkspaceIndexerConfig,
        ) -> RagTool {
            if let Some(ref opts) = config.rag_search_options {
                tool.with_default_search_options(opts.clone())
            } else {
                tool
            }
        }
        #[cfg(not(feature = "tools-rag"))]
        fn apply_rag_search_options(
            tool: RagTool,
            _config: &crate::workspace::WorkspaceIndexerConfig,
        ) -> RagTool {
            tool
        }

        let rag_tool = match &config.persistence {
            PersistenceConfig::Disk { .. } => {
                let backend = match crate::workspace::build_workspace_backend(
                    &config,
                    discovery.embedding_dimension,
                )
                .await
                {
                    Ok(backend) => backend,
                    Err(e) => {
                        tracing::warn!("Failed to create workspace vector backend: {}", e);
                        return false;
                    }
                };

                let tool = RagTool::with_backend_and_hashline_options(
                    Arc::clone(&discovery.provider),
                    backend,
                    self.rag_namespace.clone(),
                    false,
                    hashline_options,
                );
                Arc::new(apply_rag_search_options(tool, &config))
            }
            PersistenceConfig::InMemory => {
                let tool = RagTool::new_with_hashline_options(
                    Arc::clone(&discovery.provider),
                    self.rag_namespace.clone(),
                    hashline_options,
                );
                Arc::new(apply_rag_search_options(tool, &config))
            }
        };

        let indexer = WorkspaceIndexer::new(rag_tool.service(), config.clone());
        let indexing_ready = Arc::new(WorkspaceIndexingReady::new());
        let ready = Arc::clone(&indexing_ready);
        tracing::debug!("Starting workspace indexing in background...");
        tokio::spawn(async move {
            match indexer.index().await {
                Ok(res) => {
                    tracing::info!(
                        "Workspace indexing completed: {} files, {} chunks - context now available",
                        res.files_indexed,
                        res.total_chunks
                    );
                }
                Err(e) => {
                    tracing::warn!("Workspace indexing failed: {}. Continuing without.", e);
                }
            }
            ready.notify_done();
        });
        self.shared_rag_tool = Some(rag_tool);
        self.workspace_indexing_ready = Some(indexing_ready);
        self.workspace_config = Some(config);
        self.workspace_manifest = Some(manifest);
        if !self.use_all_default_tools && !self.tools.is_empty() {
            let has_fs = self
                .tools
                .iter()
                .any(|n| (*n).eq_ignore_ascii_case(TOOL_FILESYSTEM));
            if !has_fs {
                self.tools.push(Cow::borrowed(TOOL_FILESYSTEM));
            }
        }
        true
    }

    /// Add SQL generation and execution tools to the service when sql_tools config is set.
    #[cfg(not(all(feature = "tools-rag", feature = "tools-sql")))]
    #[expect(
        clippy::unused_async,
        reason = "same signature as the async SQL tools build"
    )]
    pub(crate) async fn setup_sql_tools(
        _service: &mut AgentService,
        _sql_config: SqlToolsConfig,
    ) -> anyhow::Result<Option<()>> {
        use crate::capabilities::tools;
        Err(anyhow!(
            "SQL tools require the 'tools-rag' and 'tools-sql' features to be enabled. {} {}",
            tools::tool_not_available_error("tools-rag"),
            tools::tool_not_available_error("tools-sql")
        ))
    }

    /// Add SQL generation and execution tools to the service when `sql_tools` config is set.
    #[cfg(all(feature = "tools-rag", feature = "tools-sql"))]
    pub(crate) async fn setup_sql_tools(
        service: &mut AgentService,
        sql_config: SqlToolsConfig,
    ) -> anyhow::Result<Option<Arc<dyn SqlExecutor>>> {
        {
            use dripstone::backend::memory::{InMemoryConfig, MemoryBackend};

            let embedding_provider = service.embedding_provider().ok_or_else(|| {
                anyhow!(
                    "SQL tools require an embedding-capable provider, but {:?} does not expose embeddings",
                    service.provider()
                )
            })?;

            let backend: Arc<dyn Backend + Send + Sync> =
                Arc::new(MemoryBackend::with_config(InMemoryConfig {
                    default_ttl_secs: Some(86_400),
                    max_entries: Some(50_000),
                    auto_cleanup: true,
                })) as Arc<dyn Backend + Send + Sync>;

            let introspector = PostgresSchemaIntrospector::connect(&sql_config.database_url)
                .await
                .context("Failed to connect for schema extraction")?;
            let introspector_arc: Arc<dyn DatabaseIntrospector> = Arc::new(introspector);

            let pipeline = SchemaEmbeddingPipeline::new(
                Arc::clone(&introspector_arc),
                embedding_provider,
                Arc::clone(&backend),
            );

            let sync_report = pipeline
                .sync(&sql_config.namespace)
                .await
                .context("Failed to embed database schema")?;

            if let Some(ref usage) = sync_report.token_usage {
                tracing::info!(
                    namespace = %sync_report.namespace,
                    tables = sync_report.table_count,
                    columns = sync_report.column_count,
                    embeddings = sync_report.embeddings_written,
                    tokens = usage.embedding_tokens,
                    "Loaded schema into embeddings for SQL tools"
                );
            } else {
                tracing::info!(
                    namespace = %sync_report.namespace,
                    tables = sync_report.table_count,
                    columns = sync_report.column_count,
                    embeddings = sync_report.embeddings_written,
                    "Loaded schema into embeddings for SQL tools"
                );
            }

            let embedding_provider_for_tool = service
                .embedding_provider()
                .ok_or_else(|| anyhow!("Embedding provider required for SQL generation tool"))?;

            let sql_generation_tool = SqlGenerationTool::new(
                service.llm_provider(),
                embedding_provider_for_tool,
                Arc::clone(&backend),
            )
            .with_namespace(sql_config.namespace)
            .with_top_k(sql_config.top_k);
            service.agent_mut().add_tool(Arc::new(sql_generation_tool));

            let executor = PostgresSqlExecutor::connect(&sql_config.database_url)
                .await
                .context("Failed to initialize SQL executor")?;
            let executor_arc: Arc<dyn SqlExecutor> = Arc::new(executor);
            let sql_policy = crate::agent::tools::sql_execution::SqlExecutionPolicy::from_env(
                sql_config.max_rows,
            );
            let sql_execution_tool = SqlExecutionTool::new(Arc::clone(&executor_arc))
                .with_sql_execution_policy(sql_policy)
                .with_max_rows(sql_config.max_rows);
            service.agent_mut().add_tool(Arc::new(sql_execution_tool));
            Ok(Some(executor_arc))
        }
    }

    pub fn add_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.custom_tools.push(tool);
        self
    }

    /// Register a toolkit (collection of related tools)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, Toolkit};
    /// use std::sync::Arc;
    ///
    /// // Assuming FileSystemToolkit exists
    /// # struct FileSystemToolkit;
    /// # impl Toolkit for FileSystemToolkit {
    /// #     fn get_tools(&self) -> Vec<Arc<dyn caliche::Tool>> { vec![] }
    /// #     fn name(&self) -> &str { "filesystem" }
    /// # }
    /// let toolkit = Arc::new(FileSystemToolkit);
    /// let builder = AgentBuilder::new()
    ///     .register_toolkit(toolkit);
    /// ```
    pub fn register_toolkit(mut self, toolkit: Arc<dyn crate::agent::tools::Toolkit>) -> Self {
        for tool in toolkit.get_tools() {
            self.custom_tools.push(tool);
        }
        self
    }

    /// Enable the research tool with a runner that executes research tasks (e.g. a sub-agent
    /// with workspace retrieval, fs, and workspace). The main agent can call the
    /// research tool with a query and receive a synthesized answer.
    pub fn research_runner(mut self, runner: Arc<dyn crate::agent::tools::ResearchRunner>) -> Self {
        self.research_runner = Some(runner);
        self
    }

    #[must_use]
    pub fn with_skill(mut self, skill: crate::agent::skills::Skill) -> Self {
        // Add required tools from the skill
        for tool_name in &skill.required_tools {
            if !self.tools.iter().any(|t| t == tool_name.as_str()) {
                self.tools.push(Cow::owned(tool_name.clone()));
            }
        }
        self.skills.push(skill);
        self
    }

    /// Add multiple skills to the agent.
    pub fn with_skills<I>(mut self, skills: I) -> Self
    where
        I: IntoIterator<Item = crate::agent::skills::Skill>,
    {
        for skill in skills {
            self = self.with_skill(skill);
        }
        self
    }

    /// Add a prebuilt code analysis skill
    #[must_use]
    pub fn with_code_analysis_skill(self) -> Self {
        self.with_skill(crate::agent::skills::prebuilt::code_analysis())
    }

    /// Add a prebuilt code review skill
    #[must_use]
    pub fn with_code_review_skill(self) -> Self {
        self.with_skill(crate::agent::skills::prebuilt::code_review())
    }

    /// Add a prebuilt debugging skill
    #[must_use]
    pub fn with_debugging_skill(self) -> Self {
        self.with_skill(crate::agent::skills::prebuilt::debugging())
    }

    /// Add a prebuilt research skill
    #[must_use]
    pub fn with_research_skill(self) -> Self {
        self.with_skill(crate::agent::skills::prebuilt::research())
    }

    /// Enable JSON object mode (any valid JSON response)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::OpenAI)
    ///     .model("gpt-4o")
    ///     .json_mode()
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn json_mode(mut self) -> Self {
        self.response_format = Some(crate::output::ResponseFormat::JsonObject);
        self
    }

    /// Enable JSON schema mode with a specific schema
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    /// use serde_json::json;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let schema = json!({
    ///     "type": "object",
    ///     "properties": {
    ///         "name": {"type": "string"},
    ///         "age": {"type": "integer"}
    ///     },
    ///     "required": ["name", "age"]
    /// });
    ///
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::OpenAI)
    ///     .model("gpt-4o")
    ///     .json_schema("Person", schema)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn json_schema(mut self, name: &str, schema: serde_json::Value) -> Self {
        // Normalize schema to ensure OpenAI compatibility (adds additionalProperties: false if needed)
        let normalized_schema = crate::output::JsonSchemaDefinition::normalize_schema(schema);

        self.response_format = Some(crate::output::ResponseFormat::JsonSchema {
            json_schema: crate::output::JsonSchemaDefinition {
                name: name.to_string(),
                description: None,
                schema: normalized_schema,
                strict: true,
            },
        });
        self
    }
    /// Set the list of allowed tools (if None, all tools are allowed)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType};
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::Bedrock)
    ///     .allowed_tools(&["calculator", "fs"])
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn allowed_tools(mut self, tools: &[&str]) -> Self {
        self.allowed_tools = Some(tools.iter().map(std::string::ToString::to_string).collect());
        self
    }

    #[must_use]
    pub fn allowed_builtin_tools(mut self, tools: &[crate::agent::tools::BuiltInTool]) -> Self {
        self.allowed_tools = Some(tools.iter().map(|tool| tool.as_str().to_owned()).collect());
        self
    }

    /// Register an in-process MCP server for direct tool registration
    ///
    /// In-process MCP servers execute tools directly in-process, eliminating subprocess
    /// overhead. Tools from the server are added to the agent's tool registry.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentBuilder, ProviderType, Tool, ToolResult};
    /// use caliche::mcp::InProcessMcpServer;
    /// use async_trait::async_trait;
    /// use serde_json::json;
    ///
    /// #[derive(Debug)]
    /// struct MyCustomTool;
    ///
    /// impl caliche::agent::tools::ToolMeta for MyCustomTool {
    ///     fn name(&self) -> &str { "my_tool" }
    ///     fn description(&self) -> &str { "A custom tool" }
    ///     fn parameters(&self) -> serde_json::Value { json!({}) }
    /// }
    ///
    /// #[async_trait]
    /// impl Tool for MyCustomTool {
    ///     async fn execute(&self, _args: serde_json::Value) -> ToolResult {
    ///         ToolResult::success("done".to_string())
    ///     }
    /// }
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let server = InProcessMcpServer::builder("my-tools")
    ///     .add_tool(MyCustomTool)
    ///     .build()?;
    ///
    /// let agent = AgentBuilder::new()
    ///     .provider(ProviderType::Bedrock)
    ///     .mcp_server("tools", server)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "mcp")]
    pub fn mcp_server(
        mut self,
        name: impl Into<String>,
        server: crate::mcp::InProcessMcpServer,
    ) -> Self {
        self.mcp_servers.insert(name.into(), Arc::new(server));
        self
    }

    /// Add an external MCP stdio server.
    ///
    /// When `CALICHE_MCP_COMMAND_ALLOWLIST` is set, `command` must match
    /// it. Production (`CALICHE_ENV=production`) requires the allowlist.
    /// Use [`Self::dangerously_add_stdio_mcp_server`] to skip the check.
    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    pub fn add_stdio_mcp_server(
        mut self,
        name: impl Into<String>,
        command: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.remote_mcp_servers.push(crate::mcp::McpClientConfig {
            name: name.into(),
            transport: crate::mcp::McpClientTransportConfig::Stdio {
                command: command.into(),
                args: args.into_iter().map(|a| a.into()).collect(),
                env: Vec::new(),
            },
            skip_stdio_allowlist: false,
        });
        self
    }

    /// Add an external MCP stdio server that receives `env` on top of the
    /// scrubbed host environment.
    ///
    /// The server never inherits provider keys or tokens from this process,
    /// so pass any key it needs here. The same allowlist rules as
    /// [`Self::add_stdio_mcp_server`] apply.
    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    pub fn add_stdio_mcp_server_with_env(
        mut self,
        name: impl Into<String>,
        command: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
        env: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        self.remote_mcp_servers.push(crate::mcp::McpClientConfig {
            name: name.into(),
            transport: crate::mcp::McpClientTransportConfig::Stdio {
                command: command.into(),
                args: args.into_iter().map(|a| a.into()).collect(),
                env: env
                    .into_iter()
                    .map(|(key, value)| (key.into(), value.into()))
                    .collect(),
            },
            skip_stdio_allowlist: false,
        });
        self
    }

    /// Add an MCP stdio server without applying `CALICHE_MCP_COMMAND_ALLOWLIST`.
    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    pub fn dangerously_add_stdio_mcp_server(
        mut self,
        name: impl Into<String>,
        command: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.remote_mcp_servers.push(crate::mcp::McpClientConfig {
            name: name.into(),
            transport: crate::mcp::McpClientTransportConfig::Stdio {
                command: command.into(),
                args: args.into_iter().map(|a| a.into()).collect(),
                env: Vec::new(),
            },
            skip_stdio_allowlist: true,
        });
        self
    }

    /// Pre-wire MCP servers from a workspace warm-start plan.
    ///
    /// Does not run the indexer. Repos and skills are already resolved on `plan`.
    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    pub fn apply_workspace_warm_start(
        mut self,
        plan: &crate::workspace_declaration::WarmStartPlan,
    ) -> anyhow::Result<Self> {
        for config in plan.to_mcp_client_configs()? {
            self.remote_mcp_servers.push(config);
        }
        Ok(self)
    }

    /// Add an external MCP streamable-HTTP server endpoint.
    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    pub fn add_streamable_http_mcp_server(
        mut self,
        name: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        self.remote_mcp_servers.push(crate::mcp::McpClientConfig {
            name: name.into(),
            transport: crate::mcp::McpClientTransportConfig::StreamableHttp { url: url.into() },
            skip_stdio_allowlist: false,
        });
        self
    }

    /// Enable or disable filesystem MCP auto-discovery during [`Self::build`].
    ///
    /// When enabled, the builder scans Cursor/Claude/Caliche MCP config files
    /// under the cwd, repo root, and home directory and attaches discovered remote
    /// servers. Defaults to `false` so embedding SDKs do not silently inherit host
    /// MCP tools (for example `~/.cursor/mcp.json`).
    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    pub fn auto_discover_mcp(mut self, enabled: bool) -> Self {
        self.auto_discover_mcp = enabled;
        self
    }

    /// Enable or disable on-disk skill auto-indexing during [`Self::build`].
    ///
    /// When enabled, discovered `SKILL.md` entries are registered into the
    /// progressive skill index (injected into the system prompt when `fs` is
    /// available). Defaults to `false` so SDK agents stay token-lean unless the
    /// caller opts in (CLI enables this). Explicit [`Self::with_skills`] entries
    /// are always applied regardless of this flag.
    #[must_use]
    pub const fn auto_discover_skills(mut self, enabled: bool) -> Self {
        self.auto_discover_skills = enabled;
        self
    }
}

#[cfg(all(test, feature = "profiles"))]
mod profile_permission_tests {
    use crate::AgentBuilder;
    use crate::agent::hooks::PermissionMode;
    use crate::agent::profile::{AgentProfile, ProfilePermission};

    fn profile_with_edit(edit: Option<&str>) -> AgentProfile {
        AgentProfile {
            permission: Some(ProfilePermission {
                edit: edit.map(str::to_string),
                ..ProfilePermission::default()
            }),
            ..AgentProfile::default()
        }
    }

    #[test]
    fn unknown_profile_permission_mode_is_rejected() {
        let error = match AgentBuilder::from_profile(&profile_with_edit(Some("yolo"))) {
            Err(error) => error,
            Ok(_) => panic!("unknown mode must be rejected"),
        };
        assert!(error.contains("yolo"), "{error}");
        assert!(error.contains("default"), "{error}");
        assert!(error.contains("accept-edits"), "{error}");
        assert!(error.contains("accept-all"), "{error}");
        assert!(error.contains("deny-all"), "{error}");
        assert!(error.contains("interactive"), "{error}");
    }

    #[test]
    fn known_profile_permission_mode_is_applied() {
        let builder =
            AgentBuilder::from_profile(&profile_with_edit(Some("deny-all"))).expect("known mode");
        assert_eq!(builder.permission_mode, PermissionMode::DenyAll);
    }

    #[test]
    fn empty_or_missing_profile_permission_mode_stays_default() {
        let empty = AgentBuilder::from_profile(&profile_with_edit(Some(""))).expect("empty mode");
        assert_eq!(empty.permission_mode, PermissionMode::Default);
        let missing = AgentBuilder::from_profile(&profile_with_edit(None)).expect("missing mode");
        assert_eq!(missing.permission_mode, PermissionMode::Default);
    }
}
