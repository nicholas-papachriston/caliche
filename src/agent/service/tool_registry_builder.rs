//! Tool registry builder for assembling tool collections.
//!
//! This module provides a fluent builder API for constructing tool registries with
//! configuration. It handles conditional tool registration based on provider
//! capabilities, available credentials, and user preferences.
//!
//! # Features
//!
//! - Selective tool registration based on tool names
//! - Optional semantic search registration when embedding provider is available
//! - Perplexity search tool integration (when not using Perplexity as main provider)
//! - Support for shared workspace RAG services across multiple agents
//!
//! # Examples
//!
//! ```no_run
//! use caliche::agent::service::ToolRegistryBuilder;
//! use caliche::llm::ProviderType;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Build registry with all default tools
//! let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
//!     .use_all_default_tools()
//!     .build()
//!     .await?;
//!
//! // Build registry with specific tools only
//! let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
//!     .with_tool_names(Some(vec!["calculator".to_string(), "fs".to_string()]))
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "tools-ast")]
use crate::agent::tools::AstTool;
use crate::agent::tools::names::{
    CODING_DEFAULT_TOOLS, EXTENDED_DEFAULT_TOOLS, TOOL_APPLY_PATCH, TOOL_EXECUTE_PYTHON,
    TOOL_FILESYSTEM, TOOL_GENERATE_IMAGE, TOOL_HTTP, TOOL_WEB_FETCH, TOOL_WEB_SEARCH,
    TOOL_WORKSPACE,
};
use crate::agent::tools::{
    ApplyPatchTool, BashTool, BuiltInTool, CalculatorTool, CodeExecutionTool, DataManipulationTool,
    FsTool, HttpTool, ImageGenerationTool, MutationStyle, PerplexitySearchTool, Tool, ToolRegistry,
    VerifyIntegrationTool, WebFetchTool, WebSearchTool, WorkspaceTool,
};
#[cfg(feature = "lsp")]
use crate::agent::tools::{LspDiagnosticsTool, LspHoverTool, LspSymbolsTool};
use crate::artifacts::ArtifactStore;
use crate::llm::{EmbeddingProvider, ImageGenerationProvider, ProviderType};
use crate::observability::ToolMetrics;

#[cfg(feature = "tools-git")]
use crate::agent::tools::GitTool;
#[cfg(feature = "tools-document")]
use crate::agent::tools::ProcessDocumentTool;
#[cfg(feature = "tools-rag")]
use crate::agent::tools::SemanticSearchTool;
#[cfg(feature = "tools-graph")]
use crate::agent::tools::graph::{FrontierPrioritizer, GraphExplorerTool, GraphQueryTool};
#[cfg(feature = "session-tools")]
use crate::agent::tools::session_tools::{
    SessionSpawner, SessionsHistoryTool, SessionsListTool, SessionsSendTool, SessionsSpawnTool,
};
#[cfg(feature = "tools-graph")]
use dripstone::graph::QueryableGraphBackend;
use std::sync::Arc;

/// Result of building a tool registry, including handles to concrete tools when needed.
///
/// This is primarily used to retain access to tools that are otherwise stored behind
/// `Arc<dyn Tool>` trait objects (e.g. `RagTool`), without relying on downcasting.
pub struct BuiltToolRegistry {
    pub registry: Arc<ToolRegistry>,
}

/// Builder for constructing tool registries from configuration.
#[derive(Clone)]
pub struct ToolRegistryBuilder {
    provider: ProviderType,
    tool_names: Option<Vec<String>>,
    built_in_tools: Vec<BuiltInTool>,
    use_all_default_tools: bool,
    mutation_style: MutationStyle,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    image_generation_provider: Option<Arc<dyn ImageGenerationProvider>>,
    artifact_store: Option<Arc<ArtifactStore>>,
    #[cfg(feature = "tools-graph")]
    graph_backend: Option<Arc<dyn QueryableGraphBackend>>,
    #[cfg(feature = "tools-graph")]
    frontier_prioritizer: Option<FrontierPrioritizer>,
    metrics: Option<Arc<ToolMetrics>>,
    custom_tools: Vec<Arc<dyn Tool>>,
    #[cfg(feature = "lsp")]
    lsp_manager: Option<Arc<crate::lsp::LspManager>>,
    #[cfg(feature = "session-tools")]
    session_spawner: Option<Arc<dyn SessionSpawner>>,
}

impl ToolRegistryBuilder {
    /// Create a new tool registry builder.
    /// By default, no tools are registered unless explicitly specified.
    #[must_use]
    pub fn new(provider: ProviderType) -> Self {
        Self {
            provider,
            tool_names: None,
            built_in_tools: Vec::new(),
            use_all_default_tools: false,
            mutation_style: MutationStyle::EditWrite,
            embedding_provider: None,
            image_generation_provider: None,
            artifact_store: None,
            #[cfg(feature = "tools-graph")]
            graph_backend: None,
            #[cfg(feature = "tools-graph")]
            frontier_prioritizer: None,
            metrics: None,
            custom_tools: Vec::new(),
            #[cfg(feature = "lsp")]
            lsp_manager: None,
            #[cfg(feature = "session-tools")]
            session_spawner: None,
        }
    }

    /// Select file-mutation style. `ApplyPatch` XOR `fs` write/edit.
    #[must_use]
    pub const fn with_mutation_style(mut self, style: MutationStyle) -> Self {
        self.mutation_style = style;
        self
    }

    /// Set metrics tracker for tool execution observability.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Option<Arc<ToolMetrics>>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Set specific tools to register.
    /// - If `Some(vec)` is provided, only those tools will be registered.
    /// - If `None` is provided, no tools will be registered unless `use_all_default_tools()` is called.
    #[must_use]
    pub fn with_tool_names(mut self, tool_names: Option<Vec<String>>) -> Self {
        self.tool_names = tool_names;
        self
    }

    #[must_use]
    pub fn with_built_in_tools(mut self, tools: Vec<BuiltInTool>) -> Self {
        self.built_in_tools.extend(tools);
        self
    }

    /// Register custom tools in addition to built-in tools.
    #[must_use]
    pub fn with_custom_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.custom_tools = tools;
        self
    }

    #[cfg(feature = "lsp")]
    #[must_use]
    pub fn with_lsp_manager(mut self, manager: Option<Arc<crate::lsp::LspManager>>) -> Self {
        self.lsp_manager = manager;
        self
    }

    /// Explicitly request the canonical default tools (`fs`, `bash`, `web_fetch`).
    ///
    /// Optional extras (HTTP, workspace, Python, semantic search, `web_search`,
    /// etc.) must be requested via [`Self::with_tool_names`] /
    /// [`Self::with_built_in_tools`] or [`Self::use_extended_default_tools`].
    #[must_use]
    pub const fn use_all_default_tools(mut self) -> Self {
        self.use_all_default_tools = true;
        self
    }

    /// Register the broader extended default set (`fs`, `bash`, `web_fetch`, http, workspace, `execute_python`).
    /// Does not auto-register semantic search, Perplexity, `web_search`, image generation, or graph tools.
    #[must_use]
    pub fn use_extended_default_tools(mut self) -> Self {
        self.use_all_default_tools = false;
        let mut names = self.tool_names.take().unwrap_or_default();
        for name in EXTENDED_DEFAULT_TOOLS {
            if !names
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(name))
            {
                names.push((*name).to_string());
            }
        }
        self.tool_names = Some(names);
        self
    }

    /// Set the embedding provider for RAG tools.
    #[must_use]
    pub fn with_embedding_provider(
        mut self,
        embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    ) -> Self {
        self.embedding_provider = embedding_provider;
        self
    }

    /// Set the image generation provider for the `generate_image` tool.
    #[must_use]
    pub fn with_image_generation(
        mut self,
        provider: Option<Arc<dyn ImageGenerationProvider>>,
        artifact_store: Option<Arc<ArtifactStore>>,
    ) -> Self {
        self.image_generation_provider = provider;
        self.artifact_store = artifact_store;
        self
    }

    /// Set the queryable graph backend for `graph_query` and `graph_explorer` tools.
    #[cfg(feature = "tools-graph")]
    #[must_use]
    pub fn with_graph_backend(
        mut self,
        graph_backend: Option<Arc<dyn QueryableGraphBackend>>,
    ) -> Self {
        self.graph_backend = graph_backend;
        self
    }

    /// Set the frontier prioritizer for `graph_explorer` (e.g. `FrontierPrioritizer::security_default()`).
    #[cfg(feature = "tools-graph")]
    #[must_use]
    pub fn with_frontier_prioritizer(
        mut self,
        frontier_prioritizer: Option<FrontierPrioritizer>,
    ) -> Self {
        self.frontier_prioritizer = frontier_prioritizer;
        self
    }

    /// Set the session spawner for `sessions_spawn` tool (when session-tools is enabled).
    #[cfg(feature = "session-tools")]
    #[must_use]
    pub fn session_spawner(mut self, spawner: Option<Arc<dyn SessionSpawner>>) -> Self {
        self.session_spawner = spawner;
        self
    }

    /// Build the tool registry.
    pub async fn build(self) -> anyhow::Result<Arc<ToolRegistry>> {
        Ok(self.build_with_handles().await?.registry)
    }

    /// Build the tool registry and return handles to concrete tools when applicable.
    pub async fn build_with_handles(self) -> anyhow::Result<BuiltToolRegistry> {
        let mut tool_registry = if let Some(ref metrics) = self.metrics {
            ToolRegistry::with_metrics(Arc::clone(metrics))
        } else {
            ToolRegistry::new()
        };

        let mutation_style = resolve_mutation_style(self.mutation_style, self.tool_names.as_ref());
        let all_tools = self.collect_available_tools(mutation_style);
        let names_set = self.build_requested_names(mutation_style);

        if self.use_all_default_tools {
            register_canonical_defaults(
                &mut tool_registry,
                all_tools,
                mutation_style,
                &self.custom_tools,
            );
        } else if let Some(ref names_set) = names_set {
            register_named_builtins(&mut tool_registry, all_tools, names_set);
            register_custom_tools(&mut tool_registry, &self.custom_tools);
            self.register_progressive_tools(&mut tool_registry, names_set)
                .await;
        } else {
            register_custom_tools(&mut tool_registry, &self.custom_tools);
        }

        Ok(BuiltToolRegistry {
            registry: Arc::new(tool_registry),
        })
    }

    fn collect_available_tools(
        &self,
        mutation_style: MutationStyle,
    ) -> Vec<(String, Arc<dyn Tool>)> {
        #[cfg(feature = "lsp")]
        let mut all_tools = collect_core_tools(mutation_style, self.lsp_manager_for_tools());
        #[cfg(not(feature = "lsp"))]
        let mut all_tools = collect_core_tools(mutation_style);
        self.append_optional_tools(&mut all_tools);
        for tool in &self.custom_tools {
            all_tools.push((tool.name().to_string(), Arc::clone(tool)));
        }
        all_tools
    }

    fn build_requested_names(
        &self,
        mutation_style: MutationStyle,
    ) -> Option<std::collections::HashSet<String>> {
        let mut names_set = self.tool_names.as_ref().map(|names| {
            names
                .iter()
                .map(|n| n.to_ascii_lowercase())
                .collect::<std::collections::HashSet<String>>()
        });
        if !self.built_in_tools.is_empty() {
            let names = names_set.get_or_insert_with(std::collections::HashSet::new);
            for tool in &self.built_in_tools {
                names.insert(tool.as_str().to_owned());
            }
        }
        Self::expand_canonical_aliases(&mut names_set);
        if mutation_style == MutationStyle::ApplyPatch
            && let Some(ref mut names) = names_set
        {
            names.insert(TOOL_APPLY_PATCH.to_ascii_lowercase());
        }
        names_set
    }

    async fn register_progressive_tools(
        &self,
        tool_registry: &mut ToolRegistry,
        names_set: &std::collections::HashSet<String>,
    ) {
        if names_set.contains("perplexity_search") {
            self.register_perplexity_tool(tool_registry).await;
        }
        if names_set.contains("semantic_search") {
            self.register_semantic_search_tool(tool_registry);
        }
        if names_set.contains(TOOL_GENERATE_IMAGE) {
            self.register_image_generation_tool(tool_registry);
        }
        #[cfg(feature = "tools-graph")]
        self.register_graph_tools(tool_registry, names_set);
    }

    #[cfg(feature = "tools-graph")]
    fn register_graph_tools(
        &self,
        tool_registry: &mut ToolRegistry,
        names_set: &std::collections::HashSet<String>,
    ) {
        let Some(ref graph_backend) = self.graph_backend else {
            return;
        };
        if names_set.contains("graph_query") {
            let query_tool = Arc::new(GraphQueryTool::new(Arc::clone(graph_backend)));
            tool_registry.register(query_tool);
        }
        if names_set.contains("graph_explorer") {
            let explorer_tool = match &self.frontier_prioritizer {
                Some(p) => Arc::new(GraphExplorerTool::with_prioritizer(
                    Arc::clone(graph_backend),
                    p.clone(),
                )),
                None => Arc::new(GraphExplorerTool::new(Arc::clone(graph_backend))),
            };
            tool_registry.register(explorer_tool);
        }
    }

    #[cfg(feature = "lsp")]
    fn lsp_manager_for_tools(&self) -> Option<Arc<crate::lsp::LspManager>> {
        self.lsp_manager
            .clone()
            .or_else(|| Some(Arc::new(crate::lsp::LspManager::default())))
    }

    fn append_optional_tools(&self, all_tools: &mut Vec<(String, Arc<dyn Tool>)>) {
        #[cfg(feature = "tools-document")]
        all_tools.push((
            "process_document".to_string(),
            Arc::new(ProcessDocumentTool) as Arc<dyn Tool>,
        ));

        #[cfg(feature = "session-tools")]
        {
            all_tools.push((
                "sessions_list".to_string(),
                Arc::new(SessionsListTool) as Arc<dyn Tool>,
            ));
            all_tools.push((
                "sessions_history".to_string(),
                Arc::new(SessionsHistoryTool) as Arc<dyn Tool>,
            ));
            all_tools.push((
                "sessions_send".to_string(),
                Arc::new(SessionsSendTool) as Arc<dyn Tool>,
            ));
            all_tools.push((
                "sessions_spawn".to_string(),
                Arc::new(SessionsSpawnTool::new(self.session_spawner.clone())) as Arc<dyn Tool>,
            ));
        }

        #[cfg(feature = "tools-git")]
        {
            all_tools.push(("git".to_string(), Arc::new(GitTool) as Arc<dyn Tool>));
        }

        #[cfg(feature = "lsp")]
        {
            let manager = self
                .lsp_manager_for_tools()
                .unwrap_or_else(|| Arc::new(crate::lsp::LspManager::default()));
            all_tools.push((
                "lsp_diagnostics".to_string(),
                Arc::new(LspDiagnosticsTool::new(Arc::clone(&manager))) as Arc<dyn Tool>,
            ));
            all_tools.push((
                "lsp_hover".to_string(),
                Arc::new(LspHoverTool::new(Arc::clone(&manager))) as Arc<dyn Tool>,
            ));
            all_tools.push((
                "lsp_symbols".to_string(),
                Arc::new(LspSymbolsTool::new(manager)) as Arc<dyn Tool>,
            ));
        }
    }

    fn expand_canonical_aliases(names_set: &mut Option<std::collections::HashSet<String>>) {
        let Some(names) = names_set.as_mut() else {
            return;
        };

        if names.contains("shell") {
            names.insert("bash".to_string());
        }
        if names.contains("web") {
            names.insert(TOOL_HTTP.to_string());
        }
        if names.contains("memory") {
            names.insert(TOOL_WORKSPACE.to_string());
        }
        // Do not expand bare "search" — it collided with MCP code-mode `search` and pulled
        // semantic/perplexity into coding sessions. Request `semantic_search` /
        // `perplexity_search` explicitly; use `fs` op=grep for codebase search.
        if names.contains("session") {
            names.insert("sessions_list".to_string());
            names.insert("sessions_history".to_string());
            names.insert("sessions_send".to_string());
            names.insert("sessions_spawn".to_string());
        }
    }

    /// Register Perplexity search tool if conditions are met.
    async fn register_perplexity_tool(&self, tool_registry: &mut ToolRegistry) {
        // Register Perplexity search tool if:
        // 1. Provider is NOT Perplexity (to avoid recursive calls)
        // 2. Perplexity API key is available
        if self.provider != ProviderType::Perplexity {
            if let Some(perplexity_tool) = PerplexitySearchTool::from_env().await {
                tool_registry.register(Arc::new(perplexity_tool));
            } else {
                // Warn if explicitly requested or if using all default tools (user expects it)
                if self.tool_names.is_some() || self.use_all_default_tools {
                    tracing::warn!(
                        "perplexity_search tool not available: PERPLEXITY_API_KEY not found in environment. \
                         Credentials may be configured in credential store but not loaded to environment."
                    );
                }
            }
        } else if self.tool_names.is_some() || self.use_all_default_tools {
            tracing::warn!(
                "perplexity_search tool not available when using Perplexity as provider"
            );
        }
    }

    /// Register semantic search tool if embedding provider is available.
    #[cfg(feature = "tools-rag")]
    fn register_semantic_search_tool(&self, tool_registry: &mut ToolRegistry) {
        if let Some(ref emb_provider) = self.embedding_provider {
            let semantic_search = SemanticSearchTool::with_defaults(Arc::clone(emb_provider));
            tool_registry.register(Arc::new(semantic_search));
        }
    }

    #[cfg(not(feature = "tools-rag"))]
    fn register_semantic_search_tool(&self, _tool_registry: &mut ToolRegistry) {
        // No-op when tools-rag feature is not enabled
    }

    fn register_image_generation_tool(&self, tool_registry: &mut ToolRegistry) {
        if let (Some(provider), Some(artifact_store)) =
            (&self.image_generation_provider, &self.artifact_store)
        {
            tool_registry.register(Arc::new(ImageGenerationTool::new(
                Arc::clone(provider),
                Arc::clone(artifact_store),
            )));
        }
    }

    /// Register the plugin tool (create/list/update/delete/export/import schema-based plugins).
    /// Call with the tool registry and plugin dependencies when plugins are enabled.
    #[cfg(feature = "plugins")]
    pub fn register_plugin_tool(
        _self: &ToolRegistryBuilder,
        tool_registry: &Arc<ToolRegistry>,
        plugin_registry: Arc<crate::plugins::registry::PluginRegistry>,
        runtime: Arc<crate::plugins::runtime::StepRuntime>,
        capabilities: Arc<crate::plugins::runtime::AgentCapabilities>,
    ) {
        let tool = crate::plugins::tool::PluginTool::new(
            plugin_registry,
            runtime,
            Arc::clone(tool_registry),
            capabilities,
        );
        tool_registry.register(Arc::new(tool));
    }
}

fn resolve_mutation_style(style: MutationStyle, tool_names: Option<&Vec<String>>) -> MutationStyle {
    if let Some(names) = tool_names
        && names
            .iter()
            .any(|n| n.eq_ignore_ascii_case(TOOL_APPLY_PATCH))
    {
        return MutationStyle::ApplyPatch;
    }
    style
}

fn collect_core_tools(
    mutation_style: MutationStyle,
    #[cfg(feature = "lsp")] shared_lsp_manager: Option<Arc<crate::lsp::LspManager>>,
) -> Vec<(String, Arc<dyn Tool>)> {
    #[cfg(feature = "lsp")]
    let fs_tool =
        FsTool::with_mutation_style(mutation_style).with_lsp_manager(shared_lsp_manager.clone());
    #[cfg(not(feature = "lsp"))]
    let fs_tool = FsTool::with_mutation_style(mutation_style);

    #[cfg(feature = "lsp")]
    let apply_patch_tool = ApplyPatchTool::new().with_lsp_manager(shared_lsp_manager);
    #[cfg(not(feature = "lsp"))]
    let apply_patch_tool = ApplyPatchTool::new();

    vec![
        ("bash".to_string(), Arc::new(BashTool) as Arc<dyn Tool>),
        (
            "calculator".to_string(),
            Arc::new(CalculatorTool) as Arc<dyn Tool>,
        ),
        (
            TOOL_EXECUTE_PYTHON.to_string(),
            Arc::new(CodeExecutionTool) as Arc<dyn Tool>,
        ),
        (
            TOOL_FILESYSTEM.to_string(),
            Arc::new(fs_tool) as Arc<dyn Tool>,
        ),
        (
            TOOL_APPLY_PATCH.to_string(),
            Arc::new(apply_patch_tool) as Arc<dyn Tool>,
        ),
        #[cfg(feature = "tools-ast")]
        ("ast".to_string(), Arc::new(AstTool::new()) as Arc<dyn Tool>),
        (
            TOOL_HTTP.to_string(),
            Arc::new(HttpTool::new()) as Arc<dyn Tool>,
        ),
        (
            TOOL_WEB_SEARCH.to_string(),
            Arc::new(WebSearchTool::from_env()) as Arc<dyn Tool>,
        ),
        (
            TOOL_WEB_FETCH.to_string(),
            Arc::new(WebFetchTool::from_env()) as Arc<dyn Tool>,
        ),
        (
            "data_manipulation".to_string(),
            Arc::new(DataManipulationTool) as Arc<dyn Tool>,
        ),
        (
            TOOL_WORKSPACE.to_string(),
            Arc::new(WorkspaceTool) as Arc<dyn Tool>,
        ),
        (
            "verify_integration".to_string(),
            Arc::new(VerifyIntegrationTool) as Arc<dyn Tool>,
        ),
    ]
}

fn register_canonical_defaults(
    tool_registry: &mut ToolRegistry,
    all_tools: Vec<(String, Arc<dyn Tool>)>,
    mutation_style: MutationStyle,
    custom_tools: &[Arc<dyn Tool>],
) {
    let canonical_defaults: std::collections::HashSet<&str> =
        CODING_DEFAULT_TOOLS.iter().copied().collect();
    for (name, tool) in all_tools {
        if canonical_defaults.contains(name.as_str())
            || (mutation_style == MutationStyle::ApplyPatch && name == TOOL_APPLY_PATCH)
        {
            tool_registry.register(tool);
        }
    }
    register_custom_tools(tool_registry, custom_tools);
}

fn register_named_builtins(
    tool_registry: &mut ToolRegistry,
    all_tools: Vec<(String, Arc<dyn Tool>)>,
    names_set: &std::collections::HashSet<String>,
) {
    for (name, tool) in all_tools {
        if names_set.contains(&name.to_lowercase()) {
            tool_registry.register(tool);
        }
    }
}

fn register_custom_tools(tool_registry: &mut ToolRegistry, custom_tools: &[Arc<dyn Tool>]) {
    for tool in custom_tools {
        tool_registry.register(Arc::clone(tool));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn builds_registry_from_typed_builtins() {
        let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
            .with_built_in_tools(vec![BuiltInTool::Calculator, BuiltInTool::Fs])
            .build()
            .await
            .expect("tool registry should build");

        assert!(registry.get("calculator").is_some());
        assert!(registry.get("fs").is_some());
        assert!(registry.get("http").is_none());
    }

    #[tokio::test]
    async fn merges_typed_builtins_with_string_tool_names() {
        let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
            .with_tool_names(Some(vec!["http".to_string()]))
            .with_built_in_tools(vec![BuiltInTool::Calculator])
            .build()
            .await
            .expect("tool registry should build");

        assert!(registry.get("calculator").is_some());
        assert!(registry.get("http").is_some());
    }

    #[tokio::test]
    async fn does_not_expand_removed_legacy_tool_aliases() {
        let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
            .with_tool_names(Some(vec![
                "read_file".to_string(),
                "git_status".to_string(),
            ]))
            .build()
            .await
            .expect("tool registry should build");

        assert!(registry.get("fs").is_none());
        assert!(registry.get("git").is_none());
        assert!(registry.get("http").is_none());
        assert!(registry.get("read_file").is_none());
        assert!(registry.get("git_status").is_none());
    }

    #[cfg(feature = "lsp")]
    #[tokio::test]
    async fn registers_lsp_tools_when_requested() {
        let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
            .with_tool_names(Some(vec!["lsp_diagnostics".to_string()]))
            .build()
            .await
            .expect("tool registry should build");

        assert!(registry.get("lsp_diagnostics").is_some());
        assert!(registry.get("lsp_hover").is_none());
    }

    #[tokio::test]
    async fn default_tools_include_fs_bash_and_web_fetch() {
        let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
            .use_all_default_tools()
            .build()
            .await
            .expect("tool registry should build");

        assert!(registry.get("fs").is_some());
        assert!(registry.get("bash").is_some());
        assert!(registry.get("web_fetch").is_some());
        assert!(registry.get("web_search").is_none());
        assert!(registry.get("apply_patch").is_none());
        assert!(registry.get("http").is_none());
        assert!(registry.get("workspace").is_none());
        assert!(registry.get("execute_python").is_none());
        assert!(registry.get("semantic_search").is_none());
        assert!(registry.get("calculator").is_none());
    }

    #[tokio::test]
    async fn apply_patch_style_registers_patch_and_strips_fs_write_edit() {
        let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
            .use_all_default_tools()
            .with_mutation_style(MutationStyle::ApplyPatch)
            .build()
            .await
            .expect("tool registry should build");

        assert!(registry.get("fs").is_some());
        assert!(registry.get("apply_patch").is_some());
        let schemas = registry.get_tool_schemas();
        let fs_schema = schemas
            .iter()
            .find(|s| s.name.as_ref() == "fs")
            .expect("fs schema");
        let op_enum = fs_schema
            .parameters
            .properties
            .pointer("/op/enum")
            .and_then(|v| v.as_array())
            .expect("op enum");
        let ops: Vec<&str> = op_enum.iter().filter_map(|v| v.as_str()).collect();
        assert!(ops.contains(&"read"));
        assert!(ops.contains(&"delete"));
        assert!(!ops.contains(&"write"));
        assert!(!ops.contains(&"edit"));
    }

    #[tokio::test]
    async fn requesting_apply_patch_by_name_enables_xor_style() {
        let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
            .with_tool_names(Some(vec!["fs".to_string(), "apply_patch".to_string()]))
            .build()
            .await
            .expect("tool registry should build");

        assert!(registry.get("apply_patch").is_some());
        let fs = registry.get("fs").expect("fs");
        let params = fs.parameters();
        let ops = params
            .pointer("/properties/op/enum")
            .and_then(|v| v.as_array())
            .expect("enum");
        assert!(!ops.iter().any(|v| v.as_str() == Some("write")));
    }

    #[tokio::test]
    async fn extended_defaults_include_http_workspace_python() {
        let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
            .use_extended_default_tools()
            .build()
            .await
            .expect("tool registry should build");

        assert!(registry.get("fs").is_some());
        assert!(registry.get("bash").is_some());
        assert!(registry.get("web_fetch").is_some());
        assert!(registry.get("http").is_some());
        assert!(registry.get("workspace").is_some());
        assert!(registry.get("execute_python").is_some());
        assert!(registry.get("semantic_search").is_none());
        assert!(registry.get("web_search").is_none());
    }

    #[tokio::test]
    async fn named_web_tools_register_without_keys() {
        let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
            .with_tool_names(Some(vec![
                "web_search".to_string(),
                "web_fetch".to_string(),
            ]))
            .build()
            .await
            .expect("tool registry should build");

        assert!(registry.get("web_search").is_some());
        assert!(registry.get("web_fetch").is_some());
        assert!(registry.get("http").is_none());
    }

    #[tokio::test]
    async fn coding_defaults_include_web_fetch_and_exclude_web_search() {
        let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
            .use_all_default_tools()
            .build()
            .await
            .expect("tool registry should build");

        assert!(registry.get("fs").is_some());
        assert!(registry.get("bash").is_some());
        assert!(registry.get("web_fetch").is_some());
        assert!(registry.get("web_search").is_none());
    }

    #[tokio::test]
    async fn bare_search_alias_does_not_register_semantic_tools() {
        let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
            .with_tool_names(Some(vec!["search".to_string()]))
            .build()
            .await
            .expect("tool registry should build");

        assert!(registry.get("semantic_search").is_none());
        assert!(registry.get("perplexity_search").is_none());
        assert!(registry.get("search").is_none());
    }

    #[tokio::test]
    async fn empty_tool_names_still_registers_custom_tools() {
        #[derive(Debug)]
        struct MultiplyTool;

        impl crate::agent::tools::ToolMeta for MultiplyTool {
            fn name(&self) -> &str {
                "multiply"
            }

            fn description(&self) -> &str {
                "Multiply two numbers"
            }

            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "x": { "type": "number" },
                        "y": { "type": "number" }
                    },
                    "required": ["x", "y"]
                })
            }
        }

        #[async_trait::async_trait]
        impl Tool for MultiplyTool {
            async fn execute(
                &self,
                arguments: serde_json::Value,
            ) -> crate::agent::tools::ToolResult {
                let x = arguments.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = arguments.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
                crate::agent::tools::ToolResult::success((x * y).to_string())
            }
        }

        let registry = ToolRegistryBuilder::new(ProviderType::OpenAI)
            .with_tool_names(Some(Vec::new()))
            .with_custom_tools(vec![Arc::new(MultiplyTool) as Arc<dyn Tool>])
            .build()
            .await
            .expect("tool registry should build");

        assert!(registry.get("multiply").is_some());
        assert!(registry.get("fs").is_none());
    }
}
