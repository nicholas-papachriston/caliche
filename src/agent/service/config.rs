use crate::agent::tools::BuiltInTool;
#[cfg(feature = "tools-rag")]
use crate::agent::tools::RagTool;
#[cfg(feature = "tools-graph")]
use crate::agent::tools::graph::FrontierPrioritizer;
#[cfg(feature = "session-tools")]
use crate::agent::tools::session_tools::SessionSpawner;
use crate::llm::ImageGenerationProvider;
use crate::llm::ProviderType;
use crate::llm::cache::LlmCache;
use crate::traits::credentials::{CredentialProvider, EnvironmentCredentialProvider};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct ToolSelection {
    pub names: Option<Vec<String>>,
    pub builtins: Vec<BuiltInTool>,
    pub use_all_default_tools: bool,
}

impl ToolSelection {
    #[must_use]
    pub const fn all_defaults() -> Self {
        Self {
            names: None,
            builtins: Vec::new(),
            use_all_default_tools: true,
        }
    }

    #[must_use]
    pub const fn explicit_names(names: Vec<String>) -> Self {
        Self {
            names: Some(names),
            builtins: Vec::new(),
            use_all_default_tools: false,
        }
    }

    #[must_use]
    pub const fn builtins(builtins: Vec<BuiltInTool>) -> Self {
        Self {
            names: None,
            builtins,
            use_all_default_tools: false,
        }
    }

    #[must_use]
    pub fn with_builtin(mut self, builtin: BuiltInTool) -> Self {
        self.builtins.push(builtin);
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.names.get_or_insert_with(Vec::new).push(name.into());
        self
    }
}

pub struct AgentServiceConfig {
    pub provider: Option<ProviderType>,
    pub model: Option<String>,
    pub tool_selection: ToolSelection,
    /// File mutation style (`fs` write/edit vs `apply_patch`). XOR.
    pub mutation_style: crate::agent::tools::MutationStyle,
    pub image_generation_provider: Option<Arc<dyn ImageGenerationProvider>>,
    pub artifact_store: Option<Arc<crate::artifacts::ArtifactStore>>,
    #[cfg(feature = "tools-rag")]
    pub shared_rag_tool: Option<Arc<RagTool>>,
    pub credential_provider: Arc<dyn CredentialProvider>,
    /// When true, a missing model key or an egress-blocked provider URL does not
    /// fail `AgentService::from_config`.
    ///
    /// The selected provider stays selected. Completions fail later with that
    /// provider's credential or egress error. This does not switch to another
    /// provider. A missing key is not authentication and not an isolation
    /// boundary. An egress rejection is not authentication either.
    pub defer_missing_credentials: bool,
    pub base_url: Option<String>,
    /// User-specified provider context window size. When set, this overrides
    /// provider metadata/fallback context windows while still reserving output tokens.
    pub context_window: Option<usize>,
    #[cfg(feature = "tools-graph")]
    pub graph_backend: Option<Arc<dyn dripstone::graph::QueryableGraphBackend>>,
    #[cfg(feature = "tools-graph")]
    pub frontier_prioritizer: Option<FrontierPrioritizer>,
    #[cfg(feature = "reqwest")]
    pub shared_http_client: Option<Arc<reqwest::Client>>,
    pub llm_cache: Option<Arc<LlmCache>>,
    #[cfg(feature = "local-inference-base")]
    pub local_inference_config: Option<crate::llm::LocalInferenceConfig>,
    /// When set, [`crate::agent::service::AgentService::from_config`] installs a
    /// [`crate::agent::session::BinaryFileSessionStore`] for autosave (rkyv v1 `.cls1` under this directory).
    pub session_autosave_dir: Option<PathBuf>,
    #[cfg(feature = "session-tools")]
    pub session_spawner: Option<Arc<dyn SessionSpawner>>,
    #[cfg(feature = "lsp")]
    pub lsp_config: Option<crate::lsp::LspConfig>,
    #[cfg(feature = "lsp")]
    pub lsp_manager: Option<Arc<crate::lsp::LspManager>>,
}

impl AgentServiceConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_provider(mut self, provider: Option<ProviderType>, model: Option<String>) -> Self {
        self.provider = provider;
        self.model = model;
        self
    }

    #[must_use]
    pub fn with_tool_selection(mut self, tool_selection: ToolSelection) -> Self {
        self.tool_selection = tool_selection;
        self
    }

    #[must_use]
    pub fn with_image_generation_provider(
        mut self,
        provider: Option<Arc<dyn ImageGenerationProvider>>,
    ) -> Self {
        self.image_generation_provider = provider;
        self
    }

    #[must_use]
    pub fn with_artifact_store(
        mut self,
        artifact_store: Option<Arc<crate::artifacts::ArtifactStore>>,
    ) -> Self {
        self.artifact_store = artifact_store;
        self
    }

    #[must_use]
    pub fn with_tool_names_or_defaults(mut self, tool_names: Option<Vec<String>>) -> Self {
        self.tool_selection = match tool_names {
            Some(tool_names) => ToolSelection::explicit_names(tool_names),
            None => ToolSelection::all_defaults(),
        };
        self
    }

    #[must_use]
    pub fn with_tool_names(mut self, tool_names: Vec<String>) -> Self {
        self.tool_selection = ToolSelection::explicit_names(tool_names);
        self
    }

    #[must_use]
    pub fn with_built_in_tools(mut self, builtins: Vec<BuiltInTool>) -> Self {
        self.tool_selection = ToolSelection::builtins(builtins);
        self
    }

    #[must_use]
    pub fn with_all_default_tools(mut self) -> Self {
        self.tool_selection = ToolSelection::all_defaults();
        self
    }

    #[must_use]
    pub const fn with_mutation_style(mut self, style: crate::agent::tools::MutationStyle) -> Self {
        self.mutation_style = style;
        self
    }

    pub fn with_credential_provider(
        mut self,
        credential_provider: Arc<dyn CredentialProvider>,
    ) -> Self {
        self.credential_provider = credential_provider;
        self
    }

    /// Allow service construction when the selected provider has no model key
    /// or its base URL is egress-blocked.
    ///
    /// Listen and session bookkeeping can proceed. A completion still fails
    /// with that provider's credential or egress error.
    #[must_use]
    pub const fn with_defer_missing_credentials(mut self, defer: bool) -> Self {
        self.defer_missing_credentials = defer;
        self
    }

    /// The provider's server URL. A later model change on the same provider
    /// keeps it; a change to another provider uses that provider's own
    /// environment URL or default.
    #[must_use]
    pub fn with_base_url(mut self, base_url: Option<String>) -> Self {
        self.base_url = base_url;
        self
    }

    #[must_use]
    pub const fn with_context_window(mut self, context_window: Option<usize>) -> Self {
        self.context_window = context_window;
        self
    }

    #[must_use]
    pub fn with_llm_cache(mut self, llm_cache: Option<Arc<LlmCache>>) -> Self {
        self.llm_cache = llm_cache;
        self
    }

    #[cfg(feature = "local-inference-base")]
    #[must_use]
    pub fn with_local_inference_config(
        mut self,
        config: Option<crate::llm::LocalInferenceConfig>,
    ) -> Self {
        self.local_inference_config = config;
        self
    }

    /// Enable binary (rkyv v1) session autosave under `dir` when building via [`AgentService::from_config`](crate::agent::service::AgentService::from_config).
    pub fn with_session_autosave_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.session_autosave_dir = Some(dir.into());
        self
    }

    /// Set the session spawner used by `sessions_spawn`.
    #[cfg(feature = "session-tools")]
    #[must_use]
    pub fn with_session_spawner(mut self, spawner: Option<Arc<dyn SessionSpawner>>) -> Self {
        self.session_spawner = spawner;
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
}

impl Default for AgentServiceConfig {
    fn default() -> Self {
        Self {
            provider: None,
            model: None,
            tool_selection: ToolSelection::default(),
            mutation_style: crate::agent::tools::MutationStyle::EditWrite,
            image_generation_provider: None,
            artifact_store: None,
            #[cfg(feature = "tools-rag")]
            shared_rag_tool: None,
            credential_provider: Arc::new(EnvironmentCredentialProvider::new()),
            defer_missing_credentials: false,
            base_url: None,
            context_window: None,
            #[cfg(feature = "tools-graph")]
            graph_backend: None,
            #[cfg(feature = "tools-graph")]
            frontier_prioritizer: None,
            #[cfg(feature = "reqwest")]
            shared_http_client: None,
            llm_cache: None,
            #[cfg(feature = "local-inference-base")]
            local_inference_config: None,
            session_autosave_dir: None,
            #[cfg(feature = "session-tools")]
            session_spawner: None,
            #[cfg(feature = "lsp")]
            lsp_config: None,
            #[cfg(feature = "lsp")]
            lsp_manager: None,
        }
    }
}
