//! Nested configuration section types (TOML schema building blocks).

use caliche::agent::session::{
    ShareConfig as RuntimeShareConfig, ShareMode, SharePolicy, ShareVisibility,
};
use caliche::llm::{AuthProfile, RotationStrategy};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use super::substitute::substitute_variables_in_string;

/// One credential profile for a provider (API key rotation / failover).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct AuthProfileConfig {
    /// Profile display name (e.g. "primary", "backup").
    pub name: String,
    /// API key value. Supports `{env:VAR}` placeholders (kept as EnvVar profiles).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Environment variable containing the API key.
    #[serde(default)]
    pub env_var: Option<String>,
}

impl AuthProfileConfig {
    /// Map TOML config into a runtime [`AuthProfile`].
    pub fn to_auth_profile(&self) -> AuthProfile {
        if let Some(var_name) = self.env_var.as_deref().filter(|s| !s.trim().is_empty()) {
            return AuthProfile::EnvVar {
                name: self.name.clone(),
                var_name: var_name.trim().to_string(),
            };
        }
        let key = self.api_key.as_deref().unwrap_or_default();
        if let Some(var_name) = extract_env_placeholder(key) {
            AuthProfile::EnvVar {
                name: self.name.clone(),
                var_name,
            }
        } else {
            let resolved = substitute_variables_in_string(key).into_owned();
            AuthProfile::ApiKey {
                name: self.name.clone(),
                key: resolved,
            }
        }
    }
}

/// TOML spelling requested by config documentation.
pub type AuthProfileToml = AuthProfileConfig;

/// Per-provider settings loaded from `[providers.<name>]` in TOML.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ProviderConfig {
    /// Auth profiles for credential rotation.
    #[serde(default)]
    pub auth_profiles: Vec<AuthProfileConfig>,
    /// Rotation strategy: "priority" (default), "round_robin", "least_recently_failed".
    #[serde(default)]
    pub rotation: Option<String>,
    /// Models that project or remote config may choose while this is the
    /// provider in the user config. Empty (the default) allows none.
    #[serde(default)]
    pub models: Vec<String>,
}

/// TOML spelling requested by config documentation.
pub type ProviderTomlConfig = ProviderConfig;

impl ProviderConfig {
    pub(crate) fn apply_variable_substitution(&mut self) {
        // Env placeholders are preserved as AuthProfile::EnvVar; only substitute
        // non-env api_key values at mapping time.
        if let Some(ref mut s) = self.rotation {
            *s = substitute_variables_in_string(s).into_owned();
        }
        for profile in &mut self.auth_profiles {
            if let Some(ref mut env_var) = profile.env_var {
                *env_var = substitute_variables_in_string(env_var).into_owned();
            }
        }
    }

    pub(crate) fn merge(&mut self, other: ProviderConfig) {
        if !other.auth_profiles.is_empty() {
            self.auth_profiles = other.auth_profiles;
        }
        if other.rotation.is_some() {
            self.rotation = other.rotation;
        }
        if !other.models.is_empty() {
            self.models = other.models;
        }
    }

    /// Parse configured rotation strategy (defaults to Priority).
    pub fn rotation_strategy(&self) -> RotationStrategy {
        match self
            .rotation
            .as_deref()
            .unwrap_or("priority")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "round_robin" | "roundrobin" | "rr" => RotationStrategy::RoundRobin,
            "least_recently_failed" | "lrf" => RotationStrategy::LeastRecentlyFailed,
            _ => RotationStrategy::Priority,
        }
    }

    /// Convert configured profiles to runtime auth profiles.
    pub fn to_auth_profiles(&self) -> Vec<AuthProfile> {
        self.auth_profiles
            .iter()
            .map(AuthProfileConfig::to_auth_profile)
            .collect()
    }
}

/// Convert provider TOML auth profiles into runtime auth profiles and rotation strategies.
pub fn auth_profiles_from_toml(
    providers: &HashMap<String, ProviderTomlConfig>,
) -> HashMap<String, (Vec<AuthProfile>, RotationStrategy)> {
    providers
        .iter()
        .map(|(name, cfg)| {
            (
                name.clone(),
                (cfg.to_auth_profiles(), cfg.rotation_strategy()),
            )
        })
        .collect()
}

/// Extract `VAR` from `{env:VAR}` (optionally surrounded by whitespace). Returns None otherwise.
pub(crate) fn extract_env_placeholder(value: &str) -> Option<String> {
    let trimmed = value.trim();
    let rest = trimmed.strip_prefix("{env:")?;
    let var = rest.strip_suffix('}')?;
    if var.is_empty() || var.contains('{') || var.contains('}') {
        return None;
    }
    // Only treat pure placeholders as EnvVar (no surrounding text).
    if trimmed == format!("{{env:{var}}}") {
        Some(var.to_string())
    } else {
        None
    }
}

/// Memory configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct MemoryConfig {
    /// Memory strategy: "composite" (default), "buffer", "summary", "entity", "vector", "enhanced"
    #[serde(default)]
    pub strategy: Option<String>,
    /// Enable auto-compact
    #[serde(default)]
    pub auto_compact: Option<bool>,
    /// Auto-compact threshold (0.0-1.0)
    #[serde(default)]
    pub auto_compact_threshold: Option<f64>,
    /// Route REPL turns through AgentHarness for checkpointing and discipline metrics.
    #[serde(default)]
    pub use_harness_turns: Option<bool>,
}

impl MemoryConfig {
    pub(crate) fn apply_variable_substitution(&mut self) {
        if let Some(ref mut s) = self.strategy {
            *s = substitute_variables_in_string(s).into_owned();
        }
    }

    pub(crate) fn merge(&mut self, other: MemoryConfig) {
        if other.strategy.is_some() {
            self.strategy = other.strategy;
        }
        if other.auto_compact.is_some() {
            self.auto_compact = other.auto_compact;
        }
        if other.auto_compact_threshold.is_some() {
            self.auto_compact_threshold = other.auto_compact_threshold;
        }
        if other.use_harness_turns.is_some() {
            self.use_harness_turns = other.use_harness_turns;
        }
    }
}

/// Planning and task decomposition
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct PlanningConfig {
    /// Auto-decompose complex tasks into sub-tasks (code command only; default true).
    #[serde(default)]
    pub auto_decompose: Option<bool>,
}

/// RepoMap: token-budgeted codebase overview injected into agent context (code command).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct RepoMapConfig {
    /// Enable RepoMap (default: true).
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Max tokens for the codebase overview (default: 2000).
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

impl RepoMapConfig {
    pub(crate) fn merge(&mut self, other: RepoMapConfig) {
        if other.enabled.is_some() {
            self.enabled = other.enabled;
        }
        if other.max_tokens.is_some() {
            self.max_tokens = other.max_tokens;
        }
    }
}

/// Context configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ContextConfig {
    /// Files to always include in context
    #[serde(default)]
    pub paths: Option<Vec<String>>,
}

impl ContextConfig {
    pub(crate) fn apply_variable_substitution(&mut self) {
        if let Some(ref mut paths) = self.paths {
            for p in paths.iter_mut() {
                *p = substitute_variables_in_string(p).into_owned();
            }
        }
    }

    pub(crate) fn merge(&mut self, other: ContextConfig) {
        if other.paths.is_some() {
            self.paths = other.paths;
        }
    }
}

/// Default behavior when no bash policy rules match (or when no rules are configured).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum BashDefaultMode {
    /// Allow execution when no rule matches (default).
    #[default]
    Allow,
    /// Deny all bash commands when no rule matches; use a single catch-all forbidden rule.
    Deny,
}

/// Permissions configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct PermissionsConfig {
    /// Tools to auto-approve
    #[serde(default)]
    pub allowed_tools: Option<Vec<String>>,
    /// Bash command policy rules (allow / prompt / forbidden per pattern).
    /// Last matching rule wins. Order rules with forbidden first, then prompt, then allow
    /// to approximate forbidden > prompt > allow precedence.
    /// Example: [[permissions.bash]] pattern = ["git", "status"] decision = "allow"
    #[serde(default)]
    #[schemars(skip)]
    pub bash: Option<Vec<caliche::agent::hooks::BashRule>>,
    /// Default when no bash rule matches: "allow" (default) or "deny" (deny by default).
    #[serde(default)]
    pub bash_default: Option<BashDefaultMode>,
}

impl PermissionsConfig {
    pub(crate) fn apply_variable_substitution(&mut self) {
        if let Some(ref mut arr) = self.allowed_tools {
            for s in arr.iter_mut() {
                *s = substitute_variables_in_string(s).into_owned();
            }
        }
        if let Some(ref mut rules) = self.bash {
            for rule in rules.iter_mut() {
                for pos in &mut rule.pattern {
                    match pos {
                        caliche::agent::hooks::PatternPos::One(s) => {
                            *s = substitute_variables_in_string(s).into_owned();
                        }
                        caliche::agent::hooks::PatternPos::Many(v) => {
                            for s in v.iter_mut() {
                                *s = substitute_variables_in_string(s).into_owned();
                            }
                        }
                    }
                }
                if let Some(ref mut j) = rule.justification {
                    *j = substitute_variables_in_string(j).into_owned();
                }
                if let Some(ref mut arr) = rule.match_examples {
                    for s in arr.iter_mut() {
                        *s = substitute_variables_in_string(s).into_owned();
                    }
                }
                if let Some(ref mut arr) = rule.not_match_examples {
                    for s in arr.iter_mut() {
                        *s = substitute_variables_in_string(s).into_owned();
                    }
                }
            }
        }
    }

    pub(crate) fn merge(&mut self, other: PermissionsConfig) {
        if other.allowed_tools.is_some() {
            self.allowed_tools = other.allowed_tools;
        }
        if other.bash.is_some() {
            self.bash = other.bash;
        }
        if other.bash_default.is_some() {
            self.bash_default = other.bash_default;
        }
    }

    /// Validate bash rules' match/not_match examples. Call after config load.
    pub fn validate_bash_rules(&self) -> Result<(), crate::errors::CliError> {
        if let Some(ref rules) = self.bash {
            for (i, rule) in rules.iter().enumerate() {
                rule.validate_examples().map_err(|e| {
                    crate::errors::CliError::Config(format!(
                        "permissions.bash rule {}: {}",
                        i + 1,
                        e
                    ))
                })?;
            }
        }
        Ok(())
    }
}

/// Pre-completion checklist middleware config
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct PreCompletionChecklistMiddlewareConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub verify_commands: Option<Vec<String>>,
    #[serde(default)]
    pub require_verify_within_last_n_turns: Option<usize>,
    #[serde(default)]
    pub max_injections_per_session: Option<usize>,
}

/// Loop detection middleware config
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct LoopDetectionMiddlewareConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub edit_threshold: Option<usize>,
    #[serde(default)]
    pub max_warnings: Option<usize>,
}

/// Environment onboarding middleware config
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct EnvironmentOnboardingMiddlewareConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub max_depth: Option<usize>,
    #[serde(default)]
    pub include_runtimes: Option<Vec<String>>,
    #[serde(default)]
    pub include_tools: Option<Vec<String>>,
    #[serde(default)]
    pub inject_as: Option<String>,
}

/// Middleware configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct MiddlewareConfig {
    #[serde(default)]
    pub pre_completion_checklist: Option<PreCompletionChecklistMiddlewareConfig>,
    #[serde(default)]
    pub loop_detection: Option<LoopDetectionMiddlewareConfig>,
    #[serde(default)]
    pub environment_onboarding: Option<EnvironmentOnboardingMiddlewareConfig>,
}

impl PreCompletionChecklistMiddlewareConfig {
    /// Resolve against core defaults (`enabled` defaults to true).
    pub fn resolve(&self) -> Option<caliche::agent::hooks::PreCompletionChecklistConfig> {
        if !self.enabled.unwrap_or(true) {
            return None;
        }
        let mut config = caliche::agent::hooks::PreCompletionChecklistConfig::default();
        if let Some(commands) = &self.verify_commands {
            config.verify_commands = commands.clone();
        }
        if let Some(n) = self.require_verify_within_last_n_turns {
            config.require_verify_within_last_n_turns = n;
        }
        if let Some(n) = self.max_injections_per_session {
            config.max_injections_per_session = n;
        }
        Some(config)
    }
}

impl LoopDetectionMiddlewareConfig {
    /// Resolve against core defaults (`enabled` defaults to true).
    pub fn resolve(&self) -> Option<caliche::agent::hooks::LoopDetectionConfig> {
        if !self.enabled.unwrap_or(true) {
            return None;
        }
        let mut config = caliche::agent::hooks::LoopDetectionConfig::default();
        if let Some(n) = self.edit_threshold {
            config.edit_threshold = n;
        }
        if let Some(n) = self.max_warnings {
            config.max_warnings = n;
        }
        Some(config)
    }
}

impl EnvironmentOnboardingMiddlewareConfig {
    /// Resolve against core defaults (`enabled` defaults to true).
    pub fn resolve(&self) -> Option<caliche::agent::hooks::EnvironmentOnboardingConfig> {
        if !self.enabled.unwrap_or(true) {
            return None;
        }
        let mut config = caliche::agent::hooks::EnvironmentOnboardingConfig::default();
        if let Some(n) = self.max_depth {
            config.max_depth = n;
        }
        if let Some(runtimes) = &self.include_runtimes {
            config.include_runtimes = runtimes.clone();
        }
        if let Some(tools) = &self.include_tools {
            config.include_tools = tools.clone();
        }
        Some(config)
    }
}

impl MiddlewareConfig {
    /// Resolve each middleware section, filling missing sections from core defaults.
    pub fn resolved_hooks(
        &self,
    ) -> (
        Option<caliche::agent::hooks::PreCompletionChecklistConfig>,
        Option<caliche::agent::hooks::LoopDetectionConfig>,
        Option<caliche::agent::hooks::EnvironmentOnboardingConfig>,
    ) {
        let pcc = self
            .pre_completion_checklist
            .clone()
            .unwrap_or_default()
            .resolve();
        let ld = self.loop_detection.clone().unwrap_or_default().resolve();
        let eo = self
            .environment_onboarding
            .clone()
            .unwrap_or_default()
            .resolve();
        (pcc, ld, eo)
    }
}

/// Orchestration configuration (multi-agent coordinator, supervision).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct OrchestrationConfig {
    /// OTP-style per-agent supervision (restart on failure).
    #[serde(default)]
    #[schemars(skip)]
    pub supervision: Option<caliche::config::SupervisionConfig>,
    /// Which subagents an `@mention` may start, as glob pattern -> `allow` or `deny`.
    ///
    /// Example: `task_acl = { "*" = "deny", "reviewer" = "allow" }`. The most specific
    /// pattern wins. When unset, every subagent profile can be mentioned.
    #[serde(default)]
    pub task_acl: Option<BTreeMap<String, String>>,
}

impl OrchestrationConfig {
    pub(crate) fn merge(&mut self, other: OrchestrationConfig) {
        if other.supervision.is_some() {
            self.supervision = other.supervision;
        }
        if other.task_acl.is_some() {
            self.task_acl = other.task_acl;
        }
    }

    /// Task invocation policy for REPL `@mention` dispatch.
    ///
    /// An unreadable ACL is reported and treated as deny-all, so a typo cannot silently
    /// open the gate.
    pub fn mention_policy(&self) -> caliche::orchestration::TaskInvocationPolicy {
        use caliche::orchestration::TaskInvocationPolicy;

        let Some(map) = self.task_acl.as_ref() else {
            return TaskInvocationPolicy::allow_all();
        };
        match TaskInvocationPolicy::from_config_map(map) {
            Ok(policy) => policy,
            Err(error) => {
                tracing::error!(
                    error = %error,
                    "Invalid orchestration.task_acl; refusing every @mention until it is fixed"
                );
                TaskInvocationPolicy::deny_all()
            }
        }
    }
}

/// Default provider and model for spawned subagents.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct SubagentConfig {
    /// Provider name (`anthropic`, `openai`, `openrouter`, ...).
    #[serde(default)]
    pub provider: Option<String>,
    /// Model id or `provider/model`.
    #[serde(default)]
    pub model: Option<String>,
    /// Run spawned children with the methodical complex-task loop.
    #[serde(default)]
    pub complex: Option<bool>,
}

impl SubagentConfig {
    pub(crate) fn merge(&mut self, other: SubagentConfig) {
        if other.provider.is_some() {
            self.provider = other.provider;
        }
        if other.model.is_some() {
            self.model = other.model;
        }
        if other.complex.is_some() {
            self.complex = other.complex;
        }
    }
}

/// LSP configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct LspConfig {
    /// Language server configurations
    #[serde(default)]
    pub servers: Option<HashMap<String, LspServerConfig>>,
}

/// LSP server configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct LspServerConfig {
    /// Command to run
    #[serde(default)]
    pub command: Option<String>,
    /// Arguments
    #[serde(default)]
    pub args: Option<Vec<String>>,
    /// Environment variables
    #[serde(default)]
    pub env: Option<HashMap<String, String>>,
    /// Whether enabled
    #[serde(default)]
    pub enabled: Option<bool>,
}

impl LspServerConfig {
    pub(crate) fn apply_variable_substitution(&mut self) {
        if let Some(ref mut s) = self.command {
            *s = substitute_variables_in_string(s).into_owned();
        }
        if let Some(ref mut arr) = self.args {
            for s in arr.iter_mut() {
                *s = substitute_variables_in_string(s).into_owned();
            }
        }
        if let Some(ref mut env_map) = self.env {
            for v in env_map.values_mut() {
                *v = substitute_variables_in_string(v).into_owned();
            }
        }
    }

    pub(crate) fn merge(&mut self, other: LspServerConfig) {
        if other.command.is_some() {
            self.command = other.command;
        }
        if other.args.is_some() {
            self.args = other.args;
        }
        if other.env.is_some() {
            self.env = other.env;
        }
        if other.enabled.is_some() {
            self.enabled = other.enabled;
        }
    }
}

impl LspConfig {
    pub(crate) fn apply_variable_substitution(&mut self) {
        if let Some(ref mut servers) = self.servers {
            for server in servers.values_mut() {
                server.apply_variable_substitution();
            }
        }
    }

    pub(crate) fn merge(&mut self, other: LspConfig) {
        if let Some(ref other_servers) = other.servers {
            let servers = self.servers.get_or_insert_with(HashMap::new);
            for (k, v) in other_servers {
                servers
                    .entry(k.clone())
                    .or_insert_with(LspServerConfig::default)
                    .merge(v.clone());
            }
        }
    }
}

/// Configuration for a single auto-formatter.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct FormatterEntry {
    /// Command to run. `$FILE` is replaced with the absolute file path.
    /// Example: "rustfmt $FILE"
    pub command: String,
    /// File extensions this formatter applies to (without leading dot).
    /// Example: ["rs"] or ["ts", "tsx", "js"]
    #[serde(default)]
    pub extensions: Vec<String>,
    /// Set to true to disable this formatter without removing the config.
    #[serde(default)]
    pub disabled: bool,
}

impl FormatterEntry {
    pub(crate) fn apply_variable_substitution(&mut self) {
        self.command = substitute_variables_in_string(&self.command).into_owned();
    }
}

/// Image generation provider/model configuration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct ImageGenerationConfig {
    /// Image generation provider, currently "openai" or "google".
    pub provider: Option<String>,
    /// Provider-specific image generation model ID.
    pub model: Option<String>,
}

impl ImageGenerationConfig {
    pub(crate) fn apply_variable_substitution(&mut self) {
        if let Some(ref mut provider) = self.provider {
            *provider = substitute_variables_in_string(provider).into_owned();
        }
        if let Some(ref mut model) = self.model {
            *model = substitute_variables_in_string(model).into_owned();
        }
    }

    pub(crate) fn merge(&mut self, other: ImageGenerationConfig) {
        if other.provider.is_some() {
            self.provider = other.provider;
        }
        if other.model.is_some() {
            self.model = other.model;
        }
    }
}

/// Generated artifact output configuration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct ArtifactsConfig {
    /// Directory for generated artifacts. Relative paths resolve against the workspace root.
    pub dir: Option<PathBuf>,
}

impl ArtifactsConfig {
    pub(crate) fn apply_variable_substitution(&mut self) {
        if let Some(ref mut dir) = self.dir
            && let Some(value) = dir.as_os_str().to_str()
        {
            *dir = PathBuf::from(substitute_variables_in_string(value).into_owned());
        }
    }

    pub(crate) fn merge(&mut self, other: ArtifactsConfig) {
        if other.dir.is_some() {
            self.dir = other.dir;
        }
    }
}

/// Distillation-based learning loop configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct LearningConfig {
    /// Enable observe -> distill -> reuse/refine learning.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Number of tool calls between learning checkpoints.
    #[serde(default)]
    pub interval_tool_calls: Option<usize>,
    /// Minimum tool calls before a trajectory can become a skill.
    #[serde(default)]
    pub min_tool_calls_for_skill: Option<usize>,
    /// Path for durable MEMORY.md nudges. Relative paths resolve against the workspace.
    #[serde(default)]
    pub memory_file_path: Option<PathBuf>,
    /// Automatically create new skill plugins.
    #[serde(default)]
    pub auto_create_skills: Option<bool>,
    /// Automatically patch existing skill plugins after better recoveries.
    #[serde(default)]
    pub auto_patch_skills: Option<bool>,
    /// Maximum auto-created skills per session.
    #[serde(default)]
    pub max_skills_per_session: Option<usize>,
    /// Prefer weakest valid skill hypotheses over most-specific ones.
    #[serde(default)]
    pub prefer_weak_skills: Option<bool>,
    /// Require parent-holdout generalisation before create/patch.
    #[serde(default)]
    pub require_holdout_pass: Option<bool>,
    /// Minimum holdout extent in `[0.0, 1.0]` to accept a distilled skill.
    #[serde(default)]
    pub min_holdout_extent: Option<f64>,
}

impl LearningConfig {
    pub(crate) fn apply_variable_substitution(&mut self) {
        if let Some(ref mut path) = self.memory_file_path
            && let Some(value) = path.as_os_str().to_str()
        {
            *path = PathBuf::from(substitute_variables_in_string(value).into_owned());
        }
    }

    pub(crate) fn merge(&mut self, other: LearningConfig) {
        if other.enabled.is_some() {
            self.enabled = other.enabled;
        }
        if other.interval_tool_calls.is_some() {
            self.interval_tool_calls = other.interval_tool_calls;
        }
        if other.min_tool_calls_for_skill.is_some() {
            self.min_tool_calls_for_skill = other.min_tool_calls_for_skill;
        }
        if other.memory_file_path.is_some() {
            self.memory_file_path = other.memory_file_path;
        }
        if other.auto_create_skills.is_some() {
            self.auto_create_skills = other.auto_create_skills;
        }
        if other.auto_patch_skills.is_some() {
            self.auto_patch_skills = other.auto_patch_skills;
        }
        if other.max_skills_per_session.is_some() {
            self.max_skills_per_session = other.max_skills_per_session;
        }
        if other.prefer_weak_skills.is_some() {
            self.prefer_weak_skills = other.prefer_weak_skills;
        }
        if other.require_holdout_pass.is_some() {
            self.require_holdout_pass = other.require_holdout_pass;
        }
        if other.min_holdout_extent.is_some() {
            self.min_holdout_extent = other.min_holdout_extent;
        }
    }
}

/// Session share configuration (`[share]` in TOML).
///
/// Example:
/// ```toml
/// [share]
/// policy = "manual"          # disabled | manual | auto
/// backend = "memory"         # memory | remote
/// visibility = "public"      # public | authenticated
/// backend_url = "https://share.example.com"
/// redact_patterns = ["sk-.*"]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct ShareTomlConfig {
    /// Product policy: disabled | manual | auto.
    #[serde(default)]
    pub policy: Option<String>,
    /// Storage backend: memory | remote.
    #[serde(default)]
    pub backend: Option<String>,
    /// Default share visibility: public | authenticated.
    #[serde(default)]
    pub visibility: Option<String>,
    /// Optional absolute share base URL for create responses.
    #[serde(default)]
    pub backend_url: Option<String>,
    /// Regex patterns redacted from shared transcripts.
    #[serde(default)]
    pub redact_patterns: Option<Vec<String>>,
}

impl ShareTomlConfig {
    pub(crate) fn apply_variable_substitution(&mut self) {
        if let Some(ref mut url) = self.backend_url {
            *url = substitute_variables_in_string(url).into_owned();
        }
        if let Some(ref mut patterns) = self.redact_patterns {
            for pattern in patterns.iter_mut() {
                *pattern = substitute_variables_in_string(pattern).into_owned();
            }
        }
    }

    pub(crate) fn merge(&mut self, other: ShareTomlConfig) {
        if other.policy.is_some() {
            self.policy = other.policy;
        }
        if other.backend.is_some() {
            self.backend = other.backend;
        }
        if other.visibility.is_some() {
            self.visibility = other.visibility;
        }
        if other.backend_url.is_some() {
            self.backend_url = other.backend_url;
        }
        if other.redact_patterns.is_some() {
            self.redact_patterns = other.redact_patterns;
        }
    }

    /// Map TOML config into the runtime share configuration.
    pub fn to_runtime(&self) -> anyhow::Result<RuntimeShareConfig> {
        let mut config = RuntimeShareConfig::default();
        if let Some(policy) = self.policy.as_deref() {
            config.policy = match policy.trim().to_ascii_lowercase().as_str() {
                "disabled" => SharePolicy::Disabled,
                "manual" => SharePolicy::Manual,
                "auto" => SharePolicy::Auto,
                other => anyhow::bail!("unknown share.policy '{other}'"),
            };
        }
        if let Some(backend) = self.backend.as_deref() {
            config.mode = match backend.trim().to_ascii_lowercase().as_str() {
                "memory" => ShareMode::Memory,
                "remote" => ShareMode::Remote,
                other => anyhow::bail!("unknown share.backend '{other}'"),
            };
        }
        if let Some(visibility) = self.visibility.as_deref() {
            config.default_visibility = match visibility.trim().to_ascii_lowercase().as_str() {
                "public" => ShareVisibility::Public,
                "authenticated" => ShareVisibility::Authenticated,
                other => anyhow::bail!("unknown share.visibility '{other}'"),
            };
        }
        if let Some(url) = self.backend_url.clone() {
            config.backend_url = Some(url);
        }
        if let Some(patterns) = self.redact_patterns.clone() {
            config.redact_patterns = patterns;
        }
        Ok(config)
    }
}
