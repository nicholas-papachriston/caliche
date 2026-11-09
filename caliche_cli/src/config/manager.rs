//! Configuration loading, caching, and file helpers.

use super::Config;
use super::sections::PermissionsConfig;
use super::sources::{
    ConfigSource, DatabaseConfigSource, EnvConfigSource, FileConfigSource, ProjectTomlConfigSource,
    RemoteConfigSource,
};
use crate::errors::{CliError, CliResult};
use std::path::PathBuf;

/// Configuration manager that loads from multiple sources.
pub struct ConfigManager {
    sources: Vec<Box<dyn ConfigSource>>,
    cached: Option<Config>,
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigManager {
    /// Create a new config manager with default sources.
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            cached: None,
        }
    }

    /// Create with standard sources.
    pub fn with_standard_sources() -> Self {
        let mut manager = Self::new();
        // Add sources in priority order (lowest first, highest last)
        manager.add_source(Box::new(RemoteConfigSource::new()));
        manager.add_source(Box::new(DatabaseConfigSource::new()));
        manager.add_source(Box::new(FileConfigSource::user_config()));
        manager.add_source(Box::new(ProjectTomlConfigSource::new()));
        manager.add_source(Box::new(EnvConfigSource::new()));
        manager
    }

    /// Add a configuration source.
    pub fn add_source(&mut self, source: Box<dyn ConfigSource>) {
        self.sources.push(source);
        self.cached = None; // Invalidate cache
    }

    /// Load and merge configuration from all sources.
    pub async fn load(&mut self) -> CliResult<&Config> {
        if self.cached.is_none() {
            let (merged, _) = self.merge_sources().await?;
            self.cached = Some(merged);
        }
        self.cached_or_error()
    }

    /// Load and merge configuration, returning the config and names of sources that contributed.
    pub async fn load_with_sources(&mut self) -> CliResult<(&Config, Vec<String>)> {
        let (merged, contributed) = self.merge_sources().await?;
        self.cached = Some(merged);
        Ok((self.cached_or_error()?, contributed))
    }

    /// Merge every source in order. Untrusted sources are filtered against
    /// the trusted ones merged first, so an untrusted model must be one the
    /// user config allows for its provider.
    async fn merge_sources(&self) -> CliResult<(Config, Vec<String>)> {
        let mut loaded = Vec::new();
        for source in &self.sources {
            match source.load().await {
                Ok(Some(config)) => loaded.push((source, config)),
                Ok(None) => {}
                Err(err) => return Err(source_load_error(source.name(), &err)),
            }
        }

        let mut trusted = Config::new();
        for (source, config) in &mut loaded {
            if source.is_trusted_for_local_io() {
                config.apply_variable_substitution();
                trusted.merge(config.clone());
            }
        }

        let mut merged = Config::new();
        let mut contributed = Vec::new();
        for (source, mut config) in loaded {
            if !source.is_trusted_for_local_io() {
                config.keep_untrusted_keys(&trusted);
            }
            merged.merge(config);
            contributed.push(source.name().to_string());
        }

        merged.validate()?;
        Ok((merged, contributed))
    }

    /// Names of the sources whose config is filtered as untrusted.
    #[cfg(test)]
    pub(crate) fn untrusted_source_names(&self) -> Vec<&'static str> {
        self.sources
            .iter()
            .filter(|source| !source.is_trusted_for_local_io())
            .map(|source| source.name())
            .collect()
    }

    fn cached_or_error(&self) -> CliResult<&Config> {
        self.get().ok_or_else(|| {
            crate::errors::CliError::Config(
                "internal error: configuration was not cached after successful merge".to_string(),
            )
        })
    }

    /// Force reload configuration.
    pub async fn reload(&mut self) -> CliResult<&Config> {
        self.cached = None;
        self.load().await
    }

    /// Get the cached configuration (if loaded).
    pub fn get(&self) -> Option<&Config> {
        self.cached.as_ref()
    }

    /// Get a specific value from the configuration.
    pub async fn get_value<T: serde::de::DeserializeOwned>(
        &mut self,
        key: &str,
    ) -> CliResult<Option<T>> {
        let config = self.load().await?;
        Ok(config.get_custom(key))
    }
}

fn source_load_error(source: &str, err: &CliError) -> CliError {
    CliError::Config(format!("{source}: {err}"))
}

/// Check if any project or user config file exists on disk.
pub fn has_any_file_config() -> bool {
    ProjectTomlConfigSource::find_config_file().is_some()
        || FileConfigSource::user_config_path().exists()
}

/// Load the merged standard CLI config once and return an owned snapshot.
///
/// This is the preferred runtime entrypoint for ad-hoc call sites to avoid
/// repeating `ConfigManager::with_standard_sources()` boilerplate.
pub async fn load_standard_config() -> Option<Config> {
    load_standard_config_strict().await.ok()
}

/// Load merged config and surface file or project parse errors.
///
/// Callers that build an agent must use this so a broken config cannot
/// exit 0 with a silent default.
pub async fn load_standard_config_strict() -> CliResult<Config> {
    let mut manager = ConfigManager::with_standard_sources();
    Ok(manager.load().await?.clone())
}

/// Append a bash allow rule to the config file at the given path.
/// Creates the file and [permissions.bash] section if they do not exist.
pub async fn append_bash_allow_rule(
    path: &std::path::Path,
    rule_prefix: &[String],
) -> anyhow::Result<()> {
    use caliche::agent::hooks::{BashDecision, BashRule, PatternPos};

    if rule_prefix.is_empty() {
        anyhow::bail!("Cannot persist empty rule prefix");
    }

    let mut config = if path.exists() {
        let content = tokio::fs::read_to_string(path).await?;
        match toml::from_str::<Config>(&content) {
            Ok(cfg) => cfg,
            Err(err) => {
                let backup = path.with_extension("toml.bak");
                if let Err(copy_err) = tokio::fs::copy(path, &backup).await {
                    tracing::warn!(
                        "Failed to write backup of {} to {}: {}",
                        path.display(),
                        backup.display(),
                        copy_err
                    );
                }
                anyhow::bail!(
                    "Cannot append bash rule: existing {} could not be parsed ({}). \
                     A backup was written to {}. Fix the file and retry.",
                    path.display(),
                    err,
                    backup.display()
                );
            }
        }
    } else {
        Config::new()
    };

    let perm = config
        .permissions
        .get_or_insert_with(PermissionsConfig::default);
    let rules = perm.bash.get_or_insert_with(Vec::new);
    let pattern: Vec<PatternPos> = rule_prefix
        .iter()
        .map(|s| PatternPos::One(s.clone()))
        .collect();
    rules.push(BashRule {
        pattern,
        decision: BashDecision::Allow,
        justification: Some("Added via approval flow (always)".to_string()),
        match_examples: None,
        not_match_examples: None,
    });

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let content = toml::to_string_pretty(&config)?;
    tokio::fs::write(path, content).await?;
    Ok(())
}

/// Minimal config template content.
const CONFIG_TEMPLATE: &str = r#"# Caliche configuration
# See: caliche config schema --output - for full schema

[provider]
# provider = "openai"
# model = "gpt-4o"
"#;

/// Interactive config file wizard. Prompts to create caliche.toml or user config if none exist.
pub fn config_file_wizard() -> anyhow::Result<()> {
    use std::io::{self, BufRead, Write};

    if has_any_file_config() {
        return Ok(());
    }

    println!("\n\x1b[33mNo config file found.\x1b[0m");
    print!("Create one? [Y/n]: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().lock().read_line(&mut input)?;
    let choice = input.trim().to_lowercase();
    if choice == "n" || choice == "no" {
        return Ok(());
    }

    println!("Where to create the config?");
    println!("  \x1b[1m[p]\x1b[0m Project — caliche.toml in current directory");
    println!("  \x1b[1m[g]\x1b[0m Global — ~/.config/caliche/config.toml");
    print!("Choice [p/g]: ");
    io::stdout().flush()?;
    input.clear();
    io::stdin().lock().read_line(&mut input)?;
    let scope = input.trim().to_lowercase();

    let (path, path_display) = if scope == "g" || scope == "global" {
        let p = FileConfigSource::user_config_path();
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        (p.clone(), p.display().to_string())
    } else {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let p = cwd.join("caliche.toml");
        (p.clone(), p.display().to_string())
    };

    std::fs::write(&path, CONFIG_TEMPLATE)?;
    println!("\x1b[32mCreated config at {}\x1b[0m\n", path_display);
    Ok(())
}

/// Builder for ConfigManager.
pub struct ConfigManagerBuilder {
    sources: Vec<Box<dyn ConfigSource>>,
}

impl Default for ConfigManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigManagerBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// Add an environment config source.
    pub fn with_env(mut self) -> Self {
        self.sources.push(Box::new(EnvConfigSource::new()));
        self
    }

    /// Add a project config source (caliche.toml / .caliche/config.toml).
    pub fn with_project(mut self) -> Self {
        self.sources.push(Box::new(ProjectTomlConfigSource::new()));
        self
    }

    /// Add a user config source.
    pub fn with_user_config(mut self) -> Self {
        self.sources.push(Box::new(FileConfigSource::user_config()));
        self
    }

    /// Add a database config source.
    pub fn with_database(mut self) -> Self {
        self.sources.push(Box::new(DatabaseConfigSource::new()));
        self
    }

    /// Add a custom config source.
    pub fn with_source(mut self, source: Box<dyn ConfigSource>) -> Self {
        self.sources.push(source);
        self
    }

    /// Build the config manager.
    pub fn build(self) -> ConfigManager {
        ConfigManager {
            sources: self.sources,
            cached: None,
        }
    }
}
