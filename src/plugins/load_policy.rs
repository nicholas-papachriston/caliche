//! Policy for handling plugin store load failures during agent build.

/// How [`crate::AgentBuilder`] should react when plugin definitions fail to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PluginLoadPolicy {
    /// Log a warning and continue without the failed plugins (default).
    #[default]
    Warn,
    /// Fail agent construction when plugins cannot be loaded.
    Fail,
}

impl PluginLoadPolicy {
    /// Parse from `CALICHE_PLUGIN_LOAD_POLICY` (`warn` or `fail`).
    pub fn from_env() -> Self {
        match std::env::var("CALICHE_PLUGIN_LOAD_POLICY")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("fail") => Self::Fail,
            _ => Self::Warn,
        }
    }
}
