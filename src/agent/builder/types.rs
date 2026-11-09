use crate::agent::memory::{MemoryPersistence, MemoryType};
use crate::traits::observability::ObservabilityHook;
use std::sync::Arc;

/// Configuration for AWS credentials.
///
/// This is an internal type used for bridging between `CredentialProvider`
/// and `ProviderFactory`. Prefer using the `CredentialProvider` trait instead.
#[derive(Clone)]
pub struct AwsCredentials {
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub session_token: Option<String>,
    pub region: Option<String>,
    pub bearer_token: Option<String>,
}

/// Memory configuration for [`AgentBuilder`](super::AgentBuilder).
///
/// Groups all memory-related settings so they can be passed as a single unit
/// rather than through many individual builder methods.
///
/// # Example
///
/// ```rust,no_run
/// use caliche::AgentBuilder;
/// use caliche::agent::builder::MemoryConfig;
/// use caliche::agent::memory::{MemoryBuilder, MemoryType, BufferMemoryConfig};
/// use std::sync::Arc;
///
/// # async fn example() -> anyhow::Result<()> {
/// let agent = AgentBuilder::new()
///     .with_memory_config(MemoryConfig {
///         memory_type: Some(MemoryBuilder::summary().max_tokens(8192).build()),
///         ..Default::default()
///     })
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Default)]
pub struct MemoryConfig {
    /// Memory strategy type (buffer, summary, entity, etc.).
    pub memory_type: Option<MemoryType>,
    /// Optional persistence backend for saving/loading conversation history.
    pub persistence: Option<Arc<dyn MemoryPersistence>>,
    /// Conversation ID used when persisting memory (required when `persistence` is set).
    pub conversation_id: Option<String>,
    /// Automatically save memory after each interaction when `persistence` is set.
    pub autosave: bool,
}

/// Session configuration for [`AgentBuilder`](super::AgentBuilder).
///
/// Groups session lifecycle settings (autosave, `AgentFS`) that control how
/// agent state is persisted between interactions.
///
/// # Example
///
/// ```rust,no_run
/// use caliche::AgentBuilder;
/// use caliche::agent::builder::SessionConfig;
///
/// # async fn example() -> anyhow::Result<()> {
/// let agent = AgentBuilder::new()
///     .with_session_config(SessionConfig {
///         autosave_path: Some("/tmp/sessions".into()),
///         ..Default::default()
///     })
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Default)]
pub struct SessionConfig {
    /// Directory path for session autosave snapshots.
    pub autosave_path: Option<std::path::PathBuf>,
    /// AgentFS session identifier for persistent, queryable agent state.
    #[cfg(feature = "agentfs")]
    pub agentfs_id: Option<String>,
    /// Custom AgentFS database path (overrides default `.agentfs/{id}.db`).
    #[cfg(feature = "agentfs")]
    pub agentfs_path: Option<std::path::PathBuf>,
}

/// Observability configuration for [`AgentBuilder`](super::AgentBuilder).
///
/// Groups settings for metrics collection, tracing hooks, and output observation.
///
/// # Example
///
/// ```rust,no_run
/// use caliche::AgentBuilder;
/// use caliche::agent::builder::ObservabilityConfig;
///
/// # async fn example() -> anyhow::Result<()> {
/// let agent = AgentBuilder::new()
///     .with_observability_config(ObservabilityConfig {
///         vector_metrics_enabled: true,
///         ..Default::default()
///     })
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Default)]
pub struct ObservabilityConfig {
    /// Enable metrics collection for vector operations (RAG, vector memory, etc.).
    pub vector_metrics_enabled: bool,
    /// Reserved for a future observability hook integration (currently ignored at build time).
    pub hook: Option<Arc<dyn ObservabilityHook>>,
}

// ----- Internal types (not part of the public API) -----

#[derive(Clone)]
pub(super) struct MemoryPersistenceConfig {
    pub(super) persistence: Arc<dyn MemoryPersistence>,
    pub(super) conversation_id: String,
    pub(super) autosave: bool,
}

#[derive(Clone)]
#[cfg_attr(
    not(all(feature = "tools-rag", feature = "tools-sql")),
    expect(
        dead_code,
        reason = "only the tools-rag and tools-sql build syncs the SQL schema"
    )
)]
pub struct SqlToolsConfig {
    pub(super) database_url: String,
    pub(super) namespace: String,
    pub(super) top_k: usize,
    pub(super) max_rows: usize,
}
