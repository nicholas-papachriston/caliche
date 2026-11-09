//! Shared trait abstractions for extending Caliche
//!
//! This module contains the core traits that allow you to customize:
//! - Authentication and credential management
//! - Caching backends and strategies
//! - Retry behavior and error handling
//!
//! # Implementing Custom Providers
//!
//! Each trait includes detailed documentation and examples. See the
//! individual trait modules for implementation guides:
//!
//! - [`credentials`]: Custom authentication providers
//! - [`cache`]: Custom cache backends
//! - [`retry`]: Custom retry strategies
//!
//! # Example: Custom Credential Provider
//!
//! ```rust,no_run
//! use caliche::traits::credentials::{CredentialProvider, Credentials};
//! use async_trait::async_trait;
//! use anyhow::Result;
//!
//! #[derive(Debug)]
//! struct VaultCredentialProvider {
//!     vault_url: String,
//! }
//!
//! #[async_trait]
//! impl CredentialProvider for VaultCredentialProvider {
//!     async fn get_credentials(&self, provider: &str) -> Result<Credentials> {
//!         // Fetch from HashiCorp Vault
//!         # Ok(Credentials::ApiKey("dummy".to_string()))
//!     }
//!
//!     fn supports_provider(&self, provider: &str) -> bool {
//!         true
//!     }
//!
//!     fn source_name(&self) -> &str {
//!         "vault"
//!     }
//! }
//! ```

pub mod cache;
pub mod config;
pub mod credentials;
pub mod lifecycle;
pub mod observability;
pub mod retry;
pub mod stream;
pub mod validation;

pub use cache::{CacheBackend, CacheStats, TypedCacheBackend};
pub use config::{ConfigSource, TypedConfigSource};
pub use credentials::{CredentialProvider, Credentials};
pub use lifecycle::{HealthStatus, Resource, ResourceManager};
pub use observability::ObservabilityHook;
pub use retry::{RetryError, RetryStrategy, RetryStrategyBuilder};
pub use stream::StreamTransformer;
pub use validation::{InputValidator, OutputValidator, ValidationResult};
