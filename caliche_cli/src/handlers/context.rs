//! Handler context and shared state.

use crate::storage::credentials::SecureCredentialManager;
use crate::storage::database::Database;
use crate::traits::factory::{DefaultServiceFactory, ServiceFactory};
use std::path::PathBuf;
use std::sync::Arc;

/// Context provided to command handlers.
///
/// Contains shared resources and configuration that handlers may need.
pub struct HandlerContext {
    /// Credential manager for API key access.
    pub credentials: Arc<SecureCredentialManager>,

    /// Database for session/history storage.
    pub database: Arc<Database>,

    /// Service factory for creating agent services.
    pub service_factory: Arc<dyn ServiceFactory>,

    /// Working directory for file operations.
    pub working_dir: PathBuf,

    /// Whether verbose output is enabled.
    pub verbose: bool,
}

impl HandlerContext {
    /// Create a new handler context.
    pub fn new(
        credentials: Arc<SecureCredentialManager>,
        database: Arc<Database>,
        working_dir: PathBuf,
    ) -> Self {
        Self {
            credentials,
            database,
            service_factory: Arc::new(DefaultServiceFactory::new()),
            working_dir,
            verbose: false,
        }
    }

    /// Set verbose mode.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set a custom service factory.
    pub fn with_service_factory(mut self, factory: Arc<dyn ServiceFactory>) -> Self {
        self.service_factory = factory;
        self
    }
}
