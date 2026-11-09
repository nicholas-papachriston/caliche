//! Testing utilities for Caliche

pub mod assertions;
pub mod helpers;
pub mod mock_provider;
#[cfg(test)]
pub(crate) mod planted_credential;
pub mod scenarios;

// Explicit exports from assertions module
pub use assertions::assert_contains;
#[cfg(feature = "tools-data")]
pub use assertions::assert_matches;

// Explicit exports from helpers module
pub use helpers::{assistant_message, random_string, system_message, test_message, user_message};

pub use mock_provider::MockProvider;
pub use scenarios::TestScenario;

use crate::agent::builder::AgentBuilder;
use crate::agent::service::sandbox::{AgentSandbox, MockBehavior};
use anyhow::Result;

/// Mock agent for testing
pub struct MockAgent {
    builder: AgentBuilder,
    sandbox: AgentSandbox,
}

impl MockAgent {
    pub fn new() -> Self {
        Self {
            builder: AgentBuilder::new(),
            sandbox: AgentSandbox::new(),
        }
    }

    pub fn with_mock_provider(
        mut self,
        provider: crate::llm::ProviderType,
        behavior: MockBehavior,
    ) -> Self {
        self.sandbox = self.sandbox.mock_provider(provider, behavior);
        self
    }

    pub async fn build(self) -> Result<crate::AgentService> {
        let mock_provider = self.sandbox.create_default_mock_provider();
        Box::pin(self.builder.provider_instance(mock_provider).build()).await
    }
}

impl Default for MockAgent {
    fn default() -> Self {
        Self::new()
    }
}
