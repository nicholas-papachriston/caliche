//! AgentBuilder skill auto-discovery defaults and token-lean build behavior.

use caliche::traits::credentials::{Credentials, StaticCredentialProvider};
use caliche::{AgentBuilder, ProviderType};
use std::sync::Arc;

#[test]
fn auto_discover_skills_defaults_to_disabled() {
    let _builder = AgentBuilder::new().auto_discover_skills(false);
    let _enabled = AgentBuilder::new().auto_discover_skills(true);
}

#[tokio::test]
async fn bare_builder_does_not_register_discovered_skill_index() {
    let service = AgentBuilder::new()
        .provider(ProviderType::OpenAI)
        .model("gpt-4o-mini".to_string())
        .credential_provider(Arc::new(StaticCredentialProvider::new(
            Credentials::ApiKey("test-key".to_string()),
        )))
        .build()
        .await
        .expect("bare builder should build without skill discovery");

    assert!(
        service.agent().skill_index().is_empty(),
        "SDK default must not inject workspace skill catalogs into the system prompt"
    );
}
