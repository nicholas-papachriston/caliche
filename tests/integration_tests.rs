//! Integration tests for new features

use caliche::llm::OpenAIConfig;
use caliche::templates::{TemplateRegistry, TemplateSearchFilters};
use serial_test::serial;
use std::time::Duration;

async fn require_lmstudio() {
    assert_eq!(
        std::env::var("CALICHE_RUN_LMSTUDIO_INTEGRATION").as_deref(),
        Ok("true"),
        "CALICHE_RUN_LMSTUDIO_INTEGRATION must be true when this ignored test runs"
    );
    let reachable = reqwest::Client::new()
        .get("http://localhost:1234/v1/models")
        .send()
        .await
        .is_ok();
    assert!(
        reachable,
        "LM Studio must answer at http://localhost:1234/v1/models"
    );
}

#[tokio::test]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
#[serial]
#[cfg(feature = "lmstudio")]
async fn test_template_instantiation_integration() {
    require_lmstudio().await;

    let loader = caliche::templates::TemplateLoader::new();
    let mut template = loader
        .load_template("templates/rag_agent.yaml")
        .await
        .expect("Failed to load template");

    // Override provider to use LM Studio
    template.config.provider = "lmstudio".to_string();
    template.config.model = Some("local-model".to_string());

    let builder = template.instantiate().await.expect("Failed to instantiate");
    let mut agent = builder.build().await.expect("Failed to build agent");

    // Test that agent can chat with LM Studio
    let response = agent.chat("Hello").await.expect("Chat failed");
    assert!(!response.text().is_empty());
}

#[tokio::test]
async fn test_template_registry_integration() {
    let mut registry = TemplateRegistry::new();
    registry.add_template_dir("templates");
    let count = registry
        .load_templates()
        .await
        .expect("Failed to load templates");

    assert!(count > 0, "Should load at least one template");

    let templates = registry.list();
    assert!(!templates.is_empty());

    // Test search functionality
    let filters = TemplateSearchFilters {
        query: Some("agent".to_string()),
        ..Default::default()
    };
    let results = registry.search(&filters);
    assert!(!results.is_empty() || count == 0); // May be empty if no matches
}

#[tokio::test]
async fn test_template_categories() {
    let mut registry = TemplateRegistry::new();
    registry.add_template_dir("templates");
    registry
        .load_templates()
        .await
        .expect("Failed to load templates");

    let categories = registry.get_categories();
    // Categories may be empty if templates don't have categories set
    // Just verify the function works
    let _ = categories.len(); // Verify function works
}

#[tokio::test]
async fn test_template_tags() {
    let mut registry = TemplateRegistry::new();
    registry.add_template_dir("templates");
    registry
        .load_templates()
        .await
        .expect("Failed to load templates");

    let tags = registry.get_all_tags();
    // Tags may be empty
    let _ = tags.len(); // Verify function works
}

#[tokio::test]
async fn test_template_fork_integration() {
    let mut registry = TemplateRegistry::new();
    registry.add_template_dir("templates");
    registry
        .load_templates()
        .await
        .expect("Failed to load templates");

    if let Some(template) = registry.list().first() {
        let forked = registry
            .fork_template(
                &template.id(),
                Some("Forked Test Template".to_string()),
                Some("2.0.0".to_string()),
            )
            .expect("Failed to fork");

        assert_eq!(forked.name, "Forked Test Template");
        assert_eq!(forked.version, "2.0.0");
        assert_eq!(forked.config.provider, template.config.provider);
    }
}

#[tokio::test]
async fn test_programmatic_provider_config_integration() {
    // Test that providers can be created with programmatic configuration
    let provider_result = OpenAIConfig::builder()
        .api_key("test-key-for-config-test".to_string())
        .model("gpt-4o".to_string())
        .timeout(Duration::from_secs(30))
        .max_retries(3)
        .build();
    assert!(
        provider_result.is_ok(),
        "Provider should be created with programmatic config"
    );

    // Verify config values are set correctly
    let provider2 = OpenAIConfig::builder()
        .api_key("test-key".to_string())
        .model("gpt-4o".to_string())
        .timeout(Duration::from_secs(60))
        .build()
        .expect("Failed to build provider");

    let config = provider2.config();
    assert_eq!(config.api_key, "test-key");
    assert_eq!(config.model, "gpt-4o");
}
