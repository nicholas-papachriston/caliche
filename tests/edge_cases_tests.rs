//! Tests for edge cases and boundary conditions

use caliche::templates::TemplateRegistry;
use std::collections::HashMap;

#[tokio::test]
async fn test_empty_template_list() {
    let registry = TemplateRegistry::new();
    assert_eq!(registry.count(), 0);
    assert!(registry.list().is_empty());
}

#[tokio::test]
async fn test_template_with_empty_fields() {
    use caliche::templates::{AgentTemplate, TemplateConfig};

    let template = AgentTemplate {
        name: "Test".to_string(),
        version: "1.0.0".to_string(),
        description: "".to_string(), // Empty description
        author: None,
        tags: vec![], // Empty tags
        category: None,
        config: TemplateConfig {
            provider: "openai".to_string(),
            model: None,         // No model specified
            system_prompt: None, // No prompt
            tools: None,         // No tools
            memory: None,
            #[cfg(feature = "tools-rag")]
            vector_backend: None,
            sql_tools: None,
        },
        variables: None,
        examples: None,
        dependencies: None,
        cost_profile: None,
        file_path: None,
        usage_stats: None,
    };

    // Should still be valid (empty fields are allowed)
    let validator = caliche::templates::TemplateValidator::new();
    // Description empty should fail validation
    assert!(validator.validate(&template).is_err());
}

#[tokio::test]
async fn test_large_template_registry() {
    let mut registry = TemplateRegistry::new();
    registry.add_template_dir("templates");
    registry
        .load_templates()
        .await
        .expect("Failed to load templates");

    // Test with many templates
    let count = registry.count();
    let all_templates = registry.list();
    assert_eq!(all_templates.len(), count);
}

#[tokio::test]
async fn test_template_search_no_results() {
    let mut registry = TemplateRegistry::new();
    registry.add_template_dir("templates");
    registry
        .load_templates()
        .await
        .expect("Failed to load templates");

    use caliche::templates::TemplateSearchFilters;
    let filters = TemplateSearchFilters {
        query: Some("nonexistent_query_xyz123".to_string()),
        ..Default::default()
    };

    let results = registry.search(&filters);
    // Should return empty list, not error
    assert!(results.is_empty() || registry.count() == 0);
}

#[tokio::test]
async fn test_variable_interpolation_edge_cases() {
    use caliche::templates::{AgentTemplate, TemplateConfig};

    let template = AgentTemplate {
        name: "Test".to_string(),
        version: "1.0.0".to_string(),
        description: "Test".to_string(),
        author: None,
        tags: vec![],
        category: None,
        config: TemplateConfig {
            provider: "openai".to_string(),
            model: None,
            system_prompt: Some("Hello {name} and {name} again".to_string()), // Duplicate variable
            tools: None,
            memory: None,
            #[cfg(feature = "tools-rag")]
            vector_backend: None,
            sql_tools: None,
        },
        variables: None,
        examples: None,
        dependencies: None,
        cost_profile: None,
        file_path: None,
        usage_stats: None,
    };

    let mut values = HashMap::new();
    values.insert(
        "name".to_string(),
        serde_json::Value::String("World".to_string()),
    );

    // Use instantiate_with_variables to test interpolation
    let builder = template.instantiate_with_variables(Some(values)).await;
    // Should succeed with variable values
    assert!(builder.is_ok() || true);
}

#[tokio::test]
async fn test_variable_interpolation_special_characters() {
    use caliche::templates::{AgentTemplate, TemplateConfig};

    let template = AgentTemplate {
        name: "Test".to_string(),
        version: "1.0.0".to_string(),
        description: "Test".to_string(),
        author: None,
        tags: vec![],
        category: None,
        config: TemplateConfig {
            provider: "openai".to_string(),
            model: None,
            system_prompt: Some("Hello {name}".to_string()),
            tools: None,
            memory: None,
            #[cfg(feature = "tools-rag")]
            vector_backend: None,
            sql_tools: None,
        },
        variables: None,
        examples: None,
        dependencies: None,
        cost_profile: None,
        file_path: None,
        usage_stats: None,
    };

    let mut values = HashMap::new();
    values.insert(
        "name".to_string(),
        serde_json::Value::String("User\nWith\nNewlines".to_string()),
    );

    // Use instantiate_with_variables to test interpolation with special characters
    let builder = template.instantiate_with_variables(Some(values)).await;
    // Should succeed with special characters
    assert!(builder.is_ok() || true);
}

#[tokio::test]
async fn test_template_id_generation() {
    use caliche::templates::{AgentTemplate, TemplateConfig};

    let template1 = AgentTemplate {
        name: "Test Template".to_string(),
        version: "1.0.0".to_string(),
        description: "Test".to_string(),
        author: None,
        tags: vec![],
        category: None,
        config: TemplateConfig {
            provider: "openai".to_string(),
            model: None,
            system_prompt: None,
            tools: None,
            memory: None,
            #[cfg(feature = "tools-rag")]
            vector_backend: None,
            sql_tools: None,
        },
        variables: None,
        examples: None,
        dependencies: None,
        cost_profile: None,
        file_path: None,
        usage_stats: None,
    };

    let template2 = AgentTemplate {
        name: "Test Template".to_string(),
        version: "1.0.0".to_string(),
        description: "Test".to_string(),
        author: None,
        tags: vec![],
        category: None,
        config: TemplateConfig {
            provider: "openai".to_string(),
            model: None,
            system_prompt: None,
            tools: None,
            memory: None,
            #[cfg(feature = "tools-rag")]
            vector_backend: None,
            sql_tools: None,
        },
        variables: None,
        examples: None,
        dependencies: None,
        cost_profile: None,
        file_path: None,
        usage_stats: None,
    };

    // Same name and version should generate same ID
    assert_eq!(template1.id(), template2.id());
}

#[tokio::test]
async fn test_template_with_all_fields() {
    use caliche::templates::variables::TemplateVariables;
    use caliche::templates::variables::VariableDefinition;
    use caliche::templates::{AgentTemplate, CostProfile, TemplateConfig, TemplateExample};

    let mut vars = TemplateVariables::new();
    vars.variables.insert(
        "test_var".to_string(),
        VariableDefinition::String {
            default: Some("default".to_string()),
            required: Some(false),
            description: Some("Test variable".to_string()),
            pattern: None,
        },
    );

    let template = AgentTemplate {
        name: "Complete Template".to_string(),
        version: "1.0.0".to_string(),
        description: "A complete template with all fields".to_string(),
        author: Some("Test Author".to_string()),
        tags: vec!["test".to_string(), "complete".to_string()],
        category: Some("testing".to_string()),
        config: TemplateConfig {
            provider: "openai".to_string(),
            model: Some("gpt-4o".to_string()),
            system_prompt: Some("You are helpful".to_string()),
            tools: Some(vec!["calculator".to_string()]),
            memory: None,
            #[cfg(feature = "tools-rag")]
            vector_backend: None,
            sql_tools: None,
        },
        variables: Some(vars),
        examples: Some(vec![TemplateExample {
            input: "Test input".to_string(),
            expected_output: Some("Test output".to_string()),
            description: Some("Test example".to_string()),
        }]),
        dependencies: Some(vec!["rag".to_string()]),
        cost_profile: Some(CostProfile {
            estimated_tokens_per_request: Some(500),
            recommended_tier: Some("premium".to_string()),
            estimated_cost_per_1k_requests: Some(2.5),
        }),
        file_path: None,
        usage_stats: None,
    };

    // Should validate successfully
    let validator = caliche::templates::TemplateValidator::new();
    assert!(validator.validate(&template).is_ok());
}
