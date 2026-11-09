//! Templates Example
//!
//! This example demonstrates:
//! - Template composition and chaining
//! - Template search and browsing
//! - Template variables and instantiation
//! - Template storage (filesystem, S3, R2)
//! - Prompt template storage
//!
//! Best practices shown:
//! - Composing complex prompts from simple templates
//! - Template organization and discovery
//! - Parameterized template usage
//!
//! Run with:
//! ```bash
//! cargo run --example templates --features templates,s3
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
#[cfg(feature = "prompts")]
use caliche::prompts::{PromptTemplate, PromptTemplateStore};
#[cfg(feature = "s3")]
use caliche::storage::StorageConfig;
use caliche::templates::{TemplateLoader, TemplateRegistry, TemplateSearchFilters};
use serde_json::json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Templates Examples ===\n");

    // Example 1: Template Composition
    example_template_composition()?;

    // Example 2: Template Search
    example_template_search().await?;

    // Example 3: Template Variables
    Box::pin(example_template_variables()).await?;

    // Example 4: Template Storage
    example_template_storage().await?;

    // Example 5: Prompt Template Storage
    example_prompt_storage().await?;

    Ok(())
}

fn example_template_composition() -> Result<()> {
    println!("Example 1: Template Composition\n");

    // Basic template chaining
    let analyze = PromptTemplate::new("Analyze this data: {data}");
    let summarize = PromptTemplate::new("Summarize the analysis: {text}");

    let composed = analyze.then(summarize);
    let prompt = composed.format(&json!({
        "data": "Sales figures Q1-Q4: $1M, $1.2M, $1.5M, $1.8M"
    }))?;

    println!("Chained Template:");
    println!("{}\n", prompt);

    // Multi-step composition
    let step1 = PromptTemplate::new("Extract key points from: {text}");
    let step2 = PromptTemplate::new("Categorize these points: {text}");
    let step3 = PromptTemplate::new("Create action items from: {text}");

    let workflow = step1.then(step2).then(step3);
    let prompt = workflow.format(&json!({
        "text": "Meeting notes: Discussed Q4 goals, budget approval, team expansion"
    }))?;

    println!("Multi-Step Composition:");
    println!("{}\n", prompt);

    // Template with partial variables
    let base = PromptTemplate::new("You are a {role}. {instruction}")
        .with_partial("role", "helpful assistant");

    let prompt1 = base.format(&json!({"instruction": "Explain quantum computing"}))?;
    let prompt2 = base.format(&json!({"instruction": "Write a haiku"}))?;

    println!("Template Reuse:");
    println!("1. {}", prompt1);
    println!("2. {}\n", prompt2);

    Ok(())
}

async fn example_template_search() -> Result<()> {
    println!("Example 2: Template Search\n");

    let mut registry = TemplateRegistry::new();
    registry.add_template_dir("templates");
    let count = registry.load_templates().await?;
    println!("Loaded {} templates\n", count);

    // Search by query
    println!("Searching for 'semantic_search'...");
    let filters = TemplateSearchFilters {
        query: Some("semantic_search".to_string()),
        ..Default::default()
    };
    let results = registry.search(&filters);
    if results.is_empty() {
        println!("   No results found (create templates/ directory)");
    } else {
        for template in &results {
            println!("  - {} (v{})", template.name, template.version);
        }
    }
    println!();

    // Search by category
    println!("Searching by category 'customer-service'...");
    let filters = TemplateSearchFilters {
        category: Some("customer-service".to_string()),
        ..Default::default()
    };
    let results = registry.search(&filters);
    if results.is_empty() {
        println!("   No results found");
    } else {
        for template in &results {
            println!("  - {} (v{})", template.name, template.version);
        }
    }
    println!();

    // List all categories
    println!("Available categories:");
    let categories = registry.get_categories();
    if categories.is_empty() {
        println!("   No categories found");
    } else {
        for category in categories {
            println!("  - {}", category);
        }
    }
    println!();

    // List all tags
    println!("Available tags:");
    let tags = registry.get_all_tags();
    if tags.is_empty() {
        println!("   No tags found");
    } else {
        for tag in tags {
            println!("  - {}", tag);
        }
    }
    println!();

    Ok(())
}

async fn example_template_variables() -> Result<()> {
    println!("Example 3: Template Variables\n");

    let loader = TemplateLoader::new();

    // Try to load a template (may not exist)
    if let Ok(template) = loader
        .load_template("templates/customer_support.yaml")
        .await
    {
        println!("Template: {} (v{})", template.name, template.version);
        println!("Description: {}", template.description);

        if let Some(ref vars) = template.variables {
            println!("\nTemplate Variables:");
            for (name, def) in &vars.variables {
                println!("  - {}: {:?}", name, def);
            }
        }

        let mut values = HashMap::new();
        values.insert("company_name".to_string(), json!("Acme Corp"));
        values.insert("tone".to_string(), json!("friendly"));

        let builder = template.instantiate_with_variables(Some(values)).await?;
        let agent = builder.build().await?;

        println!("\nAgent created:");
        println!("  Provider: {}", agent.provider());
        println!("  Model: {}\n", agent.model());
    } else {
        println!("(Skipped: Create templates/customer_support.yaml to see this example)\n");
    }

    Ok(())
}

async fn example_template_storage() -> Result<()> {
    println!("Example 4: Template Storage\n");

    // Filesystem storage
    println!("1. Filesystem Storage:");
    let fs_config = StorageConfig::filesystem("./template_storage")
        .with_cache(true)
        .with_cache_ttl(3600);

    let registry = TemplateRegistry::with_storage(fs_config).await?;
    registry.load_templates().await?;
    println!("   Loaded templates from filesystem\n");

    // AWS S3 storage
    #[cfg(feature = "s3")]
    {
        println!("2. AWS S3 Storage:");
        let s3_config = StorageConfig::aws_s3(
            "my-templates-bucket".to_string(),
            Some("us-east-1".to_string()),
            None,
            None,
            None,
        )
        .with_cache(true);

        match TemplateRegistry::with_storage(s3_config).await {
            Ok(registry) => match registry.load_templates().await {
                Ok(count) => println!("   Loaded {} templates from S3\n", count),
                Err(e) => println!("   Error loading from S3: {}\n", e),
            },
            Err(e) => println!("   Error creating S3 storage: {}\n", e),
        }

        // Cloudflare R2 storage
        println!("3. Cloudflare R2 Storage:");
        let _r2_config = StorageConfig::cloudflare_r2(
            "my-r2-bucket".to_string(),
            "account-id".to_string(),
            "access-key-id".to_string(),
            "secret-access-key".to_string(),
            None,
        )
        .with_cache(true)
        .with_cache_max_size(500);

        println!("   R2 config created (requires valid credentials to use)\n");
    }

    #[cfg(not(feature = "s3"))]
    {
        println!("(Skipped: Enable 's3' feature for S3/R2 storage examples)\n");
    }

    Ok(())
}

async fn example_prompt_storage() -> Result<()> {
    println!("Example 5: Prompt Template Storage\n");

    // Filesystem storage
    println!("1. Filesystem Storage:");
    let fs_config = StorageConfig::filesystem("./prompt_storage").with_cache(true);

    let store = PromptTemplateStore::new(fs_config).await?;

    // Create and save a template
    let template =
        PromptTemplate::new("Summarize this text: {text}").with_partial("style", "concise");

    store.save("summarize", &template).await?;
    println!("   Saved 'summarize' template");

    // Load the template
    let loaded = store.load("summarize").await?;
    let formatted = loaded.format(&json!({"text": "Long article here..."}))?;
    println!("   Loaded template and formatted: {}", formatted);
    println!("   Variables: {:?}\n", loaded.variables());

    // List all templates
    let templates = store.list().await?;
    println!("   Available templates: {:?}\n", templates);

    // AWS S3 storage
    #[cfg(feature = "s3")]
    {
        println!("2. AWS S3 Storage:");
        let _s3_config = StorageConfig::aws_s3(
            "my-prompts-bucket".to_string(),
            Some("us-east-1".to_string()),
            None,
            None,
            None,
        )
        .with_cache(true);

        println!("   S3 config created (requires valid credentials to use)\n");
    }

    #[cfg(not(feature = "s3"))]
    {
        println!("(Skipped: Enable 's3' feature for S3 storage examples)\n");
    }

    Ok(())
}
