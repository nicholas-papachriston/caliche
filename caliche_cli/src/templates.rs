//! CLI commands for template management.

use anyhow::{Context, Result};
use caliche::templates::{TemplateLoader, TemplateRegistry, TemplateValidator};
use std::path::PathBuf;

/// List all available templates
pub async fn list_templates(template_dir: Option<PathBuf>) -> Result<()> {
    let mut registry = TemplateRegistry::new();

    // Add default template directory if not specified
    let dir = template_dir.unwrap_or_else(|| PathBuf::from("templates"));
    registry.add_template_dir(&dir);

    let count = registry.load_templates().await?;

    if count == 0 {
        println!("No templates found in {}", dir.display());
        return Ok(());
    }

    println!("Found {} template(s):\n", count);

    let templates = registry.list();
    for template in templates {
        println!("  {} (v{})", template.name, template.version);
        println!("    ID: {}", template.id());
        println!("    Description: {}", template.description);
        if !template.tags.is_empty() {
            println!("    Tags: {}", template.tags.join(", "));
        }
        if let Some(author) = &template.author {
            println!("    Author: {}", author);
        }
        println!();
    }

    Ok(())
}

/// Show details for a specific template
pub async fn show_template(template_id: &str, template_dir: Option<PathBuf>) -> Result<()> {
    let mut registry = TemplateRegistry::new();

    let dir = template_dir.unwrap_or_else(|| PathBuf::from("templates"));
    registry.add_template_dir(&dir);

    registry.load_templates().await?;

    let template = registry
        .get(template_id)
        .ok_or_else(|| anyhow::anyhow!("Template not found: {}", template_id))?;

    println!("Template: {} (v{})", template.name, template.version);
    println!("ID: {}", template.id());
    println!("Description: {}", template.description);
    if let Some(author) = &template.author {
        println!("Author: {}", author);
    }
    if !template.tags.is_empty() {
        println!("Tags: {}", template.tags.join(", "));
    }
    println!("\nConfiguration:");
    println!("  Provider: {}", template.config.provider);
    if let Some(ref model) = template.config.model {
        println!("  Model: {}", model);
    }
    if let Some(ref prompt) = template.config.system_prompt {
        println!("  System Prompt: {}", prompt);
    }
    if let Some(ref tools) = template.config.tools {
        println!("  Tools: {}", tools.join(", "));
    }
    if let Some(ref memory) = template.config.memory {
        println!("  Memory Type: {}", memory.memory_type);
        if let Some(max_tokens) = memory.max_tokens {
            println!("  Max Tokens: {}", max_tokens);
        }
    }
    if let Some(ref vs) = template.config.vector_backend {
        println!("  Vector Store: {}", vs.backend);
    }
    if let Some(ref sql) = template.config.sql_tools {
        println!("  SQL Tools: Enabled");
        println!("    Database: {}", sql.database_url);
        println!("    Namespace: {}", sql.namespace);
    }

    Ok(())
}

/// Create an agent from a template
pub async fn use_template(
    template_id: &str,
    agent_name: Option<String>,
    template_dir: Option<PathBuf>,
) -> Result<()> {
    let mut registry = TemplateRegistry::new();

    let dir = template_dir.unwrap_or_else(|| PathBuf::from("templates"));
    registry.add_template_dir(&dir);

    registry.load_templates().await?;

    let template = registry
        .get(template_id)
        .ok_or_else(|| anyhow::anyhow!("Template not found: {}", template_id))?;

    println!("Instantiating agent from template: {}", template.name);

    let builder = template.instantiate().await?;
    let agent = Box::pin(builder.build()).await?;

    println!("Agent created successfully!");
    println!("  Provider: {}", agent.provider());
    println!("  Model: {}", agent.model());

    if let Some(name) = agent_name {
        println!("  Name: {}", name);
    }

    Ok(())
}

/// Validate a template file
pub async fn validate_template(template_path: &str) -> Result<()> {
    let loader = TemplateLoader::new();
    let template = loader
        .load_template(template_path)
        .await
        .context("Failed to load template")?;

    let validator = TemplateValidator::new();
    validator
        .validate(&template)
        .context("Template validation failed")?;

    println!("Template is valid!");
    println!("  Name: {}", template.name);
    println!("  Version: {}", template.version);
    println!("  Description: {}", template.description);

    Ok(())
}

/// Create a new template from a file
pub async fn create_template(template_path: &str) -> Result<()> {
    // Validate the template first
    validate_template(template_path).await?;

    println!("Template created successfully at: {}", template_path);
    println!("You can now use it with: caliche template use <template-id>");

    Ok(())
}

/// Interactive template creation wizard
pub async fn init_template_interactive(output_path: Option<PathBuf>) -> Result<()> {
    println!("Caliche Template Creation Wizard");
    println!("=====================================\n");

    let name = prompt_line("Template name: ")?;
    let version = prompt_line_or_default("Version (default: 1.0.0): ", "1.0.0")?;
    let description = prompt_line("Description: ")?;
    let provider = prompt_line("Provider (openai, anthropic, bedrock, etc.): ")?;
    let model = prompt_optional("Model (optional): ")?;
    let system_prompt = prompt_optional("System prompt (optional): ")?;
    let tools = prompt_optional_csv("Tools (comma-separated, optional): ")?;
    let category = prompt_optional("Category (optional): ")?;

    use caliche::templates::{AgentTemplate, TemplateConfig};
    let template = AgentTemplate {
        name,
        version,
        description,
        author: Some("User".to_string()),
        tags: vec![],
        category,
        config: TemplateConfig {
            provider,
            model,
            system_prompt,
            tools,
            memory: None,
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

    let validator = TemplateValidator::new();
    validator.validate(&template)?;
    write_template_file(&template, output_path).await
}

fn prompt_line(prompt: &str) -> Result<String> {
    use std::io::{self, Write};
    print!("{prompt}");
    io::stdout().flush()?;
    let mut buf = String::new();
    io::stdin().read_line(&mut buf)?;
    Ok(buf.trim().to_string())
}

fn prompt_line_or_default(prompt: &str, default: &str) -> Result<String> {
    let value = prompt_line(prompt)?;
    if value.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(value)
    }
}

fn prompt_optional(prompt: &str) -> Result<Option<String>> {
    let value = prompt_line(prompt)?;
    if value.is_empty() {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

fn prompt_optional_csv(prompt: &str) -> Result<Option<Vec<String>>> {
    let value = prompt_line(prompt)?;
    if value.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        value.split(',').map(|s| s.trim().to_string()).collect(),
    ))
}

async fn write_template_file(
    template: &caliche::templates::AgentTemplate,
    output_path: Option<PathBuf>,
) -> Result<()> {
    let output_path = output_path.unwrap_or_else(|| {
        let filename = format!("{}.yaml", template.id().replace(':', "_"));
        PathBuf::from("templates").join(filename)
    });
    if let Some(parent) = output_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let yaml = serde_norway::to_string(template)?;
    tokio::fs::write(&output_path, yaml).await?;
    println!(
        "\nTemplate created successfully at: {}",
        output_path.display()
    );
    println!("Template ID: {}", template.id());
    Ok(())
}

/// Test a template with example inputs
pub async fn test_template(
    template_id: &str,
    test_inputs: Option<Vec<String>>,
    template_dir: Option<PathBuf>,
) -> Result<()> {
    let mut registry = TemplateRegistry::new();
    let dir = template_dir.unwrap_or_else(|| PathBuf::from("templates"));
    registry.add_template_dir(&dir);
    registry.load_templates().await?;

    let template = registry
        .get(template_id)
        .ok_or_else(|| anyhow::anyhow!("Template not found: {}", template_id))?;

    println!(
        "Testing template: {} (v{})",
        template.name, template.version
    );
    println!("=====================================\n");

    // Use examples from template if no test inputs provided
    let inputs = if let Some(inputs) = test_inputs {
        inputs
    } else if let Some(ref examples) = template.examples {
        examples.iter().map(|e| e.input.clone()).collect()
    } else {
        println!("No test inputs provided and template has no examples.");
        println!("Provide test inputs with --test-inputs or add examples to the template.");
        return Ok(());
    };

    // Validate variables if present
    if template.variables.is_some() {
        println!("Template has variables. Using defaults for testing.\n");
    }

    // Try to instantiate
    println!("Instantiating agent from template...");
    let builder = template.instantiate().await?;
    let mut agent = Box::pin(builder.build()).await?;

    println!("Agent created successfully!\n");

    // Run test inputs
    for (i, input) in inputs.iter().enumerate() {
        println!("Test {}: {}", i + 1, input);
        println!("---");

        match agent.chat(input).await {
            Ok(chat_response) => {
                println!("Response: {}", chat_response.text());
                if let Some(total) = chat_response.usage().total_tokens {
                    println!("Tokens used: {}", total);
                }
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
        println!();
    }

    println!("Testing complete!");

    Ok(())
}

/// Benchmark a template's performance
pub async fn benchmark_template(
    template_id: &str,
    iterations: usize,
    test_input: Option<String>,
    template_dir: Option<PathBuf>,
) -> Result<()> {
    let mut registry = TemplateRegistry::new();
    let dir = template_dir.unwrap_or_else(|| PathBuf::from("templates"));
    registry.add_template_dir(&dir);
    registry.load_templates().await?;

    let template = registry
        .get(template_id)
        .ok_or_else(|| anyhow::anyhow!("Template not found: {}", template_id))?;

    println!(
        "Benchmarking template: {} (v{})",
        template.name, template.version
    );
    println!("Iterations: {}", iterations);
    println!("=====================================\n");

    let test_input = test_input.unwrap_or_else(|| "Hello, this is a test message.".to_string());

    // Instantiate once
    println!("Instantiating agent...");
    let builder = template.instantiate().await?;
    let mut agent = Box::pin(builder.build()).await?;
    println!("Agent ready.\n");

    // Warmup
    println!("Warming up...");
    let _ = agent.chat("warmup").await;

    // Benchmark
    println!("Running benchmark...\n");
    let mut durations = Vec::new();
    let mut total_tokens = 0;
    let mut errors = 0;

    for i in 0..iterations {
        let start = std::time::Instant::now();
        match agent.chat(&test_input).await {
            Ok(chat_response) => {
                let duration = start.elapsed();
                durations.push(duration);
                if let Some(tokens) = chat_response.usage().total_tokens {
                    total_tokens += tokens;
                }
            }
            Err(e) => {
                errors += 1;
                eprintln!("Iteration {} failed: {}", i + 1, e);
            }
        }
    }

    // Calculate statistics
    if durations.is_empty() {
        return Err(anyhow::anyhow!("All iterations failed"));
    }

    durations.sort();
    let min = *durations
        .first()
        .ok_or_else(|| anyhow::anyhow!("internal: benchmark durations missing min after sort"))?;
    let max = *durations
        .last()
        .ok_or_else(|| anyhow::anyhow!("internal: benchmark durations missing max after sort"))?;
    let avg: std::time::Duration =
        durations.iter().sum::<std::time::Duration>() / durations.len() as u32;
    let p50 = durations[durations.len() / 2];
    let p95 = durations[(durations.len() * 95) / 100];
    let p99 = durations[(durations.len() * 99) / 100];

    println!("Benchmark Results:");
    println!("==================");
    println!("Iterations: {}", iterations);
    println!("Errors: {}", errors);
    println!(
        "Success rate: {:.2}%",
        (iterations - errors) as f64 / iterations as f64 * 100.0
    );
    println!("\nLatency:");
    println!("  Min:    {:?}", min);
    println!("  Max:    {:?}", max);
    println!("  Avg:    {:?}", avg);
    println!("  P50:    {:?}", p50);
    println!("  P95:    {:?}", p95);
    println!("  P99:    {:?}", p99);

    if total_tokens > 0 {
        println!("\nToken Usage:");
        println!("  Total:  {}", total_tokens);
        println!(
            "  Avg:    {:.2}",
            total_tokens as f64 / (iterations - errors) as f64
        );
    }

    Ok(())
}
