//! Prompts Example
//!
//! This example demonstrates:
//! - Basic prompt templates with variable substitution
//! - Partial variables
//! - Few-shot prompting with examples
//! - Chat prompt templates
//! - Template composition
//! - Prompt management and versioning
//!
//! Best practices shown:
//! - Semantic versioning for templates
//! - Template composition for complex prompts
//! - Few-shot learning patterns
//!
//! Run with:
//! ```bash
//! cargo run --example prompts --features prompts,templates
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::prelude::ProviderType;
use caliche::{
    AgentBuilder,
    prompts::{ChatPromptTemplate, Example, FewShotPromptTemplate, PromptTemplate},
    templates::{TemplateLoader, TemplateRegistry},
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Prompts Examples ===\n");

    // Example 1: Basic Templates
    example_basic_templates()?;

    // Example 2: Few-shot Prompting
    Box::pin(example_few_shot()).await?;

    // Example 3: Chat Templates
    example_chat_templates()?;

    // Example 4: Template Composition
    example_template_composition()?;

    // Example 5: Prompt Management
    example_prompt_management().await?;

    Ok(())
}

fn example_basic_templates() -> Result<()> {
    println!("Example 1: Basic Templates\n");

    // Basic template
    println!("1. Basic Template:");
    let template = PromptTemplate::new("Translate '{text}' to {language}");
    let prompt = template.format(&json!({
        "text": "Hello, world!",
        "language": "Spanish"
    }))?;
    println!("   Prompt: {}\n", prompt);

    // Template with partial variables
    println!("2. Template with Partial Variables:");
    let template = PromptTemplate::new("You are a {role}. {instruction}")
        .with_partial("role", "helpful assistant");

    let prompt = template.format(&json!({
        "instruction": "Explain quantum computing."
    }))?;
    println!("   Prompt: {}\n", prompt);

    Ok(())
}

async fn example_few_shot() -> Result<()> {
    println!("Example 2: Few-shot Prompting\n");

    // Classification example
    println!("1. Sentiment Classification:");
    let examples = vec![
        Example::new(
            "I absolutely loved this movie! Best film of the year.",
            "Positive",
        ),
        Example::new(
            "This product is terrible. Complete waste of money.",
            "Negative",
        ),
        Example::new("It's okay, nothing special but not bad either.", "Neutral"),
    ];

    let template = FewShotPromptTemplate::new(
        examples,
        "Classify the sentiment of the following text as Positive, Negative, or Neutral.",
        "Text: {text}\nSentiment:",
    );

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .build()
        .await?;

    let test_text = "This restaurant exceeded all my expectations!";
    let prompt = template.format(&json!({"text": test_text}))?;
    let response = agent.chat(&prompt).await?;
    println!("   Input: {}", test_text);
    println!("   Classification: {}\n", response.text().trim());

    // Math example
    println!("2. Math Problem Solving:");
    let examples = vec![
        Example::new(
            "Sarah has 15 apples. She gives 6 to her friend. How many apples does Sarah have left?",
            "Step 1: Start with 15 apples\nStep 2: Subtract 6: 15 - 6 = 9\nAnswer: 9 apples",
        ),
        Example::new(
            "A train travels 60 miles per hour for 3 hours. How far does it travel?",
            "Step 1: Speed = 60 mph, Time = 3 hours\nStep 2: Distance = Speed × Time = 60 × 3 = 180\nAnswer: 180 miles",
        ),
    ];

    let template = FewShotPromptTemplate::new(
        examples,
        "Solve the following math problem step-by-step.",
        "Problem: {problem}\nSolution:",
    );

    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tool("calculator")
        .build()
        .await?;

    let problem =
        "A store sells notebooks for $3 each. If you buy 8 notebooks, how much do you pay?";
    let prompt = template.format(&json!({"problem": problem}))?;
    let response = agent.chat(&prompt).await?;
    println!("   Problem: {}", problem);
    println!("   Solution:\n{}\n", response.text().trim());

    Ok(())
}

fn example_chat_templates() -> Result<()> {
    println!("Example 3: Chat Templates\n");

    let chat_template = ChatPromptTemplate::with_system("You are an expert {domain} tutor.")
        .add_user("Please explain {concept} in simple terms.");

    let messages = chat_template.format(&json!({
        "domain": "physics",
        "concept": "relativity"
    }))?;

    println!("Chat Messages:");
    for (role, content) in messages {
        println!("   {}: {}", role.to_uppercase(), content);
    }
    println!();

    Ok(())
}

fn example_template_composition() -> Result<()> {
    println!("Example 4: Template Composition\n");

    let step1 = PromptTemplate::new("Analyze this data: {data}");
    let step2 = PromptTemplate::new("Now summarize the analysis: {text}");

    let composed = step1.then(step2);
    let prompt = composed.format(&json!({"data": "Sales figures Q1-Q4"}))?;
    println!("Composed Template:");
    println!("   Final prompt: {}\n", prompt);

    Ok(())
}

async fn example_prompt_management() -> Result<()> {
    println!("Example 5: Prompt Management and Versioning\n");

    let mut registry = TemplateRegistry::new();

    // Add template directory
    registry.add_template_dir("templates");

    let count = registry.load_templates().await?;
    println!("Loaded {} templates from templates/ directory\n", count);

    // Getting latest version
    println!("Getting Latest Version:");
    if let Some(template) = registry.get("customer_service") {
        println!("   Found: {} (v{})", template.name, template.version);
        println!("   ID: {}\n", template.id());
    } else {
        println!("   No template found (create templates/ directory with YAML files)\n");
    }

    // Getting specific version
    println!("Getting Specific Version:");
    if let Some(template) = registry.get("customer_service:1.0.0") {
        println!("   Found: {} (v{})\n", template.name, template.version);
    } else {
        println!("   Use: registry.get_version(\"customer_service\", \"1.0.0\")\n");
    }

    // Listing all versions
    println!("Listing All Versions:");
    let versions = registry.list_versions("customer_service");
    if versions.is_empty() {
        println!("   No versions found (create templates/ directory)");
        println!("   Example structure:");
        println!("     templates/customer_service/1.0.0.yaml");
        println!("     templates/customer_service/1.1.0.yaml");
        println!("     templates/customer_service/2.0.0.yaml\n");
    } else {
        for template in &versions {
            println!("     - {} (v{})", template.name, template.version);
        }
        println!();
    }

    // Creating template programmatically
    println!("Creating Templates Programmatically:");
    let loader = TemplateLoader::new();

    let template_yaml = r#"
name: Example Template
version: 1.0.0
description: An example template
author: Caliche
tags:
  - example
category: examples
config:
  provider: openai
  model: gpt-4o
  system_prompt: "You are a helpful assistant."
  tools:
    - calculator
"#;

    match loader.load_template_from_str(template_yaml, None) {
        Ok(template) => {
            println!("   Template loaded:");
            println!("     Name: {}", template.name);
            println!("     Version: {}", template.version);
            registry.register(template);
            println!("     Registered in registry\n");
        }
        Err(e) => {
            println!("   Note: Template loading requires valid YAML structure");
            println!("   Error: {}\n", e);
        }
    }

    println!("Best Practices:");
    println!("1. Use semantic versioning (MAJOR.MINOR.PATCH)");
    println!("2. Store templates in versioned directories: templates/<name>/<version>.yaml");
    println!("3. Use registry.get() for latest, registry.get_version() for specific versions");
    println!("4. Test templates before deploying: caliche prompt test <name>");

    Ok(())
}
