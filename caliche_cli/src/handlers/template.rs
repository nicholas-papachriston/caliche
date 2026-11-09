//! Template command handlers.
//!
//! Handles template listing, showing, and management.

use crate::commands::{PromptAction, TemplateAction};
use crate::errors::{CliError, CliResult};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use anyhow::Result;
use async_trait::async_trait;
use std::path::PathBuf;

/// Handler for template commands.
pub struct TemplateHandler;

#[async_trait]
impl CommandHandler for TemplateHandler {
    type Args = TemplateAction;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        Box::pin(handle_template_command(args))
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;
        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "template"
    }

    fn description(&self) -> &'static str {
        "Manage prompt templates"
    }

    fn requires_credentials(&self) -> bool {
        false
    }
}

/// Handle template subcommands
pub async fn handle_template_command(action: TemplateAction) -> Result<()> {
    use crate::templates::*;

    match action {
        TemplateAction::List { dir } => {
            let template_dir = dir.map(PathBuf::from);
            list_templates(template_dir).await
        }
        TemplateAction::Show { template_id, dir } => {
            let template_dir = dir.map(PathBuf::from);
            show_template(&template_id, template_dir).await
        }
        TemplateAction::Use {
            template_id,
            name,
            dir,
        } => {
            let template_dir = dir.map(PathBuf::from);
            Box::pin(use_template(&template_id, name, template_dir)).await
        }
        TemplateAction::Validate { template_path } => validate_template(&template_path).await,
        TemplateAction::Create { template_path } => create_template(&template_path).await,
        TemplateAction::Init { output } => {
            use crate::templates::init_template_interactive;
            let output_path = output.map(PathBuf::from);
            init_template_interactive(output_path).await
        }
        TemplateAction::Test {
            template_id,
            test_inputs,
            dir,
        } => {
            use crate::templates::test_template;
            let template_dir = dir.map(PathBuf::from);
            let inputs = test_inputs.map(|s| s.split(',').map(|s| s.trim().to_string()).collect());
            Box::pin(test_template(&template_id, inputs, template_dir)).await
        }
        TemplateAction::Benchmark {
            template_id,
            iterations,
            test_input,
            dir,
        } => {
            use crate::templates::benchmark_template;
            let template_dir = dir.map(PathBuf::from);
            Box::pin(benchmark_template(
                &template_id,
                iterations,
                test_input,
                template_dir,
            ))
            .await
        }
        TemplateAction::Get { template_id, dir } => {
            let template_dir = dir.map(PathBuf::from);
            use crate::templates::show_template;
            show_template(&template_id, template_dir).await
        }
    }
}

/// Handle prompt subcommands (alias for template commands)
pub async fn handle_prompt_command(action: PromptAction) -> Result<()> {
    use crate::templates::*;
    use anyhow::Context;
    use caliche::templates::TemplateRegistry;

    match action {
        PromptAction::List { dir } => {
            let template_dir = dir.map(PathBuf::from);
            list_templates(template_dir).await
        }
        PromptAction::Get { template_id, dir } => {
            let template_dir = dir.map(PathBuf::from);
            show_template(&template_id, template_dir).await
        }
        PromptAction::Test {
            template_id,
            input,
            dir,
        } => {
            let mut registry = TemplateRegistry::new();
            let template_dir = dir
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("templates"));
            registry.add_template_dir(&template_dir);
            registry.load_templates().await?;

            let template = registry
                .get(&template_id)
                .ok_or_else(|| anyhow::anyhow!("Template not found: {}", template_id))?;

            // Parse input JSON if provided
            let input_vars = if let Some(input_str) = input {
                let vars: std::collections::HashMap<String, serde_json::Value> =
                    serde_json::from_str(&input_str).context("Failed to parse input JSON")?;
                Some(vars)
            } else {
                None
            };

            // Instantiate template with input variables
            let agent_builder = template.instantiate_with_variables(input_vars).await?;
            let mut agent = Box::pin(agent_builder.build()).await?;

            // If template has examples, use first example input, otherwise use a default
            let test_input = if let Some(examples) = &template.examples {
                if !examples.is_empty() {
                    examples[0].input.clone()
                } else {
                    "Hello".to_string()
                }
            } else {
                "Hello".to_string()
            };

            let response = agent.chat(&test_input).await?;
            println!("Template: {} (v{})", template.name, template.version);
            println!("Input: {}", test_input);
            println!("Output: {}", response.text());
            Ok(())
        }
    }
}
