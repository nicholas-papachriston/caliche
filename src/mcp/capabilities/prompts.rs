//! Prompts capability implementation for MCP.

use crate::mcp::protocol::*;
use crate::templates::TemplateRegistry;
use anyhow::{Context, Result};
use beef::lean::Cow;
use caliche_mcp::pagination::{DEFAULT_PAGE_SIZE, cmp_name, paginate_owned};
use serde_json::Value;
use std::sync::Arc;

pub(crate) fn collect_prompts(template_registry: &TemplateRegistry) -> Vec<McpPrompt> {
    let templates = template_registry.list();
    let mut mcp_prompts = Vec::with_capacity(templates.len());

    for template in templates {
        let mut arguments = Vec::new();

        if let Some(ref vars) = template.variables {
            for (name, var_def) in &vars.variables {
                let (description, required) = match var_def {
                    crate::templates::variables::VariableDefinition::String {
                        description,
                        required,
                        ..
                    }
                    | crate::templates::variables::VariableDefinition::Integer {
                        description,
                        required,
                        ..
                    }
                    | crate::templates::variables::VariableDefinition::Enum {
                        description,
                        required,
                        ..
                    }
                    | crate::templates::variables::VariableDefinition::Boolean {
                        description,
                        required,
                        ..
                    } => (description.clone(), required.unwrap_or(false)),
                };

                arguments.push(PromptArgument {
                    name: name.clone(),
                    title: None,
                    description,
                    required: Some(required),
                });
            }
        }

        mcp_prompts.push(McpPrompt {
            name: template.id(),
            title: None,
            description: Some(template.description.clone()),
            icons: None,
            arguments: if arguments.is_empty() {
                None
            } else {
                Some(arguments)
            },
            meta: None,
        });
    }
    mcp_prompts
}

/// Handle `prompts/list` with deterministic cursor pagination.
pub async fn handle_prompts_list(
    template_registry: &Arc<TemplateRegistry>,
    params: Option<PromptsListParams>,
) -> Result<PromptsListResult> {
    let mut mcp_prompts = collect_prompts(template_registry);

    mcp_prompts.sort_by(|a, b| cmp_name(&a.name, &b.name));
    let cursor = params.as_ref().and_then(|p| p.cursor.as_deref());
    let page = paginate_owned(mcp_prompts, cursor, Some(DEFAULT_PAGE_SIZE))
        .context("invalid prompts/list cursor")?;

    Ok(PromptsListResult {
        prompts: page.items,
        next_cursor: page.next_cursor,
    })
}

/// Handle prompts/get request
pub async fn handle_prompts_get(
    template_registry: &Arc<TemplateRegistry>,
    params: PromptsGetParams,
) -> Result<PromptsGetResult> {
    let template = template_registry
        .get(&params.name)
        .ok_or_else(|| anyhow::anyhow!("Prompt not found: {}", params.name))?;

    let mut messages = Vec::new();

    if let Some(ref system_prompt) = template.config.system_prompt {
        let prompt_text = interpolate_with_optional_args(system_prompt, params.arguments.as_ref());
        messages.push(PromptMessage {
            role: Cow::borrowed("system"),
            content: Value::String(prompt_text),
        });
    }

    let user_message =
        interpolate_with_optional_args(&template.description, params.arguments.as_ref());

    messages.push(PromptMessage {
        role: Cow::borrowed("user"),
        content: Value::String(user_message),
    });

    Ok(PromptsGetResult {
        messages,
        description: Some(template.description.clone()),
    })
}

fn interpolate_with_optional_args(template: &str, args: Option<&Value>) -> String {
    let Some(obj) = args.and_then(Value::as_object) else {
        return template.to_string();
    };
    crate::templates::interpolate_string(template, obj.iter().map(|(k, v)| (k.as_str(), v)))
}
