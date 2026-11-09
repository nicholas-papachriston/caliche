//! Convert SkillPluginSpec to the existing Skill struct with template interpolation.

use crate::agent::skills::{Skill, SkillBuilder};
use crate::plugins::schema::{PluginDefinition, PluginSpec};
use std::collections::HashMap;

/// Build a Skill from a plugin definition with a skill spec.
/// Interpolates `{{param_name}}` in the system prompt template using parameter defaults or empty string.
pub fn skill_from_plugin(def: &PluginDefinition) -> Option<Skill> {
    let spec = match &def.spec {
        PluginSpec::Skill(s) => s.clone(),
        _ => return None,
    };
    let prompt = interpolate_prompt(&spec.system_prompt_template, &spec.parameters, &[]);
    let mut builder = SkillBuilder::new(def.name.clone())
        .description(def.description.clone())
        .system_prompt_addition(prompt)
        .require_tools(spec.required_tools);
    if spec.restrict_tools {
        builder = builder.restrict_to_required_tools();
    }
    if let Some(max) = spec.max_iterations {
        builder = builder.max_iterations(max);
    }
    Some(builder.build())
}

/// Interpolate `{{param}}` in the template with values from params (or defaults), then any remaining with overrides.
fn interpolate_prompt(
    template: &str,
    params: &HashMap<String, crate::plugins::schema::SkillParam>,
    overrides: &[(&str, &str)],
) -> String {
    let mut out = template.to_string();
    for (name, param) in params {
        let value = overrides
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| (*v).to_string())
            .or_else(|| {
                param
                    .default
                    .as_ref()
                    .and_then(|v| v.as_str().map(std::string::ToString::to_string))
            })
            .unwrap_or_default();
        let placeholder = format!("{{{{{}}}}}", name);
        out = out.replace(&placeholder, &value);
    }
    out
}
