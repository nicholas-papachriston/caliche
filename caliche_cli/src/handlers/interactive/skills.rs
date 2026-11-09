//! Skill command handling for interactive mode.
//!
//! Handles /skill selection and application of prebuilt agent skills.

use anyhow::Result;
use caliche::AgentService;
use dialoguer::{FuzzySelect, theme::ColorfulTheme};

pub(crate) fn skill_catalog() -> Vec<caliche::agent::skills::Skill> {
    let mut catalog = caliche::agent::skills::builtin_skills();
    for discovered in caliche::agent::skills::discovered_skills() {
        if !catalog.iter().any(|s| s.name == discovered.name) {
            catalog.push(discovered.clone());
        }
    }
    catalog
}

/// Handle /skill command for interactive skill selection.
pub fn handle_skill_command(args: &[&str], service: &mut AgentService) -> Result<()> {
    let catalog = skill_catalog();
    if args.is_empty() {
        let theme = ColorfulTheme::default();
        let items: Vec<String> = catalog
            .iter()
            .map(|skill| format!("{} - {}", skill.name, skill.description))
            .collect();

        let selection = FuzzySelect::with_theme(&theme)
            .with_prompt("Select a skill to apply")
            .items(&items)
            .default(0)
            .interact()?;

        let skill_name = &catalog[selection].name;
        apply_skill_by_name(skill_name, service)?;
    } else {
        let skill_name = args[0];
        apply_skill_by_name(skill_name, service)?;
    }

    Ok(())
}

/// Apply a skill by name.
pub fn apply_skill_by_name(name: &str, service: &mut AgentService) -> Result<()> {
    let skill = skill_catalog()
        .into_iter()
        .find(|s| s.name == name)
        .ok_or_else(|| anyhow::anyhow!("Unknown skill: {}", name))?;

    println!("Applying skill: {} - {}", skill.name, skill.description);
    service.agent_mut().apply_skill(&skill);
    println!("Skill applied successfully.");

    if !skill.required_tools.is_empty() {
        println!(
            "Required tools: {}",
            skill
                .required_tools
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    Ok(())
}
