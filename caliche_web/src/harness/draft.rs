use super::PromptPlacementKind;
use super::catalog::{HarnessPreset, command_is_forbidden_shell, preset_by_id};

/// Operator form state for one harness. Presets fill this; Custom leaves it blank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessDraft {
    pub name: String,
    pub preset_id: String,
    pub command: String,
    /// One argument per line.
    pub args_text: String,
    pub cwd: String,
    pub prompt_placement: PromptPlacementKind,
    /// Comma-separated capability tags.
    pub capabilities_text: String,
}

impl Default for HarnessDraft {
    fn default() -> Self {
        Self::from_preset(preset_by_id("custom"))
    }
}

impl HarnessDraft {
    /// Fill command, args, placement, and capabilities from a preset.
    #[must_use]
    pub fn from_preset(preset: &HarnessPreset) -> Self {
        let mut draft = Self {
            name: String::new(),
            preset_id: preset.id.to_string(),
            command: String::new(),
            args_text: String::new(),
            cwd: String::new(),
            prompt_placement: PromptPlacementKind::LastArg,
            capabilities_text: String::new(),
        };
        draft.apply_preset(preset);
        draft
    }

    /// Apply a preset. Keep an existing name so a rename survives a chip click.
    pub fn apply_preset(&mut self, preset: &HarnessPreset) {
        self.preset_id = preset.id.to_string();
        self.command = preset.command.to_string();
        self.args_text = preset.args.join("\n");
        self.prompt_placement = preset.prompt_placement;
        self.capabilities_text = preset.capabilities.join(", ");
        if self.name.trim().is_empty() && preset.id != "custom" {
            self.name = preset.label.to_string();
        }
    }

    /// Validate and build the create payload.
    pub fn to_create_input(&self) -> Result<CreateHarnessInput, String> {
        let name = require_trimmed(&self.name, "name")?;
        let command = require_trimmed(&self.command, "command")?;
        if command.contains('\0') {
            return Err("command must not contain a NUL byte".to_string());
        }
        if command_is_forbidden_shell(&command) {
            return Err(
                "command cannot be a shell interpreter; pass the harness executable directly"
                    .to_string(),
            );
        }
        let args = parse_arg_lines(&self.args_text)?;
        let cwd = optional_trimmed(&self.cwd);
        let capabilities = parse_csv_list(&self.capabilities_text);
        Ok(CreateHarnessInput {
            name,
            preset_id: self.preset_id.trim().to_string(),
            command,
            args,
            cwd,
            prompt_placement: self.prompt_placement,
            capabilities,
        })
    }
}

/// Validated create payload sent to the API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateHarnessInput {
    pub name: String,
    pub preset_id: String,
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub prompt_placement: PromptPlacementKind,
    pub capabilities: Vec<String>,
}

/// Split a textarea into argv tokens. Empty lines are dropped.
pub fn parse_arg_lines(text: &str) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    for raw in text.lines() {
        let arg = raw.trim();
        if arg.is_empty() {
            continue;
        }
        if arg.contains('\0') {
            return Err("arguments must not contain a NUL byte".to_string());
        }
        args.push(arg.to_string());
    }
    Ok(args)
}

/// Split a comma-separated list. Empty items are dropped.
#[must_use]
pub fn parse_csv_list(text: &str) -> Vec<String> {
    text.split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn require_trimmed(value: &str, field: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{field} is required"));
    }
    Ok(trimmed.to_string())
}

fn optional_trimmed(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}
