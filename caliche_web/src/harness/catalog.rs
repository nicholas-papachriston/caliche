use super::PromptPlacementKind;

/// Shells that must never be a harness executable. Matches the API allowlist policy.
pub const FORBIDDEN_SHELLS: &[&str] = &[
    "sh",
    "bash",
    "zsh",
    "fish",
    "dash",
    "ksh",
    "csh",
    "tcsh",
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
];

/// Built-in CLI preset. Selecting one fills the standard form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HarnessPreset {
    /// Stable id stored on the config (`custom` for a blank form).
    pub id: &'static str,
    /// Short chip label.
    pub label: &'static str,
    /// One sentence that teaches the preset. Shown on the chip.
    pub summary: &'static str,
    /// Executable name or path. Empty for `custom`.
    pub command: &'static str,
    /// Arguments placed before the prompt.
    pub args: &'static [&'static str],
    /// How the task text is delivered.
    pub prompt_placement: PromptPlacementKind,
    /// Capability tags used by the coordinator matcher.
    pub capabilities: &'static [&'static str],
}

/// Presets shown in the add-harness picker. `custom` is last.
pub const HARNESS_PRESETS: &[HarnessPreset] = &[
    HarnessPreset {
        id: "claude-code",
        label: "Claude Code",
        summary: "Anthropic Claude Code print mode. Task text follows -p.",
        command: "claude",
        args: &["-p"],
        prompt_placement: PromptPlacementKind::LastArg,
        capabilities: &["coding", "claude-code"],
    },
    HarnessPreset {
        id: "codex",
        label: "Codex",
        summary: "OpenAI Codex exec. Task text is the last argument.",
        command: "codex",
        args: &["exec"],
        prompt_placement: PromptPlacementKind::LastArg,
        capabilities: &["coding", "codex"],
    },
    HarnessPreset {
        id: "cursor",
        label: "Cursor",
        summary: "Cursor agent CLI. Task text follows --.",
        command: "agent",
        args: &[],
        prompt_placement: PromptPlacementKind::LastArgEoo,
        capabilities: &["coding", "cursor"],
    },
    HarnessPreset {
        id: "opencode",
        label: "OpenCode",
        summary: "OpenCode one-shot run. Task text is the last argument.",
        command: "opencode",
        args: &["run"],
        prompt_placement: PromptPlacementKind::LastArg,
        capabilities: &["coding", "opencode"],
    },
    HarnessPreset {
        id: "pi",
        label: "Pi",
        summary: "Pi print mode. Task text follows -p.",
        command: "pi",
        args: &["-p"],
        prompt_placement: PromptPlacementKind::LastArg,
        capabilities: &["coding", "pi"],
    },
    HarnessPreset {
        id: "hermes",
        label: "Hermes",
        summary: "Hermes quiet chat. Task text follows -q.",
        command: "hermes",
        args: &["chat", "-Q", "-q"],
        prompt_placement: PromptPlacementKind::LastArg,
        capabilities: &["coding", "hermes"],
    },
    HarnessPreset {
        id: "custom",
        label: "Custom",
        summary: "Any CLI. Set the command, args, and how the task text is sent.",
        command: "",
        args: &[],
        prompt_placement: PromptPlacementKind::LastArg,
        capabilities: &["coding"],
    },
];

/// Look up a preset by id. Unknown ids map to `custom`.
#[must_use]
pub fn preset_by_id(id: &str) -> &'static HarnessPreset {
    HARNESS_PRESETS
        .iter()
        .find(|preset| preset.id == id)
        .unwrap_or(&HARNESS_PRESETS[HARNESS_PRESETS.len() - 1])
}

/// True when `command` is a forbidden shell interpreter (basename match).
#[must_use]
pub fn command_is_forbidden_shell(command: &str) -> bool {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return false;
    }
    let name = std::path::Path::new(trimmed)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(trimmed);
    FORBIDDEN_SHELLS
        .iter()
        .any(|shell| name.eq_ignore_ascii_case(shell))
}

/// Sorted unique capability tags from registered harnesses.
#[must_use]
pub fn unique_capabilities<'a>(harnesses: impl IntoIterator<Item = &'a [String]>) -> Vec<String> {
    let mut tags: Vec<String> = Vec::new();
    for caps in harnesses {
        for cap in caps {
            let trimmed = cap.trim();
            if trimmed.is_empty() {
                continue;
            }
            if !tags.iter().any(|existing| existing == trimmed) {
                tags.push(trimmed.to_string());
            }
        }
    }
    tags.sort();
    tags
}

/// Add or remove `tag` from a comma-separated capability field.
#[must_use]
pub fn toggle_csv_tag(current: &str, tag: &str) -> String {
    let tag = tag.trim();
    if tag.is_empty() {
        return current.trim().to_string();
    }
    let mut items: Vec<String> = current
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToString::to_string)
        .collect();
    if let Some(idx) = items.iter().position(|item| item == tag) {
        items.remove(idx);
    } else {
        items.push(tag.to_string());
    }
    items.join(", ")
}
