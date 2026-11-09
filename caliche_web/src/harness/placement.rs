use serde::{Deserialize, Serialize};

/// How the composed prompt is delivered to an external CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PromptPlacementKind {
    /// Prompt is the last argv element (value of a flag such as `-p`).
    #[default]
    LastArg,
    /// Insert `--` immediately before the prompt so it cannot be parsed as a flag.
    LastArgEoo,
    /// Prompt is written to stdin, then stdin is closed.
    Stdin,
}

impl PromptPlacementKind {
    /// Short operator label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LastArg => "last argument",
            Self::LastArgEoo => "last argument after --",
            Self::Stdin => "stdin",
        }
    }

    /// One-line help shown next to the control.
    #[must_use]
    pub const fn help(self) -> &'static str {
        match self {
            Self::LastArg => "The task text is the last CLI argument.",
            Self::LastArgEoo => "The task text follows -- so it cannot look like a flag.",
            Self::Stdin => "The task text is written to stdin.",
        }
    }

    /// All placements in display order.
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [Self::LastArg, Self::LastArgEoo, Self::Stdin]
    }
}
