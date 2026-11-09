//! Output mode definitions for CLI modes.

/// Output mode for CLI operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMode {
    /// REPL mode (default)
    #[default]
    Repl,
    /// Streaming text with spinners
    Stream,
    /// JSON Lines output
    Jsonl,
    /// Single JSON output
    Json,
    /// Plain text output
    Text,
}
