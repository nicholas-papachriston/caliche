//! User-selectable output verbosity for REPL, CLI, and other agent surfaces.
//!
//! This is display policy, not tracing log level. `--verbose` / `RUST_LOG`
//! still control logs. Surfaces that print model text, tools, thinking, and
//! prompt-cache stats should read [`OutputVerbosity`].

use std::fmt;
use std::str::FromStr;

/// How much agent-turn detail a surface should print.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutputVerbosity {
    /// Answer text only. Hide tools, usage, thinking, and lifecycle lines.
    Quiet,
    /// Default REPL: tool summaries, usage, cache hit when present. No thinking body.
    #[default]
    Normal,
    /// Normal tools plus streamed thinking and detailed prompt-cache stats.
    Thinking,
    /// Full tool output, thinking stream, and detailed prompt-cache stats.
    Verbose,
    /// Verbose plus turn/agent lifecycle lines for debugging a session.
    Debug,
}

/// Parse error for [`OutputVerbosity::from_str`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputVerbosityParseError {
    value: String,
}

impl OutputVerbosityParseError {
    /// Values accepted by [`OutputVerbosity::from_str`].
    #[must_use]
    pub const fn usage() -> &'static str {
        "quiet, normal, thinking, verbose, or debug"
    }
}

impl OutputVerbosity {
    /// Canonical lowercase name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Quiet => "quiet",
            Self::Normal => "normal",
            Self::Thinking => "thinking",
            Self::Verbose => "verbose",
            Self::Debug => "debug",
        }
    }

    /// Print streamed model thinking/reasoning.
    #[must_use]
    pub const fn show_thinking(self) -> bool {
        matches!(self, Self::Thinking | Self::Verbose | Self::Debug)
    }

    /// Print token usage and prompt-cache counters after a turn.
    #[must_use]
    pub const fn show_usage(self) -> bool {
        !matches!(self, Self::Quiet)
    }

    /// Print uncached/write/read breakdown even when counters are zero.
    #[must_use]
    pub const fn show_cache_detail(self) -> bool {
        matches!(self, Self::Thinking | Self::Verbose | Self::Debug)
    }

    /// Print `turn.start` / `agent.start` lifecycle lines.
    #[must_use]
    pub const fn show_lifecycle(self) -> bool {
        matches!(self, Self::Debug)
    }

    /// Print full tool results instead of a one-line preview.
    #[must_use]
    pub const fn show_full_tool_output(self) -> bool {
        matches!(self, Self::Verbose | Self::Debug)
    }

    /// Print tool argument summaries.
    #[must_use]
    pub const fn show_tool_args(self) -> bool {
        matches!(self, Self::Normal | Self::Verbose | Self::Debug)
    }

    /// Print tool start/complete lines.
    #[must_use]
    pub const fn show_tools(self) -> bool {
        !matches!(self, Self::Quiet)
    }

    /// Toggle used by `/verbose` with no argument.
    #[must_use]
    pub const fn toggle_verbose(self) -> Self {
        match self {
            Self::Quiet | Self::Normal => Self::Verbose,
            Self::Thinking | Self::Verbose | Self::Debug => Self::Normal,
        }
    }

    /// Toggle used by `/compact`.
    #[must_use]
    pub const fn toggle_quiet(self) -> Self {
        if matches!(self, Self::Quiet) {
            Self::Normal
        } else {
            Self::Quiet
        }
    }
}

impl fmt::Display for OutputVerbosity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for OutputVerbosity {
    type Err = OutputVerbosityParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "quiet" | "q" => Ok(Self::Quiet),
            "normal" | "default" => Ok(Self::Normal),
            "thinking" | "think" => Ok(Self::Thinking),
            "verbose" | "v" => Ok(Self::Verbose),
            "debug" => Ok(Self::Debug),
            other => Err(OutputVerbosityParseError {
                value: other.to_string(),
            }),
        }
    }
}

impl fmt::Display for OutputVerbosityParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown output verbosity {:?}; use {}",
            self.value,
            Self::usage()
        )
    }
}

impl std::error::Error for OutputVerbosityParseError {}

#[cfg(test)]
mod tests {
    use super::OutputVerbosity;

    #[test]
    fn parse_accepts_names_and_aliases() {
        assert_eq!("quiet".parse(), Ok(OutputVerbosity::Quiet));
        assert_eq!("NORMAL".parse(), Ok(OutputVerbosity::Normal));
        assert_eq!("default".parse(), Ok(OutputVerbosity::Normal));
        assert_eq!("think".parse(), Ok(OutputVerbosity::Thinking));
        assert_eq!("v".parse(), Ok(OutputVerbosity::Verbose));
        assert_eq!("debug".parse(), Ok(OutputVerbosity::Debug));
        assert!("nope".parse::<OutputVerbosity>().is_err());
    }

    #[test]
    fn thinking_and_lifecycle_gates() {
        assert!(!OutputVerbosity::Normal.show_thinking());
        assert!(OutputVerbosity::Thinking.show_thinking());
        assert!(!OutputVerbosity::Thinking.show_full_tool_output());
        assert!(OutputVerbosity::Verbose.show_thinking());
        assert!(!OutputVerbosity::Verbose.show_lifecycle());
        assert!(OutputVerbosity::Debug.show_lifecycle());
        assert!(!OutputVerbosity::Quiet.show_usage());
        assert!(!OutputVerbosity::Quiet.show_tools());
    }

    #[test]
    fn toggles_round_trip() {
        assert_eq!(
            OutputVerbosity::Normal.toggle_verbose(),
            OutputVerbosity::Verbose
        );
        assert_eq!(
            OutputVerbosity::Thinking.toggle_verbose(),
            OutputVerbosity::Normal
        );
        assert_eq!(
            OutputVerbosity::Verbose.toggle_verbose(),
            OutputVerbosity::Normal
        );
        assert_eq!(
            OutputVerbosity::Debug.toggle_verbose(),
            OutputVerbosity::Normal
        );
        assert_eq!(
            OutputVerbosity::Normal.toggle_quiet(),
            OutputVerbosity::Quiet
        );
        assert_eq!(
            OutputVerbosity::Quiet.toggle_quiet(),
            OutputVerbosity::Normal
        );
    }
}
