//! I/O abstraction traits for CLI.
//!
//! Provides traits for abstracting input and output operations,
//! enabling testability and alternative implementations.

use crate::errors::CliResult;
use async_trait::async_trait;
use std::io::IsTerminal;

/// Style for styled output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputStyle {
    /// Normal text.
    Normal,
    /// Bold text.
    Bold,
    /// Dimmed text.
    Dim,
    /// Success (green).
    Success,
    /// Warning (yellow).
    Warning,
    /// Error (red).
    Error,
    /// Info (blue).
    Info,
    /// Code/monospace.
    Code,
}

impl OutputStyle {
    /// Get ANSI escape code for this style.
    pub fn ansi_code(&self) -> &'static str {
        match self {
            OutputStyle::Normal => "\x1b[0m",
            OutputStyle::Bold => "\x1b[1m",
            OutputStyle::Dim => "\x1b[2m",
            OutputStyle::Success => "\x1b[32m",
            OutputStyle::Warning => "\x1b[33m",
            OutputStyle::Error => "\x1b[31m",
            OutputStyle::Info => "\x1b[34m",
            OutputStyle::Code => "\x1b[36m",
        }
    }

    /// Get reset code.
    pub fn reset_code() -> &'static str {
        "\x1b[0m"
    }
}

/// Trait for output destinations.
///
/// Allows for different output targets (terminal, file, test mock).
#[async_trait]
pub trait OutputSink: Send + Sync {
    /// Write content to the output.
    fn write(&mut self, content: &str) -> CliResult<()>;

    /// Write content with a style.
    fn write_styled(&mut self, content: &str, style: OutputStyle) -> CliResult<()>;

    /// Write a line (content + newline).
    fn writeln(&mut self, content: &str) -> CliResult<()> {
        self.write(content)?;
        self.write("\n")
    }

    /// Write a styled line.
    fn writeln_styled(&mut self, content: &str, style: OutputStyle) -> CliResult<()> {
        self.write_styled(content, style)?;
        self.write("\n")
    }

    /// Flush any buffered output.
    fn flush(&mut self) -> CliResult<()>;

    /// Check if the output supports colors.
    fn supports_color(&self) -> bool {
        false
    }

    /// Check if the output is a terminal.
    fn is_terminal(&self) -> bool {
        false
    }

    /// Clear the output (if supported).
    fn clear(&mut self) -> CliResult<()> {
        Ok(())
    }
}

/// Trait for input sources.
///
/// Allows for different input sources (terminal, test mock).
#[async_trait]
pub trait InputSource: Send + Sync {
    /// Read a line of input.
    async fn read_line(&mut self, prompt: &str) -> CliResult<Option<String>>;

    /// Read a password (hidden input).
    async fn read_password(&mut self, prompt: &str) -> CliResult<String>;

    /// Add an entry to history.
    fn add_history(&mut self, line: &str);

    /// Check if input is interactive.
    fn is_interactive(&self) -> bool {
        true
    }
}

/// Terminal output sink implementation.
pub struct TerminalSink {
    use_colors: bool,
}

impl TerminalSink {
    /// Create a new terminal sink.
    pub fn new() -> Self {
        Self {
            use_colors: std::io::stdout().is_terminal(),
        }
    }

    /// Create without colors.
    pub fn plain() -> Self {
        Self { use_colors: false }
    }
}

impl Default for TerminalSink {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OutputSink for TerminalSink {
    fn write(&mut self, content: &str) -> CliResult<()> {
        print!("{}", content);
        Ok(())
    }

    fn write_styled(&mut self, content: &str, style: OutputStyle) -> CliResult<()> {
        if self.use_colors {
            print!(
                "{}{}{}",
                style.ansi_code(),
                content,
                OutputStyle::reset_code()
            );
        } else {
            print!("{}", content);
        }
        Ok(())
    }

    fn flush(&mut self) -> CliResult<()> {
        use std::io::Write;
        std::io::stdout()
            .flush()
            .map_err(crate::errors::CliError::Io)
    }

    fn supports_color(&self) -> bool {
        self.use_colors
    }

    fn is_terminal(&self) -> bool {
        std::io::stdout().is_terminal()
    }

    fn clear(&mut self) -> CliResult<()> {
        print!("\x1b[2J\x1b[H");
        self.flush()
    }
}

/// JSON output sink for structured output.
pub struct JsonSink {
    buffer: Vec<serde_json::Value>,
    pretty: bool,
}

impl JsonSink {
    /// Create a new JSON sink.
    pub fn new(pretty: bool) -> Self {
        Self {
            buffer: Vec::new(),
            pretty,
        }
    }

    /// Get buffered output as JSON array.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::Value::Array(self.buffer.clone())
    }

    /// Get buffered output as string.
    pub fn to_string(&self) -> String {
        if self.pretty {
            serde_json::to_string_pretty(&self.to_json()).unwrap_or_default()
        } else {
            serde_json::to_string(&self.to_json()).unwrap_or_default()
        }
    }
}

#[async_trait]
impl OutputSink for JsonSink {
    fn write(&mut self, content: &str) -> CliResult<()> {
        self.buffer.push(serde_json::json!({
            "type": "text",
            "content": content
        }));
        Ok(())
    }

    fn write_styled(&mut self, content: &str, style: OutputStyle) -> CliResult<()> {
        self.buffer.push(serde_json::json!({
            "type": "styled",
            "content": content,
            "style": format!("{:?}", style)
        }));
        Ok(())
    }

    fn flush(&mut self) -> CliResult<()> {
        Ok(())
    }
}

/// Test mock for capturing output.
#[derive(Default)]
pub struct TestSink {
    pub output: Vec<String>,
    pub styled_output: Vec<(String, OutputStyle)>,
}

impl TestSink {
    /// Create a new test sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get all output as a single string.
    pub fn all_output(&self) -> String {
        self.output.join("")
    }
}

#[async_trait]
impl OutputSink for TestSink {
    fn write(&mut self, content: &str) -> CliResult<()> {
        self.output.push(content.to_string());
        Ok(())
    }

    fn write_styled(&mut self, content: &str, style: OutputStyle) -> CliResult<()> {
        self.output.push(content.to_string());
        self.styled_output.push((content.to_string(), style));
        Ok(())
    }

    fn flush(&mut self) -> CliResult<()> {
        Ok(())
    }
}

/// Test mock for providing input.
pub struct TestInput {
    lines: Vec<String>,
    index: usize,
}

impl TestInput {
    /// Create from a list of input lines.
    pub fn new(lines: Vec<String>) -> Self {
        Self { lines, index: 0 }
    }

    /// Create from string slices.
    pub fn from_strs(lines: &[&str]) -> Self {
        Self::new(lines.iter().map(|s| s.to_string()).collect())
    }
}

#[async_trait]
impl InputSource for TestInput {
    async fn read_line(&mut self, _prompt: &str) -> CliResult<Option<String>> {
        if self.index < self.lines.len() {
            let line = self.lines[self.index].clone();
            self.index += 1;
            Ok(Some(line))
        } else {
            Ok(None)
        }
    }

    async fn read_password(&mut self, prompt: &str) -> CliResult<String> {
        self.read_line(prompt).await?.ok_or_else(|| {
            crate::errors::CliError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "No password input available",
            ))
        })
    }

    fn add_history(&mut self, _line: &str) {
        // No-op for test input
    }

    fn is_interactive(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_style_ansi() {
        assert_eq!(OutputStyle::Success.ansi_code(), "\x1b[32m");
        assert_eq!(OutputStyle::Error.ansi_code(), "\x1b[31m");
    }

    #[test]
    fn test_test_sink() {
        let mut sink = TestSink::new();
        if let Err(e) = sink.write("hello ") {
            panic!("sink write: {e}");
        }
        if let Err(e) = sink.write("world") {
            panic!("sink write: {e}");
        }
        assert_eq!(sink.all_output(), "hello world");
    }

    #[tokio::test]
    async fn test_test_input() {
        let mut input = TestInput::from_strs(&["line1", "line2"]);
        let Ok(line) = input.read_line("> ").await else {
            panic!("read_line should succeed");
        };
        assert_eq!(line, Some("line1".to_string()));
        let Ok(line) = input.read_line("> ").await else {
            panic!("read_line should succeed");
        };
        assert_eq!(line, Some("line2".to_string()));
        let Ok(line) = input.read_line("> ").await else {
            panic!("read_line should succeed");
        };
        assert_eq!(line, None);
    }
}
