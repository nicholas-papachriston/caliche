//! Output formatting traits.
//!
//! Provides a unified interface for formatting command output in different styles,
//! enabling consistent output across all commands.

use crate::errors::CliResult;
use std::io::{IsTerminal, Write};

/// Output style configuration.
#[derive(Debug, Clone, Default)]
pub struct OutputStyle {
    /// Use colors in output
    pub color: bool,

    /// Use unicode characters
    pub unicode: bool,

    /// Verbose output
    pub verbose: bool,

    /// Compact output (minimize whitespace)
    pub compact: bool,
}

impl OutputStyle {
    /// Create a new output style with defaults.
    pub fn new() -> Self {
        Self {
            color: std::io::stdout().is_terminal(),
            unicode: true,
            verbose: false,
            compact: false,
        }
    }

    /// Create a plain style (no colors, no unicode).
    pub fn plain() -> Self {
        Self {
            color: false,
            unicode: false,
            verbose: false,
            compact: false,
        }
    }

    /// Enable colors.
    pub fn with_color(mut self, enabled: bool) -> Self {
        self.color = enabled;
        self
    }

    /// Enable verbose output.
    pub fn with_verbose(mut self, enabled: bool) -> Self {
        self.verbose = enabled;
        self
    }

    /// Enable compact output.
    pub fn with_compact(mut self, enabled: bool) -> Self {
        self.compact = enabled;
        self
    }
}

/// Trait for formatting output.
pub trait OutputFormatter: Send + Sync {
    /// Format a success message.
    fn format_success(&self, message: &str) -> String;

    /// Format an error message.
    fn format_error(&self, message: &str) -> String;

    /// Format a warning message.
    fn format_warning(&self, message: &str) -> String;

    /// Format an info message.
    fn format_info(&self, message: &str) -> String;

    /// Format JSON data.
    fn format_json(&self, data: &serde_json::Value) -> CliResult<String>;

    /// Format a key-value pair.
    fn format_kv(&self, key: &str, value: &str) -> String;

    /// Format a list of items.
    fn format_list(&self, items: &[String]) -> String;

    /// Format a table.
    fn format_table(&self, headers: &[&str], rows: &[Vec<String>]) -> String;

    /// Get the output style.
    fn style(&self) -> &OutputStyle;
}

/// Text output formatter.
pub struct TextFormatter {
    style: OutputStyle,
}

impl Default for TextFormatter {
    fn default() -> Self {
        Self::new(OutputStyle::new())
    }
}

impl TextFormatter {
    /// Create a new text formatter.
    pub fn new(style: OutputStyle) -> Self {
        Self { style }
    }

    fn colorize(&self, text: &str, code: &str) -> String {
        if self.style.color {
            format!("\x1b[{}m{}\x1b[0m", code, text)
        } else {
            text.to_string()
        }
    }
}

impl OutputFormatter for TextFormatter {
    fn format_success(&self, message: &str) -> String {
        let prefix = if self.style.unicode { "✓" } else { "[OK]" };
        self.colorize(&format!("{} {}", prefix, message), "32") // Green
    }

    fn format_error(&self, message: &str) -> String {
        let prefix = if self.style.unicode { "✗" } else { "[ERROR]" };
        self.colorize(&format!("{} {}", prefix, message), "31") // Red
    }

    fn format_warning(&self, message: &str) -> String {
        let prefix = if self.style.unicode { "⚠" } else { "[WARN]" };
        self.colorize(&format!("{} {}", prefix, message), "33") // Yellow
    }

    fn format_info(&self, message: &str) -> String {
        let prefix = if self.style.unicode { "ℹ" } else { "[INFO]" };
        self.colorize(&format!("{} {}", prefix, message), "34") // Blue
    }

    fn format_json(&self, data: &serde_json::Value) -> CliResult<String> {
        if self.style.compact {
            serde_json::to_string(data).map_err(|e| crate::errors::CliError::Input(e.to_string()))
        } else {
            serde_json::to_string_pretty(data)
                .map_err(|e| crate::errors::CliError::Input(e.to_string()))
        }
    }

    fn format_kv(&self, key: &str, value: &str) -> String {
        let formatted_key = self.colorize(key, "1"); // Bold
        format!("{}: {}", formatted_key, value)
    }

    fn format_list(&self, items: &[String]) -> String {
        let bullet = if self.style.unicode { "•" } else { "-" };
        items
            .iter()
            .map(|item| format!("  {} {}", bullet, item))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_table(&self, headers: &[&str], rows: &[Vec<String>]) -> String {
        if headers.is_empty() || rows.is_empty() {
            return String::new();
        }

        // Calculate column widths
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        let mut output = String::new();

        // Header
        let header_line: String = headers
            .iter()
            .zip(&widths)
            .map(|(h, w)| format!("{:width$}", h, width = *w))
            .collect::<Vec<_>>()
            .join("  ");
        output.push_str(&self.colorize(&header_line, "1")); // Bold
        output.push('\n');

        // Separator
        let separator: String = widths
            .iter()
            .map(|w| "-".repeat(*w))
            .collect::<Vec<_>>()
            .join("  ");
        output.push_str(&separator);
        output.push('\n');

        // Rows
        for row in rows {
            let row_line: String = row
                .iter()
                .zip(&widths)
                .map(|(cell, w)| format!("{:width$}", cell, width = *w))
                .collect::<Vec<_>>()
                .join("  ");
            output.push_str(&row_line);
            output.push('\n');
        }

        output
    }

    fn style(&self) -> &OutputStyle {
        &self.style
    }
}

/// JSON output formatter.
pub struct JsonFormatter {
    style: OutputStyle,
    /// Whether to output as JSON Lines (one object per line)
    jsonl: bool,
}

impl JsonFormatter {
    /// Create a new JSON formatter.
    pub fn new(jsonl: bool) -> Self {
        Self {
            style: OutputStyle::plain(),
            jsonl,
        }
    }
}

impl OutputFormatter for JsonFormatter {
    fn format_success(&self, message: &str) -> String {
        serde_json::json!({
            "status": "success",
            "message": message
        })
        .to_string()
    }

    fn format_error(&self, message: &str) -> String {
        serde_json::json!({
            "status": "error",
            "message": message
        })
        .to_string()
    }

    fn format_warning(&self, message: &str) -> String {
        serde_json::json!({
            "status": "warning",
            "message": message
        })
        .to_string()
    }

    fn format_info(&self, message: &str) -> String {
        serde_json::json!({
            "status": "info",
            "message": message
        })
        .to_string()
    }

    fn format_json(&self, data: &serde_json::Value) -> CliResult<String> {
        if self.jsonl {
            serde_json::to_string(data).map_err(|e| crate::errors::CliError::Input(e.to_string()))
        } else {
            serde_json::to_string_pretty(data)
                .map_err(|e| crate::errors::CliError::Input(e.to_string()))
        }
    }

    fn format_kv(&self, key: &str, value: &str) -> String {
        serde_json::json!({ key: value }).to_string()
    }

    fn format_list(&self, items: &[String]) -> String {
        serde_json::json!({ "items": items }).to_string()
    }

    fn format_table(&self, headers: &[&str], rows: &[Vec<String>]) -> String {
        let data: Vec<serde_json::Value> = rows
            .iter()
            .map(|row| {
                let obj: serde_json::Map<String, serde_json::Value> = headers
                    .iter()
                    .zip(row.iter())
                    .map(|(h, v)| (h.to_string(), serde_json::Value::String(v.clone())))
                    .collect();
                serde_json::Value::Object(obj)
            })
            .collect();

        if self.jsonl {
            data.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            serde_json::to_string_pretty(&data).unwrap_or_default()
        }
    }

    fn style(&self) -> &OutputStyle {
        &self.style
    }
}

/// Create a formatter based on output mode.
pub fn create_formatter(mode: crate::output_mode::OutputMode) -> Box<dyn OutputFormatter> {
    use crate::output_mode::OutputMode;
    match mode {
        OutputMode::Text | OutputMode::Stream | OutputMode::Repl => {
            Box::new(TextFormatter::default())
        }
        OutputMode::Json => Box::new(JsonFormatter::new(false)),
        OutputMode::Jsonl => Box::new(JsonFormatter::new(true)),
    }
}

/// Helper to write formatted output.
pub struct OutputWriter<W: Write> {
    writer: W,
    formatter: Box<dyn OutputFormatter>,
}

impl<W: Write> OutputWriter<W> {
    /// Create a new output writer.
    pub fn new(writer: W, formatter: Box<dyn OutputFormatter>) -> Self {
        Self { writer, formatter }
    }

    /// Write a success message.
    pub fn success(&mut self, message: &str) -> std::io::Result<()> {
        writeln!(self.writer, "{}", self.formatter.format_success(message))
    }

    /// Write an error message.
    pub fn error(&mut self, message: &str) -> std::io::Result<()> {
        writeln!(self.writer, "{}", self.formatter.format_error(message))
    }

    /// Write a warning message.
    pub fn warning(&mut self, message: &str) -> std::io::Result<()> {
        writeln!(self.writer, "{}", self.formatter.format_warning(message))
    }

    /// Write an info message.
    pub fn info(&mut self, message: &str) -> std::io::Result<()> {
        writeln!(self.writer, "{}", self.formatter.format_info(message))
    }

    /// Write JSON data.
    pub fn json(&mut self, data: &serde_json::Value) -> CliResult<()> {
        let formatted = self.formatter.format_json(data)?;
        writeln!(self.writer, "{}", formatted)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_formatter_success() {
        let formatter = TextFormatter::new(OutputStyle::plain());
        let output = formatter.format_success("Operation completed");
        assert!(output.contains("OK"));
        assert!(output.contains("Operation completed"));
    }

    #[test]
    fn test_json_formatter() {
        let formatter = JsonFormatter::new(false);
        let output = formatter.format_success("Done");
        assert!(output.contains("success"));
        assert!(output.contains("Done"));
    }

    #[test]
    fn test_table_formatting() {
        let formatter = TextFormatter::new(OutputStyle::plain());
        let headers = vec!["Name", "Value"];
        let rows = vec![
            vec!["foo".to_string(), "bar".to_string()],
            vec!["baz".to_string(), "qux".to_string()],
        ];
        let output = formatter.format_table(&headers, &rows);
        assert!(output.contains("Name"));
        assert!(output.contains("foo"));
        assert!(output.contains("bar"));
    }
}
