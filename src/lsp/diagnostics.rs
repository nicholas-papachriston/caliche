//! Diagnostic collection and formatting utilities.

use crate::lsp::types::Diagnostic;

/// Formatter for diagnostics to be consumed by LLMs.
pub struct DiagnosticFormatter;

impl DiagnosticFormatter {
    /// Format diagnostics for LLM consumption.
    pub fn format_for_llm(diagnostics: &[Diagnostic]) -> String {
        if diagnostics.is_empty() {
            return "No diagnostics".to_string();
        }

        let mut sorted = diagnostics.to_vec();
        // Sort by severity (errors first, then warnings, then info, then hints)
        sorted.sort_by_key(|d| d.severity);

        let lines: Vec<String> = sorted.iter().map(|d| d.format_for_llm()).collect();
        lines.join("\n")
    }

    /// Format diagnostics as JSON for tool results.
    pub fn format_as_json(diagnostics: &[Diagnostic]) -> serde_json::Value {
        serde_json::json!({
            "count": diagnostics.len(),
            "diagnostics": diagnostics,
        })
    }

    /// Get summary of diagnostics by severity.
    pub fn summary(diagnostics: &[Diagnostic]) -> String {
        let errors = diagnostics
            .iter()
            .filter(|d| d.severity == crate::lsp::types::DiagnosticSeverity::Error)
            .count();
        let warnings = diagnostics
            .iter()
            .filter(|d| d.severity == crate::lsp::types::DiagnosticSeverity::Warning)
            .count();
        let info = diagnostics
            .iter()
            .filter(|d| d.severity == crate::lsp::types::DiagnosticSeverity::Information)
            .count();
        let hints = diagnostics
            .iter()
            .filter(|d| d.severity == crate::lsp::types::DiagnosticSeverity::Hint)
            .count();

        format!(
            "{} errors, {} warnings, {} info, {} hints",
            errors, warnings, info, hints
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsp::types::{DiagnosticSeverity, Position, Range};

    #[test]
    fn test_format_empty() {
        let formatted = DiagnosticFormatter::format_for_llm(&[]);
        assert_eq!(formatted, "No diagnostics");
    }

    #[test]
    fn test_format_single_diagnostic() {
        let diag = Diagnostic {
            range: Range::new(Position::new(5, 10), Position::new(5, 15)),
            severity: DiagnosticSeverity::Error,
            code: Some("E0425".to_string()),
            source: Some("rust-analyzer".to_string()),
            message: "cannot find value".to_string(),
            related_info: vec![],
        };

        let formatted = DiagnosticFormatter::format_for_llm(&[diag]);
        assert!(formatted.contains("error"));
        assert!(formatted.contains("E0425"));
        assert!(formatted.contains("cannot find value"));
    }

    #[test]
    fn test_format_sorted_by_severity() {
        let diags = vec![
            Diagnostic {
                range: Range::new(Position::new(1, 0), Position::new(1, 1)),
                severity: DiagnosticSeverity::Hint,
                code: None,
                source: None,
                message: "hint".to_string(),
                related_info: vec![],
            },
            Diagnostic {
                range: Range::new(Position::new(0, 0), Position::new(0, 1)),
                severity: DiagnosticSeverity::Error,
                code: None,
                source: None,
                message: "error".to_string(),
                related_info: vec![],
            },
            Diagnostic {
                range: Range::new(Position::new(2, 0), Position::new(2, 1)),
                severity: DiagnosticSeverity::Warning,
                code: None,
                source: None,
                message: "warning".to_string(),
                related_info: vec![],
            },
        ];

        let formatted = DiagnosticFormatter::format_for_llm(&diags);
        let lines: Vec<&str> = formatted.lines().collect();
        assert_eq!(lines.len(), 3);
        // Error should be first
        assert!(lines[0].contains("error"));
        // Warning should be second
        assert!(lines[1].contains("warning"));
        // Hint should be last
        assert!(lines[2].contains("hint"));
    }

    #[test]
    fn test_summary() {
        let diags = vec![
            Diagnostic {
                range: Range::new(Position::new(0, 0), Position::new(0, 1)),
                severity: DiagnosticSeverity::Error,
                code: None,
                source: None,
                message: "error".to_string(),
                related_info: vec![],
            },
            Diagnostic {
                range: Range::new(Position::new(1, 0), Position::new(1, 1)),
                severity: DiagnosticSeverity::Warning,
                code: None,
                source: None,
                message: "warning".to_string(),
                related_info: vec![],
            },
        ];

        let summary = DiagnosticFormatter::summary(&diags);
        assert!(summary.contains("1 errors"));
        assert!(summary.contains("1 warnings"));
    }
}
