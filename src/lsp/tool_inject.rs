//! Inject LSP diagnostics into filesystem tool results.

use crate::agent::tools::ToolResult;
use crate::lsp::diagnostics::DiagnosticFormatter;
use crate::lsp::manager::LspManager;
use crate::lsp::types::Diagnostic;
use serde_json::json;
use std::path::Path;
use tracing::debug;

/// Collect diagnostics for `path` when LSP is enabled and a language maps to it.
///
/// Returns `None` when injection should be skipped (disabled, unknown language,
/// or collection failure). Failures are logged and never fail the tool call.
pub async fn collect_diagnostics_for_path(
    manager: &LspManager,
    path: &Path,
) -> Option<Vec<Diagnostic>> {
    if !manager.is_enabled() {
        return None;
    }
    if manager.registry().get_language_for_path(path).is_none() {
        return None;
    }
    match manager.diagnostics_for_path(path).await {
        Ok(diagnostics) => Some(diagnostics),
        Err(error) => {
            debug!(
                path = %path.display(),
                error = %error,
                "LSP diagnostic collection skipped"
            );
            None
        }
    }
}

/// Append formatted diagnostics to a successful tool result for one path.
pub async fn inject_path_diagnostics(
    manager: &LspManager,
    path: &Path,
    result: ToolResult,
) -> ToolResult {
    if !result.success {
        return result;
    }
    let Some(diagnostics) = collect_diagnostics_for_path(manager, path).await else {
        return result;
    };
    append_diagnostics(result, path, &diagnostics)
}

/// Append diagnostics for multiple paths (for example after `apply_patch`).
pub async fn inject_paths_diagnostics(
    manager: &LspManager,
    paths: &[impl AsRef<Path>],
    mut result: ToolResult,
) -> ToolResult {
    if !result.success || !manager.is_enabled() {
        return result;
    }
    for path in paths {
        let path = path.as_ref();
        let Some(diagnostics) = collect_diagnostics_for_path(manager, path).await else {
            continue;
        };
        result = append_diagnostics(result, path, &diagnostics);
    }
    result
}

fn append_diagnostics(
    mut result: ToolResult,
    path: &Path,
    diagnostics: &[Diagnostic],
) -> ToolResult {
    if diagnostics.is_empty() {
        return result;
    }

    let formatted = DiagnosticFormatter::format_for_llm(diagnostics);
    let summary = DiagnosticFormatter::summary(diagnostics);
    let note = format!(
        "\n\nLSP diagnostics for {} ({summary}):\n{formatted}",
        path.display()
    );
    result.content.push_str(&note);

    let warning = format!("{}: {summary}", path.display());
    match &mut result.warnings {
        Some(warnings) => warnings.push(warning),
        None => result.warnings = Some(vec![warning]),
    }

    match result
        .typed_payload
        .as_mut()
        .and_then(serde_json::Value::as_object_mut)
    {
        Some(obj) => {
            let entry = obj
                .entry("lsp_diagnostics".to_string())
                .or_insert_with(|| json!([]));
            if let Some(arr) = entry.as_array_mut() {
                arr.push(json!({
                    "path": path.display().to_string(),
                    "summary": summary,
                    "diagnostics": diagnostics,
                }));
            }
        }
        None => {
            result.typed_payload = Some(json!({
                "lsp_diagnostics": [{
                    "path": path.display().to_string(),
                    "summary": summary,
                    "diagnostics": diagnostics,
                }]
            }));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsp::types::{DiagnosticSeverity, Position, Range};

    #[test]
    fn append_diagnostics_adds_content_warning_and_payload() {
        let diag = Diagnostic {
            range: Range::new(Position::new(0, 0), Position::new(0, 1)),
            severity: DiagnosticSeverity::Error,
            code: Some("E0001".to_string()),
            source: Some("test".to_string()),
            message: "boom".to_string(),
            related_info: vec![],
        };
        let result = append_diagnostics(
            ToolResult::success("wrote file"),
            Path::new("src/main.rs"),
            &[diag],
        );
        assert!(result.content.contains("LSP diagnostics for src/main.rs"));
        assert!(result.content.contains("boom"));
        assert_eq!(result.warnings.as_ref().map(Vec::len), Some(1));
        let payload = result.typed_payload.expect("payload");
        assert!(
            payload["lsp_diagnostics"]
                .as_array()
                .is_some_and(|a| !a.is_empty())
        );
    }

    #[test]
    fn append_diagnostics_skips_empty() {
        let result = append_diagnostics(ToolResult::success("ok"), Path::new("a.rs"), &[]);
        assert_eq!(result.content, "ok");
        assert!(result.warnings.is_none());
    }
}
