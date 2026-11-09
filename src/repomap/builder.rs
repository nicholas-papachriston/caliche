//! RepoMap builder: scan, rank, and format within token budget.

use crate::repomap::ranker;
use crate::repomap::scanner;
use std::path::Path;

fn estimate_tokens_for_entry(path: &Path, workspace: &Path) -> usize {
    let rel = path
        .strip_prefix(workspace)
        .unwrap_or(path)
        .to_string_lossy();
    let chars = rel.len() + 4;
    (chars / 4) + 2
}

/// Build a token-budgeted codebase map string.
pub fn build(workspace: &Path, max_tokens: usize) -> String {
    let paths = scanner::scan(workspace);
    let ranked = ranker::rank(&paths, None);

    let header = "[Codebase overview]\n";
    let header_tokens = (header.len() / 4) + 2;
    let mut remaining = max_tokens.saturating_sub(header_tokens);

    let mut lines = Vec::new();
    for p in ranked {
        let tok = estimate_tokens_for_entry(&p, workspace);
        if tok > remaining {
            break;
        }
        remaining -= tok;
        let rel = p.strip_prefix(workspace).unwrap_or(&p).to_string_lossy();
        lines.push(format!("- {rel}"));
    }

    if lines.is_empty() {
        return String::new();
    }

    format!("{header}{}", lines.join("\n"))
}
