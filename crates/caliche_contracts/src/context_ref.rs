//! Shared `@` context-reference parsing for CLI / UI / API hosts (ROADMAP 2.10).
//!
//! Resolution (file I/O, MCP, artifacts) stays in host crates; this module only
//! classifies the mention text after the `@` prefix.

use serde::{Deserialize, Serialize};

/// Parsed `@` mention kind without host-specific payload resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ContextRefKind {
    File { path: String },
    Directory { path: String },
    Url { url: String },
    Symbol { name: String },
    Clipboard,
    GitDiff,
    GitStatus,
    Artifact { selector: String },
    Resource { server: Option<String>, uri: String },
}

/// Parse a context reference from a string without the leading `@`.
#[must_use]
pub fn parse_context_ref_kind(input: &str) -> Option<ContextRefKind> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }

    if let Some(selector) = input.strip_prefix("artifact:") {
        return Some(ContextRefKind::Artifact {
            selector: selector.to_string(),
        });
    }

    if let Some(selector) = input.strip_prefix("resource:") {
        return parse_resource_kind(selector);
    }

    match input {
        "clipboard" | "clip" => return Some(ContextRefKind::Clipboard),
        "git:diff" | "diff" => return Some(ContextRefKind::GitDiff),
        "git:status" | "status" => return Some(ContextRefKind::GitStatus),
        _ => {}
    }

    if input.starts_with("http://") || input.starts_with("https://") {
        return Some(ContextRefKind::Url {
            url: input.to_string(),
        });
    }

    if input.ends_with('/') {
        Some(ContextRefKind::Directory {
            path: input.to_string(),
        })
    } else if input.contains('.') || input.contains('/') {
        Some(ContextRefKind::File {
            path: input.to_string(),
        })
    } else {
        Some(ContextRefKind::Symbol {
            name: input.to_string(),
        })
    }
}

fn parse_resource_kind(selector: &str) -> Option<ContextRefKind> {
    let selector = selector.trim();
    if selector.is_empty() {
        return None;
    }
    if let Some((server, uri)) = selector.split_once("::") {
        Some(ContextRefKind::Resource {
            server: Some(server.to_string()),
            uri: uri.to_string(),
        })
    } else {
        Some(ContextRefKind::Resource {
            server: None,
            uri: selector.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_context_ref_kind_covers_common_mentions() {
        assert_eq!(
            parse_context_ref_kind("diff"),
            Some(ContextRefKind::GitDiff)
        );
        assert_eq!(
            parse_context_ref_kind("clip"),
            Some(ContextRefKind::Clipboard)
        );
        assert_eq!(
            parse_context_ref_kind("https://example.com"),
            Some(ContextRefKind::Url {
                url: "https://example.com".into()
            })
        );
        assert_eq!(
            parse_context_ref_kind("src/main.rs"),
            Some(ContextRefKind::File {
                path: "src/main.rs".into()
            })
        );
        assert_eq!(
            parse_context_ref_kind("MyType"),
            Some(ContextRefKind::Symbol {
                name: "MyType".into()
            })
        );
        assert_eq!(
            parse_context_ref_kind("artifact:latest"),
            Some(ContextRefKind::Artifact {
                selector: "latest".into()
            })
        );
        assert_eq!(
            parse_context_ref_kind("resource:srv::uri://x"),
            Some(ContextRefKind::Resource {
                server: Some("srv".into()),
                uri: "uri://x".into()
            })
        );
    }
}
