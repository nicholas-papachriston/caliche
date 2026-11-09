//! Context reference parsing and handling for @ mentions.
//!
//! Allows users to attach context using @ symbol:
//! - `@file.rs` - Attach a file
//! - `@src/` - List a directory
//! - `@diff` - Attach git diff
//! - `@status` - Attach git status
//! - `@clipboard` - Attach clipboard content
//! - `@https://...` - Fetch URL content

use serde::{Deserialize, Serialize};

/// Available context reference types with descriptions
pub const CONTEXT_REFERENCES: &[(&str, &str)] = &[
    ("@diff", "Git diff (uncommitted changes)"),
    ("@status", "Git status"),
    ("@clipboard", "Clipboard content"),
    ("@staged", "Git staged changes"),
];

/// A context reference parsed from @ symbol
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ContextReference {
    /// Reference to a file path
    File(String),
    /// Reference to a directory
    Directory(String),
    /// Reference to a URL
    Url(String),
    /// Reference to clipboard content
    Clipboard,
    /// Reference to git diff
    GitDiff,
    /// Reference to git staged changes
    GitStaged,
    /// Reference to git status
    GitStatus,
}

impl ContextReference {
    /// Parse a context reference from a string (without the @ prefix)
    pub fn parse(input: &str) -> Option<Self> {
        let input = input.trim();

        if input.is_empty() {
            return None;
        }

        // Special references
        match input.to_lowercase().as_str() {
            "clipboard" | "clip" => return Some(ContextReference::Clipboard),
            "git:diff" | "diff" => return Some(ContextReference::GitDiff),
            "git:staged" | "staged" => return Some(ContextReference::GitStaged),
            "git:status" | "status" => return Some(ContextReference::GitStatus),
            _ => {}
        }

        // URL reference
        if input.starts_with("http://") || input.starts_with("https://") {
            return Some(ContextReference::Url(input.to_string()));
        }

        // File or directory reference
        if input.ends_with('/') {
            Some(ContextReference::Directory(input.to_string()))
        } else if input.contains('.') || input.contains('/') {
            Some(ContextReference::File(input.to_string()))
        } else {
            // Could be a file without extension
            Some(ContextReference::File(input.to_string()))
        }
    }

    /// Get a display name for this reference
    pub fn display_name(&self) -> String {
        match self {
            ContextReference::File(path) => format!("@{}", path),
            ContextReference::Directory(path) => format!("@{}", path),
            ContextReference::Url(url) => format!("@{}", url),
            ContextReference::Clipboard => "@clipboard".to_string(),
            ContextReference::GitDiff => "@diff".to_string(),
            ContextReference::GitStaged => "@staged".to_string(),
            ContextReference::GitStatus => "@status".to_string(),
        }
    }

    /// Get a short label for display in UI
    pub fn short_label(&self) -> &str {
        match self {
            ContextReference::File(_) => "[file]",
            ContextReference::Directory(_) => "[dir]",
            ContextReference::Url(_) => "[url]",
            ContextReference::Clipboard => "[clip]",
            ContextReference::GitDiff => "[diff]",
            ContextReference::GitStaged => "[staged]",
            ContextReference::GitStatus => "[status]",
        }
    }
}

/// Parsed input with context references extracted
#[derive(Debug, Clone, Default)]
pub struct ParsedInput {
    /// The clean message text (with @ references removed)
    pub message: String,
    /// Extracted context references
    pub references: Vec<ContextReference>,
}

/// Parse all @ references from input and return parsed result
pub fn parse_context_references(input: &str) -> ParsedInput {
    let mut references = Vec::new();
    let mut clean_parts = Vec::new();
    let mut current_pos = 0;

    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Check for @ at start of word
        if chars[i] == '@' && (i == 0 || chars[i - 1].is_whitespace()) {
            let start = i + 1;
            let mut end = start;

            // Collect until whitespace or end
            while end < chars.len() && !chars[end].is_whitespace() {
                end += 1;
            }

            if end > start {
                let ref_str: String = chars[start..end].iter().collect();
                if let Some(context_ref) = ContextReference::parse(&ref_str) {
                    references.push(context_ref);
                }
                // Add text before the @ reference
                if i > current_pos {
                    clean_parts.push(chars[current_pos..i].iter().collect::<String>());
                }
                current_pos = end;
                i = end;
                continue;
            }
        }
        i += 1;
    }

    // Add remaining text
    if current_pos < chars.len() {
        clean_parts.push(chars[current_pos..].iter().collect::<String>());
    }

    let message = clean_parts
        .join("")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    ParsedInput {
        message,
        references,
    }
}

/// Resolve a list of [`ContextReference`]s into an expanded text block that
/// can be prepended to an outgoing chat message.
///
/// This delegates to the API client for anything that needs server-side
/// knowledge (`@file`, `@directory`, `@diff`, `@status`, `@staged`), handles
/// `@clipboard` via the browser clipboard when running in WASM, and fetches
/// `@url` references through the shared [`reqwest::Client`] owned by the API
/// client. Errors are captured inline so one failing reference does not
/// silently drop the whole message.
///
/// The returned [`ResolvedContext`] contains a human-readable block to prepend
/// to the user's message plus structured entries so callers can render badges
/// or attachments later without re-parsing.
pub async fn resolve_context(
    references: &[ContextReference],
    client: &crate::api::ApiClient,
    session_id: Option<&str>,
) -> ResolvedContext {
    let mut entries: Vec<ResolvedEntry> = Vec::with_capacity(references.len());

    for reference in references {
        let entry = match reference {
            ContextReference::File(path) => match client.read_file(path, Some(4_000)).await {
                Ok(resp) => ResolvedEntry::ok(
                    reference.clone(),
                    format!("# File: {path}\n\n```\n{}\n```", resp.content),
                ),
                Err(err) => ResolvedEntry::err(reference.clone(), err.to_string()),
            },
            ContextReference::Directory(path) => match client.list_directory(Some(path)).await {
                Ok(resp) => {
                    let names: Vec<String> = resp
                        .entries
                        .iter()
                        .map(|e| {
                            if e.is_dir {
                                format!("{}/", e.name)
                            } else {
                                e.name.clone()
                            }
                        })
                        .collect();
                    ResolvedEntry::ok(
                        reference.clone(),
                        format!("# Directory: {path}\n\n{}", names.join("\n")),
                    )
                }
                Err(err) => ResolvedEntry::err(reference.clone(), err.to_string()),
            },
            ContextReference::Url(url) => {
                let http = reqwest::Client::new();
                match http.get(url).send().await {
                    Ok(resp) => match resp.error_for_status() {
                        Ok(resp) => match resp.text().await {
                            Ok(body) => {
                                let trimmed = truncate_for_context(&body, 16 * 1024);
                                ResolvedEntry::ok(
                                    reference.clone(),
                                    format!("# URL: {url}\n\n{trimmed}"),
                                )
                            }
                            Err(err) => ResolvedEntry::err(reference.clone(), err.to_string()),
                        },
                        Err(err) => ResolvedEntry::err(reference.clone(), err.to_string()),
                    },
                    Err(err) => ResolvedEntry::err(reference.clone(), err.to_string()),
                }
            }
            ContextReference::Clipboard => match read_clipboard().await {
                Ok(text) => ResolvedEntry::ok(
                    reference.clone(),
                    format!("# Clipboard\n\n```\n{text}\n```"),
                ),
                Err(err) => ResolvedEntry::err(reference.clone(), err),
            },
            ContextReference::GitDiff
            | ContextReference::GitStaged
            | ContextReference::GitStatus => {
                let Some(session_id) = session_id else {
                    entries.push(ResolvedEntry::err(
                        reference.clone(),
                        "diff/status references require an active session".to_string(),
                    ));
                    continue;
                };
                match client.get_diff(session_id).await {
                    Ok(resp) => {
                        let heading = match reference {
                            ContextReference::GitDiff => "Git diff",
                            ContextReference::GitStaged => "Git diff (staged)",
                            ContextReference::GitStatus => "Git status",
                            _ => unreachable!(),
                        };
                        let files = if resp.files_changed.is_empty() {
                            "(no files changed)".to_string()
                        } else {
                            resp.files_changed.join("\n")
                        };
                        ResolvedEntry::ok(reference.clone(), format!("# {heading}\n\n{files}"))
                    }
                    Err(err) => ResolvedEntry::err(reference.clone(), err.to_string()),
                }
            }
        };
        entries.push(entry);
    }

    let mut block = String::new();
    for entry in &entries {
        if !block.is_empty() {
            block.push_str("\n\n");
        }
        match &entry.body {
            Ok(body) => block.push_str(body),
            Err(err) => block.push_str(&format!(
                "# {}\n\n_error resolving reference: {err}_",
                entry.reference.display_name()
            )),
        }
    }

    ResolvedContext { block, entries }
}

fn truncate_for_context(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    format!(
        "{}\n…truncated",
        caliche_contracts::text::truncate_utf8(text, max_bytes)
    )
}

async fn read_clipboard() -> Result<String, String> {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen_futures::JsFuture;

        let window = match web_sys::window() {
            Some(w) => w,
            None => return Err("no window available".to_string()),
        };
        let nav = window.navigator();
        let clipboard = nav.clipboard();
        let promise = clipboard.read_text();
        let js = JsFuture::from(promise)
            .await
            .map_err(|e| format!("{:?}", e))?;
        js.as_string()
            .ok_or_else(|| "clipboard value not a string".to_string())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        // On native, we don't bundle a system clipboard crate by default.
        // Rather than failing the whole message, return a friendly placeholder
        // so callers can see that @clipboard is WASM-only right now.
        Err("clipboard access is only available in the WASM UI".to_string())
    }
}

/// Result of resolving a single [`ContextReference`].
#[derive(Debug, Clone)]
pub struct ResolvedEntry {
    pub reference: ContextReference,
    pub body: Result<String, String>,
}

impl ResolvedEntry {
    fn ok(reference: ContextReference, body: String) -> Self {
        Self {
            reference,
            body: Ok(body),
        }
    }

    fn err(reference: ContextReference, err: String) -> Self {
        Self {
            reference,
            body: Err(err),
        }
    }
}

/// Combined result of resolving every reference for a single user turn.
#[derive(Debug, Clone, Default)]
pub struct ResolvedContext {
    /// Pre-formatted markdown block to prepend to the outgoing message.
    pub block: String,
    /// Per-reference resolution outcomes (for badges, tooltips, retries).
    #[allow(
        dead_code,
        reason = "nothing reads it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub entries: Vec<ResolvedEntry>,
}
