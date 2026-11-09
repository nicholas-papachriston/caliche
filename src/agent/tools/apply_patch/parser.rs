//! Parse Codex-style `apply_patch` text into hunks.

use std::path::PathBuf;

use thiserror::Error;

pub const BEGIN_PATCH_MARKER: &str = "*** Begin Patch";
pub const END_PATCH_MARKER: &str = "*** End Patch";
// The policy gate reads paths from these same headers, so a header the
// parser writes through is one the gate checks.
pub use caliche_policy::{ADD_FILE_MARKER, DELETE_FILE_MARKER, MOVE_TO_MARKER, UPDATE_FILE_MARKER};
pub const EOF_MARKER: &str = "*** End of File";
pub const CHANGE_CONTEXT_MARKER: &str = "@@ ";
pub const EMPTY_CHANGE_CONTEXT_MARKER: &str = "@@";

#[derive(Debug, PartialEq, Eq, Error, Clone)]
pub enum ParseError {
    #[error("invalid patch: {0}")]
    InvalidPatch(String),
    #[error("invalid hunk at line {line_number}: {message}")]
    InvalidHunk { message: String, line_number: usize },
}

#[derive(Debug, PartialEq, Clone)]
pub enum Hunk {
    AddFile {
        path: PathBuf,
        contents: String,
    },
    DeleteFile {
        path: PathBuf,
    },
    UpdateFile {
        path: PathBuf,
        move_path: Option<PathBuf>,
        chunks: Vec<UpdateFileChunk>,
    },
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct UpdateFileChunk {
    pub change_context: Option<String>,
    pub old_lines: Vec<String>,
    pub new_lines: Vec<String>,
    pub is_end_of_file: bool,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ApplyPatchArgs {
    pub patch: String,
    pub hunks: Vec<Hunk>,
}

/// Parse a complete `apply_patch` document.
pub fn parse_patch(patch: &str) -> Result<ApplyPatchArgs, ParseError> {
    let trimmed = strip_heredoc(patch.trim());
    let lines: Vec<&str> = trimmed.lines().collect();
    let patch_lines = check_boundaries(&lines)?;
    let hunks = parse_hunks(patch_lines)?;
    if hunks.is_empty() {
        return Err(ParseError::InvalidPatch(
            "patch must contain at least one hunk".to_string(),
        ));
    }
    Ok(ApplyPatchArgs {
        patch: patch_lines.join("\n"),
        hunks,
    })
}

fn strip_heredoc(patch: &str) -> String {
    let trimmed = patch.trim();
    // Lenient mode for heredoc wrappers produced by weak models.
    if let Some(rest) = trimmed.strip_prefix("<<'EOF'")
        && let Some(body) = rest.strip_suffix("EOF")
    {
        return body.trim().to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("<<EOF")
        && let Some(body) = rest.strip_suffix("EOF")
    {
        return body.trim().to_string();
    }
    trimmed.to_string()
}

fn check_boundaries<'a>(lines: &'a [&'a str]) -> Result<&'a [&'a str], ParseError> {
    if lines.is_empty() {
        return Err(ParseError::InvalidPatch("empty patch".to_string()));
    }
    let first = lines[0].trim();
    if first != BEGIN_PATCH_MARKER {
        return Err(ParseError::InvalidPatch(format!(
            "patch must start with '{BEGIN_PATCH_MARKER}'"
        )));
    }
    let last_nonempty = lines
        .iter()
        .rposition(|line| !line.trim().is_empty())
        .unwrap_or(0);
    if lines[last_nonempty].trim() != END_PATCH_MARKER {
        return Err(ParseError::InvalidPatch(format!(
            "patch must end with '{END_PATCH_MARKER}'"
        )));
    }
    if last_nonempty <= 1 {
        return Err(ParseError::InvalidPatch(
            "patch has no content between markers".to_string(),
        ));
    }
    Ok(&lines[1..last_nonempty])
}

fn parse_hunks(lines: &[&str]) -> Result<Vec<Hunk>, ParseError> {
    let mut hunks = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty() {
            i += 1;
            continue;
        }
        if let Some(path) = line.strip_prefix(ADD_FILE_MARKER) {
            let (contents, next) = parse_add_lines(lines, i + 1)?;
            hunks.push(Hunk::AddFile {
                path: PathBuf::from(path.trim()),
                contents,
            });
            i = next;
            continue;
        }
        if let Some(path) = line.strip_prefix(DELETE_FILE_MARKER) {
            hunks.push(Hunk::DeleteFile {
                path: PathBuf::from(path.trim()),
            });
            i += 1;
            continue;
        }
        if let Some(path) = line.strip_prefix(UPDATE_FILE_MARKER) {
            let (hunk, next) = parse_update_hunk(PathBuf::from(path.trim()), lines, i + 1)?;
            hunks.push(hunk);
            i = next;
            continue;
        }
        return Err(ParseError::InvalidHunk {
            message: format!("unexpected line '{line}'"),
            line_number: i + 2, // +1 for Begin Patch, +1 for 1-index
        });
    }
    Ok(hunks)
}

fn parse_add_lines(lines: &[&str], start: usize) -> Result<(String, usize), ParseError> {
    let mut contents = String::new();
    let mut i = start;
    let mut saw_line = false;
    while i < lines.len() {
        let raw = lines[i];
        let trimmed = raw.trim_start();
        if trimmed.starts_with("*** ") {
            break;
        }
        if let Some(rest) = raw.strip_prefix('+') {
            contents.push_str(rest);
            contents.push('\n');
            saw_line = true;
            i += 1;
            continue;
        }
        if raw.trim().is_empty() {
            i += 1;
            continue;
        }
        return Err(ParseError::InvalidHunk {
            message: "add-file lines must start with '+'".to_string(),
            line_number: i + 2,
        });
    }
    if !saw_line {
        return Err(ParseError::InvalidHunk {
            message: "add-file hunk requires at least one '+' line".to_string(),
            line_number: start + 1,
        });
    }
    Ok((contents, i))
}

fn parse_update_hunk(
    path: PathBuf,
    lines: &[&str],
    start: usize,
) -> Result<(Hunk, usize), ParseError> {
    let mut i = start;
    let mut move_path = None;
    if i < lines.len()
        && let Some(dest) = lines[i].trim().strip_prefix(MOVE_TO_MARKER)
    {
        move_path = Some(PathBuf::from(dest.trim()));
        i += 1;
    }

    let mut chunks = Vec::new();
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.starts_with("*** ") && !trimmed.starts_with(EOF_MARKER) {
            break;
        }
        if trimmed.is_empty() {
            i += 1;
            continue;
        }

        let mut change_context = None;
        if trimmed == EMPTY_CHANGE_CONTEXT_MARKER {
            i += 1;
        } else if let Some(ctx) = trimmed.strip_prefix(CHANGE_CONTEXT_MARKER) {
            change_context = Some(ctx.to_string());
            i += 1;
        }

        let (chunk_lines, next, is_end_of_file) = parse_chunk_lines(lines, i)?;
        i = next;
        if change_context.is_some() || chunk_lines.saw_change {
            chunks.push(UpdateFileChunk {
                change_context,
                old_lines: chunk_lines.old_lines,
                new_lines: chunk_lines.new_lines,
                is_end_of_file,
            });
        } else {
            break;
        }
    }

    if chunks.is_empty() && move_path.is_none() {
        return Err(ParseError::InvalidHunk {
            message: "update hunk requires changes or a move".to_string(),
            line_number: start + 1,
        });
    }

    Ok((
        Hunk::UpdateFile {
            path,
            move_path,
            chunks,
        },
        i,
    ))
}

struct ChunkLines {
    old_lines: Vec<String>,
    new_lines: Vec<String>,
    saw_change: bool,
}

fn parse_chunk_lines(
    lines: &[&str],
    start: usize,
) -> Result<(ChunkLines, usize, bool), ParseError> {
    let mut old_lines = Vec::new();
    let mut new_lines = Vec::new();
    let mut is_end_of_file = false;
    let mut saw_change = false;
    let mut i = start;

    while i < lines.len() {
        let raw = lines[i];
        let t = raw.trim();
        if t == EOF_MARKER {
            is_end_of_file = true;
            i += 1;
            break;
        }
        if t == EMPTY_CHANGE_CONTEXT_MARKER
            || t.starts_with(CHANGE_CONTEXT_MARKER)
            || (t.starts_with("*** ") && t != EOF_MARKER)
        {
            break;
        }
        if let Some(rest) = raw.strip_prefix('+') {
            new_lines.push(rest.to_string());
            saw_change = true;
            i += 1;
        } else if let Some(rest) = raw.strip_prefix('-') {
            old_lines.push(rest.to_string());
            saw_change = true;
            i += 1;
        } else if let Some(rest) = raw.strip_prefix(' ') {
            old_lines.push(rest.to_string());
            new_lines.push(rest.to_string());
            saw_change = true;
            i += 1;
        } else if t.is_empty() {
            i += 1;
        } else {
            return Err(ParseError::InvalidHunk {
                message: format!("invalid update line '{raw}'"),
                line_number: i + 2,
            });
        }
    }

    Ok((
        ChunkLines {
            old_lines,
            new_lines,
            saw_change,
        },
        i,
        is_end_of_file,
    ))
}
