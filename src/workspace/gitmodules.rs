//! Parse `.gitmodules` without walking the worktree.

use std::path::{Component, Path};

/// One `[submodule]` entry from `.gitmodules`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GitmoduleEntry {
    pub(crate) name: String,
    pub(crate) relative_path: String,
}

/// Parse `[submodule "name"]` / `path = ...` entries.
///
/// Absolute paths and `..` components are omitted. Quoted git-config values
/// are accepted. Only a `path` key inside a submodule section counts.
#[must_use]
pub(crate) fn parse_gitmodules(contents: &str) -> Vec<GitmoduleEntry> {
    let mut entries = Vec::new();
    let mut in_submodule = false;
    let mut name = None;
    let mut relative_path = None;
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }
        if trimmed.starts_with('[') {
            push_gitmodules_entry(&mut entries, &mut name, &mut relative_path);
            in_submodule = false;
            if let Some(section_name) = submodule_section_name(trimmed) {
                in_submodule = true;
                name = Some(section_name);
            }
            continue;
        }
        if !in_submodule {
            continue;
        }
        let Some(value) = value_for_key(trimmed, "path") else {
            continue;
        };
        let unquoted = unquote_git_value(value);
        if is_safe_relative(&unquoted) {
            relative_path = Some(unquoted);
        } else {
            relative_path = None;
        }
    }
    push_gitmodules_entry(&mut entries, &mut name, &mut relative_path);
    entries
}

fn push_gitmodules_entry(
    entries: &mut Vec<GitmoduleEntry>,
    name: &mut Option<String>,
    relative_path: &mut Option<String>,
) {
    if let (Some(name), Some(relative_path)) = (name.take(), relative_path.take()) {
        entries.push(GitmoduleEntry {
            name,
            relative_path,
        });
    }
}

fn submodule_section_name(header: &str) -> Option<String> {
    let inner = header.strip_prefix('[')?.strip_suffix(']')?.trim();
    let rest = inner.strip_prefix("submodule")?.trim();
    let quoted = rest.strip_prefix('"')?.strip_suffix('"')?;
    let name = unescape_git_chars(quoted);
    if name.is_empty() { None } else { Some(name) }
}

fn value_for_key<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (found, value) = line.split_once('=')?;
    if found.trim() != key {
        return None;
    }
    Some(value.trim())
}

fn unquote_git_value(value: &str) -> String {
    let trimmed = value.trim();
    let Some(inner) = trimmed
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
    else {
        return trimmed.to_string();
    };
    unescape_git_chars(inner)
}

fn unescape_git_chars(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(next) = chars.next() {
                out.push(next);
            }
        } else {
            out.push(ch);
        }
    }
    out
}

pub(crate) fn is_safe_relative(value: &str) -> bool {
    if value.is_empty() {
        return false;
    }
    let path = Path::new(value);
    if path.is_absolute() {
        return false;
    }
    path.components()
        .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}
