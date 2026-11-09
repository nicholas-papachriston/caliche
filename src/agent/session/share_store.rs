//! Share payload storage backends.

use crate::agent::session::share::ShareVisibility;
use anyhow::{Context, Result};
use caliche_contracts::sessions::ShareMessageDto;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

/// Stored redacted transcript share payload.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SharePayload {
    pub slug: String,
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub messages: Vec<ShareMessageDto>,
    #[serde(default)]
    pub redaction_count: usize,
    #[serde(default)]
    pub visibility: ShareVisibility,
}

/// Storage abstraction for redacted transcript shares.
pub trait ShareStore: Send + Sync {
    /// Store or replace a share payload.
    fn put(&self, payload: SharePayload) -> Result<SharePayload>;

    /// Resolve a share slug.
    fn get(&self, slug: &str) -> Option<SharePayload>;

    /// Delete all shares for a session.
    fn delete_session(&self, session_id: &str) -> bool;

    /// Delete one share by slug.
    fn delete_slug(&self, slug: &str) -> bool;
}

/// Share slugs are URL path segments and on-disk filenames.
///
/// Only ASCII alphanumeric characters, `-`, and `_` are allowed so `../`
/// cannot escape `CALICHE_SHARE_DIR`.
#[must_use]
pub fn is_safe_share_slug(slug: &str) -> bool {
    let len = slug.len();
    (1..=128).contains(&len)
        && slug
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

/// Process-local share store used by default.
#[derive(Debug, Default)]
pub struct MemoryShareStore {
    records: DashMap<String, SharePayload>,
}

impl MemoryShareStore {
    /// Create an empty memory share store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl ShareStore for MemoryShareStore {
    fn put(&self, payload: SharePayload) -> Result<SharePayload> {
        if !is_safe_share_slug(&payload.slug) {
            anyhow::bail!("share slug is not a safe filename");
        }
        self.records.insert(payload.slug.clone(), payload.clone());
        Ok(payload)
    }

    fn get(&self, slug: &str) -> Option<SharePayload> {
        if !is_safe_share_slug(slug) {
            return None;
        }
        self.records.get(slug).map(|entry| entry.value().clone())
    }

    fn delete_session(&self, session_id: &str) -> bool {
        let mut removed = false;
        self.records.retain(|_, payload| {
            let keep = payload.session_id != session_id;
            removed |= !keep;
            keep
        });
        removed
    }

    fn delete_slug(&self, slug: &str) -> bool {
        self.records.remove(slug).is_some()
    }
}

/// Durable share store that mirrors records to JSON files under a directory.
///
/// Each share is written to `{dir}/{slug}.json`. Reads prefer the in-memory
/// cache and fall back to disk so process restarts can recover shares.
#[derive(Debug)]
pub struct DirectoryShareStore {
    dir: PathBuf,
    memory: MemoryShareStore,
}

impl DirectoryShareStore {
    /// Create a store rooted at `dir`, creating the directory when needed.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)
            .with_context(|| format!("create share directory {}", dir.display()))?;
        let store = Self {
            dir,
            memory: MemoryShareStore::new(),
        };
        store.load_existing()?;
        Ok(store)
    }

    /// Build a store from `CALICHE_SHARE_DIR`, or return `None` when unset.
    pub fn from_env() -> Result<Option<Self>> {
        match std::env::var_os("CALICHE_SHARE_DIR") {
            Some(value) if !value.is_empty() => Ok(Some(Self::new(PathBuf::from(value))?)),
            _ => Ok(None),
        }
    }

    fn path_for_slug(&self, slug: &str) -> Option<PathBuf> {
        if !is_safe_share_slug(slug) {
            return None;
        }
        Some(self.dir.join(format!("{slug}.json")))
    }

    fn load_existing(&self) -> Result<()> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("read share directory {}", self.dir.display()));
            }
        };
        for entry in entries {
            let entry = entry.with_context(|| {
                format!("read share directory entry under {}", self.dir.display())
            })?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let bytes =
                fs::read(&path).with_context(|| format!("read share file {}", path.display()))?;
            let payload: SharePayload = serde_json::from_slice(&bytes)
                .with_context(|| format!("decode share file {}", path.display()))?;
            self.memory.put(payload)?;
        }
        Ok(())
    }

    fn write_payload(&self, payload: &SharePayload) -> Result<()> {
        let path = self
            .path_for_slug(&payload.slug)
            .ok_or_else(|| anyhow::anyhow!("share slug is not a safe filename"))?;
        let bytes = serde_json::to_vec_pretty(payload).context("serialize share payload")?;
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, &bytes)
            .with_context(|| format!("write temp share file {}", tmp.display()))?;
        fs::rename(&tmp, &path).with_context(|| format!("commit share file {}", path.display()))?;
        Ok(())
    }

    fn remove_file(&self, slug: &str) {
        let Some(path) = self.path_for_slug(slug) else {
            return;
        };
        let _ = fs::remove_file(path);
    }
}

impl ShareStore for DirectoryShareStore {
    fn put(&self, payload: SharePayload) -> Result<SharePayload> {
        self.write_payload(&payload)?;
        self.memory.put(payload)
    }

    fn get(&self, slug: &str) -> Option<SharePayload> {
        if !is_safe_share_slug(slug) {
            return None;
        }
        if let Some(payload) = self.memory.get(slug) {
            return Some(payload);
        }
        let path = self.path_for_slug(slug)?;
        let bytes = fs::read(&path).ok()?;
        let payload: SharePayload = serde_json::from_slice(&bytes).ok()?;
        let _ = self.memory.put(payload.clone());
        Some(payload)
    }

    fn delete_session(&self, session_id: &str) -> bool {
        let slugs: Vec<String> = self
            .memory
            .records
            .iter()
            .filter(|entry| entry.value().session_id == session_id)
            .map(|entry| entry.key().clone())
            .collect();
        let mut removed = false;
        for slug in &slugs {
            self.remove_file(slug);
            removed |= self.memory.delete_slug(slug);
        }
        // Also scan disk for orphans not in memory.
        if let Ok(entries) = fs::read_dir(&self.dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                    continue;
                }
                if let Ok(bytes) = fs::read(&path)
                    && let Ok(payload) = serde_json::from_slice::<SharePayload>(&bytes)
                    && payload.session_id == session_id
                {
                    let _ = fs::remove_file(&path);
                    removed |= self.memory.delete_slug(&payload.slug);
                }
            }
        }
        removed
    }

    fn delete_slug(&self, slug: &str) -> bool {
        self.remove_file(slug);
        self.memory.delete_slug(slug)
    }
}

/// Build the default share store for API hosts.
///
/// Uses [`DirectoryShareStore`] when `CALICHE_SHARE_DIR` is set; otherwise
/// falls back to an in-memory store.
pub fn default_share_store() -> Result<Arc<dyn ShareStore>> {
    if let Some(store) = DirectoryShareStore::from_env()? {
        return Ok(Arc::new(store));
    }
    Ok(Arc::new(MemoryShareStore::new()))
}

/// Escape helper kept next to HTML rendering for share views.
#[must_use]
pub fn escape_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

/// Render a redacted share payload as a minimal public HTML page.
#[must_use]
pub fn render_share_html(payload: &SharePayload) -> String {
    let mut messages = String::new();
    for message in &payload.messages {
        let role = escape_html(&message.role);
        let content = escape_html(&message.content);
        messages.push_str(&format!(
            "<article class=\"message\" data-role=\"{role}\">\
               <header>{role}</header>\
               <pre>{content}</pre>\
             </article>"
        ));
    }
    format!(
        "<!DOCTYPE html>\
<html lang=\"en\">\
<head>\
<meta charset=\"utf-8\">\
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
<title>Shared session {slug}</title>\
<style>\
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#111;color:#eee;}}\
main{{max-width:52rem;margin:0 auto;padding:1.5rem;}}\
h1{{font-size:1.25rem;margin:0 0 0.25rem;}}\
.meta{{color:#9aa;margin-bottom:1.5rem;font-size:0.9rem;}}\
.message{{border-top:1px solid #333;padding:1rem 0;}}\
.message header{{font-weight:600;text-transform:uppercase;letter-spacing:0.04em;font-size:0.75rem;color:#9aa;margin-bottom:0.5rem;}}\
pre{{white-space:pre-wrap;word-break:break-word;margin:0;font:inherit;}}\
</style>\
</head>\
<body>\
<main>\
<h1>Shared session</h1>\
<p class=\"meta\">slug {slug} · session {session} · created {created} · redactions {redactions}</p>\
{messages}\
</main>\
</body>\
</html>",
        slug = escape_html(&payload.slug),
        session = escape_html(&payload.session_id),
        created = escape_html(&payload.created_at.to_rfc3339()),
        redactions = payload.redaction_count,
        messages = messages,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn payload(slug: &str, session_id: &str, content: &str) -> SharePayload {
        SharePayload {
            slug: slug.to_string(),
            session_id: session_id.to_string(),
            created_at: Utc::now(),
            messages: vec![ShareMessageDto {
                role: "user".to_string(),
                content: content.to_string(),
            }],
            redaction_count: 1,
            visibility: ShareVisibility::Public,
        }
    }

    #[test]
    fn memory_store_keeps_payload_copy_after_source_changes() -> Result<()> {
        let store = MemoryShareStore::new();
        let mut original = payload("abc", "session-1", "[REDACTED]");

        store.put(original.clone())?;
        original.messages[0].content = "secret-later".to_string();

        let stored = store
            .get("abc")
            .ok_or_else(|| anyhow::anyhow!("stored payload missing"))?;
        assert_eq!(stored.messages[0].content, "[REDACTED]");
        assert_eq!(stored.redaction_count, 1);
        Ok(())
    }

    #[test]
    fn memory_store_deletes_by_session_and_slug() -> Result<()> {
        let store = MemoryShareStore::new();

        store.put(payload("one", "session-1", "a"))?;
        store.put(payload("two", "session-1", "b"))?;
        store.put(payload("three", "session-2", "c"))?;

        assert!(store.delete_slug("three"));
        assert!(store.get("three").is_none());
        assert!(store.delete_session("session-1"));
        assert!(store.get("one").is_none());
        assert!(store.get("two").is_none());
        assert!(!store.delete_session("session-1"));
        Ok(())
    }

    #[test]
    fn directory_store_survives_reload() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().to_path_buf();
        {
            let store = DirectoryShareStore::new(&path)?;
            store.put(payload("slug-a", "session-1", "hello <world>"))?;
        }
        let reloaded = DirectoryShareStore::new(&path)?;
        let stored = reloaded
            .get("slug-a")
            .ok_or_else(|| anyhow::anyhow!("missing after reload"))?;
        assert_eq!(stored.messages[0].content, "hello <world>");
        assert!(path.join("slug-a.json").exists());
        Ok(())
    }

    #[test]
    fn render_share_html_escapes_content() {
        let html = render_share_html(&payload("s1", "sess", "<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;"));
        assert!(!html.contains("<script>alert"));
        assert!(html.contains("slug s1"));
    }

    #[test]
    fn share_slug_rejects_path_escape() {
        assert!(is_safe_share_slug("abc"));
        assert!(is_safe_share_slug("slug-a_1"));
        assert!(!is_safe_share_slug("../etc/passwd"));
        assert!(!is_safe_share_slug("foo/bar"));
        assert!(!is_safe_share_slug("foo.json"));
        assert!(!is_safe_share_slug(""));
        assert!(!is_safe_share_slug(&"a".repeat(129)));
    }

    #[test]
    fn directory_store_ignores_traversal_slug() -> Result<()> {
        let dir = tempdir()?;
        let store = DirectoryShareStore::new(dir.path())?;
        let mut evil = payload("../secret", "session-1", "nope");
        evil.slug = "../secret".to_string();
        assert!(store.put(evil).is_err());
        assert!(store.get("../secret").is_none());
        Ok(())
    }
}
