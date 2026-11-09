//! Pi-inspired JSONL session tree with id/parentId branching.

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// Current JSONL session format version.
pub const JSONL_SESSION_VERSION: u32 = 3;

/// Header stored as the first line of a JSONL session file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JsonlSessionHeader {
    #[serde(rename = "type")]
    pub entry_type: String,
    pub version: u32,
    pub id: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
}

impl JsonlSessionHeader {
    pub fn new(id: impl Into<String>, cwd: Option<String>) -> Self {
        Self {
            entry_type: "session".to_string(),
            version: JSONL_SESSION_VERSION,
            id: id.into(),
            timestamp: Utc::now(),
            cwd,
            parent_session: None,
        }
    }
}

/// Tree-linked session entry (everything except the header).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JsonlSessionEntry {
    Message {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: DateTime<Utc>,
        message: serde_json::Value,
    },
    CompactionSummary {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: DateTime<Utc>,
        summary: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens_before: Option<usize>,
    },
    BranchSummary {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: DateTime<Utc>,
        summary: String,
        #[serde(rename = "fromId")]
        from_id: String,
    },
    ModelChange {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: DateTime<Utc>,
        provider: String,
        #[serde(rename = "modelId")]
        model_id: String,
    },
}

impl JsonlSessionEntry {
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Message { id, .. }
            | Self::CompactionSummary { id, .. }
            | Self::BranchSummary { id, .. }
            | Self::ModelChange { id, .. } => id,
        }
    }

    #[must_use]
    pub fn parent_id(&self) -> Option<&str> {
        match self {
            Self::Message { parent_id, .. }
            | Self::CompactionSummary { parent_id, .. }
            | Self::BranchSummary { parent_id, .. }
            | Self::ModelChange { parent_id, .. } => parent_id.as_deref(),
        }
    }

    #[must_use]
    pub fn with_parent(mut self, parent_id: Option<String>) -> Self {
        match &mut self {
            Self::Message { parent_id: p, .. }
            | Self::CompactionSummary { parent_id: p, .. }
            | Self::BranchSummary { parent_id: p, .. }
            | Self::ModelChange { parent_id: p, .. } => *p = parent_id,
        }
        self
    }

    pub fn message(
        id: impl Into<String>,
        parent_id: Option<String>,
        message: serde_json::Value,
    ) -> Self {
        Self::Message {
            id: id.into(),
            parent_id,
            timestamp: Utc::now(),
            message,
        }
    }
}

/// In-memory view of a JSONL session tree.
#[derive(Debug, Clone)]
pub struct JsonlSessionTree {
    pub header: JsonlSessionHeader,
    pub entries: Vec<JsonlSessionEntry>,
    by_id: HashMap<String, usize>,
}

impl JsonlSessionTree {
    #[must_use]
    pub fn new(header: JsonlSessionHeader) -> Self {
        Self {
            header,
            entries: Vec::new(),
            by_id: HashMap::new(),
        }
    }

    pub fn append(&mut self, entry: JsonlSessionEntry) -> Result<()> {
        if self.by_id.contains_key(entry.id()) {
            bail!("duplicate entry id '{}'", entry.id());
        }
        if let Some(parent) = entry.parent_id()
            && !self.by_id.contains_key(parent)
        {
            bail!("unknown parentId '{parent}'");
        }
        let id = entry.id().to_string();
        self.by_id.insert(id, self.entries.len());
        self.entries.push(entry);
        Ok(())
    }

    /// Leaf entries: ids that are not referenced as a parent by any other entry.
    pub fn leaves(&self) -> Vec<&str> {
        let mut referenced = std::collections::HashSet::new();
        for entry in &self.entries {
            if let Some(parent) = entry.parent_id() {
                referenced.insert(parent.to_string());
            }
        }
        self.entries
            .iter()
            .filter(|e| !referenced.contains(e.id()))
            .map(JsonlSessionEntry::id)
            .collect()
    }

    /// Walk from `leaf_id` to root, returning the linear branch (root-first).
    pub fn branch_path(&self, leaf_id: &str) -> Result<Vec<&JsonlSessionEntry>> {
        let mut path = Vec::new();
        let mut current = Some(leaf_id.to_string());
        let mut guard = 0usize;
        while let Some(id) = current {
            guard += 1;
            if guard > self.entries.len().saturating_add(1) {
                bail!("cycle detected while resolving branch from '{leaf_id}'");
            }
            let idx = self
                .by_id
                .get(&id)
                .copied()
                .with_context(|| format!("unknown entry id '{id}'"))?;
            let entry = &self.entries[idx];
            current = entry.parent_id().map(str::to_string);
            path.push(entry);
        }
        path.reverse();
        Ok(path)
    }

    /// Message payloads along the branch to `leaf_id` (root-first).
    pub fn restore_branch_messages(&self, leaf_id: &str) -> Result<Vec<serde_json::Value>> {
        let path = self.branch_path(leaf_id)?;
        Ok(path
            .into_iter()
            .filter_map(|entry| match entry {
                JsonlSessionEntry::Message { message, .. } => Some(message.clone()),
                _ => None,
            })
            .collect())
    }

    /// Create a branch from `from_id` by appending a branch summary + first message.
    pub fn branch_from(
        &mut self,
        from_id: &str,
        branch_entry_id: impl Into<String>,
        summary: impl Into<String>,
    ) -> Result<String> {
        if !self.by_id.contains_key(from_id) {
            bail!("unknown fromId '{from_id}'");
        }
        let branch_id = branch_entry_id.into();
        self.append(JsonlSessionEntry::BranchSummary {
            id: branch_id.clone(),
            parent_id: Some(from_id.to_string()),
            timestamp: Utc::now(),
            summary: summary.into(),
            from_id: from_id.to_string(),
        })?;
        Ok(branch_id)
    }
}

/// Append-only JSONL session store on disk.
#[derive(Debug, Clone)]
pub struct JsonlSessionStore {
    path: PathBuf,
}

impl JsonlSessionStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn create(&self, header: &JsonlSessionHeader) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .await
                .with_context(|| format!("create session dir {}", parent.display()))?;
        }
        let mut file = fs::File::create(&self.path)
            .await
            .with_context(|| format!("create {}", self.path.display()))?;
        let line = serde_json::to_string(header)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        file.flush().await?;
        Ok(())
    }

    pub async fn append_entry(&self, entry: &JsonlSessionEntry) -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await
            .with_context(|| format!("open {}", self.path.display()))?;
        let line = serde_json::to_string(entry)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
        file.flush().await?;
        Ok(())
    }

    pub async fn load(&self) -> Result<JsonlSessionTree> {
        let text = fs::read_to_string(&self.path)
            .await
            .with_context(|| format!("read {}", self.path.display()))?;
        parse_jsonl_session(&text)
    }
}

/// Parse a full JSONL session document into a tree.
pub fn parse_jsonl_session(text: &str) -> Result<JsonlSessionTree> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header_line = lines
        .next()
        .context("JSONL session is empty (missing header)")?;
    let header: JsonlSessionHeader =
        serde_json::from_str(header_line).context("invalid JSONL session header")?;
    if header.entry_type != "session" {
        bail!("first line must be a session header");
    }
    let mut tree = JsonlSessionTree::new(header);
    for (idx, line) in lines.enumerate() {
        let entry: JsonlSessionEntry = serde_json::from_str(line)
            .with_context(|| format!("invalid JSONL entry at line {}", idx + 2))?;
        tree.append(entry)?;
    }
    Ok(tree)
}

/// Generate a short hex id (8 chars) suitable for session entries.
#[must_use]
pub fn new_entry_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
        .chars()
        .rev()
        .take(8)
        .collect::<String>()
        .chars()
        .rev()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn branch_restore_returns_linear_history() {
        let header = JsonlSessionHeader::new("sess-1", Some("/tmp".into()));
        let mut tree = JsonlSessionTree::new(header);
        tree.append(JsonlSessionEntry::message(
            "a",
            None,
            json!({"role":"user","content":"hi"}),
        ))
        .unwrap();
        tree.append(JsonlSessionEntry::message(
            "b",
            Some("a".into()),
            json!({"role":"assistant","content":"hello"}),
        ))
        .unwrap();
        tree.append(JsonlSessionEntry::message(
            "c",
            Some("a".into()),
            json!({"role":"assistant","content":"alt"}),
        ))
        .unwrap();

        let leaves = tree.leaves();
        assert_eq!(leaves.len(), 2);

        let main = tree.restore_branch_messages("b").unwrap();
        assert_eq!(main.len(), 2);
        assert_eq!(main[1]["content"], "hello");

        let alt = tree.restore_branch_messages("c").unwrap();
        assert_eq!(alt.len(), 2);
        assert_eq!(alt[1]["content"], "alt");
    }

    #[test]
    fn branch_from_adds_summary_node() {
        let mut tree = JsonlSessionTree::new(JsonlSessionHeader::new("s", None));
        tree.append(JsonlSessionEntry::message("root", None, json!("x")))
            .unwrap();
        let branch_id = tree.branch_from("root", "br1", "retry").unwrap();
        assert_eq!(branch_id, "br1");
        tree.append(JsonlSessionEntry::message(
            "child",
            Some(branch_id),
            json!("y"),
        ))
        .unwrap();
        let path = tree.branch_path("child").unwrap();
        assert_eq!(path.len(), 3);
    }

    #[tokio::test]
    async fn round_trips_through_jsonl_store() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let store = JsonlSessionStore::new(&path);
        let header = JsonlSessionHeader::new("abc", Some("/proj".into()));
        store.create(&header).await.unwrap();
        store
            .append_entry(&JsonlSessionEntry::message(
                "m1",
                None,
                json!({"role":"user","content":"ping"}),
            ))
            .await
            .unwrap();
        store
            .append_entry(&JsonlSessionEntry::message(
                "m2",
                Some("m1".into()),
                json!({"role":"assistant","content":"pong"}),
            ))
            .await
            .unwrap();

        let loaded = store.load().await.unwrap();
        assert_eq!(loaded.header.id, "abc");
        assert_eq!(loaded.entries.len(), 2);
        let msgs = loaded.restore_branch_messages("m2").unwrap();
        assert_eq!(msgs[0]["content"], "ping");
        assert_eq!(msgs[1]["content"], "pong");
    }

    #[test]
    fn rejects_unknown_parent() {
        let mut tree = JsonlSessionTree::new(JsonlSessionHeader::new("s", None));
        let err = tree
            .append(JsonlSessionEntry::message(
                "x",
                Some("missing".into()),
                json!(1),
            ))
            .unwrap_err();
        assert!(err.to_string().contains("unknown parentId"));
    }
}
