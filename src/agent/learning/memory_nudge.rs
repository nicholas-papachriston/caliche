//! MEMORY.md persistence: durable fact extraction and system-prompt injection.
//!
//! After each checkpoint, facts worth persisting (user corrections, recovery
//! paths, environment details) are appended to a Markdown file. The file's
//! contents are loaded into the system prompt on subsequent sessions so the
//! agent's accumulated knowledge is available without re-discovery.

use anyhow::{Context, Result};
use chrono::Utc;
use std::path::{Path, PathBuf};

/// Manages the MEMORY.md lifecycle.
#[derive(Debug)]
pub struct MemoryStore {
    path: PathBuf,
}

impl MemoryStore {
    /// Create a store backed by the given file path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The file path used by this store.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one or more facts to the MEMORY.md file.
    ///
    /// Each fact becomes a timestamped bullet point under a `## Learned Facts`
    /// heading. The file is created if it does not exist.
    pub async fn append_facts(&self, facts: &[String]) -> Result<()> {
        if facts.is_empty() {
            return Ok(());
        }

        let timestamp = Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
        let mut section = String::new();
        for fact in facts {
            section.push_str(&format!("- [{timestamp}] {fact}\n"));
        }

        let needs_header = !self.path.exists()
            || tokio::fs::read_to_string(&self.path)
                .await
                .map(|c| !c.contains("## Learned Facts"))
                .unwrap_or(true);

        if needs_header && let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .context("creating MEMORY.md parent directory")?;
        }

        let mut content = if self.path.exists() {
            tokio::fs::read_to_string(&self.path)
                .await
                .unwrap_or_default()
        } else {
            String::new()
        };

        if needs_header {
            if content.is_empty() {
                content.push_str("# Agent Memory\n\nPersisted facts from the learning loop.\n\n");
            }
            content.push_str("\n## Learned Facts\n\n");
        }

        content.push_str(&section);

        tokio::fs::write(&self.path, &content)
            .await
            .context("writing MEMORY.md")?;

        tracing::info!(
            facts_count = facts.len(),
            path = %self.path.display(),
            "Memory nudge: appended facts to MEMORY.md"
        );

        Ok(())
    }

    /// Load the MEMORY.md contents for injection into a system prompt.
    ///
    /// Returns `None` when the file does not exist or is empty.
    pub async fn load_for_prompt(&self) -> Option<String> {
        let content = tokio::fs::read_to_string(&self.path).await.ok()?;
        if content.trim().is_empty() {
            return None;
        }
        Some(content)
    }

    /// Load only the "Learned Facts" section for a concise system-prompt addition.
    pub async fn load_facts_section(&self) -> Option<String> {
        let content = self.load_for_prompt().await?;
        let marker = "## Learned Facts";
        let start = content.find(marker)?;
        let section = &content[start..];
        if section.trim().len() <= marker.len() {
            return None;
        }
        Some(section.to_string())
    }
}
