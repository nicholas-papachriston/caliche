//! MEMORY.md persistence: durable fact extraction and system-prompt injection.
//!
//! After each checkpoint, facts worth persisting (user corrections, recovery
//! paths, environment details) are appended to a Markdown file. The file's
//! contents are loaded into the system prompt on subsequent sessions so the
//! agent's accumulated knowledge is available without re-discovery.

use anyhow::{Context, Result};
use chrono::Utc;
use std::path::{Component, Path, PathBuf};

/// Manages the MEMORY.md lifecycle.
///
/// A file inside the workspace is read and written below the workspace
/// without following any link, so a link committed into a project cannot
/// put an outside file into the prompt or take the write outside. A file
/// configured outside the workspace is used as named.
#[derive(Debug, Clone)]
pub struct MemoryStore {
    path: PathBuf,
    workspace: Option<PathBuf>,
}

/// How [`MemoryStore`] opens its file.
#[derive(Clone, Copy)]
enum Open {
    Read,
    Append,
}

impl MemoryStore {
    /// Create a store backed by the given file path, in `workspace` when
    /// one is known.
    pub fn new(path: impl Into<PathBuf>, workspace: Option<&Path>) -> Self {
        Self {
            path: path.into(),
            workspace: workspace.map(Path::to_path_buf),
        }
    }

    /// The file path used by this store.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Open the file, below the canonical workspace when it is inside it.
    fn open(&self, how: Open) -> std::io::Result<std::fs::File> {
        let below = self.workspace.as_deref().and_then(|workspace| {
            let below = self.path.strip_prefix(workspace).ok()?;
            below
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
                .then(|| (workspace, below))
        });
        let Some((workspace, below)) = below else {
            if matches!(how, Open::Append)
                && let Some(parent) = self.path.parent()
            {
                std::fs::create_dir_all(parent)?;
            }
            return match how {
                Open::Read => std::fs::File::open(&self.path),
                Open::Append => std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path),
            };
        };
        let root = workspace.canonicalize()?;
        let target = root.join(below);
        match how {
            Open::Read => caliche_sandbox::open_file_beneath(&root, &target),
            Open::Append => caliche_sandbox::append_file_beneath(&root, &target),
        }
    }

    /// The file's text, or `None` when it does not exist.
    fn read(&self) -> std::io::Result<Option<String>> {
        use std::io::Read as _;
        let mut file = match self.open(Open::Read) {
            Ok(file) => file,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err),
        };
        let mut text = String::new();
        file.read_to_string(&mut text)?;
        Ok(Some(text))
    }

    /// Append one or more facts to the MEMORY.md file.
    ///
    /// Each fact becomes a timestamped bullet point under a `## Learned Facts`
    /// heading. The file is created if it does not exist. A file that
    /// cannot be read is left as it is.
    pub async fn append_facts(&self, facts: &[String]) -> Result<()> {
        if facts.is_empty() {
            return Ok(());
        }

        let timestamp = Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
        let mut section = String::new();
        for fact in facts {
            section.push_str(&format!("- [{timestamp}] {fact}\n"));
        }

        let store = self.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            use std::io::Write as _;
            let existing = store
                .read()
                .with_context(|| format!("reading {}", store.path.display()))?;
            let mut addition = String::new();
            match existing.as_deref() {
                Some(text) if text.contains("## Learned Facts") => {}
                Some(text) if !text.is_empty() => addition.push_str("\n## Learned Facts\n\n"),
                _ => addition.push_str(
                    "# Agent Memory\n\nPersisted facts from the learning loop.\n\n\n## Learned Facts\n\n",
                ),
            }
            addition.push_str(&section);
            store
                .open(Open::Append)
                .and_then(|mut file| file.write_all(addition.as_bytes()))
                .context("writing MEMORY.md")
        })
        .await??;

        tracing::info!(
            facts_count = facts.len(),
            path = %self.path.display(),
            "Memory nudge: appended facts to MEMORY.md"
        );

        Ok(())
    }

    /// Load the MEMORY.md contents for injection into a system prompt.
    ///
    /// Returns `None` when the file does not exist, cannot be read, or is
    /// empty.
    pub async fn load_for_prompt(&self) -> Option<String> {
        let store = self.clone();
        let content = tokio::task::spawn_blocking(move || store.read())
            .await
            .ok()?
            .ok()??;
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

#[cfg(all(test, unix))]
mod tests {
    use super::MemoryStore;
    use std::os::unix::fs::PermissionsExt as _;

    const HEADER: &str =
        "# Agent Memory\n\nPersisted facts from the learning loop.\n\n\n## Learned Facts\n\n- [";

    #[expect(clippy::expect_used, reason = "test helper panics with context")]
    fn workspace() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("workspace");
        let root = dir.path().canonicalize().expect("canonical workspace");
        (dir, root)
    }

    #[tokio::test]
    async fn facts_are_appended_under_one_heading() {
        let (_dir, root) = workspace();
        let store = MemoryStore::new(root.join(".caliche/MEMORY.md"), Some(&root));

        store
            .append_facts(&["first".to_string()])
            .await
            .expect("first");
        store
            .append_facts(&["second".to_string()])
            .await
            .expect("second");

        let text = store.load_for_prompt().await.expect("memory text");
        assert!(text.starts_with(HEADER), "{text}");
        assert!(text.ends_with("] second\n"), "{text}");
        assert_eq!(text.matches("## Learned Facts").count(), 1, "{text}");
        assert_eq!(text.matches("] first\n- [").count(), 1, "{text}");
    }

    /// A `.caliche` directory or a `MEMORY.md` that is a link to a place
    /// outside the workspace is neither read into the prompt nor written.
    #[tokio::test]
    async fn a_linked_memory_file_is_not_read_or_written() {
        for layout in ["linked dir", "linked file"] {
            let (_dir, root) = workspace();
            let (_outside_dir, outside) = workspace();
            std::fs::write(outside.join("MEMORY.md"), "OUTSIDE-SECRET\n").expect("outside");
            if layout == "linked dir" {
                std::os::unix::fs::symlink(&outside, root.join(".caliche")).expect("dir link");
            } else {
                std::fs::create_dir(root.join(".caliche")).expect(".caliche");
                std::os::unix::fs::symlink(
                    outside.join("MEMORY.md"),
                    root.join(".caliche/MEMORY.md"),
                )
                .expect("file link");
            }
            let store = MemoryStore::new(root.join(".caliche/MEMORY.md"), Some(&root));

            let appended = store.append_facts(&["PWNED".to_string()]).await;
            let loaded = store.load_for_prompt().await;

            assert!(appended.is_err(), "{layout}: the append went through");
            assert_eq!(loaded, None, "{layout}");
            assert_eq!(
                std::fs::read_to_string(outside.join("MEMORY.md")).expect("outside"),
                "OUTSIDE-SECRET\n",
                "{layout}"
            );
        }
    }

    /// A MEMORY.md that cannot be read is left as it is, not replaced by
    /// the new facts alone.
    #[tokio::test]
    async fn an_unreadable_memory_file_is_left_as_it_is() {
        let (_dir, root) = workspace();
        let path = root.join("MEMORY.md");
        std::fs::write(&path, "# Kept\n\nolder facts\n").expect("memory");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o200)).expect("chmod");
        let store = MemoryStore::new(&path, Some(&root));

        let appended = store.append_facts(&["new".to_string()]).await;

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod");
        assert!(appended.is_err(), "the append went through");
        assert_eq!(
            std::fs::read_to_string(&path).expect("memory"),
            "# Kept\n\nolder facts\n"
        );
    }
}
