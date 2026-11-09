//! State Manager for checkpointing and undo/rollback support.
//!
//! Tracks file modifications and enables reverting changes.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Unique identifier for a checkpoint
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointId(String);

impl CheckpointId {
    /// Create a new checkpoint ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a new unique checkpoint ID
    pub fn generate() -> Self {
        Self(crate::utils::time_id::generate_time_id().to_string())
    }
}

impl std::fmt::Display for CheckpointId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for CheckpointId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for CheckpointId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// A saved version of a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileVersion {
    /// The file path
    pub path: PathBuf,
    /// Version number (0 = original, 1 = first modification, etc.)
    pub version: usize,
    /// The content at this version (None if file didn't exist)
    pub content: Option<String>,
    /// Timestamp of this version
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Associated checkpoint ID (if any)
    pub checkpoint_id: Option<CheckpointId>,
    /// Turn number when this was recorded
    pub turn: usize,
}

/// A checkpoint representing a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Unique checkpoint ID
    pub id: CheckpointId,
    /// Human-readable label
    pub label: String,
    /// Timestamp when checkpoint was created
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Turn number at checkpoint
    pub turn: usize,
    /// File versions at this checkpoint
    pub file_versions: HashMap<PathBuf, usize>,
}

/// Action to perform during rollback
#[derive(Debug, Clone)]
pub struct RollbackAction {
    /// File path
    pub path: PathBuf,
    /// Version being rolled back to
    pub version: usize,
    /// Content to restore (None = delete file)
    pub content: Option<String>,
    /// Whether the file needs to be deleted
    pub delete: bool,
}

/// State Manager for file modifications and rollback
#[derive(Debug)]
pub struct StateManager {
    /// All checkpoints in chronological order
    checkpoints: Vec<Checkpoint>,
    /// File versions by path
    file_versions: HashMap<PathBuf, Vec<FileVersion>>,
    /// Current turn number
    current_turn: usize,
    /// Modifications since last checkpoint
    pending_modifications: Vec<(PathBuf, usize)>,
}

impl StateManager {
    /// Create a new state manager
    pub fn new() -> Self {
        Self {
            checkpoints: Vec::new(),
            file_versions: HashMap::new(),
            current_turn: 0,
            pending_modifications: Vec::new(),
        }
    }

    /// Set the current turn number
    pub fn set_turn(&mut self, turn: usize) {
        self.current_turn = turn;
    }

    /// Create a checkpoint at the current state
    pub fn checkpoint(&mut self, label: &str) -> CheckpointId {
        let id = CheckpointId::generate();

        // Capture current version of each file
        let file_versions: HashMap<PathBuf, usize> = self
            .file_versions
            .iter()
            .map(|(path, versions)| (path.clone(), versions.len().saturating_sub(1)))
            .collect();

        let checkpoint = Checkpoint {
            id: id.clone(),
            label: label.to_string(),
            timestamp: chrono::Utc::now(),
            turn: self.current_turn,
            file_versions,
        };

        self.checkpoints.push(checkpoint);
        self.pending_modifications.clear();

        id
    }

    /// Get the number of files at a checkpoint
    pub fn checkpoint_file_count(&self, id: &CheckpointId) -> usize {
        self.checkpoints
            .iter()
            .find(|c| &c.id == id)
            .map(|c| c.file_versions.len())
            .unwrap_or(0)
    }

    /// Record a file modification
    ///
    /// Call this BEFORE modifying a file to save the old content.
    pub fn record_file_write(
        &mut self,
        path: &Path,
        old_content: Option<&str>,
        _new_content: &str,
    ) {
        let path_buf = path.to_path_buf();
        let versions = self.file_versions.entry(path_buf.clone()).or_default();

        // If this is the first version, record the original content
        if versions.is_empty() {
            versions.push(FileVersion {
                path: path_buf.clone(),
                version: 0,
                content: old_content.map(|s| s.to_string()),
                timestamp: chrono::Utc::now(),
                checkpoint_id: None,
                turn: self.current_turn,
            });
        }

        // Record the new state (we record old content before writing)
        let new_version = versions.len();
        versions.push(FileVersion {
            path: path_buf.clone(),
            version: new_version,
            content: old_content.map(|s| s.to_string()), // Store old content so we can restore
            timestamp: chrono::Utc::now(),
            checkpoint_id: self.checkpoints.last().map(|c| c.id.clone()),
            turn: self.current_turn,
        });

        self.pending_modifications.push((path_buf, new_version));
    }

    /// Record a file deletion
    pub fn record_file_delete(&mut self, path: &Path, old_content: &str) {
        self.record_file_write(path, Some(old_content), "");
    }

    /// Get files modified since a specific turn
    pub fn files_modified_since(&self, turn: usize) -> Vec<PathBuf> {
        self.file_versions
            .iter()
            .filter_map(|(path, versions)| {
                if versions.iter().any(|v| v.turn > turn) {
                    Some(path.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Rollback to a specific checkpoint
    pub fn rollback_to(&mut self, checkpoint_id: &CheckpointId) -> Result<Vec<RollbackAction>> {
        // Find the checkpoint
        let checkpoint = self
            .checkpoints
            .iter()
            .find(|c| &c.id == checkpoint_id)
            .ok_or_else(|| anyhow!("Checkpoint not found: {}", checkpoint_id))?
            .clone();

        let mut actions = Vec::new();

        // For each file with versions, determine what to restore
        for (path, versions) in &self.file_versions {
            let target_version = checkpoint.file_versions.get(path).copied();

            if let Some(target_ver) = target_version {
                // File existed at checkpoint, restore to that version
                if let Some(version) = versions.get(target_ver)
                    && let Some(ref content) = version.content
                {
                    actions.push(RollbackAction {
                        path: path.clone(),
                        version: target_ver,
                        content: Some(content.clone()),
                        delete: false,
                    });
                }
            } else if !versions.is_empty() {
                // File was created after checkpoint - check if original didn't exist
                if versions[0].content.is_none() {
                    actions.push(RollbackAction {
                        path: path.clone(),
                        version: 0,
                        content: None,
                        delete: true,
                    });
                }
            }
        }

        // Apply the rollback actions
        for action in &actions {
            self.apply_rollback_action(action)?;
        }

        // Remove checkpoints after this one
        let checkpoint_idx = self
            .checkpoints
            .iter()
            .position(|c| &c.id == checkpoint_id)
            .unwrap_or(self.checkpoints.len());
        self.checkpoints.truncate(checkpoint_idx + 1);

        Ok(actions)
    }

    /// Rollback to a checkpoint by string ID
    pub fn rollback_to_str(&mut self, checkpoint_id: &str) -> Result<Vec<RollbackAction>> {
        self.rollback_to(&CheckpointId::from(checkpoint_id))
    }

    /// Undo the last file modification
    pub fn undo_last(&mut self) -> Result<RollbackAction> {
        // Find the most recent modification
        let (path, version) = self
            .pending_modifications
            .pop()
            .ok_or_else(|| anyhow!("No file modifications to undo"))?;

        let versions = self
            .file_versions
            .get(&path)
            .ok_or_else(|| anyhow!("No versions found for file: {:?}", path))?;

        // Get the version before the current one
        let restore_version = version.saturating_sub(1);
        let file_version = versions
            .get(restore_version)
            .ok_or_else(|| anyhow!("Cannot find version {} for {:?}", restore_version, path))?;

        let action = RollbackAction {
            path: path.clone(),
            version: restore_version,
            content: file_version.content.clone(),
            delete: file_version.content.is_none(),
        };

        // Apply the undo
        self.apply_rollback_action(&action)?;

        Ok(action)
    }

    /// Apply a rollback action (restore or delete file)
    fn apply_rollback_action(&self, action: &RollbackAction) -> Result<()> {
        if action.delete {
            // Delete the file if it exists
            if action.path.exists() {
                std::fs::remove_file(&action.path)?;
            }
        } else if let Some(ref content) = action.content {
            // Restore content
            if let Some(parent) = action.path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&action.path, content)?;
        }
        Ok(())
    }

    /// Replace checkpoint metadata (durable resume). File contents are not restored.
    pub fn restore_checkpoints(&mut self, checkpoints: Vec<Checkpoint>) {
        if let Some(last) = checkpoints.last() {
            self.current_turn = last.turn;
        }
        self.checkpoints = checkpoints;
    }

    /// Latest checkpoint id, if any.
    #[must_use]
    pub fn last_checkpoint_id(&self) -> Option<&CheckpointId> {
        self.checkpoints.last().map(|checkpoint| &checkpoint.id)
    }

    /// List all checkpoints
    pub fn list_checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    /// Get file version history
    pub fn file_history(&self, path: &Path) -> Option<&Vec<FileVersion>> {
        self.file_versions.get(path)
    }

    /// Get all modified files
    pub fn modified_files(&self) -> Vec<PathBuf> {
        self.file_versions.keys().cloned().collect()
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.checkpoints.clear();
        self.file_versions.clear();
        self.pending_modifications.clear();
        self.current_turn = 0;
    }
}

impl Default for StateManager {
    fn default() -> Self {
        Self::new()
    }
}
