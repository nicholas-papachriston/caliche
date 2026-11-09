//! Machine-readable record of omitted, truncated, or compacted context.
//!
//! Mirrors Unreal Agent's context-builder `Report` (`kind` / `source` / `reason`)
//! so Jev hybrid compaction can persist an audit trail alongside the summary.

use serde::{Deserialize, Serialize};

/// How a conversation fragment changed during compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionChangeKind {
    /// Dropped from the summarizer input or the compacted transcript.
    Omitted,
    /// Kept a prefix (or similar) and discarded the rest.
    Truncated,
    /// Replaced by an LLM / Jev summary.
    Compacted,
}

/// One omitted, truncated, or compacted change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionChange {
    /// Kind of mutation.
    pub kind: CompactionChangeKind,
    /// Producer (`jev`, `llm_summarization`, `sliding_window`, …).
    pub source: String,
    /// Why the change was applied.
    pub reason: String,
}

impl CompactionChange {
    /// Construct a change with owned `source` / `reason`.
    #[must_use]
    pub fn new(
        kind: CompactionChangeKind,
        source: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            source: source.into(),
            reason: reason.into(),
        }
    }
}

/// Audit trail persisted with a compaction summary.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionReport {
    /// Ordered mutations applied while building the compacted prompt.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub changes: Vec<CompactionChange>,
}

impl CompactionReport {
    /// Empty report.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            changes: Vec::new(),
        }
    }

    /// True when no mutations were recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Append one change.
    pub fn push(&mut self, change: CompactionChange) {
        self.changes.push(change);
    }

    /// Append an omitted / truncated / compacted entry.
    pub fn record(
        &mut self,
        kind: CompactionChangeKind,
        source: impl Into<String>,
        reason: impl Into<String>,
    ) {
        self.changes
            .push(CompactionChange::new(kind, source, reason));
    }

    /// Move `other`'s changes onto the end of this report.
    pub fn extend(&mut self, mut other: Self) {
        self.changes.append(&mut other.changes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_round_trips_json() {
        let mut report = CompactionReport::new();
        report.record(CompactionChangeKind::Omitted, "jev", "drop tool result c1");
        report.record(
            CompactionChangeKind::Truncated,
            "jev",
            "truncate tool result c2",
        );
        let json = serde_json::to_string(&report).expect("serialize");
        let loaded: CompactionReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.changes.len(), 2);
        assert_eq!(loaded.changes[0].kind, CompactionChangeKind::Omitted);
        assert_eq!(loaded.changes[1].source, "jev");
    }
}
