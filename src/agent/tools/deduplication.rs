use crate::agent::tools::selection::ToolExecutionRecord;
use std::collections::VecDeque;
use std::path::Path;

/// Detects and prevents redundant tool calls
#[derive(Debug, Clone)]
pub struct CallDeduplicator {
    /// Recent tool calls to check against (FIFO queue)
    recent_calls: VecDeque<DeduplicationEntry>,
    /// Maximum number of recent calls to track
    window_size: usize,
    /// Similarity threshold (0.0-1.0) for considering calls identical
    similarity_threshold: f32,
}

/// Entry in the deduplication history
#[derive(Debug, Clone)]
struct DeduplicationEntry {
    tool_name: String,
    arguments: serde_json::Value,
    timestamp: chrono::DateTime<chrono::Utc>,
}

impl CallDeduplicator {
    /// Create a new deduplicator with default settings
    #[must_use]
    pub const fn new() -> Self {
        Self {
            recent_calls: VecDeque::new(),
            window_size: 50,
            similarity_threshold: 0.95,
        }
    }

    /// Create a deduplicator with custom window size
    #[must_use]
    pub fn with_window_size(window_size: usize) -> Self {
        Self {
            recent_calls: VecDeque::with_capacity(window_size),
            window_size,
            similarity_threshold: 0.95,
        }
    }

    /// Set the similarity threshold
    #[must_use]
    pub const fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Check if a tool call is redundant
    #[must_use]
    pub fn is_redundant(&self, tool_name: &str, arguments: &serde_json::Value) -> RedundancyReport {
        // Check recent calls
        for entry in &self.recent_calls {
            if entry.tool_name == tool_name {
                let similarity = Self::calculate_similarity(&entry.arguments, arguments);

                if similarity >= self.similarity_threshold {
                    let age_seconds = chrono::Utc::now()
                        .signed_duration_since(entry.timestamp)
                        .num_seconds();

                    return RedundancyReport {
                        is_redundant: true,
                        similar_call_id: Some(format!("{tool_name}_{age_seconds}")),
                        similarity,
                        age_seconds,
                        suggestion: Some(format!(
                            "Identical call to '{tool_name}' was made {age_seconds} seconds ago"
                        )),
                    };
                }
            }
        }

        RedundancyReport {
            is_redundant: false,
            similar_call_id: None,
            similarity: 0.0,
            age_seconds: 0,
            suggestion: None,
        }
    }

    /// Record a tool call for future deduplication checks
    pub fn record_call(&mut self, tool_name: impl Into<String>, arguments: serde_json::Value) {
        let entry = DeduplicationEntry {
            tool_name: tool_name.into(),
            arguments,
            timestamp: chrono::Utc::now(),
        };

        self.recent_calls.push_back(entry);

        // Maintain window size
        while self.recent_calls.len() > self.window_size {
            self.recent_calls.pop_front();
        }
    }

    /// Calculate similarity between two JSON values (0.0-1.0)
    fn calculate_similarity(a: &serde_json::Value, b: &serde_json::Value) -> f32 {
        // Exact match
        if a == b {
            return 1.0;
        }

        // Both objects
        if let (Some(a_obj), Some(b_obj)) = (a.as_object(), b.as_object()) {
            return Self::object_similarity(a_obj, b_obj);
        }

        // Both arrays
        if let (Some(a_arr), Some(b_arr)) = (a.as_array(), b.as_array()) {
            if a_arr.len() != b_arr.len() {
                return 0.0;
            }
            let matches = a_arr
                .iter()
                .zip(b_arr.iter())
                .filter(|(x, y)| x == y)
                .count();
            return (matches as f32) / (a_arr.len() as f32);
        }

        // Different types
        0.0
    }

    /// Calculate similarity between two JSON objects
    fn object_similarity(
        a: &serde_json::Map<String, serde_json::Value>,
        b: &serde_json::Map<String, serde_json::Value>,
    ) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }

        let mut matching_keys = 0;
        let total_keys = a.len().max(b.len());

        for (key, a_val) in a {
            if let Some(b_val) = b.get(key)
                && a_val == b_val
            {
                matching_keys += 1;
            }
        }

        if total_keys == 0 {
            return 1.0;
        }

        (matching_keys as f32) / (total_keys as f32)
    }

    /// Clear all recorded calls
    pub fn clear(&mut self) {
        self.recent_calls.clear();
    }

    /// Get the number of recorded calls
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.recent_calls.len()
    }
}

impl Default for CallDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

/// Report of redundancy analysis
#[derive(Debug, Clone)]
pub struct RedundancyReport {
    /// Whether the call is considered redundant
    pub is_redundant: bool,
    /// ID of the similar previous call (if any)
    pub similar_call_id: Option<String>,
    /// Similarity score (0.0-1.0)
    pub similarity: f32,
    /// Age of the similar call in seconds
    pub age_seconds: i64,
    /// Suggestion for the user
    pub suggestion: Option<String>,
}

impl RedundancyReport {
    #[must_use]
    pub const fn not_redundant() -> Self {
        Self {
            is_redundant: false,
            similar_call_id: None,
            similarity: 0.0,
            age_seconds: 0,
            suggestion: None,
        }
    }
}

/// Execution history store for tracking tool calls
#[derive(Debug)]
pub struct ExecutionHistoryStore {
    /// All recorded executions
    executions: VecDeque<ToolExecutionRecord>,
    /// Maximum number of executions to store
    max_size: usize,
}

impl ExecutionHistoryStore {
    /// Create a new execution history store
    #[must_use]
    pub const fn new() -> Self {
        Self {
            executions: VecDeque::new(),
            max_size: 1000,
        }
    }

    /// Create with custom max size
    #[must_use]
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            executions: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Save an execution record
    pub fn save_execution(&mut self, record: ToolExecutionRecord) {
        self.executions.push_back(record);

        // Maintain max size
        while self.executions.len() > self.max_size {
            self.executions.pop_front();
        }
    }

    /// Get recent executions for a tool
    #[must_use]
    pub fn get_recent(&self, tool_name: &str, limit: usize) -> Vec<ToolExecutionRecord> {
        self.executions
            .iter()
            .rev()
            .filter(|r| r.tool_name == tool_name)
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get recent executions across all tools (most recent first).
    #[must_use]
    pub fn get_recent_global(&self, limit: usize) -> Vec<ToolExecutionRecord> {
        self.executions.iter().rev().take(limit).cloned().collect()
    }

    /// Get all executions for a tool
    #[must_use]
    pub fn get_all(&self, tool_name: &str) -> Vec<ToolExecutionRecord> {
        self.executions
            .iter()
            .filter(|r| r.tool_name == tool_name)
            .cloned()
            .collect()
    }

    /// Get execution metrics for a tool
    #[must_use]
    pub fn get_metrics(&self, tool_name: &str) -> ExecutionMetrics {
        let executions = self.get_all(tool_name);

        if executions.is_empty() {
            return ExecutionMetrics::default();
        }

        let successful = executions.iter().filter(|e| e.success).count();
        let total = executions.len();
        let success_rate = (successful as f32) / (total as f32);
        let avg_time = executions.iter().map(|e| e.execution_time_ms).sum::<u64>() as f32
            / executions.len() as f32;
        let avg_output = executions.iter().map(|e| e.output_length).sum::<usize>() as f32
            / executions.len() as f32;

        ExecutionMetrics {
            tool_name: tool_name.to_string(),
            execution_count: total,
            success_count: successful,
            success_rate,
            avg_execution_time_ms: avg_time as u64,
            avg_output_size: avg_output as usize,
        }
    }

    /// Clean up old records (older than hours)
    pub fn cleanup_old_records(&mut self, hours: i64) {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(hours);
        self.executions.retain(|r| r.timestamp > cutoff);
    }

    /// Get total number of stored executions
    #[must_use]
    pub fn total_executions(&self) -> usize {
        self.executions.len()
    }

    /// Clear all records
    pub fn clear(&mut self) {
        self.executions.clear();
    }

    /// Check if a prior grep call covers the given path and pattern.
    /// A covering grep has:
    /// - Same or broader path (parent directory of the new path)
    /// - Pattern that subsumes or is similar to the new pattern
    #[must_use]
    pub fn has_covering_grep(&self, new_path: &str, new_pattern: &str) -> bool {
        let new_path_normalized = if new_path == "." { "" } else { new_path };

        for record in self.executions.iter().rev() {
            if record.tool_name != "fs" {
                continue;
            }

            // Check if this was a grep operation
            let Some(op) = record.arguments.get("op").and_then(|v| v.as_str()) else {
                continue;
            };
            if op != "grep" {
                continue;
            }

            let Some(prev_path) = record.arguments.get("path").and_then(|v| v.as_str()) else {
                continue;
            };
            let Some(prev_pattern) = record.arguments.get("pattern").and_then(|v| v.as_str())
            else {
                continue;
            };

            // Normalize paths for comparison
            let prev_path_normalized = if prev_path == "." { "" } else { prev_path };

            // Check if prev_path covers new_path (prev is parent of or equal to new)
            let path_covered =
                Path::new(new_path_normalized).starts_with(Path::new(prev_path_normalized));

            if !path_covered {
                continue;
            }

            // Check if patterns overlap (prev subsumes new or vice versa)
            let pattern_covered =
                prev_pattern.contains(new_pattern) || new_pattern.contains(prev_pattern);

            if pattern_covered {
                return true;
            }
        }

        false
    }
}

impl Default for ExecutionHistoryStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics about tool execution performance
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    pub tool_name: String,
    pub execution_count: usize,
    pub success_count: usize,
    pub success_rate: f32,
    pub avg_execution_time_ms: u64,
    pub avg_output_size: usize,
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self {
            tool_name: String::new(),
            execution_count: 0,
            success_count: 0,
            success_rate: 0.0,
            avg_execution_time_ms: 0,
            avg_output_size: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deduplicator_exact_match() {
        let mut dedup = CallDeduplicator::new();
        let args = serde_json::json!({"path": "/test/file.txt"});

        dedup.record_call("fs", args.clone());

        let report = dedup.is_redundant("fs", &args);
        assert!(report.is_redundant);
        assert_eq!(report.similarity, 1.0);
    }

    #[test]
    fn test_deduplicator_different_tools() {
        let mut dedup = CallDeduplicator::new();
        let args = serde_json::json!({"path": "/test/file.txt"});

        dedup.record_call("fs", args.clone());

        // A call to a *different* tool with the same args should not be redundant.
        let report = dedup.is_redundant("bash", &args);
        assert!(!report.is_redundant);
    }

    #[test]
    fn test_deduplicator_different_args() {
        let mut dedup = CallDeduplicator::new();
        let args1 = serde_json::json!({"path": "/test/file1.txt"});
        let args2 = serde_json::json!({"path": "/test/file2.txt"});

        dedup.record_call("fs", args1);

        let report = dedup.is_redundant("fs", &args2);
        assert!(!report.is_redundant);
    }

    #[test]
    fn test_deduplicator_window_size() {
        let mut dedup = CallDeduplicator::with_window_size(5);

        for i in 0..10 {
            dedup.record_call("test_tool", serde_json::json!({"id": i}));
        }

        assert_eq!(dedup.call_count(), 5);
    }

    #[test]
    fn test_execution_history_store() {
        let mut store = ExecutionHistoryStore::new();
        let record = ToolExecutionRecord::new("fs", serde_json::json!({}), true)
            .with_timing(100)
            .with_output_size(500);

        store.save_execution(record);
        assert_eq!(store.total_executions(), 1);

        let recent = store.get_recent("fs", 10);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_execution_metrics() {
        let mut store = ExecutionHistoryStore::new();

        for i in 0..10 {
            let success = i % 2 == 0;
            let record = ToolExecutionRecord::new("test_tool", serde_json::json!({}), success)
                .with_timing(100);
            store.save_execution(record);
        }

        let metrics = store.get_metrics("test_tool");
        assert_eq!(metrics.execution_count, 10);
        assert_eq!(metrics.success_count, 5);
        assert!((metrics.success_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_object_similarity() {
        let a = serde_json::json!({"key1": "value1", "key2": "value2"});
        let b = serde_json::json!({"key1": "value1", "key2": "different"});

        let sim = CallDeduplicator::calculate_similarity(&a, &b);
        assert!(sim > 0.0 && sim < 1.0);
    }

    fn grep_record(path: &str, pattern: &str) -> ToolExecutionRecord {
        ToolExecutionRecord::new(
            "fs",
            serde_json::json!({
                "op": "grep",
                "path": path,
                "pattern": pattern,
            }),
            true,
        )
    }

    #[test]
    fn covering_grep_matches_equal_paths_and_patterns() {
        let mut store = ExecutionHistoryStore::new();
        store.save_execution(grep_record("src/agent", "needle"));

        assert!(store.has_covering_grep("src/agent", "needle"));
    }

    #[test]
    fn covering_grep_matches_parent_paths_and_related_patterns_symmetrically() {
        let mut store = ExecutionHistoryStore::new();
        store.save_execution(grep_record("src", "needle"));

        assert!(store.has_covering_grep("src/agent/tools", "needle suffix"));
        assert!(store.has_covering_grep("src/agent/tools", "need"));
    }

    #[test]
    fn covering_grep_rejects_unrelated_paths() {
        let mut store = ExecutionHistoryStore::new();
        store.save_execution(grep_record("src/agent", "needle"));

        assert!(!store.has_covering_grep("src/agents", "needle"));
        assert!(!store.has_covering_grep("tests", "needle"));
    }
}
