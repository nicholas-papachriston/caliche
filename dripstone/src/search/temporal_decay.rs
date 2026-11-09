//! Temporal decay scoring for search results.
//!
//! Applies a time-based score multiplier so older content is down-weighted
//! relative to newer content, with configurable half-life and evergreen paths.

use crate::core::SearchResult;
use crate::utils::numeric::{checked_f64_to_f32, i64_to_f64};
use chrono::{DateTime, Utc};
use regex::Regex;
use std::path::Path;

/// Configuration for temporal decay.
#[derive(Debug, Clone)]
pub struct TemporalDecayConfig {
    /// Whether temporal decay is enabled.
    pub enabled: bool,
    /// Half-life in days. After this many days, score is halved. Default: 30.
    pub half_life_days: f64,
    /// Metadata key for timestamp extraction. Default: "`created_at`".
    pub timestamp_key: String,
    /// Regex to extract date from `doc_id/path` (e.g. memory/2025-01-15.md).
    pub path_date_pattern: Option<Regex>,
    /// Paths (or `doc_id` prefixes) that are immune to decay (evergreen content).
    pub evergreen_paths: Vec<String>,
}

impl Default for TemporalDecayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            half_life_days: 30.0,
            timestamp_key: "created_at".into(),
            path_date_pattern: None,
            evergreen_paths: vec!["MEMORY.md".into()],
        }
    }
}

/// Decay rate lambda from half-life: lambda = ln(2) / `half_life_days`.
#[must_use]
pub fn decay_lambda(half_life_days: f64) -> f64 {
    if half_life_days <= 0.0 {
        return 0.0;
    }
    std::f64::consts::LN_2 / half_life_days
}

/// Temporal decay multiplier: exp(-lambda * `age_days`).
#[must_use]
pub fn decay_multiplier(age_days: f64, half_life_days: f64) -> f64 {
    if half_life_days <= 0.0 || age_days <= 0.0 {
        return 1.0;
    }
    let lambda = decay_lambda(half_life_days);
    (-lambda * age_days).exp()
}

/// Extract timestamp from a search result using metadata or path pattern.
pub fn extract_timestamp(
    result: &SearchResult,
    config: &TemporalDecayConfig,
) -> Option<DateTime<Utc>> {
    if let Some(ref dt) = result
        .chunk
        .metadata
        .get(&config.timestamp_key)
        .and_then(parse_metadata_datetime)
    {
        return Some(*dt);
    }
    if let Some(ref re) = config.path_date_pattern
        && let Some(caps) = re.captures(&result.chunk.doc_id)
        && let Some(m) = caps.get(1)
        && let Ok(dt) = chrono::NaiveDate::parse_from_str(m.as_str(), "%Y-%m-%d")
        && let Some(ndt) = dt.and_hms_opt(0, 0, 0)
    {
        return Some(ndt.and_utc());
    }
    None
}

fn parse_metadata_datetime(v: &serde_json::Value) -> Option<DateTime<Utc>> {
    let s = v.as_str()?;
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
                .ok()
                .map(|n| n.and_utc())
        })
}

fn is_evergreen(doc_id: &str, evergreen_paths: &[String]) -> bool {
    let doc_id_lower = doc_id.to_lowercase();
    for path in evergreen_paths {
        if doc_id_lower.contains(&path.to_lowercase()) {
            return true;
        }
        if Path::new(doc_id).file_name().map(|n| n.to_string_lossy())
            == Path::new(path).file_name().map(|n| n.to_string_lossy())
        {
            return true;
        }
    }
    false
}

/// Apply temporal decay to search results in place, then re-sort by score.
///
/// Results without timestamps are unchanged. Evergreen paths are unchanged.
pub fn apply_temporal_decay(
    results: &mut [SearchResult],
    config: &TemporalDecayConfig,
    now: DateTime<Utc>,
) {
    if !config.enabled || !config.half_life_days.is_finite() || config.half_life_days <= 0.0 {
        return;
    }
    for r in results.iter_mut() {
        if is_evergreen(&r.chunk.doc_id, &config.evergreen_paths) {
            continue;
        }
        let Some(ts) = extract_timestamp(r, config) else {
            continue;
        };
        let age_days = i64_to_f64((now - ts).num_seconds()) / 86_400.0;
        if age_days > 0.0 {
            let mult = decay_multiplier(age_days, config.half_life_days);
            if let Some(score) = checked_f64_to_f32(f64::from(r.score) * mult) {
                r.score = score;
            }
        }
    }
    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (rank, result) in results.iter_mut().enumerate() {
        result.rank = rank;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Chunk, SearchResult};
    use serde_json::json;

    fn result_with_date(doc_id: &str, score: f32, rank: usize) -> SearchResult {
        SearchResult::new(
            Chunk::new(
                doc_id,
                0,
                "notes",
                vec![0.1_f32; 4],
                0,
                5,
                1,
                json!({}),
                None,
            ),
            score,
            rank,
        )
    }

    #[test]
    fn decay_reorders_older_dated_doc_ids_behind_newer_ones() {
        let config = TemporalDecayConfig {
            enabled: true,
            half_life_days: 30.0,
            timestamp_key: "created_at".into(),
            path_date_pattern: Regex::new(r"(\d{4}-\d{2}-\d{2})").ok(),
            evergreen_paths: vec!["MEMORY.md".into()],
        };
        let now = DateTime::parse_from_rfc3339("2026-08-18T00:00:00Z")
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());
        let mut results = vec![
            result_with_date("journal/2020-01-01.md", 1.0, 0),
            result_with_date("journal/2026-08-01.md", 1.0, 1),
        ];
        apply_temporal_decay(&mut results, &config, now);
        assert_eq!(results[0].chunk.doc_id, "journal/2026-08-01.md");
        assert_eq!(results[0].rank, 0);
        assert!(results[0].score > results[1].score);
    }
}
