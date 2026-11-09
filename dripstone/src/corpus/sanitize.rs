//! Drop train and query rows that cosine cannot score.
//!
//! Applied once, before [`super::layout::VECTORS_FILE`] is written, so upsert
//! and a later smaller-cap read never see the omitted rows. Neighbor indices
//! are remapped onto the kept training order. A neighbor that pointed at a
//! dropped row, or past the scanned prefix, becomes [`DROPPED_NEIGHBOR`].

use crate::errors::DripstoneError;

use super::layout::{CorpusRowPolicy, MIN_L2_NORM_FOR_COSINE, MaterializedRows};
use super::relevance::{QueryJudgments, remap_graded};

/// Neighbor slot whose training row was not stored.
const DROPPED_NEIGHBOR: i32 = -1;

/// Counts from one [`apply_row_policy`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RowPolicyReport {
    /// Policy that was applied.
    pub policy: CorpusRowPolicy,
    /// Training rows examined.
    pub scanned_train_rows: usize,
    /// Query rows examined.
    pub scanned_query_rows: usize,
    /// Training rows omitted.
    pub dropped_train_rows: usize,
    /// Query rows omitted.
    pub dropped_query_rows: usize,
}

/// True when `vector` has a finite L2 norm of at least [`MIN_L2_NORM_FOR_COSINE`].
#[must_use]
pub fn row_ok_for_cosine(vector: &[f32]) -> bool {
    let mut squared = 0.0_f32;
    for value in vector {
        squared += value * value;
    }
    if !squared.is_finite() || squared <= 0.0 {
        return false;
    }
    let norm = squared.sqrt();
    norm.is_finite() && norm >= MIN_L2_NORM_FOR_COSINE
}

/// Filter `rows` according to `policy`.
///
/// [`CorpusRowPolicy::Keep`] leaves every row in place and records a zero drop
/// count. [`CorpusRowPolicy::DropBelowMinL2`] removes near-zero, all-zero, and
/// non-finite train and query rows, then rewrites ids, metadata, and neighbors
/// so they still line up.
///
/// # Errors
///
/// Returns an error when ids, metadata, or neighbors are not aligned with the
/// matrices, when every training row is dropped, when every query row is
/// dropped after at least one query was scanned, or when `policy` is
/// [`CorpusRowPolicy::Unspecified`].
pub(super) fn apply_row_policy(
    rows: &mut MaterializedRows,
    policy: CorpusRowPolicy,
) -> Result<RowPolicyReport, DripstoneError> {
    let scanned_train_rows = rows.train.len();
    let scanned_query_rows = rows.queries.len();
    match policy {
        CorpusRowPolicy::Keep => Ok(RowPolicyReport {
            policy,
            scanned_train_rows,
            scanned_query_rows,
            dropped_train_rows: 0,
            dropped_query_rows: 0,
        }),
        CorpusRowPolicy::DropBelowMinL2 => {
            drop_below_min_l2(rows, scanned_train_rows, scanned_query_rows)
        }
        CorpusRowPolicy::Unspecified => Err(DripstoneError::storage(
            "corpus row policy was not set; normalize must apply Keep or DropBelowMinL2 before writing",
        )),
    }
}

fn drop_below_min_l2(
    rows: &mut MaterializedRows,
    scanned_train_rows: usize,
    scanned_query_rows: usize,
) -> Result<RowPolicyReport, DripstoneError> {
    validate_alignment(rows)?;
    if !rows.train.iter().any(|vector| row_ok_for_cosine(vector)) {
        return Err(DripstoneError::storage(format!(
            "cosine corpus: all {scanned_train_rows} training vectors are below the minimum L2 norm ({MIN_L2_NORM_FOR_COSINE})"
        )));
    }
    if scanned_query_rows > 0 && !rows.queries.iter().any(|vector| row_ok_for_cosine(vector)) {
        return Err(DripstoneError::storage(format!(
            "cosine corpus: all {scanned_query_rows} query vectors are below the minimum L2 norm ({MIN_L2_NORM_FOR_COSINE})"
        )));
    }

    let had_meta = !rows.train_meta.is_empty();
    let had_neighbors = rows.neighbors.is_some();
    let had_judgments = rows.judgments.is_some();
    let neighbor_width = neighbor_width(rows.neighbors.as_ref());
    let mut old_to_new = vec![None; scanned_train_rows];
    let mut dropped_train_rows = 0_usize;

    let source_train = std::mem::take(&mut rows.train);
    let source_ids = std::mem::take(&mut rows.train_ids);
    let source_meta = std::mem::take(&mut rows.train_meta);
    let mut train = Vec::with_capacity(scanned_train_rows);
    let mut train_ids = Vec::with_capacity(scanned_train_rows);
    let mut train_meta = Vec::new();
    let mut meta_iter = source_meta.into_iter();
    for (index, (vector, id)) in source_train.into_iter().zip(source_ids).enumerate() {
        let meta = if had_meta { meta_iter.next() } else { None };
        if row_ok_for_cosine(&vector) {
            old_to_new[index] = Some(train.len());
            train.push(vector);
            train_ids.push(id);
            if had_meta {
                train_meta.push(meta.flatten());
            }
        } else {
            dropped_train_rows += 1;
        }
    }

    let source_queries = std::mem::take(&mut rows.queries);
    let source_query_ids = std::mem::take(&mut rows.query_ids);
    let source_neighbors = rows.neighbors.take();
    let source_judgments = rows.judgments.take();
    let mut neighbor_iter = source_neighbors.into_iter().flatten();
    let mut judgment_iter = source_judgments.into_iter().flatten();
    let mut queries = Vec::with_capacity(scanned_query_rows);
    let mut query_ids = Vec::with_capacity(scanned_query_rows);
    let mut neighbors = Vec::new();
    let mut judgments = Vec::new();
    let mut dropped_query_rows = 0_usize;
    for (vector, id) in source_queries.into_iter().zip(source_query_ids) {
        let neighbor_row = neighbor_iter.next();
        let judgment_row = judgment_iter.next();
        if !row_ok_for_cosine(&vector) {
            dropped_query_rows += 1;
            continue;
        }
        queries.push(vector);
        query_ids.push(id);
        if had_neighbors {
            let neighbor_row = neighbor_row.ok_or_else(|| {
                DripstoneError::storage(
                    "cosine corpus: neighbor row missing for a kept query".to_string(),
                )
            })?;
            neighbors.push(remap_neighbors(&neighbor_row, &old_to_new, neighbor_width));
        }
        if had_judgments {
            let judgment_row = judgment_row.ok_or_else(|| {
                DripstoneError::storage(
                    "cosine corpus: judgment row missing for a kept query".to_string(),
                )
            })?;
            judgments.push(remap_judgment(&judgment_row, &old_to_new));
        }
    }

    rows.train = train;
    rows.train_ids = train_ids;
    rows.train_meta = train_meta;
    rows.queries = queries;
    rows.query_ids = query_ids;
    rows.neighbors = if had_neighbors { Some(neighbors) } else { None };
    rows.judgments = if had_judgments { Some(judgments) } else { None };

    Ok(RowPolicyReport {
        policy: CorpusRowPolicy::DropBelowMinL2,
        scanned_train_rows,
        scanned_query_rows,
        dropped_train_rows,
        dropped_query_rows,
    })
}

fn validate_alignment(rows: &MaterializedRows) -> Result<(), DripstoneError> {
    if rows.train_ids.len() != rows.train.len() {
        return Err(DripstoneError::storage(format!(
            "cosine corpus: {} training ids for {} vectors",
            rows.train_ids.len(),
            rows.train.len()
        )));
    }
    if !rows.train_meta.is_empty() && rows.train_meta.len() != rows.train.len() {
        return Err(DripstoneError::storage(format!(
            "cosine corpus: {} metadata rows for {} vectors",
            rows.train_meta.len(),
            rows.train.len()
        )));
    }
    if rows.query_ids.len() != rows.queries.len() {
        return Err(DripstoneError::storage(format!(
            "cosine corpus: {} query ids for {} queries",
            rows.query_ids.len(),
            rows.queries.len()
        )));
    }
    if let Some(neighbors) = &rows.neighbors
        && neighbors.len() != rows.queries.len()
    {
        return Err(DripstoneError::storage(format!(
            "cosine corpus: {} neighbor rows for {} queries",
            neighbors.len(),
            rows.queries.len()
        )));
    }
    if let Some(judgments) = &rows.judgments
        && judgments.len() != rows.queries.len()
    {
        return Err(DripstoneError::storage(format!(
            "cosine corpus: {} judgment rows for {} queries",
            judgments.len(),
            rows.queries.len()
        )));
    }
    Ok(())
}

fn remap_judgment(row: &QueryJudgments, old_to_new: &[Option<usize>]) -> QueryJudgments {
    QueryJudgments {
        query_id: row.query_id.clone(),
        neighbors: remap_graded(&row.neighbors, old_to_new),
    }
}

fn neighbor_width(neighbors: Option<&Vec<Vec<i32>>>) -> usize {
    neighbors
        .and_then(|rows| rows.iter().map(Vec::len).max())
        .unwrap_or(0)
}

fn remap_neighbors(row: &[i32], old_to_new: &[Option<usize>], width: usize) -> Vec<i32> {
    let mut kept = Vec::with_capacity(width);
    for &old_id in row {
        if old_id < 0 || kept.len() == width {
            continue;
        }
        let Ok(index) = usize::try_from(old_id) else {
            continue;
        };
        let Some(mapped) = old_to_new.get(index).copied().flatten() else {
            continue;
        };
        let Ok(id) = i32::try_from(mapped) else {
            continue;
        };
        kept.push(id);
    }
    kept.resize(width, DROPPED_NEIGHBOR);
    kept
}
