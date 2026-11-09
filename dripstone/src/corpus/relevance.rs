//! Graded multi-positive judgments on top of hard neighbor indices.
//!
//! `neighbors.i32.bin` stays the positive index matrix (answer and support,
//! padded with `-1`) so existing `recall@k` keeps its inputs. `qrels.jsonl`
//! stores the same positives with a [`RelevanceClass`] and an optional score.
//! Binary qrels, including CoIR CoSQA, are answer-class and leave support empty.
//!
//! [`RetrievalPlugin`] is the extension point for a later technique. The
//! stubs do not embed. They keep chunk text and source labels unchanged.

use std::borrow::Cow;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::errors::DripstoneError;

/// Score at or above this value is an answer under [`GradePolicy::Threshold`].
///
/// A lower positive score is support. Binary corpora do not use this cutoff:
/// [`GradePolicy::BinaryAnswer`] maps every positive to answer.
pub const DEFAULT_ANSWER_MIN_SCORE: i64 = 2;

/// Role of a labeled training row for one query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelevanceClass {
    /// Explicit negative. Omitted rows are also irrelevant.
    Irrelevant,
    /// Supporting context that is not the answer span.
    Support,
    /// Chunk that answers the query. Binary positives use this class.
    Answer,
}

impl RelevanceClass {
    /// Stable name stored in `qrels.jsonl`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Irrelevant => "irrelevant",
            Self::Support => "support",
            Self::Answer => "answer",
        }
    }

    /// Higher wins when one training row is labeled twice.
    #[must_use]
    pub const fn rank(self) -> u8 {
        match self {
            Self::Irrelevant => 0,
            Self::Support => 1,
            Self::Answer => 2,
        }
    }

    /// Parse `answer`, `support` / `context`, or `irrelevant` / `negative`.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "answer" | "gold" => Some(Self::Answer),
            "support" | "context" | "supporting" => Some(Self::Support),
            "irrelevant" | "negative" | "none" => Some(Self::Irrelevant),
            _ => None,
        }
    }
}

/// How a source qrel becomes a [`RelevanceClass`].
///
/// Stored on the corpus manifest. A layout is reused only when the policy matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GradePolicy {
    /// Score `<= 0` is dropped. Every positive is [`RelevanceClass::Answer`].
    ///
    /// A `class` field is ignored. `CoSQA`'s binary test qrels use this.
    BinaryAnswer,
    /// `class` / `role` / `relevance` wins. A missing class uses [`Self::BinaryAnswer`].
    Explicit,
    /// Numeric bands, unless an explicit class is present.
    ///
    /// `score <= 0` is irrelevant, `score >= answer_min` is answer, and a
    /// positive score below `answer_min` is support.
    Threshold {
        /// Inclusive answer cutoff. See [`DEFAULT_ANSWER_MIN_SCORE`].
        answer_min: i64,
    },
}

impl Default for GradePolicy {
    fn default() -> Self {
        Self::BinaryAnswer
    }
}

impl GradePolicy {
    /// Header label. Threshold includes its cutoff.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::BinaryAnswer => "binary_answer".to_string(),
            Self::Explicit => "explicit".to_string(),
            Self::Threshold { answer_min } => format!("threshold:{answer_min}"),
        }
    }
}

impl std::fmt::Display for GradePolicy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.label())
    }
}

/// Technique a later ingest can plug in. Metrics do not depend on it.
///
/// [`RetrievalTechnique::HardNeighbors`] and [`RetrievalTechnique::GradedContext`]
/// are implemented. The other variants are stubs: they do not change vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalTechnique {
    /// Hard neighbor indices. Existing recall@k.
    #[default]
    HardNeighbors,
    /// Multi-positive answer / support judgments.
    GradedContext,
    /// Reserved: prepend document context, then embed the chunk.
    ContextualEmbed,
    /// Reserved: embed the document, then pool token spans into chunks.
    LateChunking,
    /// Reserved: distill token scores into a chunk label.
    TokenChunkTeacher,
}

impl RetrievalTechnique {
    /// Stable config name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HardNeighbors => "hard_neighbors",
            Self::GradedContext => "graded_context",
            Self::ContextualEmbed => "contextual_embed",
            Self::LateChunking => "late_chunking",
            Self::TokenChunkTeacher => "token_chunk_teacher",
        }
    }
}

/// One labeled neighbor. `index` points at the training matrix (`-1` is dropped).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradedNeighbor {
    /// Row in `vectors.f32.bin`.
    pub index: i32,
    /// Answer, support, or an explicit negative.
    pub class: RelevanceClass,
    /// Publisher grade, when the source had one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

/// Judgments for one query, aligned with `query_ids`.
///
/// `qrels.jsonl` is one object per line:
/// `{"query_id":"q1","neighbors":[{"index":0,"class":"answer","score":1.0}]}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryJudgments {
    /// Query id from `query_ids.jsonl`.
    pub query_id: String,
    /// Labeled training rows. Empty when this query has no positives in the cap.
    pub neighbors: Vec<GradedNeighbor>,
}

/// Answer@k and Support@k for one retrieved list.
///
/// Either field is `None` when that class has no positives, so a binary corpus
/// reports Answer@k and leaves Support@k empty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradedRecall {
    /// Set recall of answer-class indices inside the top `k`.
    pub answer_at_k: Option<f64>,
    /// Set recall of support-class indices inside the top `k`.
    pub support_at_k: Option<f64>,
}

/// Contract for a retrieval technique.
///
/// Add a [`RetrievalTechnique`] variant and a [`RetrievalPlugin`] arm. Do not
/// add a second neighbor file or a second recall table. `prepare_text` is what
/// a future embedder would see. `label_chunk` is how a split document inherits
/// the source qrel. `aggregate_teacher` folds token scores into one class.
pub trait RetrievalExtension {
    /// Which technique this plugin implements.
    fn technique(&self) -> RetrievalTechnique;

    /// Text a later embedder would consume. Stubs return `text` unchanged.
    fn prepare_text<'a>(&self, text: &'a str, context: Option<&str>) -> Cow<'a, str>;

    /// Class written for one chunk of a labeled source document.
    fn label_chunk(
        &self,
        source: RelevanceClass,
        chunk_ordinal: usize,
        chunk_count: usize,
    ) -> RelevanceClass;

    /// Fold token-level teacher scores into a chunk class.
    ///
    /// The token-chunk teacher stub returns `None` until a teacher exists.
    fn aggregate_teacher(&self, token_scores: &[f32]) -> Option<RelevanceClass>;
}

/// In-repo plugins. New techniques are new variants, matched exhaustively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalPlugin {
    /// Hard neighbor indices only.
    HardNeighbors,
    /// Graded answer / support judgments. Labels pass through.
    GradedContext,
    /// Stub for contextual embeddings.
    ContextualEmbed,
    /// Stub for late chunking.
    LateChunking,
    /// Stub for a token-to-chunk teacher.
    TokenChunkTeacher,
}

impl RetrievalPlugin {
    /// Plugin for a technique. Stubs still return this enum.
    #[must_use]
    pub const fn from_technique(technique: RetrievalTechnique) -> Self {
        match technique {
            RetrievalTechnique::HardNeighbors => Self::HardNeighbors,
            RetrievalTechnique::GradedContext => Self::GradedContext,
            RetrievalTechnique::ContextualEmbed => Self::ContextualEmbed,
            RetrievalTechnique::LateChunking => Self::LateChunking,
            RetrievalTechnique::TokenChunkTeacher => Self::TokenChunkTeacher,
        }
    }
}

impl RetrievalExtension for RetrievalPlugin {
    fn technique(&self) -> RetrievalTechnique {
        match self {
            Self::HardNeighbors => RetrievalTechnique::HardNeighbors,
            Self::GradedContext => RetrievalTechnique::GradedContext,
            Self::ContextualEmbed => RetrievalTechnique::ContextualEmbed,
            Self::LateChunking => RetrievalTechnique::LateChunking,
            Self::TokenChunkTeacher => RetrievalTechnique::TokenChunkTeacher,
        }
    }

    fn prepare_text<'a>(&self, text: &'a str, context: Option<&str>) -> Cow<'a, str> {
        let _ = context;
        match self {
            Self::HardNeighbors
            | Self::GradedContext
            | Self::ContextualEmbed
            | Self::LateChunking
            | Self::TokenChunkTeacher => Cow::Borrowed(text),
        }
    }

    fn label_chunk(
        &self,
        source: RelevanceClass,
        chunk_ordinal: usize,
        chunk_count: usize,
    ) -> RelevanceClass {
        let _ = (chunk_ordinal, chunk_count);
        match self {
            Self::HardNeighbors
            | Self::GradedContext
            | Self::ContextualEmbed
            | Self::LateChunking
            | Self::TokenChunkTeacher => source,
        }
    }

    fn aggregate_teacher(&self, token_scores: &[f32]) -> Option<RelevanceClass> {
        let _ = token_scores;
        match self {
            Self::HardNeighbors
            | Self::GradedContext
            | Self::ContextualEmbed
            | Self::LateChunking
            | Self::TokenChunkTeacher => None,
        }
    }
}

/// Parse [`crate::corpus::ENV_CORPUS_GRADE_POLICY`].
///
/// # Errors
///
/// Returns an error when `value` is not `binary`, `explicit`, or `threshold` / `threshold:N`.
pub fn parse_grade_policy(value: &str) -> Result<GradePolicy, DripstoneError> {
    let token = value.trim().to_ascii_lowercase();
    match token.as_str() {
        "binary" | "binary_answer" => Ok(GradePolicy::BinaryAnswer),
        "explicit" => Ok(GradePolicy::Explicit),
        "threshold" => Ok(GradePolicy::Threshold {
            answer_min: DEFAULT_ANSWER_MIN_SCORE,
        }),
        other => parse_threshold_cutoff(other),
    }
}

fn parse_threshold_cutoff(token: &str) -> Result<GradePolicy, DripstoneError> {
    let Some(raw) = token.strip_prefix("threshold:") else {
        return Err(DripstoneError::storage(format!(
            "DRIPSTONE_CORPUS_GRADE_POLICY must be binary, explicit, or threshold[:N]; got {token}"
        )));
    };
    let answer_min: i64 = raw.parse().map_err(|_| {
        DripstoneError::storage(format!(
            "DRIPSTONE_CORPUS_GRADE_POLICY threshold cutoff `{raw}` is not an integer"
        ))
    })?;
    if answer_min <= 0 {
        return Err(DripstoneError::storage(
            "DRIPSTONE_CORPUS_GRADE_POLICY threshold cutoff must be positive".to_string(),
        ));
    }
    Ok(GradePolicy::Threshold { answer_min })
}

/// Parse [`crate::corpus::ENV_CORPUS_TECHNIQUE`].
///
/// # Errors
///
/// Returns an error when `value` is not a [`RetrievalTechnique`] name.
pub fn parse_technique(value: &str) -> Result<RetrievalTechnique, DripstoneError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "hard" | "hard_neighbors" => Ok(RetrievalTechnique::HardNeighbors),
        "graded" | "graded_context" => Ok(RetrievalTechnique::GradedContext),
        "contextual" | "contextual_embed" => Ok(RetrievalTechnique::ContextualEmbed),
        "late" | "late_chunking" => Ok(RetrievalTechnique::LateChunking),
        "teacher" | "token_chunk_teacher" => Ok(RetrievalTechnique::TokenChunkTeacher),
        other => Err(DripstoneError::storage(format!(
            "DRIPSTONE_CORPUS_TECHNIQUE must be hard_neighbors, graded_context, contextual_embed, late_chunking, or token_chunk_teacher; got {other}"
        ))),
    }
}

/// Class for one source qrel, or `None` when the row is not a positive.
#[must_use]
pub fn relevance_of(
    policy: GradePolicy,
    class_hint: Option<&str>,
    score: i64,
) -> Option<RelevanceClass> {
    match policy {
        GradePolicy::BinaryAnswer => binary_positive(score),
        GradePolicy::Explicit => explicit_class(class_hint, score),
        GradePolicy::Threshold { answer_min } => threshold_class(class_hint, score, answer_min),
    }
}

const fn binary_positive(score: i64) -> Option<RelevanceClass> {
    if score > 0 {
        Some(RelevanceClass::Answer)
    } else {
        None
    }
}

fn explicit_class(class_hint: Option<&str>, score: i64) -> Option<RelevanceClass> {
    match class_hint.and_then(RelevanceClass::parse) {
        Some(RelevanceClass::Irrelevant) => None,
        Some(class) => Some(class),
        None => binary_positive(score),
    }
}

fn threshold_class(
    class_hint: Option<&str>,
    score: i64,
    answer_min: i64,
) -> Option<RelevanceClass> {
    if let Some(class) = class_hint.and_then(RelevanceClass::parse) {
        return match class {
            RelevanceClass::Irrelevant => None,
            RelevanceClass::Support | RelevanceClass::Answer => Some(class),
        };
    }
    if score <= 0 {
        None
    } else if score >= answer_min {
        Some(RelevanceClass::Answer)
    } else {
        Some(RelevanceClass::Support)
    }
}

/// Keep the higher class when `index` is already present.
pub(super) fn merge_graded(neighbors: &mut Vec<GradedNeighbor>, next: GradedNeighbor) {
    if next.index < 0 {
        return;
    }
    if let Some(existing) = neighbors.iter_mut().find(|kept| kept.index == next.index) {
        if next.class.rank() > existing.class.rank() {
            existing.class = next.class;
            existing.score = next.score;
        }
        return;
    }
    neighbors.push(next);
}

/// Positive indices in label order. Irrelevant and `-1` are omitted.
///
/// This is the hard-neighbor row written to `neighbors.i32.bin`.
#[must_use]
pub fn positive_indices(neighbors: &[GradedNeighbor]) -> Vec<i32> {
    let mut indices = Vec::new();
    for neighbor in neighbors {
        if neighbor.class == RelevanceClass::Irrelevant || neighbor.index < 0 {
            continue;
        }
        if !indices.contains(&neighbor.index) {
            indices.push(neighbor.index);
        }
    }
    indices
}

/// Answer-class judgments for a hard neighbor matrix. Support stays empty.
///
/// Used when an older layout set `has_qrels` but did not write `qrels.jsonl`.
#[must_use]
pub(super) fn judgments_from_hard_neighbors(
    query_ids: &[String],
    neighbors: &[Vec<i32>],
) -> Vec<QueryJudgments> {
    let mut rows = Vec::with_capacity(neighbors.len());
    for (offset, indices) in neighbors.iter().enumerate() {
        let query_id = query_ids
            .get(offset)
            .cloned()
            .unwrap_or_else(|| format!("q{offset}"));
        let mut graded = Vec::new();
        for index in indices {
            merge_graded(
                &mut graded,
                GradedNeighbor {
                    index: *index,
                    class: RelevanceClass::Answer,
                    score: None,
                },
            );
        }
        rows.push(QueryJudgments {
            query_id,
            neighbors: graded,
        });
    }
    rows
}

/// Rewrite indices after cosine row drops. Dropped targets disappear.
#[must_use]
pub fn remap_graded(
    neighbors: &[GradedNeighbor],
    old_to_new: &[Option<usize>],
) -> Vec<GradedNeighbor> {
    let mut kept = Vec::new();
    for neighbor in neighbors {
        let Some(index) = remap_index(neighbor.index, old_to_new) else {
            continue;
        };
        merge_graded(
            &mut kept,
            GradedNeighbor {
                index,
                class: neighbor.class,
                score: neighbor.score,
            },
        );
    }
    kept
}

fn remap_index(index: i32, old_to_new: &[Option<usize>]) -> Option<i32> {
    if index < 0 {
        return None;
    }
    let position = usize::try_from(index).ok()?;
    let mapped = old_to_new.get(position).copied().flatten()?;
    i32::try_from(mapped).ok()
}

/// Set recall of one class inside `retrieved`'s first `k` indices.
///
/// `None` when that class has no positives. Duplicate retrieved ids count once.
#[must_use]
pub fn class_recall_at_k(
    retrieved: &[i32],
    neighbors: &[GradedNeighbor],
    class: RelevanceClass,
    k: usize,
) -> Option<f64> {
    let relevant = positive_class_indices(neighbors, class);
    if relevant.is_empty() {
        return None;
    }
    let hit = overlap_count(retrieved, &relevant, k);
    Some(count_ratio(hit, relevant.len()))
}

/// Answer@k and Support@k for one query.
#[must_use]
pub fn graded_recall_at_k(
    retrieved: &[i32],
    neighbors: &[GradedNeighbor],
    k: usize,
) -> GradedRecall {
    GradedRecall {
        answer_at_k: class_recall_at_k(retrieved, neighbors, RelevanceClass::Answer, k),
        support_at_k: class_recall_at_k(retrieved, neighbors, RelevanceClass::Support, k),
    }
}

/// Append present per-query ratios. A missing class does not push `0`.
pub fn record_graded_recall(
    retrieved: &[i32],
    neighbors: &[GradedNeighbor],
    k: usize,
    answers: &mut Vec<f64>,
    supports: &mut Vec<f64>,
) {
    let graded = graded_recall_at_k(retrieved, neighbors, k);
    if let Some(answer) = graded.answer_at_k {
        answers.push(answer);
    }
    if let Some(support) = graded.support_at_k {
        supports.push(support);
    }
}

/// Mean of the samples that were present. Empty input is `None`.
#[must_use]
pub fn mean_ratio(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let sum: f64 = samples.iter().copied().sum();
    let width = u32::try_from(samples.len()).map_or(1.0, f64::from);
    Some(sum / width)
}

fn positive_class_indices(neighbors: &[GradedNeighbor], class: RelevanceClass) -> Vec<i32> {
    let mut indices = Vec::new();
    for neighbor in neighbors {
        if neighbor.class == class && neighbor.index >= 0 && !indices.contains(&neighbor.index) {
            indices.push(neighbor.index);
        }
    }
    indices
}

fn overlap_count(retrieved: &[i32], relevant: &[i32], k: usize) -> usize {
    let mut seen = Vec::new();
    let mut hit = 0_usize;
    for index in retrieved.iter().take(k) {
        if seen.contains(index) {
            continue;
        }
        seen.push(*index);
        if relevant.contains(index) {
            hit += 1;
        }
    }
    hit
}

fn count_ratio(hit: usize, total: usize) -> f64 {
    let hit = u32::try_from(hit).map_or(0.0, f64::from);
    let total = u32::try_from(total).map_or(0.0, f64::from);
    if total == 0.0 { 0.0 } else { hit / total }
}

pub(super) fn write_judgments(path: &Path, rows: &[QueryJudgments]) -> Result<(), DripstoneError> {
    let mut file = File::create(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("create {}", path.display()), error)
    })?;
    for row in rows {
        let line = serde_json::to_string(row)?;
        writeln!(file, "{line}").map_err(|error| {
            DripstoneError::storage_with_source(format!("write {}", path.display()), error)
        })?;
    }
    Ok(())
}

pub(super) fn read_judgments(
    path: &Path,
    cap: usize,
) -> Result<Vec<QueryJudgments>, DripstoneError> {
    if cap == 0 {
        return Ok(Vec::new());
    }
    let file = File::open(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("open {}", path.display()), error)
    })?;
    let mut rows = Vec::new();
    for (offset, line) in BufReader::new(file).lines().enumerate() {
        if rows.len() >= cap {
            break;
        }
        let line = line.map_err(|error| {
            DripstoneError::storage_with_source(format!("read {}", path.display()), error)
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: QueryJudgments = serde_json::from_str(trimmed).map_err(|error| {
            DripstoneError::storage(format!(
                "parse {} line {}: {error}",
                path.display(),
                offset + 1
            ))
        })?;
        rows.push(row);
    }
    if rows.len() != cap {
        return Err(DripstoneError::storage(format!(
            "qrels {} has {} query rows, expected {cap}",
            path.display(),
            rows.len()
        )));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neighbor(index: i32, class: RelevanceClass) -> GradedNeighbor {
        GradedNeighbor {
            index,
            class,
            score: None,
        }
    }

    #[test]
    fn binary_policy_is_answer_only_and_ignores_a_support_hint() {
        assert_eq!(
            relevance_of(GradePolicy::BinaryAnswer, Some("support"), 1),
            Some(RelevanceClass::Answer)
        );
        assert_eq!(relevance_of(GradePolicy::BinaryAnswer, None, 0), None);
        assert_eq!(
            relevance_of(GradePolicy::Explicit, Some("context"), 1),
            Some(RelevanceClass::Support)
        );
        assert_eq!(
            relevance_of(
                GradePolicy::Threshold {
                    answer_min: DEFAULT_ANSWER_MIN_SCORE
                },
                None,
                1
            ),
            Some(RelevanceClass::Support)
        );
        assert_eq!(
            relevance_of(
                GradePolicy::Threshold {
                    answer_min: DEFAULT_ANSWER_MIN_SCORE
                },
                None,
                2
            ),
            Some(RelevanceClass::Answer)
        );
    }

    #[test]
    fn answer_and_support_recall_stay_independent_of_hard_positives() {
        let neighbors = vec![
            neighbor(0, RelevanceClass::Answer),
            neighbor(1, RelevanceClass::Answer),
            neighbor(4, RelevanceClass::Support),
        ];
        assert_eq!(positive_indices(&neighbors), vec![0, 1, 4]);
        let graded = graded_recall_at_k(&[4, 9, 0], &neighbors, 3);
        assert!((graded.answer_at_k.expect("answer") - 0.5).abs() < 1e-9);
        assert!((graded.support_at_k.expect("support") - 1.0).abs() < 1e-9);

        let binary = vec![
            neighbor(0, RelevanceClass::Answer),
            neighbor(2, RelevanceClass::Answer),
        ];
        let only_answer = graded_recall_at_k(&[0, 2], &binary, 10);
        assert!((only_answer.answer_at_k.expect("answer") - 1.0).abs() < 1e-9);
        assert!(only_answer.support_at_k.is_none());
    }

    #[test]
    fn aggregation_skips_queries_that_lack_a_class() {
        let mut answers = Vec::new();
        let mut supports = Vec::new();
        record_graded_recall(
            &[0],
            &[
                neighbor(0, RelevanceClass::Answer),
                neighbor(1, RelevanceClass::Answer),
            ],
            10,
            &mut answers,
            &mut supports,
        );
        record_graded_recall(
            &[3],
            &[
                neighbor(2, RelevanceClass::Answer),
                neighbor(3, RelevanceClass::Support),
            ],
            10,
            &mut answers,
            &mut supports,
        );
        let answer_mean = mean_ratio(&answers).expect("answers");
        let support_mean = mean_ratio(&supports).expect("support");
        assert!((answer_mean - 0.25).abs() < 1e-9);
        assert!((support_mean - 1.0).abs() < 1e-9);
        assert!(mean_ratio(&[]).is_none());
    }

    #[test]
    fn empty_relevant_set_and_duplicate_hits_do_not_inflate_recall() {
        assert!(class_recall_at_k(&[0], &[], RelevanceClass::Answer, 10).is_none());
        let neighbors = vec![neighbor(1, RelevanceClass::Answer)];
        let graded = graded_recall_at_k(&[1, 1, 1], &neighbors, 3);
        assert!((graded.answer_at_k.expect("answer") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn qrels_jsonl_roundtrip_preserves_class_and_score() {
        let dir = tempfile::tempdir().expect("temp");
        let path = dir.path().join("qrels.jsonl");
        let rows = vec![QueryJudgments {
            query_id: "q1".to_string(),
            neighbors: vec![GradedNeighbor {
                index: 3,
                class: RelevanceClass::Support,
                score: Some(1.0),
            }],
        }];
        write_judgments(&path, &rows).expect("write");
        let loaded = read_judgments(&path, 1).expect("read");
        assert_eq!(loaded, rows);
    }

    #[test]
    fn hard_neighbors_become_answer_judgments_without_support() {
        let rows = judgments_from_hard_neighbors(&["q0".to_string()], &[vec![2, -1, 0]]);
        assert_eq!(rows[0].query_id, "q0");
        assert_eq!(positive_indices(&rows[0].neighbors), vec![2, 0]);
        assert!(
            rows[0]
                .neighbors
                .iter()
                .all(|row| row.class == RelevanceClass::Answer)
        );
        assert!(
            graded_recall_at_k(&[2], &rows[0].neighbors, 1)
                .support_at_k
                .is_none()
        );
    }

    #[test]
    fn remap_drops_indices_whose_training_row_was_removed() {
        let neighbors = vec![
            neighbor(0, RelevanceClass::Answer),
            neighbor(1, RelevanceClass::Support),
            neighbor(3, RelevanceClass::Answer),
        ];
        let old_to_new = vec![Some(0), None, None, Some(1)];
        let mapped = remap_graded(&neighbors, &old_to_new);
        assert_eq!(positive_indices(&mapped), vec![0, 1]);
        assert_eq!(mapped[0].class, RelevanceClass::Answer);
        assert_eq!(mapped[1].class, RelevanceClass::Answer);
    }

    #[test]
    fn extension_stubs_do_not_rewrite_text_or_labels() {
        let techniques = [
            RetrievalTechnique::HardNeighbors,
            RetrievalTechnique::GradedContext,
            RetrievalTechnique::ContextualEmbed,
            RetrievalTechnique::LateChunking,
            RetrievalTechnique::TokenChunkTeacher,
        ];
        for technique in techniques {
            let plugin = RetrievalPlugin::from_technique(technique);
            assert_eq!(plugin.technique(), technique);
            let text = plugin.prepare_text("fn answer() {}", Some("file context"));
            assert_eq!(text.as_ref(), "fn answer() {}");
            assert_eq!(
                plugin.label_chunk(RelevanceClass::Answer, 1, 4),
                RelevanceClass::Answer
            );
            assert!(plugin.aggregate_teacher(&[0.1, 0.9]).is_none());
        }
    }

    #[test]
    fn policy_and_technique_names_parse() {
        assert_eq!(
            parse_grade_policy("threshold:3").expect("threshold"),
            GradePolicy::Threshold { answer_min: 3 }
        );
        assert_eq!(
            parse_technique("late").expect("late"),
            RetrievalTechnique::LateChunking
        );
        assert!(parse_grade_policy("ndcg").is_err());
        assert!(parse_technique("bm25").is_err());
        assert!(parse_grade_policy("threshold:0").is_err());
    }
}
