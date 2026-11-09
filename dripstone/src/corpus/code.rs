//! CoIR CoSQA download, AST split, and feature-vector normalize.
//!
//! [`CorpusName::Code`](super::catalog::CorpusName::Code) fetches the pinned
//! `CoIR-Retrieval/cosqa` revision into `source/`, then writes the same capped
//! layout as the HDF5 and Parquet catalogs. Each code row is split with
//! [`crate::rag::ast::AstChunker`] when a grammar is available. Queries stay
//! whole. Vectors are [`super::features::code_feature_vector`] so a soak can
//! upsert without a neural embedder. Test qrels become neighbor indices into
//! the kept training rows and graded judgments in `qrels.jsonl`. Binary
//! positives are answer-class. A `class` or numeric threshold can mark support.
//!
//! A staged tree (`DRIPSTONE_CORPUS_CODE_DIR`) skips the download. It may hold
//! the Hugging Face parquet paths or `corpus.jsonl`, `queries.jsonl`, and
//! `qrels.jsonl`.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use serde::Deserialize;
use serde_json::Value;

use crate::errors::DripstoneError;
use crate::loader::download::{DownloadOptions, ensure_downloaded, sanitize_relative_path};
use crate::rag::ast::{AstChunk, AstChunker, AstChunking, META_SOURCE_ID, META_TEXT, META_TITLE};

use super::catalog::{
    CODE_FEATURE_DIMENSION, COSQA_CORPUS_FILE, COSQA_QRELS_FILE, COSQA_QUERIES_FILE, COSQA_REPO,
    COSQA_REVISION,
};
use super::env::ENV_CORPUS_CODE_DIR;
use super::features::code_feature_vector;
use super::layout::{CodeRowMeta, MaterializedRows};
use super::parquet::{huggingface_dataset_url, huggingface_token, string_cell};
use super::relevance::{
    GradePolicy, GradedNeighbor, QueryJudgments, RelevanceClass, merge_graded, positive_indices,
    relevance_of,
};

const ID_COLUMN: &str = "_id";
const TEXT_COLUMN: &str = "text";
const TITLE_COLUMN: &str = "title";
const LANGUAGE_COLUMN: &str = "language";
const QUERY_ID_COLUMN: &str = "query-id";
const CORPUS_ID_COLUMN: &str = "corpus-id";
const SCORE_COLUMN: &str = "score";
const CLASS_COLUMN: &str = "class";
const DROPPED_NEIGHBOR: i32 = -1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodeTable {
    Corpus,
    Queries,
    Qrels,
}

impl CodeTable {
    const fn candidates(self) -> &'static [&'static str] {
        match self {
            Self::Corpus => &["corpus.jsonl", "corpus.parquet", COSQA_CORPUS_FILE],
            Self::Queries => &["queries.jsonl", "queries.parquet", COSQA_QUERIES_FILE],
            Self::Qrels => &["qrels.jsonl", "qrels.parquet", COSQA_QRELS_FILE],
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Corpus => "corpus rows (corpus.jsonl or corpus/*.parquet)",
            Self::Queries => "queries (queries.jsonl or queries/*.parquet)",
            Self::Qrels => "qrels (qrels.jsonl or data/test-*.parquet)",
        }
    }
}

struct HeldDoc {
    id: String,
    text: String,
    title: Option<String>,
    language: Option<String>,
}

struct QrelHit {
    corpus_id: String,
    class: RelevanceClass,
    score: i64,
}

struct LoadedQrels {
    corpus_ids: HashSet<String>,
    order: Vec<String>,
    by_query: HashMap<String, Vec<QrelHit>>,
}

struct SelectedDocs {
    docs: Vec<HeldDoc>,
    scanned_all: bool,
}

struct QueryPick {
    id: String,
    text: String,
    judgments: Vec<GradedNeighbor>,
}

enum DocReader {
    Jsonl(JsonlDocs),
    Parquet(ParquetDocs),
}

struct JsonlDocs {
    lines: std::io::Lines<BufReader<File>>,
    path: String,
    line_no: usize,
}

struct ParquetDocs {
    reader: ParquetRecordBatchReader,
    batch: Option<arrow_array::RecordBatch>,
    index: usize,
    path: String,
}

/// Download the pinned CoSQA corpus, queries, and test qrels into `source_root`.
///
/// # Errors
///
/// Returns an error when a file cannot be fetched. Stage the tree and set
/// [`ENV_CORPUS_CODE_DIR`] to skip the network.
pub(super) fn download_cosqa(source_root: &Path) -> Result<(), DripstoneError> {
    let token = huggingface_token();
    let options = DownloadOptions::default().with_huggingface_token(token.as_deref());
    for relative in [COSQA_CORPUS_FILE, COSQA_QUERIES_FILE, COSQA_QRELS_FILE] {
        let url = huggingface_dataset_url(COSQA_REPO, COSQA_REVISION, relative);
        let dest = sanitize_relative_path(source_root, relative)?;
        ensure_downloaded(&url, &dest, &options).map_err(|error| {
            DripstoneError::storage(format!(
                "{error}; stage CoIR CoSQA and set {ENV_CORPUS_CODE_DIR} to that directory"
            ))
        })?;
    }
    Ok(())
}

/// Read a CoSQA-shaped directory into capped feature rows.
///
/// # Errors
///
/// Returns an error when the corpus file is missing or a row cannot be decoded.
pub(super) fn read_code_dir(
    root: &Path,
    train_max: usize,
    query_max: usize,
    source_url: Option<String>,
    grade_policy: GradePolicy,
) -> Result<MaterializedRows, DripstoneError> {
    let corpus_path = require_table(root, CodeTable::Corpus)?;
    let qrels = match find_table(root, CodeTable::Qrels)? {
        Some(path) => load_qrels(&path, grade_policy)?,
        None => LoadedQrels {
            corpus_ids: HashSet::new(),
            order: Vec::new(),
            by_query: HashMap::new(),
        },
    };
    let mut corpus_reader = open_docs(&corpus_path)?;
    let selected = select_docs(&mut corpus_reader, &qrels.corpus_ids, train_max)?;
    let expanded = expand_docs(&selected.docs, train_max)?;
    let queries_path = if query_max == 0 {
        None
    } else {
        Some(require_table(root, CodeTable::Queries)?)
    };
    let (picks, queries_complete) = match queries_path {
        Some(path) => pick_queries(&path, &qrels, &expanded.by_source, query_max)?,
        None => (Vec::new(), true),
    };
    assemble_rows(
        expanded,
        picks,
        selected.scanned_all && queries_complete,
        source_url,
        Some(root.display().to_string()),
    )
}

fn assemble_rows(
    expanded: ExpandedDocs,
    picks: Vec<QueryPick>,
    mut source_exhausted: bool,
    source_url: Option<String>,
    source_path: Option<String>,
) -> Result<MaterializedRows, DripstoneError> {
    if !expanded.chunks_complete {
        source_exhausted = false;
    }
    if expanded.texts.is_empty() {
        return Err(DripstoneError::storage(
            "code corpus produced no training rows at the configured cap".to_string(),
        ));
    }
    let train: Vec<Vec<f32>> = expanded
        .texts
        .iter()
        .map(|text| code_feature_vector(text))
        .collect();
    let mut query_ids = Vec::with_capacity(picks.len());
    let mut queries = Vec::with_capacity(picks.len());
    let mut query_judgments = Vec::with_capacity(picks.len());
    let mut neighbor_lists = Vec::with_capacity(picks.len());
    for pick in picks {
        neighbor_lists.push(positive_indices(&pick.judgments));
        queries.push(code_feature_vector(&pick.text));
        query_ids.push(pick.id.clone());
        query_judgments.push(QueryJudgments {
            query_id: pick.id,
            neighbors: pick.judgments,
        });
    }
    let has_qrels = neighbor_lists.iter().any(|row| !row.is_empty());
    let neighbors = pad_neighbors(neighbor_lists);
    let neighbors_width = neighbors
        .as_ref()
        .and_then(|rows| rows.first().map(Vec::len))
        .unwrap_or(0);
    let dimension = CODE_FEATURE_DIMENSION;
    if train.iter().any(|row| row.len() != dimension)
        || queries.iter().any(|row| row.len() != dimension)
    {
        return Err(DripstoneError::storage(
            "code feature width changed while materializing".to_string(),
        ));
    }
    Ok(MaterializedRows {
        train_ids: expanded.ids,
        train,
        train_meta: expanded.metas,
        query_ids,
        queries,
        neighbors,
        neighbors_width,
        dimension,
        source_exhausted,
        source_url,
        source_path,
        has_qrels,
        judgments: Some(query_judgments),
    })
}

struct ExpandedDocs {
    ids: Vec<String>,
    texts: Vec<String>,
    metas: Vec<Option<Value>>,
    by_source: HashMap<String, Vec<i32>>,
    chunks_complete: bool,
}

fn expand_docs(docs: &[HeldDoc], train_max: usize) -> Result<ExpandedDocs, DripstoneError> {
    let mut chunker = AstChunker::new();
    let mut ids = Vec::new();
    let mut texts = Vec::new();
    let mut metas = Vec::new();
    let mut by_source: HashMap<String, Vec<i32>> = HashMap::new();
    let mut chunks_complete = true;
    for doc in docs {
        let pieces = split_doc(&mut chunker, doc);
        let total = pieces.len();
        for (offset, chunk) in pieces.into_iter().enumerate() {
            if texts.len() >= train_max {
                chunks_complete = false;
                break;
            }
            let index = i32::try_from(texts.len()).map_err(|_| {
                DripstoneError::storage("code corpus row index exceeds i32".to_string())
            })?;
            by_source.entry(doc.id.clone()).or_default().push(index);
            ids.push(chunk_id(&doc.id, offset, total));
            texts.push(chunk.text.clone());
            metas.push(Some(row_meta(&chunk, &doc.id, doc.title.as_deref())?));
        }
        if !chunks_complete {
            break;
        }
    }
    Ok(ExpandedDocs {
        ids,
        texts,
        metas,
        by_source,
        chunks_complete,
    })
}

fn split_doc(chunker: &mut AstChunker, doc: &HeldDoc) -> Vec<AstChunk> {
    match chunker.chunk(&doc.text, doc.language.as_deref(), None) {
        AstChunking::Parsed(chunks) if !chunks.is_empty() => chunks,
        AstChunking::Parsed(_) | AstChunking::Unavailable => vec![fallback_chunk(doc)],
    }
}

fn fallback_chunk(doc: &HeldDoc) -> AstChunk {
    let lines = doc.text.lines().count().max(1);
    let end_line = u32::try_from(lines).unwrap_or(u32::MAX);
    AstChunk {
        text: doc.text.clone(),
        start_offset: 0,
        end_offset: doc.text.len(),
        language: doc.language.clone(),
        path: None,
        symbol: doc.title.clone(),
        start_line: Some(1),
        end_line: Some(end_line),
    }
}

fn chunk_id(source_id: &str, offset: usize, total: usize) -> String {
    if total <= 1 {
        source_id.to_string()
    } else {
        format!("{source_id}#{offset}")
    }
}

fn row_meta(
    chunk: &AstChunk,
    source_id: &str,
    title: Option<&str>,
) -> Result<Value, DripstoneError> {
    let mut value = serde_json::to_value(CodeRowMeta {
        language: chunk.language.clone(),
        path: chunk.path.clone(),
        symbol: chunk.symbol.clone(),
        start_line: chunk.start_line,
        end_line: chunk.end_line,
    })?;
    if let Some(object) = value.as_object_mut() {
        object.insert(META_TEXT.to_string(), Value::String(chunk.text.clone()));
        object.insert(
            META_SOURCE_ID.to_string(),
            Value::String(source_id.to_string()),
        );
        if let Some(title) = title.map(str::trim).filter(|title| !title.is_empty()) {
            object.insert(META_TITLE.to_string(), Value::String(title.to_string()));
        }
    }
    Ok(value)
}

fn select_docs(
    reader: &mut DocReader,
    relevant: &HashSet<String>,
    train_max: usize,
) -> Result<SelectedDocs, DripstoneError> {
    let mut unseen = relevant.clone();
    let mut labeled = Vec::new();
    let mut filler = Vec::new();
    let mut seen = HashSet::new();
    let mut saw_unkept = false;
    while let Some(doc) = reader.next_doc()? {
        if doc.id.is_empty() {
            continue;
        }
        if !seen.insert(doc.id.clone()) {
            continue;
        }
        if train_max == 0 {
            saw_unkept = true;
            break;
        }
        if unseen.remove(&doc.id) {
            if labeled.len() < train_max {
                labeled.push(doc);
            } else {
                saw_unkept = true;
            }
        } else {
            let reserved = unseen.len().min(train_max.saturating_sub(labeled.len()));
            let used = labeled.len().saturating_add(filler.len());
            if used.saturating_add(reserved) < train_max {
                filler.push(doc);
            } else {
                saw_unkept = true;
                if unseen.is_empty() {
                    break;
                }
            }
        }
    }
    labeled.append(&mut filler);
    labeled.truncate(train_max);
    Ok(SelectedDocs {
        docs: labeled,
        scanned_all: !saw_unkept,
    })
}

fn pick_queries(
    path: &Path,
    qrels: &LoadedQrels,
    by_source: &HashMap<String, Vec<i32>>,
    query_max: usize,
) -> Result<(Vec<QueryPick>, bool), DripstoneError> {
    let scan = scan_query_texts(path, qrels, query_max)?;
    let mut picks = labeled_picks(qrels, by_source, &scan.texts, query_max);
    append_unlabeled(&mut picks, scan.extras, query_max);
    Ok((picks, !scan.stopped_early))
}

/// Query texts read from the docs file: the labeled ones, plus up to
/// `query_max` unlabeled ones in file order.
struct QueryScan {
    texts: HashMap<String, String>,
    extras: Vec<(String, String)>,
    stopped_early: bool,
}

fn scan_query_texts(
    path: &Path,
    qrels: &LoadedQrels,
    query_max: usize,
) -> Result<QueryScan, DripstoneError> {
    let wanted: HashSet<&str> = qrels.order.iter().map(String::as_str).collect();
    let mut reader = open_docs(path)?;
    let mut scan = QueryScan {
        texts: HashMap::new(),
        extras: Vec::new(),
        stopped_early: false,
    };
    while let Some(doc) = reader.next_doc()? {
        if doc.id.is_empty() {
            continue;
        }
        let want_this = wanted.contains(doc.id.as_str());
        if want_this && !scan.texts.contains_key(&doc.id) {
            scan.texts.insert(doc.id, doc.text);
            continue;
        }
        if !want_this && scan.extras.len() < query_max {
            scan.extras.push((doc.id, doc.text));
        }
        let have_labeled = wanted.is_empty() || scan.texts.len() >= wanted.len();
        if have_labeled && scan.extras.len() >= query_max {
            scan.stopped_early = true;
            break;
        }
    }
    Ok(scan)
}

fn labeled_picks(
    qrels: &LoadedQrels,
    by_source: &HashMap<String, Vec<i32>>,
    texts: &HashMap<String, String>,
    query_max: usize,
) -> Vec<QueryPick> {
    let mut picks = Vec::new();
    for query_id in &qrels.order {
        if picks.len() >= query_max {
            break;
        }
        let Some(text) = texts.get(query_id) else {
            continue;
        };
        let Some(corpus_ids) = qrels.by_query.get(query_id) else {
            continue;
        };
        let judgments = graded_for_hits(corpus_ids, by_source);
        if judgments.is_empty() {
            continue;
        }
        picks.push(QueryPick {
            id: query_id.clone(),
            text: text.clone(),
            judgments,
        });
    }
    picks
}

fn append_unlabeled(picks: &mut Vec<QueryPick>, extras: Vec<(String, String)>, query_max: usize) {
    for (id, text) in extras {
        if picks.len() >= query_max {
            break;
        }
        if picks.iter().any(|pick| pick.id == id) {
            continue;
        }
        picks.push(QueryPick {
            id,
            text,
            judgments: Vec::new(),
        });
    }
}

fn graded_for_hits(hits: &[QrelHit], by_source: &HashMap<String, Vec<i32>>) -> Vec<GradedNeighbor> {
    let mut judgments = Vec::new();
    for hit in hits {
        let Some(indices) = by_source.get(&hit.corpus_id) else {
            continue;
        };
        for index in indices {
            merge_graded(
                &mut judgments,
                GradedNeighbor {
                    index: *index,
                    class: hit.class,
                    score: stored_score(hit.score),
                },
            );
        }
    }
    judgments
}

fn stored_score(score: i64) -> Option<f32> {
    let narrow = i16::try_from(score).ok()?;
    Some(f32::from(narrow))
}

fn pad_neighbors(rows: Vec<Vec<i32>>) -> Option<Vec<Vec<i32>>> {
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    if width == 0 {
        return None;
    }
    Some(
        rows.into_iter()
            .map(|mut row| {
                row.resize(width, DROPPED_NEIGHBOR);
                row
            })
            .collect(),
    )
}

fn load_qrels(path: &Path, policy: GradePolicy) -> Result<LoadedQrels, DripstoneError> {
    let mut loaded = LoadedQrels {
        corpus_ids: HashSet::new(),
        order: Vec::new(),
        by_query: HashMap::new(),
    };
    match table_kind(path)? {
        TableKind::Jsonl => {
            for record in read_jsonl_qrels(path)? {
                push_qrel(&mut loaded, policy, record);
            }
        }
        TableKind::Parquet => {
            scan_parquet(
                path,
                &[QUERY_ID_COLUMN, CORPUS_ID_COLUMN],
                &[SCORE_COLUMN, CLASS_COLUMN],
                |batch| {
                    let queries = batch.column_by_name(QUERY_ID_COLUMN);
                    let corpus = batch.column_by_name(CORPUS_ID_COLUMN);
                    let scores = batch.column_by_name(SCORE_COLUMN);
                    let (Some(queries), Some(corpus)) = (queries, corpus) else {
                        return Err(DripstoneError::storage(format!(
                            "qrels {} is missing {QUERY_ID_COLUMN} or {CORPUS_ID_COLUMN}",
                            path.display()
                        )));
                    };
                    for row in 0..batch.num_rows() {
                        let Some(query_id) = string_cell(queries, row) else {
                            continue;
                        };
                        let Some(corpus_id) = string_cell(corpus, row) else {
                            continue;
                        };
                        push_qrel(
                            &mut loaded,
                            policy,
                            QrelRecord {
                                query_id,
                                corpus_id,
                                score: score_at(scores, row),
                                class: class_at(batch, row),
                            },
                        );
                    }
                    Ok(true)
                },
            )?;
        }
    }
    Ok(loaded)
}

fn push_qrel(loaded: &mut LoadedQrels, policy: GradePolicy, record: QrelRecord) {
    if record.query_id.is_empty() || record.corpus_id.is_empty() {
        return;
    }
    let Some(class) = relevance_of(policy, record.class.as_deref(), record.score) else {
        return;
    };
    loaded.corpus_ids.insert(record.corpus_id.clone());
    let slot = loaded.by_query.entry(record.query_id.clone()).or_default();
    if slot.is_empty() {
        loaded.order.push(record.query_id);
    }
    push_hit(
        slot,
        QrelHit {
            corpus_id: record.corpus_id,
            class,
            score: record.score,
        },
    );
}

fn push_hit(slot: &mut Vec<QrelHit>, hit: QrelHit) {
    if let Some(existing) = slot.iter_mut().find(|kept| kept.corpus_id == hit.corpus_id) {
        if hit.class.rank() > existing.class.rank() {
            existing.class = hit.class;
            existing.score = hit.score;
        }
        return;
    }
    slot.push(hit);
}

fn class_at(batch: &arrow_array::RecordBatch, row: usize) -> Option<String> {
    optional_string(batch.column_by_name(CLASS_COLUMN), row)
}

fn require_table(root: &Path, table: CodeTable) -> Result<PathBuf, DripstoneError> {
    find_table(root, table)?.ok_or_else(|| {
        DripstoneError::storage(format!(
            "code corpus is missing {} under {}; stage CoIR CoSQA and set {ENV_CORPUS_CODE_DIR}",
            table.label(),
            root.display()
        ))
    })
}

fn find_table(root: &Path, table: CodeTable) -> Result<Option<PathBuf>, DripstoneError> {
    for relative in table.candidates() {
        let path = root.join(relative);
        if path.is_file() {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn open_docs(path: &Path) -> Result<DocReader, DripstoneError> {
    match table_kind(path)? {
        TableKind::Jsonl => Ok(DocReader::Jsonl(JsonlDocs {
            lines: BufReader::new(File::open(path).map_err(|error| {
                DripstoneError::storage_with_source(format!("open {}", path.display()), error)
            })?)
            .lines(),
            path: path.display().to_string(),
            line_no: 0,
        })),
        TableKind::Parquet => Ok(DocReader::Parquet(ParquetDocs {
            reader: parquet_reader(
                path,
                &[ID_COLUMN, TEXT_COLUMN],
                &[TITLE_COLUMN, LANGUAGE_COLUMN],
            )?,
            batch: None,
            index: 0,
            path: path.display().to_string(),
        })),
    }
}

impl DocReader {
    fn next_doc(&mut self) -> Result<Option<HeldDoc>, DripstoneError> {
        match self {
            Self::Jsonl(reader) => reader.next_doc(),
            Self::Parquet(reader) => reader.next_doc(),
        }
    }
}

impl JsonlDocs {
    fn next_doc(&mut self) -> Result<Option<HeldDoc>, DripstoneError> {
        loop {
            let Some(line) = self.lines.next() else {
                return Ok(None);
            };
            self.line_no = self.line_no.saturating_add(1);
            let line = line.map_err(|error| {
                DripstoneError::storage_with_source(format!("read {}", self.path), error)
            })?;
            let trimmed = line.trim().trim_start_matches('\u{feff}');
            if trimmed.is_empty() {
                continue;
            }
            let record: CodeLine = serde_json::from_str(trimmed).map_err(|error| {
                DripstoneError::storage(format!(
                    "parse {} line {}: {error}",
                    self.path, self.line_no
                ))
            })?;
            return Ok(Some(HeldDoc {
                id: record.id,
                text: record.text,
                title: record.title,
                language: record.language,
            }));
        }
    }
}

impl ParquetDocs {
    fn next_doc(&mut self) -> Result<Option<HeldDoc>, DripstoneError> {
        loop {
            let ready = self
                .batch
                .as_ref()
                .is_some_and(|batch| self.index < batch.num_rows());
            if ready {
                let row = self.index;
                let doc = doc_from_batch(
                    self.batch.as_ref().ok_or_else(|| {
                        DripstoneError::storage(format!("missing parquet batch in {}", self.path))
                    })?,
                    row,
                    &self.path,
                )?;
                self.index = self.index.saturating_add(1);
                return Ok(Some(doc));
            }
            match self.reader.next() {
                None => return Ok(None),
                Some(Err(error)) => {
                    return Err(DripstoneError::storage(format!(
                        "parquet batch {}: {error}",
                        self.path
                    )));
                }
                Some(Ok(batch)) => {
                    self.batch = Some(batch);
                    self.index = 0;
                }
            }
        }
    }
}

fn doc_from_batch(
    batch: &arrow_array::RecordBatch,
    row: usize,
    path: &str,
) -> Result<HeldDoc, DripstoneError> {
    let ids = batch.column_by_name(ID_COLUMN).ok_or_else(|| {
        DripstoneError::storage(format!("code corpus {path} is missing `{ID_COLUMN}`"))
    })?;
    let texts = batch.column_by_name(TEXT_COLUMN).ok_or_else(|| {
        DripstoneError::storage(format!("code corpus {path} is missing `{TEXT_COLUMN}`"))
    })?;
    let id = string_cell(ids, row).ok_or_else(|| {
        DripstoneError::storage(format!(
            "code corpus {path} row {row} has a null or non-string `{ID_COLUMN}` ({:?})",
            ids.data_type()
        ))
    })?;
    let text = if texts.is_null(row) {
        String::new()
    } else {
        string_cell(texts, row).ok_or_else(|| {
            DripstoneError::storage(format!(
                "code corpus {path} row {row} `{TEXT_COLUMN}` type {:?} is not a string",
                texts.data_type()
            ))
        })?
    };
    Ok(HeldDoc {
        id,
        text,
        title: optional_string(batch.column_by_name(TITLE_COLUMN), row),
        language: optional_string(batch.column_by_name(LANGUAGE_COLUMN), row),
    })
}

fn optional_string(column: Option<&arrow_array::ArrayRef>, row: usize) -> Option<String> {
    let column = column?;
    if column.is_null(row) {
        None
    } else {
        string_cell(column, row)
    }
}

fn score_at(column: Option<&arrow_array::ArrayRef>, row: usize) -> i64 {
    let Some(column) = column else {
        return 1;
    };
    if column.is_null(row) {
        return 1;
    }
    if let Some(array) = column.as_primitive_opt::<Int64Type>() {
        return array.value(row);
    }
    1
}

enum TableKind {
    Jsonl,
    Parquet,
}

fn table_kind(path: &Path) -> Result<TableKind, DripstoneError> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    match extension.to_ascii_lowercase().as_str() {
        "jsonl" | "json" => Ok(TableKind::Jsonl),
        "parquet" => Ok(TableKind::Parquet),
        other => Err(DripstoneError::storage(format!(
            "code corpus file {} has unsupported extension `{other}`",
            path.display()
        ))),
    }
}

fn parquet_reader(
    path: &Path,
    required: &[&str],
    optional: &[&str],
) -> Result<ParquetRecordBatchReader, DripstoneError> {
    let file = File::open(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("open {}", path.display()), error)
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|error| {
        DripstoneError::storage(format!("parquet reader {}: {error}", path.display()))
    })?;
    let projection = projection_for(&builder, path, required, optional)?;
    builder
        .with_projection(projection)
        .build()
        .map_err(|error| {
            DripstoneError::storage(format!("parquet build {}: {error}", path.display()))
        })
}

fn projection_for(
    builder: &ParquetRecordBatchReaderBuilder<File>,
    path: &Path,
    required: &[&str],
    optional: &[&str],
) -> Result<ProjectionMask, DripstoneError> {
    let schema = builder.schema();
    let mut indices = Vec::new();
    for name in required {
        let index = schema.index_of(name).map_err(|_| {
            DripstoneError::storage(format!(
                "code corpus {} is missing column `{name}`",
                path.display()
            ))
        })?;
        indices.push(index);
    }
    for name in optional {
        if let Ok(index) = schema.index_of(name) {
            indices.push(index);
        }
    }
    Ok(ProjectionMask::roots(builder.parquet_schema(), indices))
}

fn scan_parquet(
    path: &Path,
    required: &[&str],
    optional: &[&str],
    mut on_batch: impl FnMut(&arrow_array::RecordBatch) -> Result<bool, DripstoneError>,
) -> Result<(), DripstoneError> {
    let reader = parquet_reader(path, required, optional)?;
    for batch in reader {
        let batch = batch.map_err(|error| {
            DripstoneError::storage(format!("parquet batch {}: {error}", path.display()))
        })?;
        if !on_batch(&batch)? {
            break;
        }
    }
    Ok(())
}

fn read_jsonl_qrels(path: &Path) -> Result<Vec<QrelRecord>, DripstoneError> {
    let file = File::open(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("open {}", path.display()), error)
    })?;
    let mut records = Vec::new();
    for (offset, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| {
            DripstoneError::storage_with_source(format!("read {}", path.display()), error)
        })?;
        let trimmed = line.trim().trim_start_matches('\u{feff}');
        if trimmed.is_empty() {
            continue;
        }
        let record: QrelRecord = serde_json::from_str(trimmed).map_err(|error| {
            DripstoneError::storage(format!(
                "parse {} line {}: {error}",
                path.display(),
                offset + 1
            ))
        })?;
        records.push(record);
    }
    Ok(records)
}

#[derive(Debug, Deserialize)]
struct CodeLine {
    #[serde(rename = "_id", alias = "id")]
    id: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    language: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QrelRecord {
    #[serde(rename = "query-id", alias = "query_id")]
    query_id: String,
    #[serde(rename = "corpus-id", alias = "corpus_id")]
    corpus_id: String,
    #[serde(default = "default_qrel_score")]
    score: i64,
    #[serde(default, rename = "class", alias = "role", alias = "relevance")]
    class: Option<String>,
}

fn default_qrel_score() -> i64 {
    1
}
