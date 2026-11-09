//! Optional real-corpus workload for the ColdForge live scene.
//!
//! Unset corpus env keeps the synthetic microbench. `DRIPSTONE_CORPUS` (and the
//! URL / Parquet / code-dir aliases, including `DRIPSTONE_INT_*`) loads that
//! catalog at the soak caps. `code` is CoIR CoSQA.

use std::collections::HashMap;

use crate::core::{Chunk, Metric};
use crate::corpus::{self, CorpusFormat, CorpusProfile, QueryJudgments, RetrievalTechnique};
use crate::errors::DripstoneError;
use crate::rag::ast::META_TEXT;

use super::live::{DIMENSION, SCENE_NAMESPACE, SIZES, sample_chunks};

/// Header fields printed above the live table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SceneHeader {
    /// `synthetic` or the corpus name.
    pub corpus: String,
    /// Vector width.
    pub dimension: usize,
    /// Metric label.
    pub metric: String,
    /// Comma-separated N values.
    pub sizes: String,
    /// Training rows omitted before upsert. Zero for the synthetic scene.
    pub dropped_train_rows: usize,
    /// Query rows omitted before upsert.
    pub dropped_query_rows: usize,
    /// Grade policy recorded on the corpus manifest. `binary_answer` for the synthetic scene.
    pub grade_policy: String,
    /// Technique plugin selected for this process. Does not change recall@k.
    pub technique: String,
}

/// One measured N.
#[derive(Debug, Clone)]
pub(super) struct LiveSlice {
    /// Vectors upserted for this N.
    pub chunks: Vec<Chunk>,
    /// Query passed to search and to brute-force recall.
    pub query: Vec<f32>,
    /// Judgments for `query`. `None` when the corpus has no qrels.
    pub judgments: Option<QueryJudgments>,
}

/// What [`load_scene`] decided to measure.
#[derive(Debug, Clone)]
pub(super) struct LiveScene {
    /// Lines for the report header.
    pub header: SceneHeader,
    /// Width stored in ColdForge.
    pub dimension: usize,
    /// Search metric.
    pub metric: Metric,
    /// One slice per measured N.
    pub slices: Vec<LiveSlice>,
}

/// Synthetic sizes, or one capped corpus slice when a corpus env is set.
///
/// # Errors
///
/// Returns an error when a requested corpus cannot be downloaded or normalized.
pub(super) async fn load_scene() -> Result<LiveScene, DripstoneError> {
    let technique = corpus::technique_from_env()?;
    if corpus::corpus_requested() {
        load_corpus_scene(technique).await
    } else {
        Ok(synthetic_scene(technique))
    }
}

/// Header used by render tests. Matches [`synthetic_scene`].
pub(super) fn synthetic_header() -> SceneHeader {
    SceneHeader {
        corpus: "synthetic".to_string(),
        dimension: DIMENSION,
        metric: corpus::metric_label(Metric::Cosine).to_string(),
        sizes: size_list(&SIZES),
        dropped_train_rows: 0,
        dropped_query_rows: 0,
        grade_policy: corpus::GradePolicy::BinaryAnswer.label(),
        technique: RetrievalTechnique::HardNeighbors.as_str().to_string(),
    }
}

fn synthetic_scene(technique: RetrievalTechnique) -> LiveScene {
    let mut header = synthetic_header();
    header.technique = technique.as_str().to_string();
    let slices = SIZES
        .into_iter()
        .map(|count| {
            let chunks = sample_chunks(count);
            let query = chunks
                .first()
                .map(|chunk| chunk.vector.to_vec())
                .unwrap_or_else(|| vec![0.0; DIMENSION]);
            LiveSlice {
                chunks,
                query,
                judgments: None,
            }
        })
        .collect();
    LiveScene {
        header,
        dimension: DIMENSION,
        metric: Metric::Cosine,
        slices,
    }
}

async fn load_corpus_scene(technique: RetrievalTechnique) -> Result<LiveScene, DripstoneError> {
    let request = corpus::request_from_env(CorpusProfile::Soak)?;
    let prepared = corpus::prepare_corpus(request).await?;
    let metric = corpus::metric_from_env()?.unwrap_or(prepared.loaded.manifest.metric);
    if prepared.loaded.base.is_empty() {
        return Err(DripstoneError::storage(format!(
            "corpus {} produced no training rows at the configured cap",
            prepared.loaded.manifest.name
        )));
    }
    let query = prepared
        .loaded
        .queries
        .first()
        .cloned()
        .or_else(|| prepared.loaded.base.first().cloned())
        .ok_or_else(|| DripstoneError::storage("corpus has no query or training vector"))?;
    let dimension = prepared.loaded.dimension;
    if query.len() != dimension {
        return Err(DripstoneError::storage(format!(
            "corpus query dim {} != {dimension}",
            query.len()
        )));
    }
    let format = prepared.loaded.manifest.format;
    let metas = prepared.loaded.train_meta;
    let mut chunks = Vec::with_capacity(prepared.loaded.base.len());
    for (index, vector) in prepared.loaded.base.into_iter().enumerate() {
        let doc_id = prepared
            .loaded
            .ids
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("doc{index}"));
        let meta = metas.get(index).and_then(Option::as_ref);
        let (content, metadata) = chunk_payload(format, meta, index);
        let end_offset = content.len();
        chunks.push(Chunk::new(
            doc_id,
            0,
            content,
            vector,
            0,
            end_offset,
            1,
            metadata,
            Some(SCENE_NAMESPACE.to_string()),
        ));
    }
    let n = chunks.len();
    let name = prepared.loaded.manifest.name.clone();
    let judgments = prepared
        .loaded
        .judgments
        .as_ref()
        .and_then(|rows| rows.first().cloned());
    Ok(LiveScene {
        header: SceneHeader {
            corpus: name,
            dimension,
            metric: corpus::metric_label(metric).to_string(),
            sizes: n.to_string(),
            dropped_train_rows: prepared.loaded.manifest.dropped_train_rows,
            dropped_query_rows: prepared.loaded.manifest.dropped_query_rows,
            grade_policy: prepared.loaded.manifest.grade_policy.label(),
            technique: technique.as_str().to_string(),
        },
        dimension,
        metric,
        slices: vec![LiveSlice {
            chunks,
            query,
            judgments,
        }],
    })
}

/// Training indices for retrieved chunk ids. Chunk id is `{train_id}#0`.
pub(super) fn retrieved_train_indices(chunks: &[Chunk], retrieved: &[String]) -> Vec<i32> {
    let mut index_of = HashMap::with_capacity(chunks.len());
    for (index, chunk) in chunks.iter().enumerate() {
        let Ok(id) = i32::try_from(index) else {
            continue;
        };
        index_of.insert(chunk.id.as_str(), id);
    }
    let mut indices = Vec::with_capacity(retrieved.len());
    for id in retrieved {
        if let Some(index) = index_of.get(id.as_str()) {
            indices.push(*index);
        }
    }
    indices
}

fn chunk_payload(
    format: CorpusFormat,
    meta: Option<&serde_json::Value>,
    index: usize,
) -> (String, serde_json::Value) {
    match format {
        CorpusFormat::Code => code_payload(meta, index),
        CorpusFormat::Hdf5 | CorpusFormat::Parquet => {
            (String::new(), serde_json::json!({ "idx": index }))
        }
    }
}

fn code_payload(meta: Option<&serde_json::Value>, index: usize) -> (String, serde_json::Value) {
    let Some(meta) = meta else {
        return (String::new(), serde_json::json!({ "idx": index }));
    };
    let content = meta
        .get(META_TEXT)
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string();
    let mut metadata = meta.clone();
    if let Some(object) = metadata.as_object_mut() {
        object.remove(META_TEXT);
        object.insert("idx".to_string(), serde_json::json!(index));
    }
    (content, metadata)
}

fn size_list(sizes: &[usize]) -> String {
    sizes
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_rows_keep_snippet_text_off_the_metadata_blob() {
        let meta = serde_json::json!({
            "text": "def alpha():\n    return 1\n",
            "symbol": "alpha",
            "language": "python"
        });
        let (content, metadata) = code_payload(Some(&meta), 3);
        assert!(content.contains("def alpha"));
        assert!(metadata.get(META_TEXT).is_none());
        assert_eq!(
            metadata.get("symbol").and_then(serde_json::Value::as_str),
            Some("alpha")
        );
        assert_eq!(
            metadata.get("idx").and_then(serde_json::Value::as_u64),
            Some(3)
        );
        let (plain, _) = chunk_payload(CorpusFormat::Hdf5, Some(&meta), 1);
        assert!(plain.is_empty());
    }
}
