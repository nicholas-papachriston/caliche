//! Download, normalize, and stream vector corpora.
//!
//! Catalog entries cover ann-benchmarks HDF5 sets, the English shard of
//! Cohere's Wikipedia embeddings, and CoIR CoSQA (`code`). Bytes land under
//! `$DRIPSTONE_CACHE_DIR/corpora/<name>/<rev>/` as `manifest.json`,
//! `vectors.f32.bin`, `ids.jsonl`, and optional queries, neighbors, and
//! `meta.jsonl`. Caps (`DRIPSTONE_CORPUS_TRAIN_MAX`, `DRIPSTONE_CORPUS_QUERY_MAX`)
//! bound what is materialized so a multi-gigabyte source is not loaded whole.
//! `DRIPSTONE_INT_*` names remain as aliases for the integration harness.
//!
//! `code` downloads the pinned CoSQA revision, splits functions with
//! tree-sitter ([`crate::config::ChunkStrategy::AstAware`]), and stores hashed
//! feature vectors. [`CodeRowMeta`] is written to `meta.jsonl`. Test qrels
//! become hard neighbor indices plus graded judgments in `qrels.jsonl`.
//! Binary positives are answer-class; support stays empty until a qrel says
//! otherwise. [`RetrievalPlugin`] is the stub for a later embedding technique.

mod catalog;
mod code;
mod env;
mod features;
mod hdf5;
mod layout;
mod normalize;
mod parquet;
mod relevance;
mod sanitize;

pub use catalog::{
    CODE_FEATURE_DIMENSION, COHERE_LANG_PREFIX, COHERE_REPO, COSQA_CORPUS_FILE, COSQA_LANGUAGE,
    COSQA_QRELS_FILE, COSQA_QUERIES_FILE, COSQA_REPO, COSQA_REVISION, CorpusFormat, CorpusName,
    CorpusSelector, GIST_URL, NYTIMES_URL, SIFT_URL,
};
pub use env::{
    CorpusProfile, CorpusRequest, DEFAULT_NEIGHBORS_K, DEFAULT_SHARD_MAX, DEFAULT_UPSERT_BATCH,
    EMB_COLUMN, ENV_CORPUS, ENV_CORPUS_BATCH, ENV_CORPUS_CODE_DIR, ENV_CORPUS_GRADE_POLICY,
    ENV_CORPUS_METRIC, ENV_CORPUS_PARQUET_DIR, ENV_CORPUS_PUT_CONCURRENCY, ENV_CORPUS_QUERY_MAX,
    ENV_CORPUS_SHARD_MAX, ENV_CORPUS_TECHNIQUE, ENV_CORPUS_TRAIN_MAX, ENV_CORPUS_URL,
    ENV_INT_DATASET_NAME, ENV_INT_DATASET_URL, ENV_INT_NEIGHBORS_K, ENV_INT_PARQUET_DIR,
    ENV_INT_QUERY_MAX, ENV_INT_SEARCH_METRIC, ENV_INT_TRAIN_MAX, INTEGRATION_QUERY_MAX,
    INTEGRATION_TRAIN_MAX, SOAK_QUERY_MAX, SOAK_TRAIN_MAX, body_put_concurrency_from_env,
    corpus_requested, grade_policy_from_env, metric_from_env, parse_metric_name, request_from_env,
    resolve_cap, technique_from_env, upsert_batch_from_env,
};
pub use features::code_feature_vector;
pub use layout::{
    CodeRowMeta, CorpusCaps, CorpusManifest, CorpusRowPolicy, IDS_FILE, MANIFEST_FILE, META_FILE,
    MIN_L2_NORM_FOR_COSINE, NEIGHBORS_FILE, NormalizedCorpus, QRELS_FILE, QUERIES_FILE,
    QUERY_IDS_FILE, VECTORS_FILE, VectorBatch, VectorBatchIter,
};
pub use normalize::{
    LoadedCorpus, PreparedCorpus, metric_label, prepare_corpus, prepare_corpus_blocking,
};
pub use relevance::{
    DEFAULT_ANSWER_MIN_SCORE, GradePolicy, GradedNeighbor, GradedRecall, QueryJudgments,
    RelevanceClass, RetrievalExtension, RetrievalPlugin, RetrievalTechnique, class_recall_at_k,
    graded_recall_at_k, mean_ratio, parse_grade_policy, parse_technique, positive_indices,
    record_graded_recall, relevance_of, remap_graded,
};
pub use sanitize::row_ok_for_cosine;

#[cfg(test)]
mod tests;
