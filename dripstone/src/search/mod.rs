//! Search utilities for hybrid and text-based retrieval.

pub mod bm25;
pub mod filter;
#[cfg(feature = "chroma-backend")]
pub mod filter_to_chroma;
#[cfg(feature = "milvus-backend")]
pub mod filter_to_milvus;
#[cfg(feature = "qdrant-backend")]
pub mod filter_to_qdrant;
#[cfg(feature = "postgres-backend")]
pub mod filter_to_sql;
#[cfg(feature = "weaviate-backend")]
pub mod filter_to_weaviate;
pub mod fusion;
pub mod maxsim;
pub mod mmr;
pub mod native_filter;
pub mod post_filter;
pub mod tokenizer;

#[cfg(feature = "search-advanced")]
pub mod query_expansion;
#[cfg(feature = "search-advanced")]
pub mod temporal_decay;

pub use bm25::{Bm25Config, Bm25Scorer};
pub use filter::{Filter, FilterBuilder};
pub use fusion::{
    FusionConfig, FusionStrategy, fuse_search_results, reciprocal_rank_fusion, weighted_fusion,
};
pub use maxsim::{
    DEFAULT_MAX_TOKEN_VECTORS, MaxSimOptions, MultiVectors, exact_maxsim_search, maxsim_score,
    multi_vectors_from_owned, overfetch_limit, pooled_vector_from_multi, rerank_with_maxsim,
    tokens_for_late_interaction,
};
pub use mmr::{MmrConfig, MmrSimilarity, mmr_rerank};
pub use native_filter::{
    EqualityConjunctionPlan, EqualityTerm, NativeFilterKey, NativeFilterPlan, collect_filter_keys,
    compile_native_filter_plan, json_key,
};
pub use post_filter::{SearchPostFilter, native_plan_covers_namespace, search_post_filter_passes};
pub use tokenizer::{SimpleTokenizer, Tokenizer};

#[cfg(feature = "search-advanced")]
pub use query_expansion::{
    ExpandedQuery, QueryExpansionConfig, expand_query_for_fts, extract_keywords,
};
#[cfg(feature = "search-advanced")]
pub use temporal_decay::{
    TemporalDecayConfig, apply_temporal_decay, decay_lambda, decay_multiplier, extract_timestamp,
};
