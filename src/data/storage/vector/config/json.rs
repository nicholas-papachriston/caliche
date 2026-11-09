//! JSON / dict parsing for [`VectorBackendConfig`].
//!
//! Shared by the JS and Python bindings (and usable from Rust) so tiered
//! backend specs cannot drift across languages. Accepts both snake_case and
//! camelCase field names.

use super::VectorBackendConfig;
use anyhow::{Context, Result, anyhow};
use serde_json::{Map, Value};

impl VectorBackendConfig {
    /// Parse a single backend object (`{ "type": "...", ... }`).
    pub fn from_json_value(value: &Value) -> Result<Self> {
        vector_backend_from_json_value(value)
    }

    /// Parse a JSON string containing a single backend object.
    pub fn from_json_str(s: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(s).context("invalid vector backend JSON")?;
        Self::from_json_value(&value)
    }
}

/// Parse an array of `{ backend, tierConfig|tier_config }` objects.
pub fn tiered_tiers_from_json_value(
    value: &Value,
) -> Result<Vec<(VectorBackendConfig, dripstone::TierConfig)>> {
    let arr = value
        .as_array()
        .ok_or_else(|| anyhow!("tiered tiers must be a JSON array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let obj = item
            .as_object()
            .ok_or_else(|| anyhow!("tier[{i}]: expected object"))?;
        let backend_val = obj
            .get("backend")
            .ok_or_else(|| anyhow!("tier[{i}]: missing 'backend'"))?;
        let tier_config_val = obj
            .get("tierConfig")
            .or_else(|| obj.get("tier_config"))
            .ok_or_else(|| anyhow!("tier[{i}]: missing 'tierConfig'"))?;
        let backend_config = vector_backend_from_json_value(backend_val)
            .with_context(|| format!("tier[{i}] backend"))?;
        let tier_config = tier_config_from_json_value(tier_config_val)
            .with_context(|| format!("tier[{i}] tierConfig"))?;
        out.push((backend_config, tier_config));
    }
    Ok(out)
}

/// Parse a JSON string of tiered tiers.
pub fn tiered_tiers_from_json_str(
    s: &str,
) -> Result<Vec<(VectorBackendConfig, dripstone::TierConfig)>> {
    let value: Value = serde_json::from_str(s).context("invalid tiered tiers JSON")?;
    tiered_tiers_from_json_value(&value)
}

fn vector_backend_from_json_value(value: &Value) -> Result<VectorBackendConfig> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("backend must be object"))?;
    let backend_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("backend.type required"))?
        .to_lowercase();
    if backend_type == "memory" {
        return memory_from_json(object);
    }
    if let Some(cfg) = decorator_from_json(&backend_type, object)? {
        return Ok(cfg);
    }
    if let Some(cfg) = redis_family_from_json(&backend_type, object)? {
        return Ok(cfg);
    }
    if let Some(cfg) = sql_and_qdrant_from_json(&backend_type, object)? {
        return Ok(cfg);
    }
    if let Some(cfg) = cloud_backend_from_json(&backend_type, object)? {
        return Ok(cfg);
    }
    Err(anyhow!(
        "unknown backend type '{backend_type}'; expected one of: memory, indexed, full_text_indexed, tiered, redis, valkey, postgres, qdrant, qdrant_grpc, pinecone, chroma, weaviate, milvus, s3, coldforge, dynamodb"
    ))
}

fn decorator_from_json(
    backend_type: &str,
    object: &Map<String, Value>,
) -> Result<Option<VectorBackendConfig>> {
    #[cfg(feature = "vector-memory")]
    {
        if backend_type == "indexed" {
            return indexed_from_json(object).map(Some);
        }
        if backend_type == "full_text_indexed" || backend_type == "full-text-indexed" {
            return full_text_indexed_from_json(object).map(Some);
        }
        if backend_type == "tiered" {
            return tiered_from_json(object).map(Some);
        }
    }
    let _ = (backend_type, object);
    Ok(None)
}

#[cfg(feature = "vector-memory")]
fn indexed_from_json(object: &Map<String, Value>) -> Result<VectorBackendConfig> {
    let inner_val = object
        .get("inner")
        .ok_or_else(|| anyhow!("indexed.inner required"))?;
    let inner = Box::new(vector_backend_from_json_value(inner_val)?);
    let dimension = json_require_usize(object, "dimension", "dimension")?;
    Ok(VectorBackendConfig::Indexed {
        inner,
        dimension,
        hnsw_config: json_get_hnsw_config(object)?,
        config: json_get_indexed_backend_config(object)?,
    })
}

#[cfg(feature = "vector-memory")]
fn full_text_indexed_from_json(object: &Map<String, Value>) -> Result<VectorBackendConfig> {
    let inner_val = object
        .get("inner")
        .ok_or_else(|| anyhow!("full_text_indexed.inner required"))?;
    let inner = Box::new(vector_backend_from_json_value(inner_val)?);
    Ok(VectorBackendConfig::FullTextIndexed {
        inner,
        config: json_get_full_text_indexed_backend_config(object)?,
    })
}

#[cfg(feature = "vector-memory")]
fn tiered_from_json(object: &Map<String, Value>) -> Result<VectorBackendConfig> {
    let tiers_val = object
        .get("tiers")
        .ok_or_else(|| anyhow!("tiered.tiers required"))?;
    let tiers = tiered_tiers_from_json_value(tiers_val)?;
    Ok(VectorBackendConfig::Tiered {
        tiers,
        write_strategy: json_get_write_strategy(object)?,
        search_strategy: json_get_search_strategy(object)?,
        background_sync: json_get_bool(object, "background_sync", "backgroundSync")
            .unwrap_or(false),
        sync_interval_secs: json_get_u64(object, "sync_interval_secs", "syncIntervalSecs")
            .unwrap_or(60),
    })
}

#[cfg(feature = "vector-memory")]
fn json_get_hnsw_config(
    object: &Map<String, Value>,
) -> Result<Option<crate::data::storage::vector::HnswConfig>> {
    let Some(hnsw) = object.get("hnsw").or_else(|| object.get("hnswConfig")) else {
        return Ok(None);
    };
    let obj = hnsw
        .as_object()
        .ok_or_else(|| anyhow!("hnsw must be an object"))?;
    let mut config = crate::data::storage::vector::HnswConfig::default();
    if let Some(m) = json_get_usize(obj, "m", "m")? {
        config.m = m;
    }
    if let Some(ef_construction) = json_get_usize(obj, "ef_construction", "efConstruction")? {
        config.ef_construction = ef_construction;
    }
    if let Some(ef_search) = json_get_usize(obj, "ef_search", "efSearch")? {
        config.ef_search = ef_search;
    }
    if let Some(rerank) = json_get_usize(obj, "rerank_overfetch", "rerankOverfetch")? {
        config.rerank_overfetch = rerank;
    }
    Ok(Some(config))
}

#[cfg(feature = "vector-memory")]
fn json_get_indexed_backend_config(
    object: &Map<String, Value>,
) -> Result<Option<dripstone::IndexedBackendConfig>> {
    const KEYS: &[&str] = &[
        "deferred",
        "scan_unindexed_tail",
        "scanUnindexedTail",
        "deferred_flush_threshold",
        "deferredFlushThreshold",
        "deferred_search_flush_threshold",
        "deferredSearchFlushThreshold",
        "max_tail_scan_records",
        "maxTailScanRecords",
        "auto_rebuild_threshold",
        "autoRebuildThreshold",
        "fallback_to_brute_force",
        "fallbackToBruteForce",
        "overfetch_factor",
        "overfetchFactor",
    ];
    if !KEYS.iter().any(|key| object.contains_key(*key)) {
        return Ok(None);
    }
    let mut config = dripstone::IndexedBackendConfig::default();
    apply_indexed_deferred_options(object, &mut config)?;
    apply_indexed_search_options(object, &mut config)?;
    Ok(Some(config))
}

#[cfg(feature = "vector-memory")]
fn apply_indexed_deferred_options(
    object: &Map<String, Value>,
    config: &mut dripstone::IndexedBackendConfig,
) -> Result<()> {
    if json_get_bool(object, "deferred", "deferred").unwrap_or(false) {
        config.index_update_mode = dripstone::IndexUpdateMode::Deferred;
    }
    if let Some(scan) = json_get_bool(object, "scan_unindexed_tail", "scanUnindexedTail") {
        config.scan_unindexed_tail = scan;
    }
    if let Some(threshold) =
        json_get_usize(object, "deferred_flush_threshold", "deferredFlushThreshold")?
    {
        config.deferred_flush_threshold = threshold;
    }
    if let Some(threshold) = json_get_usize(
        object,
        "deferred_search_flush_threshold",
        "deferredSearchFlushThreshold",
    )? {
        config.deferred_search_flush_threshold = threshold;
    }
    if let Some(max_tail) = json_get_usize(object, "max_tail_scan_records", "maxTailScanRecords")? {
        config.max_tail_scan_records = max_tail;
    }
    Ok(())
}

#[cfg(feature = "vector-memory")]
fn apply_indexed_search_options(
    object: &Map<String, Value>,
    config: &mut dripstone::IndexedBackendConfig,
) -> Result<()> {
    if let Some(threshold) =
        json_get_usize(object, "auto_rebuild_threshold", "autoRebuildThreshold")?
    {
        config.auto_rebuild_threshold = threshold;
    }
    if let Some(fallback) = json_get_bool(object, "fallback_to_brute_force", "fallbackToBruteForce")
    {
        config.fallback_to_brute_force = fallback;
    }
    if let Some(factor) = json_get_usize(object, "overfetch_factor", "overfetchFactor")? {
        config.overfetch_factor = factor;
    }
    Ok(())
}

#[cfg(feature = "vector-memory")]
fn json_get_full_text_indexed_backend_config(
    object: &Map<String, Value>,
) -> Result<Option<dripstone::FullTextIndexedBackendConfig>> {
    let has_config = object.contains_key("auto_rebuild_threshold")
        || object.contains_key("autoRebuildThreshold")
        || object.contains_key("snapshot_path")
        || object.contains_key("snapshotPath")
        || object.contains_key("rebuild_on_search")
        || object.contains_key("rebuildOnSearch");
    if !has_config {
        return Ok(None);
    }
    let mut config = dripstone::FullTextIndexedBackendConfig::default();
    if let Some(threshold) =
        json_get_usize(object, "auto_rebuild_threshold", "autoRebuildThreshold")?
    {
        config.auto_rebuild_threshold = threshold;
    }
    if let Some(path) = json_get_str(object, "snapshot_path", "snapshotPath") {
        config.snapshot_path = Some(path);
    }
    if let Some(rebuild) = json_get_bool(object, "rebuild_on_search", "rebuildOnSearch") {
        config.rebuild_on_search = rebuild;
    }
    Ok(Some(config))
}

#[cfg(feature = "vector-memory")]
fn json_get_write_strategy(
    object: &Map<String, Value>,
) -> Result<Option<dripstone::WriteStrategy>> {
    let Some(raw) = json_get_str(object, "write_strategy", "writeStrategy") else {
        return Ok(None);
    };
    match raw.to_ascii_lowercase().as_str() {
        "write_to_source_and_hot" | "source_and_hot" => {
            Ok(Some(dripstone::WriteStrategy::WriteToSourceAndHot))
        }
        "write_to_source" | "source_only" => Ok(Some(dripstone::WriteStrategy::WriteToSource)),
        other => Err(anyhow!("unknown write_strategy '{other}'")),
    }
}

#[cfg(feature = "vector-memory")]
fn json_get_search_strategy(
    object: &Map<String, Value>,
) -> Result<Option<dripstone::SearchStrategy>> {
    let Some(raw) = json_get_str(object, "search_strategy", "searchStrategy") else {
        return Ok(None);
    };
    match raw.to_ascii_lowercase().as_str() {
        "search_all_tiers" | "all_tiers" => Ok(Some(dripstone::SearchStrategy::SearchAllTiers)),
        "hot_then_source_fallback" | "hot_fallback" => {
            Ok(Some(dripstone::SearchStrategy::HotThenSourceFallback))
        }
        other => Err(anyhow!("unknown search_strategy '{other}'")),
    }
}

fn memory_from_json(object: &Map<String, Value>) -> Result<VectorBackendConfig> {
    Ok(VectorBackendConfig::Memory {
        ttl_secs: json_get_u64(object, "ttl_secs", "ttlSecs").unwrap_or(86400),
        max_entries: json_get_usize(object, "max_entries", "maxEntries")?,
    })
}

fn redis_family_from_json(
    backend_type: &str,
    object: &Map<String, Value>,
) -> Result<Option<VectorBackendConfig>> {
    #[cfg(feature = "vector-store-redis")]
    {
        if backend_type == "redis" || backend_type == "valkey" {
            let url = json_require_str(object, "url", "url")?;
            let prefix = json_get_str(object, "prefix", "prefix")
                .unwrap_or_else(|| "embeddings".to_string());
            if backend_type == "valkey" {
                return Ok(Some(VectorBackendConfig::Valkey { url, prefix }));
            }
            return Ok(Some(VectorBackendConfig::Redis { url, prefix }));
        }
    }
    let _ = (backend_type, object);
    Ok(None)
}

fn sql_and_qdrant_from_json(
    backend_type: &str,
    object: &Map<String, Value>,
) -> Result<Option<VectorBackendConfig>> {
    #[cfg(feature = "vector-store-postgres")]
    if backend_type == "postgres" || backend_type == "postgresql" {
        return Ok(Some(VectorBackendConfig::Postgres {
            connection_string: json_require_str(object, "connection_string", "connectionString")?,
            table_name: json_get_str(object, "table_name", "tableName")
                .unwrap_or_else(|| "embeddings".to_string()),
            vector_dimension: json_get_usize(object, "vector_dimension", "vectorDimension")?,
            max_connections: json_get_u64(object, "max_connections", "maxConnections").unwrap_or(10)
                as u32,
            hnsw_config: None,
        }));
    }
    #[cfg(feature = "vector-store-qdrant")]
    if backend_type == "qdrant" || backend_type == "qdrant-http" || backend_type == "qdrant_http" {
        return Ok(Some(VectorBackendConfig::Qdrant {
            url: json_require_str(object, "url", "url")?,
            collection_prefix: json_get_str(object, "collection_prefix", "collectionPrefix"),
            vector_dimension: json_require_usize(object, "vector_dimension", "vectorDimension")?,
            api_key: json_get_str(object, "api_key", "apiKey"),
            batch_size: json_get_usize(object, "batch_size", "batchSize")?,
            timeout: json_get_u64(object, "timeout_secs", "timeoutSecs")
                .map(std::time::Duration::from_secs),
        }));
    }
    #[cfg(feature = "vector-store-qdrant-grpc")]
    if backend_type == "qdrant_grpc" || backend_type == "qdrant-grpc" {
        return Ok(Some(VectorBackendConfig::QdrantGrpc {
            grpc_url: json_require_str(object, "grpc_url", "grpcUrl")?,
            collection_prefix: json_get_str(object, "collection_prefix", "collectionPrefix"),
            vector_dimension: json_require_usize(object, "vector_dimension", "vectorDimension")?,
            api_key: json_get_str(object, "api_key", "apiKey"),
            batch_size: json_get_usize(object, "batch_size", "batchSize")?,
        }));
    }
    let _ = (backend_type, object);
    Ok(None)
}

fn cloud_backend_from_json(
    backend_type: &str,
    object: &Map<String, Value>,
) -> Result<Option<VectorBackendConfig>> {
    if let Some(cfg) = hosted_vector_from_json(backend_type, object)? {
        return Ok(Some(cfg));
    }
    object_store_from_json(backend_type, object)
}

fn hosted_vector_from_json(
    backend_type: &str,
    object: &Map<String, Value>,
) -> Result<Option<VectorBackendConfig>> {
    #[cfg(feature = "vector-store-pinecone")]
    if backend_type == "pinecone" {
        return Ok(Some(VectorBackendConfig::Pinecone {
            api_key: json_require_str(object, "api_key", "apiKey")?,
            base_url: json_require_str(object, "base_url", "baseUrl")?,
            dimension: json_require_usize(object, "dimension", "dimension")?,
        }));
    }
    #[cfg(feature = "vector-store-chroma")]
    if backend_type == "chroma" {
        return Ok(Some(VectorBackendConfig::Chroma {
            url: json_require_str(object, "url", "url")?,
            collection_name: json_get_str(object, "collection_name", "collectionName")
                .unwrap_or_else(|| "embeddings".to_string()),
            dimension: json_require_usize(object, "dimension", "dimension")?,
        }));
    }
    #[cfg(feature = "vector-store-weaviate")]
    if backend_type == "weaviate" {
        return Ok(Some(VectorBackendConfig::Weaviate {
            url: json_require_str(object, "url", "url")?,
            class_name: json_get_str(object, "class_name", "className")
                .or_else(|| json_get_str(object, "collection_name", "collectionName"))
                .unwrap_or_else(|| "Embeddings".to_string()),
            dimension: json_require_usize(object, "dimension", "dimension")
                .or_else(|_| json_require_usize(object, "vector_dimension", "vectorDimension"))?,
            api_key: json_get_str(object, "api_key", "apiKey"),
        }));
    }
    #[cfg(feature = "vector-store-milvus")]
    if backend_type == "milvus" {
        return Ok(Some(VectorBackendConfig::Milvus {
            url: json_require_str(object, "url", "url")
                .or_else(|_| json_require_str(object, "endpoint", "endpoint"))?,
            collection_name: json_get_str(object, "collection_name", "collectionName")
                .or_else(|| json_get_str(object, "collection_prefix", "collectionPrefix"))
                .unwrap_or_else(|| "embeddings".to_string()),
            dimension: json_require_usize(object, "dimension", "dimension")
                .or_else(|_| json_require_usize(object, "vector_dimension", "vectorDimension"))?,
        }));
    }
    let _ = (backend_type, object);
    Ok(None)
}

fn object_store_from_json(
    backend_type: &str,
    object: &Map<String, Value>,
) -> Result<Option<VectorBackendConfig>> {
    #[cfg(feature = "vector-store-s3-backend")]
    if backend_type == "s3" {
        return Ok(Some(VectorBackendConfig::S3 {
            bucket: json_require_str(object, "bucket", "bucket")?,
            prefix: json_get_str(object, "prefix", "prefix")
                .unwrap_or_else(|| "vectors/".to_string()),
            region: json_get_str(object, "region", "region"),
            dimension: json_require_usize(object, "dimension", "dimension")?,
        }));
    }
    #[cfg(feature = "vector-store-coldforge")]
    if backend_type == "coldforge" {
        return Ok(Some(VectorBackendConfig::ColdForge {
            bucket: json_require_str(object, "bucket", "bucket")?,
            prefix: json_get_str(object, "prefix", "prefix").unwrap_or_else(|| "cold/".to_string()),
            region: json_get_str(object, "region", "region"),
            dimension: json_require_usize(object, "dimension", "dimension")?,
            compression: json_get_bool(object, "compression", "compression").unwrap_or(true),
        }));
    }
    #[cfg(feature = "vector-store-dynamodb")]
    if backend_type == "dynamodb"
        || backend_type == "dynamodb-vector"
        || backend_type == "dynamodb_vector"
    {
        return Ok(Some(VectorBackendConfig::DynamoDb {
            table_name: json_require_str(object, "table_name", "tableName")?,
            index_name: json_get_str(object, "index_name", "indexName")
                .unwrap_or_else(|| "embeddings".to_string()),
            region: json_get_str(object, "region", "region"),
            endpoint_url: json_get_str(object, "endpoint_url", "endpointUrl"),
            dimension: json_require_usize(object, "dimension", "dimension")?,
            ensure_table: json_get_bool(object, "ensure_table", "ensureTable").unwrap_or(false),
        }));
    }
    let _ = (backend_type, object);
    Ok(None)
}

/// Parse a single `TierConfig` object (`name`, optional capacity fields).
pub fn tier_config_from_json_value(value: &Value) -> Result<dripstone::TierConfig> {
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow!("tierConfig must be object"))?;
    let name = json_require_str(obj, "name", "name")?;
    let mut cfg = dripstone::TierConfig::new(name);
    if let Some(m) = json_get_usize(obj, "max_vectors", "maxVectors")? {
        cfg = cfg.max_vectors(m);
    }
    if let Some(b) = json_get_usize(obj, "max_bytes", "maxBytes")? {
        cfg = cfg.max_bytes(b);
    }
    if let Some(t) = json_get_usize(obj, "promotion_threshold", "promotionThreshold")? {
        cfg = cfg.promotion_threshold(t);
    }
    Ok(cfg)
}

fn json_get_str(obj: &Map<String, Value>, snake: &str, camel: &str) -> Option<String> {
    obj.get(snake)
        .or_else(|| obj.get(camel))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn json_require_str(obj: &Map<String, Value>, snake: &str, camel: &str) -> Result<String> {
    json_get_str(obj, snake, camel).ok_or_else(|| anyhow!("{snake} (or {camel}) required"))
}

fn json_get_u64(obj: &Map<String, Value>, snake: &str, camel: &str) -> Option<u64> {
    obj.get(snake)
        .or_else(|| obj.get(camel))
        .and_then(Value::as_u64)
}

fn json_get_bool(obj: &Map<String, Value>, snake: &str, camel: &str) -> Option<bool> {
    obj.get(snake)
        .or_else(|| obj.get(camel))
        .and_then(Value::as_bool)
}

fn json_get_usize(obj: &Map<String, Value>, snake: &str, camel: &str) -> Result<Option<usize>> {
    match json_get_u64(obj, snake, camel) {
        Some(v) => {
            Ok(Some(usize::try_from(v).map_err(|_| {
                anyhow!("{snake} exceeds the platform size limit")
            })?))
        }
        None => Ok(None),
    }
}

fn json_require_usize(obj: &Map<String, Value>, snake: &str, camel: &str) -> Result<usize> {
    json_get_usize(obj, snake, camel)?.ok_or_else(|| anyhow!("{snake} (or {camel}) required"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_memory_camel_and_snake() {
        let cfg = VectorBackendConfig::from_json_value(&json!({
            "type": "memory",
            "ttlSecs": 3600,
            "maxEntries": 10
        }))
        .expect("parse");
        match cfg {
            VectorBackendConfig::Memory {
                ttl_secs,
                max_entries,
            } => {
                assert_eq!(ttl_secs, 3600);
                assert_eq!(max_entries, Some(10));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[cfg(feature = "vector-store-dynamodb")]
    #[test]
    fn parses_dynamodb_aliases_and_ensure_table() {
        let cfg = VectorBackendConfig::from_json_value(&json!({
            "type": "dynamodb-vector",
            "tableName": "vectors",
            "indexName": "embeddings",
            "dimension": 8,
            "region": "us-east-1",
            "ensureTable": true
        }))
        .expect("parse");
        match cfg {
            VectorBackendConfig::DynamoDb {
                table_name,
                index_name,
                dimension,
                ensure_table,
                ..
            } => {
                assert_eq!(table_name, "vectors");
                assert_eq!(index_name, "embeddings");
                assert_eq!(dimension, 8);
                assert!(ensure_table);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_tiered_tiers_array() {
        let tiers = tiered_tiers_from_json_value(&json!([
            {
                "backend": { "type": "memory", "ttl_secs": 60, "max_entries": 100 },
                "tierConfig": { "name": "hot", "maxVectors": 100, "promotionThreshold": 1 }
            },
            {
                "backend": { "type": "memory", "ttlSecs": 3600 },
                "tier_config": { "name": "warm", "max_bytes": 1024 }
            }
        ]))
        .expect("tiers");
        assert_eq!(tiers.len(), 2);
        assert_eq!(tiers[0].1.name, "hot");
        assert_eq!(tiers[1].1.name, "warm");
    }

    #[test]
    fn rejects_unknown_backend_type() {
        let err = VectorBackendConfig::from_json_value(&json!({ "type": "mystery" }))
            .expect_err("unknown");
        assert!(err.to_string().contains("unknown backend type"));
    }

    #[cfg(feature = "vector-memory")]
    #[test]
    fn parses_indexed_backend_with_deferred_config() {
        let cfg = VectorBackendConfig::from_json_value(&json!({
            "type": "indexed",
            "inner": { "type": "memory", "ttl_secs": 60 },
            "dimension": 8,
            "deferred": true,
            "scanUnindexedTail": true,
            "deferredFlushThreshold": 128,
            "deferredSearchFlushThreshold": 64,
            "maxTailScanRecords": 32
        }))
        .expect("parse indexed");
        match cfg {
            VectorBackendConfig::Indexed {
                dimension, config, ..
            } => {
                assert_eq!(dimension, 8);
                let config = config.expect("indexed config");
                assert_eq!(
                    config.index_update_mode,
                    dripstone::IndexUpdateMode::Deferred
                );
                assert!(config.scan_unindexed_tail);
                assert_eq!(config.deferred_flush_threshold, 128);
                assert_eq!(config.deferred_search_flush_threshold, 64);
                assert_eq!(config.max_tail_scan_records, 32);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[cfg(feature = "vector-memory")]
    #[tokio::test]
    #[serial_test::serial(env)]
    async fn builds_indexed_backend_from_json_with_deferred_env() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("CALICHE_DEFERRED_ANN_INDEX", "true");
        let cfg = VectorBackendConfig::from_json_value(&json!({
            "type": "indexed",
            "inner": { "type": "memory", "ttl_secs": 60, "max_entries": 100 },
            "dimension": 4
        }))
        .expect("parse");
        let backend = cfg.build_backend().await.expect("build");
        let chunk = dripstone::Chunk::new(
            "doc",
            0,
            "hello",
            vec![1.0_f32, 0.0, 0.0, 0.0],
            0,
            5,
            1,
            serde_json::json!({}),
            None,
        );
        backend.upsert(&[chunk]).await.expect("upsert");
        assert_eq!(backend.count(None).await.expect("count"), 1);
    }
}
