use async_trait::async_trait;
use redis::{AsyncCommands, IntoConnectionInfo, aio::MultiplexedConnection, cmd, pipe};
use std::collections::BTreeSet;

use crate::embedding::{
    DEFAULT_NAMESPACE, EmbeddingRecord, SearchOptions, SimilarityMetric, SimilarityResult,
};
use crate::errors::DripstoneError;

pub struct RedisEmbeddingBackend {
    connection: MultiplexedConnection,
    key_prefix: String,
}

impl RedisEmbeddingBackend {
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn new(
        connection_info: impl IntoConnectionInfo,
        key_prefix: impl Into<String>,
    ) -> Result<Self, DripstoneError> {
        let client = redis::Client::open(connection_info).map_err(DripstoneError::from)?;
        let mut connection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(DripstoneError::from)?;
        let _: () = cmd("PING")
            .query_async(&mut connection)
            .await
            .map_err(DripstoneError::from)?;
        Ok(Self {
            connection,
            key_prefix: normalize_prefix(key_prefix.into()),
        })
    }

    fn build_key(&self, namespace: &str, id: &str) -> String {
        embedding_key(&self.key_prefix, namespace, id)
    }

    fn build_doc_key(&self, namespace: &str, doc_id: &str) -> String {
        // Format: "{prefix}{namespace}:doc:{doc_id}"
        // Embedding hashes are "{prefix}{namespace}:vec:{id}" so an id of `doc:1`
        // cannot collide with the document-id set.
        format!("{}{}:doc:{}", self.key_prefix, namespace, doc_id)
    }

    fn namespace_from_key<'a>(&self, key: &'a str) -> Result<(&'a str, &'a str), DripstoneError> {
        parse_embedding_key(&self.key_prefix, key)
            .ok_or_else(|| DripstoneError::storage("unexpected redis embedding key"))
    }

    /// Embedding hashes use `:vec:`. Document-id sets use `:doc:` and must not be
    /// counted or read with HGET (WRONGTYPE).
    fn is_embedding_hash_key(&self, key: &str) -> bool {
        parse_embedding_key(&self.key_prefix, key).is_some()
    }

    async fn scan_keys(&self, pattern: &str) -> Result<Vec<String>, DripstoneError> {
        let mut conn = self.connection.clone();
        let mut cursor = 0_u64;
        let mut keys = Vec::new();
        loop {
            let (next, batch): (u64, Vec<String>) = redis::cmd("SCAN")
                .cursor_arg(cursor)
                .arg("MATCH")
                .arg(pattern)
                .arg("COUNT")
                .arg(256_u64)
                .query_async(&mut conn)
                .await
                .map_err(DripstoneError::from)?;
            keys.extend(batch);
            if next == 0 {
                break;
            }
            cursor = next;
        }
        Ok(keys)
    }

    async fn persist_records(
        &self,
        mut records: Vec<EmbeddingRecord>,
    ) -> Result<(), DripstoneError> {
        if records.is_empty() {
            return Ok(());
        }

        let mut conn = self.connection.clone();
        let mut pipeline = pipe();
        pipeline.atomic();

        for record in &mut records {
            let payload = self.prepare_payload(record)?;

            // Maintain a doc_id -> embedding_ids index for efficient delete_document.
            // Must happen before we move `payload.namespace` into the HSET command.
            if let Some(doc_id) = record
                .metadata
                .as_ref()
                .and_then(|m| m.get("doc_id"))
                .and_then(|v| v.as_str())
            {
                let doc_key = self.build_doc_key(payload.namespace.as_str(), doc_id);
                let mut sadd = cmd("SADD");
                sadd.arg(doc_key).arg(record.id.as_ref());
                pipeline.add_command(sadd);
            }

            let mut command = cmd("HSET");
            command
                .arg(payload.key)
                .arg("id")
                .arg(payload.id)
                .arg("namespace")
                .arg(payload.namespace)
                .arg("vector")
                .arg(payload.vector)
                .arg("metadata")
                .arg(payload.metadata)
                .arg("created_at")
                .arg(payload.created_at)
                .arg("updated_at")
                .arg(payload.updated_at);
            pipeline.add_command(command);
        }

        let _: Vec<usize> = pipeline.query_async(&mut conn).await?;
        Ok(())
    }

    fn prepare_payload(&self, record: &mut EmbeddingRecord) -> Result<HashPayload, DripstoneError> {
        record.touch();
        let namespace_value = record.namespace().to_owned();
        let key = self.build_key(namespace_value.as_str(), &record.id);
        let vector_vec: Vec<f32> = record.vector.iter().copied().collect();
        let vector_payload = rkyv::to_bytes::<rkyv::rancor::Error>(&vector_vec)
            .map(rkyv::util::AlignedVec::into_vec)
            .map_err(|err| DripstoneError::storage(format!("failed to encode vector: {err}")))?;
        let metadata_payload = record
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?
            .unwrap_or_default();
        let created_at = EmbeddingRecord::format_timestamp(&record.created_at)?;
        let updated_at = EmbeddingRecord::format_timestamp(&record.updated_at)?;

        Ok(HashPayload {
            key,
            id: record.id.to_string(),
            namespace: namespace_value,
            vector: vector_payload,
            metadata: metadata_payload,
            created_at,
            updated_at,
        })
    }

    async fn collect_embeddings(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<EmbeddingRecord>, DripstoneError> {
        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let pattern = format!("{}{}:vec:*", self.key_prefix, ns);
        let keys = self.scan_keys(&pattern).await?;
        let mut conn = self.connection.clone();
        let mut records = Vec::with_capacity(keys.len());
        for key in keys {
            if !self.is_embedding_hash_key(&key) {
                continue;
            }
            if let Some(record) = self.fetch_record(&mut conn, &key).await? {
                records.push(record);
            }
        }
        Ok(records)
    }

    async fn fetch_record(
        &self,
        conn: &mut MultiplexedConnection,
        key: &str,
    ) -> Result<Option<EmbeddingRecord>, DripstoneError> {
        let exists: bool = conn.exists(key).await.map_err(DripstoneError::from)?;
        if !exists {
            return Ok(None);
        }

        let (namespace, identifier) = self.namespace_from_key(key)?;

        let vector_bytes: Vec<u8> = conn
            .hget(key, "vector")
            .await
            .map_err(DripstoneError::from)?;
        let vector: Vec<f32> = rkyv::from_bytes::<_, rkyv::rancor::Error>(&vector_bytes)
            .map_err(|err| DripstoneError::storage(format!("failed to decode vector: {err}")))?;

        let metadata_str: String = conn.hget(key, "metadata").await.unwrap_or_default();
        let metadata = if metadata_str.is_empty() {
            None
        } else {
            Some(serde_json::from_str(&metadata_str)?)
        };

        let id: String = conn
            .hget(key, "id")
            .await
            .unwrap_or_else(|_| identifier.to_owned());
        let created_at_raw: String = conn
            .hget(key, "created_at")
            .await
            .map_err(DripstoneError::from)?;
        let updated_at_raw: String = conn
            .hget(key, "updated_at")
            .await
            .map_err(DripstoneError::from)?;

        let created_at = EmbeddingRecord::parse_timestamp(&created_at_raw)?;
        let updated_at = EmbeddingRecord::parse_timestamp(&updated_at_raw)?;

        Ok(Some(EmbeddingRecord {
            id: std::sync::Arc::from(id),
            vector: vector.into(),
            namespace: Some(std::sync::Arc::from(namespace)),
            metadata,
            sparse_vector: None,
            multi_vectors: None,
            created_at,
            updated_at,
        }))
    }

    // Helper methods used by RedisBackend for delegation
    pub(crate) async fn count_embeddings(
        &self,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let pattern = namespace.map_or_else(
            || format!("{}*:vec:*", self.key_prefix),
            |ns| format!("{}{ns}:vec:*", self.key_prefix),
        );
        let keys = self.scan_keys(&pattern).await?;
        Ok(keys
            .into_iter()
            .filter(|k| self.is_embedding_hash_key(k))
            .count())
    }

    pub(crate) async fn list_namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let pattern = format!("{}*:vec:*", self.key_prefix);
        let keys = self.scan_keys(&pattern).await?;
        let mut namespaces = BTreeSet::new();
        for key in keys {
            let (namespace, _) = self.namespace_from_key(key.as_str())?;
            namespaces.insert(namespace.to_owned());
        }
        Ok(namespaces.into_iter().collect())
    }

    pub(crate) async fn flush_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let mut keys = self
            .scan_keys(&format!("{}{namespace}:vec:*", self.key_prefix))
            .await?;
        keys.extend(
            self.scan_keys(&format!("{}{namespace}:doc:*", self.key_prefix))
                .await?,
        );
        let mut conn = self.connection.clone();
        if keys.is_empty() {
            return Ok(());
        }
        let _: () = conn.del(keys).await?;
        Ok(())
    }

    pub(crate) async fn advanced_search(
        &self,
        vector: &[f32],
        mut options: SearchOptions,
    ) -> Result<Vec<SimilarityResult>, DripstoneError> {
        if options.namespace.is_none() {
            options.namespace = Some(DEFAULT_NAMESPACE.to_string());
        }
        let records = self.collect_embeddings(options.namespace_ref()).await?;

        let mut results = Vec::with_capacity(records.len());
        for record in records {
            if let Some(filter) = options.filter.as_ref() {
                if let Some(metadata) = &record.metadata {
                    if !filter.matches(metadata) {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            let score = options.metric.compare(vector, &record.vector)?;
            if !options.passes_threshold(options.metric, score) {
                continue;
            }
            results.push(SimilarityResult::new(std::sync::Arc::new(record), score));
        }

        if options.metric.prefers_lower_scores() {
            results.sort_by(|a, b| a.score.total_cmp(&b.score));
        } else {
            results.sort_by(|a, b| b.score.total_cmp(&a.score));
        }

        if results.len() > options.limit {
            results.truncate(options.limit);
        }
        Ok(results)
    }
}

struct HashPayload {
    key: String,
    id: String,
    namespace: String,
    vector: Vec<u8>,
    metadata: String,
    created_at: String,
    updated_at: String,
}

fn embedding_key(prefix: &str, namespace: &str, id: &str) -> String {
    format!("{prefix}{namespace}:vec:{id}")
}

fn parse_embedding_key<'a>(prefix: &str, key: &'a str) -> Option<(&'a str, &'a str)> {
    let rest = key.strip_prefix(prefix)?;
    let (namespace, id) = rest.rsplit_once(":vec:")?;
    if namespace.is_empty() || id.is_empty() {
        None
    } else {
        Some((namespace, id))
    }
}

fn doc_id_from_metadata_json(metadata: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(metadata).ok()?;
    value.get("doc_id")?.as_str().map(ToOwned::to_owned)
}

fn normalize_prefix(prefix: String) -> String {
    if prefix.is_empty() {
        return prefix;
    }
    if prefix.ends_with(':') {
        prefix
    } else {
        format!("{prefix}:")
    }
}

// =============================================================================
// Backend trait implementation
// =============================================================================

use crate::backend::chunk_envelope::{self, filter_document_metadata};
use crate::core::{Capabilities, Chunk, SearchQuery, SearchResult};
use crate::traits::Backend;
use std::fmt;

impl fmt::Debug for RedisEmbeddingBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RedisBackend")
            .field("key_prefix", &self.key_prefix)
            .finish_non_exhaustive()
    }
}

/// Redis backend for the new `Dripstone` API.
#[derive(Debug)]
pub struct RedisBackend {
    inner: RedisEmbeddingBackend,
}

impl RedisBackend {
    /// Create a new Redis backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn connect(connection_string: &str) -> Result<Self, DripstoneError> {
        let inner = RedisEmbeddingBackend::new(connection_string, "dripstone").await?;
        Ok(Self { inner })
    }

    /// Create a new Redis backend with prefix.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn with_prefix(
        connection_string: &str,
        prefix: &str,
    ) -> Result<Self, DripstoneError> {
        let inner = RedisEmbeddingBackend::new(connection_string, prefix).await?;
        Ok(Self { inner })
    }
}

#[async_trait]
impl Backend for RedisBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: false,
            requires_exact_filter_pass: false,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: None,
            max_batch: None,
            supports_streaming: false,
        }
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        let mut records = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            records.push(chunk_envelope::embedding_record_from_chunk(chunk));
        }
        self.inner.persist_records(records).await
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let key = self.inner.build_key(ns, id);
        let mut conn = self.inner.connection.clone();
        let record = self.inner.fetch_record(&mut conn, &key).await?;
        match record {
            Some(r) => Ok(Some(chunk_envelope::decode_chunk(&r, None)?)),
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let key = self.inner.build_key(ns, id);
        let mut conn = self.inner.connection.clone();
        let metadata: Option<String> = conn
            .hget(&key, "metadata")
            .await
            .map_err(DripstoneError::from)?;
        let doc_id = metadata.as_deref().and_then(doc_id_from_metadata_json);

        let mut pipeline = pipe();
        pipeline.atomic();
        if let Some(doc_id) = doc_id {
            let mut srem = cmd("SREM");
            srem.arg(self.inner.build_doc_key(ns, &doc_id)).arg(id);
            pipeline.add_command(srem);
        }
        let mut del = cmd("DEL");
        del.arg(key);
        pipeline.add_command(del);
        let _: Vec<usize> = pipeline.query_async(&mut conn).await?;
        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let doc_key = self.inner.build_doc_key(ns, doc_id);
        let mut conn = self.inner.connection.clone();

        let ids: Vec<String> = conn.smembers(doc_key.as_str()).await?;
        if ids.is_empty() {
            return Ok(0);
        }

        let mut pipeline = pipe();
        pipeline.atomic();
        for id in &ids {
            let key = self.inner.build_key(ns, id.as_str());
            let mut del = cmd("DEL");
            del.arg(key);
            pipeline.add_command(del);
        }
        let mut del_doc = cmd("DEL");
        del_doc.arg(doc_key);
        pipeline.add_command(del_doc);

        let _: Vec<usize> = pipeline.query_async(&mut conn).await?;
        Ok(ids.len())
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let metric = SimilarityMetric::from(query.metric);

        let namespace = query.namespace.as_deref().unwrap_or(DEFAULT_NAMESPACE);

        let mut oversample_limit = query.limit;
        if query.filter.is_some() {
            oversample_limit = oversample_limit.saturating_mul(5).max(query.limit);
            // Hard cap to avoid unbounded work on large corpora
            oversample_limit = oversample_limit.min(2000);
        }

        let options =
            SearchOptions::new(metric, oversample_limit).with_namespace(namespace.to_string());
        let mut inner_results = self.inner.advanced_search(&query.vector, options).await?;

        // If filter is present, apply it client-side (Redis backend has no native filter support).
        if let Some(ref filter) = query.filter {
            inner_results.retain(|r| {
                r.record
                    .metadata
                    .as_ref()
                    .is_some_and(|metadata| filter.matches(filter_document_metadata(metadata)))
            });
        }

        if inner_results.len() > query.limit {
            inner_results.truncate(query.limit);
        }

        let mut results = Vec::with_capacity(inner_results.len());
        for (i, r) in inner_results.into_iter().enumerate() {
            let chunk = chunk_envelope::decode_chunk(&r.record, None)?;
            results.push(SearchResult::new(chunk, r.score, i));
        }
        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        self.inner.list_namespaces().await
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        self.inner.flush_namespace(namespace).await
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        self.inner.count_embeddings(namespace).await
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let records = self.inner.collect_embeddings(namespace).await?;
        let mut chunks = Vec::with_capacity(records.len());

        for record in records {
            chunks.push(chunk_envelope::decode_chunk(&record, None)?);
        }

        if let Some(limit) = limit {
            chunks.truncate(limit);
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod key_tests {
    use super::{doc_id_from_metadata_json, embedding_key, parse_embedding_key};

    #[test]
    fn embedding_ids_that_look_like_doc_keys_stay_embeddings() {
        let prefix = "coldforge:hot:";
        let key = embedding_key(prefix, "demo", "doc:10");
        assert_eq!(key, "coldforge:hot:demo:vec:doc:10");
        assert_eq!(parse_embedding_key(prefix, &key), Some(("demo", "doc:10")));
        let doc_set = format!("{prefix}demo:doc:document-1");
        assert_eq!(parse_embedding_key(prefix, &doc_set), None);
    }

    #[test]
    fn namespace_may_contain_colons() {
        let key = embedding_key("vf:", "tenant:a", "chunk");
        assert_eq!(
            parse_embedding_key("vf:", &key),
            Some(("tenant:a", "chunk"))
        );
    }

    #[test]
    fn doc_id_is_read_from_the_envelope() {
        let metadata = r#"{"doc_id":"document-1","title":"Hello"}"#;
        assert_eq!(
            doc_id_from_metadata_json(metadata).as_deref(),
            Some("document-1")
        );
        assert_eq!(doc_id_from_metadata_json("{}"), None);
    }
}
