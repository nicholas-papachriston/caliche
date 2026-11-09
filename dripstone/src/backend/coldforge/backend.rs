//! [`Backend`](crate::traits::Backend) trait implementation for `ColdForge`.

use super::ColdForgeBackend;
use crate::backend::chunk_envelope::{self, doc_id_from_envelope, filter_document_metadata};
use crate::core::{Capabilities, Chunk, Metric, SearchQuery, SearchResult};
use crate::embedding::{DEFAULT_NAMESPACE, SimilarityMetric};
use crate::errors::DripstoneError;
use crate::index::{IndexSearchRequest, VectorIndex};
use crate::search::compile_native_filter_plan;
use crate::traits::Backend;
use async_trait::async_trait;
use std::fmt;

impl fmt::Debug for ColdForgeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColdForgeBackend")
            .field("bucket", &self.config.bucket)
            .field("key_prefix", &self.config.key_prefix)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Backend for ColdForgeBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true,
            requires_exact_filter_pass: true,
            supports_namespaces: true,
            native_maxsim: true,
            max_dimension: Some(self.config.vector_dimension),
            max_batch: None,
            supports_streaming: false,
        }
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        if chunks.is_empty() {
            return Ok(());
        }

        let grouped = group_chunks_by_namespace(chunks);
        for (namespace, group) in &grouped {
            self.upsert_namespace(namespace, group).await?;
        }

        // Hot tier is a shared get-cache. Search still uses this process's index
        // plus flushed WAL segments. A hot-tier error fails the upsert so the
        // caller can retry; the WAL buffer already holds the operations.
        if let Some(ref hot_tier) = self.hot_tier {
            hot_tier.upsert(chunks).await?;
        }

        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let ns = namespace.unwrap_or(DEFAULT_NAMESPACE);
        self.sync_namespace(Some(ns)).await?;

        if let Some(record) = self.cache.get(id, Some(ns)) {
            if self.index_contains(ns, id).await {
                return Ok(Some(chunk_envelope::decode_chunk(&record, Some(ns))?));
            }
            self.cache.remove_in_namespace(id, Some(ns));
        }

        if let Some(ref hot_tier) = self.hot_tier {
            match hot_tier.get(id, Some(ns)).await {
                Ok(Some(chunk)) if self.index_contains(ns, id).await => {
                    return Ok(Some(chunk));
                }
                Ok(_) => {}
                Err(err) => {
                    tracing::warn!(
                        "ColdForge hot-tier get failed for {id} in {ns}; reading the chunk body: {err}"
                    );
                }
            }
        }

        if self.index_contains(ns, id).await {
            let record = self.require_body(ns, id).await?;
            let chunk = chunk_envelope::decode_chunk(&record, Some(ns))?;
            self.cache.put(record);
            return Ok(Some(chunk));
        }

        Ok(None)
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        use crate::backend::s3::wal::WalOperation;

        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        self.sync_namespace(Some(namespace)).await?;

        let index = self.get_index(Some(namespace)).await;
        let wal_writer = self.get_wal_writer(Some(namespace)).await;

        self.cache.remove_in_namespace(id, Some(namespace));

        {
            let idx = index.write().await;
            VectorIndex::remove(&*idx, id).await?;
            drop(idx);
        }
        self.bodies.remove(namespace, id).await?;

        wal_writer
            .append(WalOperation::Delete {
                id: id.to_string(),
                namespace: Some(namespace.to_string()),
            })
            .await?;

        if let Some(ref hot_tier) = self.hot_tier {
            hot_tier.delete(id, Some(namespace)).await?;
        }

        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        self.sync_namespace(Some(namespace)).await?;

        let index = self.get_index(Some(namespace)).await;
        let candidate_ids: Vec<String> = {
            let idx = index.read().await;
            let records = idx.iter_records_by_namespace(Some(namespace)).await?;
            records
                .into_iter()
                .map(|record| record.id.to_string())
                .collect()
        };
        let mut matching_ids = Vec::new();
        for id in candidate_ids {
            let record = self.require_body(namespace, &id).await?;
            let envelope = record
                .metadata
                .clone()
                .unwrap_or_else(|| serde_json::json!({}));
            if doc_id_from_envelope(&envelope).is_some_and(|found| found == doc_id) {
                matching_ids.push(id);
            }
        }

        let count = matching_ids.len();
        for id in matching_ids {
            self.delete(&id, Some(namespace)).await?;
        }

        Ok(count)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let namespace = query.namespace.as_deref().unwrap_or(DEFAULT_NAMESPACE);
        self.sync_namespace(Some(namespace)).await?;

        let index = self.get_index(Some(namespace)).await;
        let metric = match query.metric {
            Metric::Cosine => SimilarityMetric::Cosine,
            Metric::Euclidean => SimilarityMetric::Euclidean,
            Metric::DotProduct => SimilarityMetric::Dot,
            Metric::Manhattan => SimilarityMetric::Manhattan,
        };

        let native_filter_plan =
            compile_native_filter_plan(query.namespace.as_deref(), query.filter.as_ref());
        let ann_limit = if query.query_multi_vectors.is_some() {
            crate::search::overfetch_limit(query)
        } else {
            query.limit
        };
        let inner_results = {
            let idx = index.read().await;
            VectorIndex::search_request(
                &*idx,
                IndexSearchRequest {
                    query: &query.vector,
                    limit: ann_limit,
                    metric,
                    native_filter: native_filter_plan
                        .is_supported()
                        .then_some(&native_filter_plan),
                },
            )
            .await?
        };

        let mut results = Vec::with_capacity(inner_results.len());
        for r in &inner_results {
            let id = r.record.id.as_ref();
            let record = self.require_body(namespace, id).await?;
            let envelope = record
                .metadata
                .clone()
                .unwrap_or_else(|| serde_json::json!({}));
            // This metadata filter pass is local to ColdForge. A shared post_filter helper does not exist yet.
            if query
                .filter
                .as_ref()
                .is_some_and(|filter| !filter.matches(filter_document_metadata(&envelope)))
            {
                continue;
            }

            let chunk = chunk_envelope::decode_chunk(&record, Some(namespace))?;
            results.push(SearchResult::new(chunk, r.score, results.len()));
        }

        if query.query_multi_vectors.is_some() {
            return crate::search::rerank_with_maxsim(results, query);
        }

        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let indexes = self.indexes.read().await;
        Ok(indexes.keys().cloned().collect())
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        use crate::backend::s3::wal::WalOperation;

        self.sync_namespace(Some(namespace)).await?;

        let index = self.get_index(Some(namespace)).await;
        let wal_writer = self.get_wal_writer(Some(namespace)).await;

        {
            let idx = index.write().await;
            idx.build(&[]).await?;
            drop(idx);
        }

        self.cache.clear_namespace(Some(namespace));
        self.bodies.clear_namespace(namespace).await?;

        wal_writer
            .append(WalOperation::Clear {
                namespace: Some(namespace.to_string()),
            })
            .await?;

        if let Some(ref hot_tier) = self.hot_tier {
            hot_tier.delete_namespace(namespace).await?;
        }

        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        self.sync_namespace(Some(namespace)).await?;

        let index = self.get_index(Some(namespace)).await;
        let count = {
            let idx = index.read().await;
            VectorIndex::len(&*idx)
        };

        Ok(count)
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        self.sync_namespace(Some(namespace)).await?;

        let index = self.get_index(Some(namespace)).await;

        let ids = {
            let idx = index.read().await;
            let records = idx.iter_records_by_namespace(Some(namespace)).await?;
            records
                .into_iter()
                .map(|record| record.id.to_string())
                .collect::<Vec<_>>()
        };

        let capacity = limit.map_or(ids.len(), |l| l.min(ids.len()));
        let mut chunks = Vec::with_capacity(capacity);
        for id in ids {
            if let Some(l) = limit
                && chunks.len() >= l
            {
                break;
            }
            let record = self.require_body(namespace, &id).await?;
            chunks.push(chunk_envelope::decode_chunk(&record, Some(namespace))?);
        }

        Ok(chunks)
    }

    async fn flush(&self) -> Result<(), DripstoneError> {
        let writers = {
            let writers = self.wal_writers.read().await;
            writers.values().cloned().collect::<Vec<_>>()
        };
        for w in &writers {
            w.flush_batch().await?;
        }
        if let Some(ref hot) = self.hot_tier {
            hot.flush().await?;
        }
        Ok(())
    }

    async fn close(&self) -> Result<(), DripstoneError> {
        self.flush().await?;
        if let Some(ref hot) = self.hot_tier {
            hot.close().await?;
        }
        Ok(())
    }
}

impl ColdForgeBackend {
    async fn upsert_namespace(
        &self,
        namespace: &str,
        chunks: &[Chunk],
    ) -> Result<(), DripstoneError> {
        use crate::backend::s3::wal::WalOperation;

        self.sync_namespace(Some(namespace)).await?;
        let index = self.get_index(Some(namespace)).await;
        let wal_writer = self.get_wal_writer(Some(namespace)).await;

        let records = chunks
            .iter()
            .map(chunk_envelope::embedding_record_from_chunk)
            .collect::<Vec<_>>();
        // Bodies go out together. Index and WAL stay in input order, and only
        // after every PUT in this batch has succeeded, so a cold open replays
        // the segment without writing those keys again.
        self.bodies
            .put_many(namespace, &records, self.config.body_put_concurrency)
            .await?;
        for record in records {
            self.cache.put(record.clone());
            let idx = index.write().await;
            VectorIndex::add(&*idx, record.clone()).await?;
            drop(idx);
            wal_writer.append(WalOperation::Upsert(record)).await?;
        }

        if self.config.wal_batch_interval_ms == 0 {
            wal_writer.flush_batch().await?;
        }
        Ok(())
    }

    /// Drop in-process envelopes and the LRU so the next `get` hydrates again.
    ///
    /// The namespace index and the durable body objects stay. Used by the live
    /// scene so get samples measure object storage or the hot tier.
    pub(super) async fn discard_local_envelopes(&self) {
        self.bodies.forget_cached().await;
        self.cache.clear_all();
    }

    async fn require_body(
        &self,
        namespace: &str,
        id: &str,
    ) -> Result<crate::embedding::EmbeddingRecord, DripstoneError> {
        self.bodies.get(namespace, id).await?.ok_or_else(|| {
            DripstoneError::storage(format!(
                "ColdForge chunk body missing for {id} in {namespace}"
            ))
        })
    }
}

fn group_chunks_by_namespace(chunks: &[Chunk]) -> Vec<(String, Vec<Chunk>)> {
    let mut order = Vec::new();
    let mut groups: std::collections::HashMap<String, Vec<Chunk>> =
        std::collections::HashMap::new();
    for chunk in chunks {
        let namespace = chunk
            .namespace
            .as_deref()
            .unwrap_or(DEFAULT_NAMESPACE)
            .to_string();
        if let Some(group) = groups.get_mut(&namespace) {
            group.push(chunk.clone());
        } else {
            order.push(namespace.clone());
            groups.insert(namespace, vec![chunk.clone()]);
        }
    }
    let mut grouped = Vec::with_capacity(order.len());
    for namespace in order {
        if let Some(group) = groups.remove(&namespace) {
            grouped.push((namespace, group));
        }
    }
    grouped
}
