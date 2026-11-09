use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::core::{Capabilities, Chunk, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::traits::Backend;

use super::{ActiveMmapData, MmapBackend};

#[async_trait]
impl Backend for MmapBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: None,
            max_batch: None,
            supports_streaming: true,
            requires_exact_filter_pass: false,
        }
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        if let Some(w) = &self.writable {
            let flushed = w.write().upsert(chunks)?;
            self.record_snapshot(flushed);
            return Ok(());
        }
        Err(DripstoneError::storage(
            "MmapBackend is read-only; use write_dripstone_mmap to build files".to_string(),
        ))
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        if let Some(w) = &self.writable {
            return Ok(w.read().get(id, namespace));
        }

        match &self.data {
            ActiveMmapData::Empty => Err(Self::not_ready()),
            ActiveMmapData::AlignedV1 { mmap, header } => Self::get_v1(mmap, header, id, namespace),
            ActiveMmapData::AlignedV2 { mmap, header } => Self::get_v2(mmap, header, id, namespace),
        }
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        if let Some(w) = &self.writable {
            let flushed = w.write().delete(id, namespace)?;
            self.record_snapshot(flushed);
            return Ok(());
        }
        Err(DripstoneError::storage(
            "MmapBackend is read-only".to_string(),
        ))
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        if let Some(w) = &self.writable {
            let (deleted, flushed) = w.write().delete_document(doc_id, namespace)?;
            self.record_snapshot(flushed);
            return Ok(deleted);
        }
        Err(DripstoneError::storage(
            "MmapBackend is read-only".to_string(),
        ))
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        if let Some(w) = &self.writable {
            return w.read().search(query);
        }
        if query.limit == 0 {
            return Ok(Vec::new());
        }

        match &self.data {
            ActiveMmapData::Empty => Err(Self::not_ready()),
            ActiveMmapData::AlignedV1 { mmap, header } => Self::search_v1(mmap, header, query),
            ActiveMmapData::AlignedV2 { mmap, header } => Self::search_v2(mmap, header, query),
        }
    }

    fn search_stream<'a>(
        &'a self,
        query: &'a SearchQuery,
    ) -> BoxStream<'a, Result<SearchResult, DripstoneError>> {
        use futures::StreamExt;

        let query = query.clone();
        Box::pin(
            futures::stream::once(async move { self.search(&query).await }).flat_map(|result| {
                match result {
                    Ok(results) => futures::stream::iter(results.into_iter().map(Ok)).boxed(),
                    Err(e) => futures::stream::once(async move { Err(e) }).boxed(),
                }
            }),
        )
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        if let Some(w) = &self.writable {
            return Ok(w.read().namespaces());
        }

        let mut out: Vec<String> = Vec::new();
        match &self.data {
            ActiveMmapData::Empty => Err(Self::not_ready()),
            ActiveMmapData::AlignedV1 { mmap, header } => {
                let row_meta_table = Self::rkyv_row_meta_v1_unchecked(mmap, header);
                for row in row_meta_table.rows.iter() {
                    if let Some(namespace) = row.namespace.as_ref() {
                        out.push(namespace.as_str().to_string());
                    }
                }
                out.sort();
                out.dedup();
                Ok(out)
            }
            ActiveMmapData::AlignedV2 { mmap, header } => {
                let row_meta_table = Self::rkyv_row_meta_v2_unchecked(mmap, header);
                for row in row_meta_table.rows.iter() {
                    if let Some(namespace) = row.namespace.as_ref() {
                        out.push(namespace.as_str().to_string());
                    }
                }
                out.sort();
                out.dedup();
                Ok(out)
            }
        }
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        if let Some(w) = &self.writable {
            let flushed = w.write().delete_namespace(namespace)?;
            self.record_snapshot(flushed);
            return Ok(());
        }
        Err(DripstoneError::storage(
            "MmapBackend is read-only".to_string(),
        ))
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        if let Some(w) = &self.writable {
            return Ok(w.read().count(namespace));
        }

        match &self.data {
            ActiveMmapData::Empty => Err(Self::not_ready()),
            ActiveMmapData::AlignedV1 { mmap, header } => {
                let row_meta_table = Self::rkyv_row_meta_v1_unchecked(mmap, header);
                namespace.map_or_else(
                    || Ok(row_meta_table.rows.len()),
                    |namespace| {
                        Ok(row_meta_table
                            .rows
                            .iter()
                            .filter(|row| Self::namespace_matches_v1(row, Some(namespace)))
                            .count())
                    },
                )
            }
            ActiveMmapData::AlignedV2 { mmap, header } => {
                let row_meta_table = Self::rkyv_row_meta_v2_unchecked(mmap, header);
                namespace.map_or_else(
                    || Ok(row_meta_table.rows.len()),
                    |namespace| {
                        Ok(row_meta_table
                            .rows
                            .iter()
                            .filter(|row| Self::namespace_matches_v1(row, Some(namespace)))
                            .count())
                    },
                )
            }
        }
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        if let Some(w) = &self.writable {
            return Ok(w.read().list(namespace, limit));
        }

        match &self.data {
            ActiveMmapData::Empty => Err(Self::not_ready()),
            ActiveMmapData::AlignedV1 { mmap, header } => {
                let row_meta_table = Self::rkyv_row_meta_v1_unchecked(mmap, header);
                let mut out = Vec::new();
                for (row_index, row_meta) in row_meta_table.rows.iter().enumerate() {
                    if !Self::namespace_matches_v1(row_meta, namespace) {
                        continue;
                    }
                    out.push(Self::materialize_v1_chunk(
                        mmap, header, row_meta, row_index, None, true, true,
                    )?);
                    if let Some(limit) = limit
                        && out.len() >= limit
                    {
                        break;
                    }
                }
                Ok(out)
            }
            ActiveMmapData::AlignedV2 { mmap, header } => {
                let row_meta_table = Self::rkyv_row_meta_v2_unchecked(mmap, header);
                let mut out = Vec::new();
                for (row_index, row_meta) in row_meta_table.rows.iter().enumerate() {
                    if !Self::namespace_matches_v1(row_meta, namespace) {
                        continue;
                    }
                    out.push(Self::materialize_v2_chunk(
                        mmap, header, row_meta, row_index, None, true, true,
                    )?);
                    if let Some(limit) = limit
                        && out.len() >= limit
                    {
                        break;
                    }
                }
                Ok(out)
            }
        }
    }

    async fn flush(&self) -> Result<(), DripstoneError> {
        if let Some(w) = &self.writable {
            let flushed = w.write().flush_to_disk()?;
            self.record_snapshot(flushed);
        }
        Ok(())
    }
}
