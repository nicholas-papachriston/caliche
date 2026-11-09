//! Canonical encode/decode for [`Chunk`] fields stored in [`EmbeddingRecord::metadata`].
//!
//! Remote backends persist chunk text, offsets, and document metadata in a single JSON
//! envelope so vectors and auxiliary fields stay in one record. Field names are stable
//! for wire compatibility across Redis, Qdrant, S3, and other stores.

use serde_json::{Value, json};

use crate::core::Chunk;
use crate::embedding::{DEFAULT_NAMESPACE, EmbeddingRecord, EmbeddingVector};
use crate::errors::DripstoneError;

/// Envelope key for [`Chunk::metadata`] (document-level JSON).
pub const KEY_ORIGINAL_METADATA: &str = "original_metadata";
pub const KEY_DOC_ID: &str = "doc_id";
pub const KEY_CONTENT: &str = "content";
pub const KEY_START_OFFSET: &str = "start_offset";
pub const KEY_END_OFFSET: &str = "end_offset";
pub const KEY_CHUNK_INDEX: &str = "chunk_index";
pub const KEY_TOTAL_CHUNKS: &str = "total_chunks";

/// Build the metadata JSON object written to [`EmbeddingRecord::metadata`].
#[must_use]
pub fn encode_chunk_metadata(chunk: &Chunk) -> Value {
    json!({
        KEY_DOC_ID: chunk.doc_id,
        KEY_CONTENT: chunk.content,
        KEY_START_OFFSET: chunk.start_offset,
        KEY_END_OFFSET: chunk.end_offset,
        KEY_CHUNK_INDEX: chunk.chunk_index,
        KEY_TOTAL_CHUNKS: chunk.total_chunks,
        KEY_ORIGINAL_METADATA: chunk.metadata,
    })
}

/// Apply chunk envelope fields to an existing record (vector and id must already be set).
pub fn apply_chunk_metadata(record: &mut EmbeddingRecord, chunk: &Chunk) {
    record.metadata = Some(encode_chunk_metadata(chunk));
}

/// Build an [`EmbeddingRecord`] from a [`Chunk`] using the chunk's namespace (or default).
pub fn embedding_record_from_chunk(chunk: &Chunk) -> EmbeddingRecord {
    let namespace = chunk.namespace.as_deref().unwrap_or(DEFAULT_NAMESPACE);
    let mut record = EmbeddingRecord::new(&chunk.id, std::sync::Arc::clone(&chunk.vector))
        .with_namespace(namespace);
    apply_chunk_metadata(&mut record, chunk);
    if let Some(ref mv) = chunk.multi_vectors {
        record = record.with_multi_vectors(std::sync::Arc::clone(mv));
    }
    record
}

/// Document id stored in the envelope, if present.
#[must_use]
pub fn doc_id_from_envelope(envelope: &Value) -> Option<&str> {
    envelope.get(KEY_DOC_ID).and_then(Value::as_str)
}

/// Metadata object passed to [`crate::search::Filter::matches`] for chunk records.
#[must_use]
pub fn filter_document_metadata(envelope: &Value) -> &Value {
    envelope.get(KEY_ORIGINAL_METADATA).unwrap_or(envelope)
}

fn envelope_from_record(record: &EmbeddingRecord) -> Value {
    record.metadata.clone().unwrap_or_else(|| json!({}))
}

fn str_field(envelope: &Value, key: &str) -> String {
    envelope
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn usize_field(envelope: &Value, key: &str, default: usize) -> usize {
    envelope
        .get(key)
        .and_then(json_number_to_usize)
        .unwrap_or(default)
}

fn json_number_to_usize(value: &Value) -> Option<usize> {
    value
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .or_else(|| value.as_i64().and_then(|n| usize::try_from(n).ok()))
}

fn original_metadata(envelope: &Value) -> Value {
    envelope
        .get(KEY_ORIGINAL_METADATA)
        .cloned()
        .unwrap_or_else(|| json!({}))
}

fn resolve_namespace(record: &EmbeddingRecord, namespace_override: Option<&str>) -> Option<String> {
    namespace_override.map(str::to_string).or_else(|| {
        record
            .namespace
            .as_ref()
            .map(std::string::ToString::to_string)
    })
}

fn chunk_from_envelope(
    id: String,
    vector: EmbeddingVector,
    envelope: &Value,
    namespace: Option<String>,
    multi_vectors: Option<crate::search::MultiVectors>,
) -> Chunk {
    Chunk {
        id,
        doc_id: str_field(envelope, KEY_DOC_ID),
        content: str_field(envelope, KEY_CONTENT),
        vector,
        start_offset: usize_field(envelope, KEY_START_OFFSET, 0),
        end_offset: usize_field(envelope, KEY_END_OFFSET, 0),
        chunk_index: usize_field(envelope, KEY_CHUNK_INDEX, 0),
        total_chunks: usize_field(envelope, KEY_TOTAL_CHUNKS, 1),
        metadata: original_metadata(envelope),
        namespace,
        multi_vectors,
    }
}

/// Rehydrate a [`Chunk`] from envelope JSON and vector (no [`EmbeddingRecord`] wrapper).
pub fn decode_chunk_from_envelope(
    id: impl Into<String>,
    vector: EmbeddingVector,
    envelope: &Value,
    namespace: Option<String>,
) -> Chunk {
    chunk_from_envelope(id.into(), vector, envelope, namespace, None)
}

/// Rehydrate a [`Chunk`] from an [`EmbeddingRecord`].
///
/// When `namespace_override` is set (e.g. search/list scoped to a namespace), it wins over
/// `record.namespace`.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn decode_chunk(
    record: &EmbeddingRecord,
    namespace_override: Option<&str>,
) -> Result<Chunk, DripstoneError> {
    let envelope = envelope_from_record(record);
    Ok(chunk_from_envelope(
        record.id.to_string(),
        std::sync::Arc::clone(&record.vector),
        &envelope,
        resolve_namespace(record, namespace_override),
        record.multi_vectors.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::embedding::EmbeddingRecord;

    fn sample_chunk() -> Chunk {
        Chunk {
            id: "doc1#0".to_string(),
            doc_id: "doc1".to_string(),
            content: "hello world".to_string(),
            vector: Arc::from([0.1_f32, 0.2, 0.3]),
            start_offset: 10,
            end_offset: 21,
            chunk_index: 0,
            total_chunks: 3,
            metadata: json!({"category": "greeting"}),
            namespace: Some("tenant-a".to_string()),
            multi_vectors: None,
        }
    }

    #[test]
    fn encode_decode_round_trip() {
        let chunk = sample_chunk();
        let record = embedding_record_from_chunk(&chunk);
        let decoded = decode_chunk(&record, None).expect("decode");
        assert_eq!(decoded.id, chunk.id);
        assert_eq!(decoded.doc_id, chunk.doc_id);
        assert_eq!(decoded.content, chunk.content);
        assert_eq!(decoded.start_offset, chunk.start_offset);
        assert_eq!(decoded.end_offset, chunk.end_offset);
        assert_eq!(decoded.chunk_index, chunk.chunk_index);
        assert_eq!(decoded.total_chunks, chunk.total_chunks);
        assert_eq!(decoded.metadata, chunk.metadata);
        assert_eq!(decoded.namespace, chunk.namespace);
        assert_eq!(decoded.vector.as_ref(), chunk.vector.as_ref());
    }

    #[test]
    fn decode_namespace_override() {
        let chunk = sample_chunk();
        let record = embedding_record_from_chunk(&chunk);
        let decoded = decode_chunk(&record, Some("forced-ns")).expect("decode");
        assert_eq!(decoded.namespace.as_deref(), Some("forced-ns"));
    }

    #[test]
    fn filter_document_metadata_prefers_original() {
        let envelope = encode_chunk_metadata(&sample_chunk());
        let target = filter_document_metadata(&envelope);
        assert_eq!(
            target.get("category").and_then(Value::as_str),
            Some("greeting")
        );
    }

    #[test]
    fn doc_id_from_envelope_extracts() {
        let envelope = encode_chunk_metadata(&sample_chunk());
        assert_eq!(doc_id_from_envelope(&envelope), Some("doc1"));
    }

    #[test]
    fn decode_tolerates_i64_offsets() {
        let mut record = EmbeddingRecord::new("id", Arc::from([1.0_f32]));
        record.metadata = Some(json!({
            KEY_DOC_ID: "d",
            KEY_CONTENT: "c",
            KEY_START_OFFSET: 5_i64,
            KEY_END_OFFSET: 9_i64,
            KEY_CHUNK_INDEX: 1_i64,
            KEY_TOTAL_CHUNKS: 2_i64,
            KEY_ORIGINAL_METADATA: json!({}),
        }));
        let chunk = decode_chunk(&record, None).expect("decode");
        assert_eq!(chunk.start_offset, 5);
        assert_eq!(chunk.end_offset, 9);
        assert_eq!(chunk.chunk_index, 1);
        assert_eq!(chunk.total_chunks, 2);
    }
}
