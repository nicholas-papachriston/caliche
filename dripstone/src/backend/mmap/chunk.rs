use std::sync::Arc;

use half::f16;
use memmap2::Mmap;

use crate::backend::mmap::format::{
    ArchivedMmapRowMetaV1, ParsedMmapHeaderV1, ParsedMmapHeaderV2, VectorEncoding,
};
use crate::core::Chunk;
use crate::errors::DripstoneError;
use crate::utils::numeric::i32_to_f32;

use super::MmapBackend;

impl MmapBackend {
    pub(super) fn vector_bytes_v2<'a>(
        mmap: &'a Mmap,
        header: &ParsedMmapHeaderV1,
        row_index: usize,
    ) -> Result<&'a [u8], DripstoneError> {
        let chunk_count = usize::try_from(header.chunk_count)
            .map_err(|_| DripstoneError::storage("mmap v1 chunk count exceeds addressable size"))?;
        if row_index >= chunk_count {
            return Err(DripstoneError::storage(format!(
                "mmap v1 row index {row_index} out of bounds for {} rows",
                header.chunk_count
            )));
        }

        let dimension = usize::try_from(header.dimension)
            .map_err(|_| DripstoneError::storage("mmap v1 dimension exceeds addressable size"))?;
        let logical_len = header.vector_encoding.encoded_vector_bytes(dimension)?;
        let start = header
            .vector_offset
            .checked_add(
                row_index
                    .checked_mul(header.vector_stride_bytes)
                    .ok_or_else(|| {
                        DripstoneError::storage("mmap v1 vector row offset overflow".to_string())
                    })?,
            )
            .ok_or_else(|| {
                DripstoneError::storage("mmap v1 vector row offset overflow".to_string())
            })?;
        let end = start.checked_add(logical_len).ok_or_else(|| {
            DripstoneError::storage("mmap v1 vector row length overflow".to_string())
        })?;
        let slab_end = header
            .vector_offset
            .checked_add(header.vector_len)
            .ok_or_else(|| {
                DripstoneError::storage("mmap v1 vector slab end overflow".to_string())
            })?;
        if end > slab_end {
            return Err(DripstoneError::storage(
                "mmap v1 vector row exceeds the declared vector slab",
            ));
        }
        Ok(&mmap[start..end])
    }

    pub(super) fn vector_bytes_v3<'a>(
        mmap: &'a Mmap,
        header: &ParsedMmapHeaderV2,
        row_index: usize,
    ) -> Result<&'a [u8], DripstoneError> {
        let chunk_count = usize::try_from(header.chunk_count)
            .map_err(|_| DripstoneError::storage("mmap v2 chunk count exceeds addressable size"))?;
        if row_index >= chunk_count {
            return Err(DripstoneError::storage(format!(
                "mmap v2 row index {row_index} out of bounds for {} rows",
                header.chunk_count
            )));
        }

        let dimension = usize::try_from(header.dimension)
            .map_err(|_| DripstoneError::storage("mmap v2 dimension exceeds addressable size"))?;
        let logical_len = header.vector_encoding.encoded_vector_bytes(dimension)?;
        let start = header
            .vector_offset
            .checked_add(
                row_index
                    .checked_mul(header.vector_stride_bytes)
                    .ok_or_else(|| {
                        DripstoneError::storage("mmap v2 vector row offset overflow".to_string())
                    })?,
            )
            .ok_or_else(|| {
                DripstoneError::storage("mmap v2 vector row offset overflow".to_string())
            })?;
        let end = start.checked_add(logical_len).ok_or_else(|| {
            DripstoneError::storage("mmap v2 vector row length overflow".to_string())
        })?;
        let slab_end = header
            .vector_offset
            .checked_add(header.vector_len)
            .ok_or_else(|| {
                DripstoneError::storage("mmap v2 vector slab end overflow".to_string())
            })?;
        if end > slab_end {
            return Err(DripstoneError::storage(
                "mmap v2 vector row exceeds the declared vector slab",
            ));
        }
        Ok(&mmap[start..end])
    }

    pub(super) fn decode_vector_v2(
        candidate_bytes: &[u8],
        encoding: VectorEncoding,
        dimension: usize,
    ) -> Result<Arc<[f32]>, DripstoneError> {
        match encoding {
            VectorEncoding::F32 => {
                let mut values = Vec::with_capacity(dimension);
                for chunk in candidate_bytes.chunks_exact(std::mem::size_of::<f32>()) {
                    let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    values.push(f32::from_bits(bits));
                }
                Ok(Arc::from(values))
            }
            VectorEncoding::F16 => {
                let mut values = Vec::with_capacity(dimension);
                for chunk in candidate_bytes.chunks_exact(std::mem::size_of::<u16>()) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    values.push(f16::from_bits(bits).to_f32());
                }
                Ok(Arc::from(values))
            }
            VectorEncoding::Int8 | VectorEncoding::ProductQuantized => {
                Err(DripstoneError::storage(format!(
                    "mmap v1 encoding {encoding:?} is reserved but not implemented"
                )))
            }
        }
    }

    pub(super) fn decode_vector_v3(
        candidate_bytes: &[u8],
        header: &ParsedMmapHeaderV2,
    ) -> Result<Arc<[f32]>, DripstoneError> {
        match header.vector_encoding {
            VectorEncoding::Int8 => {
                let zero_point = i32_to_f32(header.quant_zero_point);
                let values = candidate_bytes
                    .iter()
                    .map(|byte| {
                        (f32::from(i8::from_le_bytes([*byte])) - zero_point) * header.quant_scale
                    })
                    .collect::<Vec<_>>();
                Ok(Arc::from(values))
            }
            VectorEncoding::F32 | VectorEncoding::F16 | VectorEncoding::ProductQuantized => {
                Err(DripstoneError::storage(format!(
                    "mmap v2 encoding {:?} is not supported",
                    header.vector_encoding
                )))
            }
        }
    }

    pub(super) fn materialize_v1_chunk(
        mmap: &Mmap,
        header: &ParsedMmapHeaderV1,
        row_meta: &ArchivedMmapRowMetaV1,
        row_index: usize,
        metadata: Option<serde_json::Value>,
        include_content: bool,
        include_vectors: bool,
    ) -> Result<Chunk, DripstoneError> {
        let metadata = match metadata {
            Some(metadata) => metadata,
            None => Self::metadata_from_row_meta_v1(row_meta)?,
        };
        let vector = if include_vectors {
            let bytes = Self::vector_bytes_v2(mmap, header, row_index)?;
            Self::decode_vector_v2(bytes, header.vector_encoding, header.dimension as usize)?
        } else {
            Arc::from([])
        };

        Ok(Chunk {
            id: row_meta.id.as_str().to_string(),
            doc_id: row_meta.doc_id.as_str().to_string(),
            content: if include_content {
                row_meta.content.as_str().to_string()
            } else {
                String::new()
            },
            vector,
            start_offset: row_meta.start_offset.to_native() as usize,
            end_offset: row_meta.end_offset.to_native() as usize,
            chunk_index: row_meta.chunk_index.to_native() as usize,
            total_chunks: row_meta.total_chunks.to_native() as usize,
            metadata,
            namespace: row_meta
                .namespace
                .as_ref()
                .map(|namespace| namespace.as_str().to_string()),
            multi_vectors: None,
        })
    }

    pub(super) fn materialize_v2_chunk(
        mmap: &Mmap,
        header: &ParsedMmapHeaderV2,
        row_meta: &ArchivedMmapRowMetaV1,
        row_index: usize,
        metadata: Option<serde_json::Value>,
        include_content: bool,
        include_vectors: bool,
    ) -> Result<Chunk, DripstoneError> {
        let metadata = match metadata {
            Some(metadata) => metadata,
            None => Self::metadata_from_row_meta_v1(row_meta)?,
        };
        let vector = if include_vectors {
            let bytes = Self::vector_bytes_v3(mmap, header, row_index)?;
            Self::decode_vector_v3(bytes, header)?
        } else {
            Arc::from([])
        };

        Ok(Chunk {
            id: row_meta.id.as_str().to_string(),
            doc_id: row_meta.doc_id.as_str().to_string(),
            content: if include_content {
                row_meta.content.as_str().to_string()
            } else {
                String::new()
            },
            vector,
            start_offset: row_meta.start_offset.to_native() as usize,
            end_offset: row_meta.end_offset.to_native() as usize,
            chunk_index: row_meta.chunk_index.to_native() as usize,
            total_chunks: row_meta.total_chunks.to_native() as usize,
            metadata,
            namespace: row_meta
                .namespace
                .as_ref()
                .map(|namespace| namespace.as_str().to_string()),
            multi_vectors: None,
        })
    }
}
