//! Encode `Dripstone` mmap files (see [`super::format`]).

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::backend::mmap::format::{
    MMAP_HEADER_V1_LEN, MMAP_HEADER_V2_LEN, MMAP_V1_DEFAULT_ALIGNMENT, MmapHeaderV1Spec,
    MmapHeaderV2Spec, MmapIdIndexV1, MmapRowMetaTableV1, MmapRowMetaV1, VectorEncoding,
    align_offset, build_header_bytes_v1, build_header_bytes_v2, default_mmap_vector_encoding,
};
use crate::core::Chunk;
use crate::errors::DripstoneError;
use crate::utils::numeric::i32_to_f32;
use half::f16;

fn validate_chunks(chunks: &[Chunk]) -> Result<usize, DripstoneError> {
    if chunks.is_empty() {
        return Err(DripstoneError::storage(
            "write_dripstone_mmap: empty chunk list".to_string(),
        ));
    }

    let dim = chunks[0].dimension();
    for chunk in chunks {
        if chunk.dimension() != dim {
            return Err(DripstoneError::storage(
                "write_dripstone_mmap: mixed embedding dimensions".to_string(),
            ));
        }
    }

    Ok(dim)
}

fn write_zero_padding(file: &mut File, len: usize) -> Result<(), DripstoneError> {
    if len == 0 {
        return Ok(());
    }

    let zeros = vec![0_u8; len];
    file.write_all(&zeros)
        .map_err(|e| DripstoneError::storage_with_source("mmap write padding", e))
}

fn write_padding_until(
    file: &mut File,
    current_offset: &mut usize,
    target_offset: usize,
) -> Result<(), DripstoneError> {
    if *current_offset > target_offset {
        return Err(DripstoneError::storage(
            "mmap section offsets overlap while writing".to_string(),
        ));
    }

    let padding = target_offset - *current_offset;
    write_zero_padding(file, padding)?;
    *current_offset = target_offset;
    Ok(())
}

fn compute_int8_quant_params(chunks: &[Chunk]) -> (f32, i32) {
    let mut max_abs = 0.0_f32;
    for chunk in chunks {
        for value in chunk.vector.iter() {
            max_abs = max_abs.max(value.abs());
        }
    }

    if max_abs <= f32::EPSILON {
        (1.0, 0)
    } else {
        (max_abs / 127.0, 0)
    }
}

fn quantize_i8(value: f32, scale: f32, zero_point: i32) -> i8 {
    let fallback =
        i8::try_from(zero_point.clamp(i32::from(i8::MIN), i32::from(i8::MAX))).unwrap_or_default();
    if !value.is_finite() || !scale.is_finite() || scale <= 0.0 {
        return fallback;
    }

    let quantized = (value / scale).round() + i32_to_f32(zero_point);
    let clamped = quantized.clamp(f32::from(i8::MIN), f32::from(i8::MAX));
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the rounded quantized value is finite, integral, and clamped to the i8 range"
    )]
    {
        clamped as i8
    }
}

struct MmapSlabLayout {
    vector_stride_bytes: usize,
    index_offset: usize,
    index_len: usize,
    row_meta_offset: usize,
    row_meta_len: usize,
    norms_offset: usize,
    norms_len: usize,
    vector_offset: usize,
    vector_len: usize,
}

fn as_u64(value: usize, field: &str) -> Result<u64, DripstoneError> {
    u64::try_from(value).map_err(|_| DripstoneError::storage(format!("{field} does not fit u64")))
}

fn build_v1_header(
    dim: usize,
    chunk_count: usize,
    encoding: VectorEncoding,
    layout: &MmapSlabLayout,
) -> Result<[u8; MMAP_HEADER_V1_LEN], DripstoneError> {
    Ok(build_header_bytes_v1(MmapHeaderV1Spec {
        dimension: u32::try_from(dim)
            .map_err(|_| DripstoneError::storage("mmap v1 dimension does not fit u32"))?,
        chunk_count: as_u64(chunk_count, "mmap v1 chunk count")?,
        vector_encoding: encoding,
        section_alignment: MMAP_V1_DEFAULT_ALIGNMENT,
        vector_stride_bytes: as_u64(layout.vector_stride_bytes, "mmap v1 vector stride")?,
        index_offset: as_u64(layout.index_offset, "mmap v1 index offset")?,
        index_len: as_u64(layout.index_len, "mmap v1 index length")?,
        row_meta_offset: as_u64(layout.row_meta_offset, "mmap v1 row metadata offset")?,
        row_meta_len: as_u64(layout.row_meta_len, "mmap v1 row metadata length")?,
        norms_offset: as_u64(layout.norms_offset, "mmap v1 norms offset")?,
        norms_len: as_u64(layout.norms_len, "mmap v1 norms length")?,
        vector_offset: as_u64(layout.vector_offset, "mmap v1 vector offset")?,
        vector_len: as_u64(layout.vector_len, "mmap v1 vector length")?,
    }))
}

fn build_v2_header(
    dim: usize,
    chunk_count: usize,
    encoding: VectorEncoding,
    layout: &MmapSlabLayout,
    quant_scale: f32,
    quant_zero_point: i32,
) -> Result<[u8; MMAP_HEADER_V2_LEN], DripstoneError> {
    Ok(build_header_bytes_v2(MmapHeaderV2Spec {
        dimension: u32::try_from(dim)
            .map_err(|_| DripstoneError::storage("mmap v2 dimension does not fit u32"))?,
        chunk_count: as_u64(chunk_count, "mmap v2 chunk count")?,
        vector_encoding: encoding,
        section_alignment: MMAP_V1_DEFAULT_ALIGNMENT,
        vector_stride_bytes: as_u64(layout.vector_stride_bytes, "mmap v2 vector stride")?,
        index_offset: as_u64(layout.index_offset, "mmap v2 index offset")?,
        index_len: as_u64(layout.index_len, "mmap v2 index length")?,
        row_meta_offset: as_u64(layout.row_meta_offset, "mmap v2 row metadata offset")?,
        row_meta_len: as_u64(layout.row_meta_len, "mmap v2 row metadata length")?,
        norms_offset: as_u64(layout.norms_offset, "mmap v2 norms offset")?,
        norms_len: as_u64(layout.norms_len, "mmap v2 norms length")?,
        vector_offset: as_u64(layout.vector_offset, "mmap v2 vector offset")?,
        vector_len: as_u64(layout.vector_len, "mmap v2 vector length")?,
        quant_scale,
        quant_zero_point,
    }))
}

fn serialize_slab_metadata(
    chunks: &[Chunk],
    version: &str,
) -> Result<(rkyv::util::AlignedVec, rkyv::util::AlignedVec), DripstoneError> {
    let mut rows = Vec::with_capacity(chunks.len());
    let mut entries = Vec::with_capacity(chunks.len());
    for (index, chunk) in chunks.iter().enumerate() {
        let vector_row = u32::try_from(index)
            .map_err(|_| DripstoneError::storage(format!("{version}: chunk count exceeds u32")))?;
        rows.push(MmapRowMetaV1 {
            id: chunk.id.clone(),
            doc_id: chunk.doc_id.clone(),
            content: chunk.content.clone(),
            start_offset: chunk.start_offset,
            end_offset: chunk.end_offset,
            chunk_index: chunk.chunk_index,
            total_chunks: chunk.total_chunks,
            metadata_json: serde_json::to_string(&chunk.metadata).map_err(|error| {
                DripstoneError::storage(format!("failed to serialize chunk metadata: {error}"))
            })?,
            namespace: chunk.namespace.clone(),
            vector_row,
        });
        entries.push((chunk.id.clone(), vector_row));
    }
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let index_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&MmapIdIndexV1 { entries })
        .map_err(|error| DripstoneError::storage(format!("{version} index serialize: {error}")))?;
    let row_meta_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&MmapRowMetaTableV1 { rows })
        .map_err(|error| {
            DripstoneError::storage(format!("{version} row metadata serialize: {error}"))
        })?;
    Ok((index_bytes, row_meta_bytes))
}

fn plan_aligned_slab(
    chunks: &[Chunk],
    encoding: VectorEncoding,
    header_len: usize,
    version: &str,
) -> Result<
    (
        MmapSlabLayout,
        rkyv::util::AlignedVec,
        rkyv::util::AlignedVec,
        usize,
    ),
    DripstoneError,
> {
    let alignment = usize::try_from(MMAP_V1_DEFAULT_ALIGNMENT)
        .map_err(|_| DripstoneError::storage(format!("{version} alignment conversion failed")))?;
    let encoded_vector_bytes = encoding.encoded_vector_bytes(chunks[0].dimension())?;
    let vector_stride_bytes = align_offset(encoded_vector_bytes, alignment)?;
    let (index_bytes, row_meta_bytes) = serialize_slab_metadata(chunks, version)?;
    let norms_len = chunks
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| DripstoneError::storage(format!("{version} norms length overflow")))?;
    let index_offset = align_offset(header_len, alignment)?;
    let row_meta_offset = align_offset(index_offset + index_bytes.len(), alignment)?;
    let norms_offset = align_offset(row_meta_offset + row_meta_bytes.len(), alignment)?;
    let vector_offset = align_offset(norms_offset + norms_len, alignment)?;
    let vector_len = vector_stride_bytes
        .checked_mul(chunks.len())
        .ok_or_else(|| DripstoneError::storage(format!("{version} vector slab length overflow")))?;
    Ok((
        MmapSlabLayout {
            vector_stride_bytes,
            index_offset,
            index_len: index_bytes.len(),
            row_meta_offset,
            row_meta_len: row_meta_bytes.len(),
            norms_offset,
            norms_len,
            vector_offset,
            vector_len,
        },
        index_bytes,
        row_meta_bytes,
        encoded_vector_bytes,
    ))
}

fn write_slab_sidecars(
    file: &mut File,
    version: &str,
    header: &[u8],
    layout: &MmapSlabLayout,
    index_bytes: &rkyv::util::AlignedVec,
    row_meta_bytes: &rkyv::util::AlignedVec,
    chunks: &[Chunk],
) -> Result<(), DripstoneError> {
    let mut current_offset = 0usize;
    file.write_all(header)
        .map_err(|e| DripstoneError::storage_with_source(format!("{version} write header"), e))?;
    current_offset += header.len();

    write_padding_until(file, &mut current_offset, layout.index_offset)?;
    file.write_all(index_bytes.as_slice())
        .map_err(|e| DripstoneError::storage_with_source(format!("{version} write index"), e))?;
    current_offset += index_bytes.len();

    write_padding_until(file, &mut current_offset, layout.row_meta_offset)?;
    file.write_all(row_meta_bytes.as_slice()).map_err(|e| {
        DripstoneError::storage_with_source(format!("{version} write row metadata"), e)
    })?;
    current_offset += row_meta_bytes.len();

    write_padding_until(file, &mut current_offset, layout.norms_offset)?;
    for chunk in chunks {
        let squared_sum = chunk.vector.iter().map(|value| value * value).sum::<f32>();
        let norm = squared_sum.sqrt();
        file.write_all(&norm.to_le_bytes()).map_err(|e| {
            DripstoneError::storage_with_source(format!("{version} write norms"), e)
        })?;
    }
    current_offset += layout.norms_len;
    write_padding_until(file, &mut current_offset, layout.vector_offset)
}

fn write_v1_vector_row(
    file: &mut File,
    encoding: VectorEncoding,
    chunk: &Chunk,
) -> Result<(), DripstoneError> {
    match encoding {
        VectorEncoding::F32 => {
            for value in chunk.vector.iter() {
                file.write_all(&value.to_le_bytes()).map_err(|e| {
                    DripstoneError::storage_with_source("mmap v1 write f32 vector", e)
                })?;
            }
            Ok(())
        }
        VectorEncoding::F16 => {
            for value in chunk.vector.iter() {
                let encoded = f16::from_f32(*value);
                file.write_all(&encoded.to_bits().to_le_bytes())
                    .map_err(|e| {
                        DripstoneError::storage_with_source("mmap v1 write f16 vector", e)
                    })?;
            }
            Ok(())
        }
        VectorEncoding::Int8 | VectorEncoding::ProductQuantized => Err(DripstoneError::storage(
            format!("mmap v1 encoding {encoding:?} is reserved but not implemented"),
        )),
    }
}

/// Write a new memory-mapped index file using the v1 aligned slab format.
///
/// The encoded vector slab is separated from row metadata so search can scan vectors
/// without traversing archived `Vec<f32>` structures.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn write_dripstone_mmap_v1(
    path: &Path,
    chunks: &[Chunk],
    encoding: VectorEncoding,
) -> Result<(), DripstoneError> {
    let dim = validate_chunks(chunks)?;
    let (layout, index_bytes, row_meta_bytes, encoded_vector_bytes) =
        plan_aligned_slab(chunks, encoding, MMAP_HEADER_V1_LEN, "mmap v1")?;
    let header = build_v1_header(dim, chunks.len(), encoding, &layout)?;

    let mut file = File::create(path)
        .map_err(|e| DripstoneError::storage_with_source("mmap v1 create file", e))?;
    write_slab_sidecars(
        &mut file,
        "mmap v1",
        &header,
        &layout,
        &index_bytes,
        &row_meta_bytes,
        chunks,
    )?;

    let padding = layout.vector_stride_bytes - encoded_vector_bytes;
    for chunk in chunks {
        write_v1_vector_row(&mut file, encoding, chunk)?;
        write_zero_padding(&mut file, padding)?;
    }

    file.sync_all()
        .map_err(|e| DripstoneError::storage_with_source("mmap v1 sync", e))?;
    Ok(())
}

/// Write a new memory-mapped index file using the v2 int8 slab format.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn write_dripstone_mmap_v2(
    path: &Path,
    chunks: &[Chunk],
    encoding: VectorEncoding,
) -> Result<(), DripstoneError> {
    if !matches!(encoding, VectorEncoding::Int8) {
        return Err(DripstoneError::storage(format!(
            "mmap v2 only supports Int8 encoding, got {encoding:?}"
        )));
    }

    let dim = validate_chunks(chunks)?;
    let (layout, index_bytes, row_meta_bytes, encoded_vector_bytes) =
        plan_aligned_slab(chunks, encoding, MMAP_HEADER_V2_LEN, "mmap v2")?;
    let (quant_scale, quant_zero_point) = compute_int8_quant_params(chunks);
    let header = build_v2_header(
        dim,
        chunks.len(),
        encoding,
        &layout,
        quant_scale,
        quant_zero_point,
    )?;

    let mut file = File::create(path)
        .map_err(|e| DripstoneError::storage_with_source("mmap v2 create file", e))?;
    write_slab_sidecars(
        &mut file,
        "mmap v2",
        &header,
        &layout,
        &index_bytes,
        &row_meta_bytes,
        chunks,
    )?;

    let padding = layout.vector_stride_bytes - encoded_vector_bytes;
    for chunk in chunks {
        for value in chunk.vector.iter() {
            let encoded = quantize_i8(*value, quant_scale, quant_zero_point);
            file.write_all(&encoded.to_le_bytes())
                .map_err(|e| DripstoneError::storage_with_source("mmap v2 write int8 vector", e))?;
        }
        write_zero_padding(&mut file, padding)?;
    }

    file.sync_all()
        .map_err(|e| DripstoneError::storage_with_source("mmap v2 sync", e))?;
    Ok(())
}

/// Write a new mmap index using the aligned v1 format with an explicit encoding.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn write_dripstone_mmap_with_encoding(
    path: &Path,
    chunks: &[Chunk],
    encoding: VectorEncoding,
) -> Result<(), DripstoneError> {
    match encoding {
        VectorEncoding::Int8 => write_dripstone_mmap_v2(path, chunks, encoding),
        _ => write_dripstone_mmap_v1(path, chunks, encoding),
    }
}

/// Write a new mmap index using the aligned v1 format.
///
/// The default v1 layout is architecture-aware:
/// `f32` on `aarch64`, `f16` elsewhere.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn write_dripstone_mmap(path: &Path, chunks: &[Chunk]) -> Result<(), DripstoneError> {
    write_dripstone_mmap_v1(path, chunks, default_mmap_vector_encoding())
}
