use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::Mmap;
use parking_lot::RwLock;

use crate::backend::mmap::format::{
    ArchivedMmapIdIndexV1, ArchivedMmapRowMetaTableV1, ArchivedMmapRowMetaV1,
    MMAP_V1_DEFAULT_ALIGNMENT, ParsedMmapHeaderV1, ParsedMmapHeaderV2, VectorEncoding,
    is_rkyv_mmap_v1, is_rkyv_mmap_v2, parse_mmap_header_v1, parse_mmap_header_v2,
};
use crate::backend::mmap::writable::WritableStore;

use crate::core::Chunk;
use crate::errors::DripstoneError;

use super::{ActiveMmapData, MmapBackend, MmapBackendConfig};

impl MmapBackend {
    /// Opens a memory-mapped `Dripstone` index file (read-only) or creates an empty file (read-write truncate).
    ///
    /// # Errors
    ///
    /// Fails if the file exists but does not start with a supported `Dripstone` mmap magic or fails validation.
    pub fn new(config: MmapBackendConfig) -> Result<Self, DripstoneError> {
        let path = Path::new(&config.file_path);

        if config.read_only {
            if !path.exists() {
                return Err(DripstoneError::storage(format!(
                    "mmap file does not exist: {}",
                    config.file_path
                )));
            }

            let file = File::open(path)
                .map_err(|e| DripstoneError::storage_with_source("failed to open mmap file", e))?;

            let mmap = unsafe {
                Mmap::map(&file).map_err(|e| {
                    DripstoneError::storage_with_source("failed to create memory map", e)
                })?
            };

            let data = if is_rkyv_mmap_v1(&mmap) {
                let header = Self::validate_v1(&mmap, config.dimension)?;
                ActiveMmapData::AlignedV1 { mmap, header }
            } else if is_rkyv_mmap_v2(&mmap) {
                let header = Self::validate_v2(&mmap, config.dimension)?;
                ActiveMmapData::AlignedV2 { mmap, header }
            } else {
                return Err(DripstoneError::storage(format!(
                    "mmap file is not a supported Dripstone index (expected magic DRIPST01 or DRIPST02): {}",
                    config.file_path
                )));
            };

            Ok(Self {
                config,
                data,
                _file: Some(file),
                writable: None,
                snapshot_bytes: AtomicU64::new(0),
            })
        } else {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)
                .map_err(|e| {
                    DripstoneError::storage_with_source("failed to create mmap file", e)
                })?;
            drop(file);

            let flush_every = config.flush_every_ops;
            let path_str = config.file_path.clone();
            let dim = config.dimension;
            let vector_encoding = config.vector_encoding;
            Ok(Self {
                config,
                data: ActiveMmapData::Empty,
                _file: None,
                writable: Some(Arc::new(RwLock::new(WritableStore::new(
                    path_str,
                    dim,
                    flush_every,
                    vector_encoding,
                )))),
                snapshot_bytes: AtomicU64::new(0),
            })
        }
    }

    /// Byte length of the on-disk snapshot after the last flush (read-write mode only).
    #[must_use]
    pub fn last_snapshot_len(&self) -> u64 {
        self.snapshot_bytes.load(Ordering::Relaxed)
    }

    pub(super) fn validate_v1(
        mmap: &Mmap,
        expected_dimension: usize,
    ) -> Result<ParsedMmapHeaderV1, DripstoneError> {
        let header = parse_mmap_header_v1(mmap)?;
        if usize::try_from(header.dimension).ok() != Some(expected_dimension) {
            return Err(DripstoneError::storage(format!(
                "mmap format version {}: file dimension {} does not match config dimension {}",
                header.version, header.dimension, expected_dimension
            )));
        }

        let index_slice = &mmap[header.index_offset..header.index_offset + header.index_len];
        let row_meta_slice =
            &mmap[header.row_meta_offset..header.row_meta_offset + header.row_meta_len];
        rkyv::access::<ArchivedMmapIdIndexV1, rkyv::rancor::Error>(index_slice)
            .map_err(|e| DripstoneError::storage(format!("mmap v1 id index validation: {e}")))?;
        let row_meta_table =
            rkyv::access::<ArchivedMmapRowMetaTableV1, rkyv::rancor::Error>(row_meta_slice)
                .map_err(|e| {
                    DripstoneError::storage(format!("mmap v1 row metadata validation: {e}"))
                })?;
        if row_meta_table.rows.len() as u64 != header.chunk_count {
            return Err(DripstoneError::storage(
                "mmap v1 chunk_count does not match row metadata rows".to_string(),
            ));
        }

        let expected_alignment = usize::try_from(header.section_alignment)
            .map_err(|_| DripstoneError::storage("mmap v1 alignment conversion failed"))?;
        if expected_alignment < usize::try_from(MMAP_V1_DEFAULT_ALIGNMENT).unwrap_or(64) {
            return Err(DripstoneError::storage(
                "mmap v1 alignment is smaller than the minimum supported alignment",
            ));
        }
        if matches!(header.vector_encoding, VectorEncoding::F32)
            && header.vector_offset % std::mem::align_of::<f32>() != 0
        {
            return Err(DripstoneError::storage(
                "mmap v1 f32 vector slab is not aligned to f32 boundaries",
            ));
        }
        for (row_index, row_meta) in row_meta_table.rows.iter().enumerate() {
            let vector_row = row_meta.vector_row.to_native() as usize;
            if vector_row != row_index {
                return Err(DripstoneError::storage(
                    "mmap v1 row metadata vector_row must match row order",
                ));
            }
        }

        Ok(header)
    }

    pub(super) fn validate_v2(
        mmap: &Mmap,
        expected_dimension: usize,
    ) -> Result<ParsedMmapHeaderV2, DripstoneError> {
        let header = parse_mmap_header_v2(mmap)?;
        if usize::try_from(header.dimension).ok() != Some(expected_dimension) {
            return Err(DripstoneError::storage(format!(
                "mmap format version {}: file dimension {} does not match config dimension {}",
                header.version, header.dimension, expected_dimension
            )));
        }

        let index_slice = &mmap[header.index_offset..header.index_offset + header.index_len];
        let row_meta_slice =
            &mmap[header.row_meta_offset..header.row_meta_offset + header.row_meta_len];
        rkyv::access::<ArchivedMmapIdIndexV1, rkyv::rancor::Error>(index_slice)
            .map_err(|e| DripstoneError::storage(format!("mmap v2 id index validation: {e}")))?;
        let row_meta_table =
            rkyv::access::<ArchivedMmapRowMetaTableV1, rkyv::rancor::Error>(row_meta_slice)
                .map_err(|e| {
                    DripstoneError::storage(format!("mmap v2 row metadata validation: {e}"))
                })?;
        if row_meta_table.rows.len() as u64 != header.chunk_count {
            return Err(DripstoneError::storage(
                "mmap v2 chunk_count does not match row metadata rows".to_string(),
            ));
        }

        let expected_alignment = usize::try_from(header.section_alignment)
            .map_err(|_| DripstoneError::storage("mmap v2 alignment conversion failed"))?;
        if expected_alignment < usize::try_from(MMAP_V1_DEFAULT_ALIGNMENT).unwrap_or(64) {
            return Err(DripstoneError::storage(
                "mmap v2 alignment is smaller than the minimum supported alignment",
            ));
        }
        for (row_index, row_meta) in row_meta_table.rows.iter().enumerate() {
            let vector_row = row_meta.vector_row.to_native() as usize;
            if vector_row != row_index {
                return Err(DripstoneError::storage(
                    "mmap v2 row metadata vector_row must match row order",
                ));
            }
        }

        Ok(header)
    }

    pub(super) fn rkyv_index_v1_unchecked<'a>(
        mmap: &'a Mmap,
        header: &ParsedMmapHeaderV1,
    ) -> &'a ArchivedMmapIdIndexV1 {
        let slice = &mmap[header.index_offset..header.index_offset + header.index_len];
        // SAFETY: the slice offsets come from a header that `validate_v1` checked with
        // `rkyv::access` when the map was opened, and the mapping is read-only, so the
        // archive bytes have not changed.
        unsafe { rkyv::access_unchecked(slice) }
    }

    pub(super) fn rkyv_row_meta_v1_unchecked<'a>(
        mmap: &'a Mmap,
        header: &ParsedMmapHeaderV1,
    ) -> &'a ArchivedMmapRowMetaTableV1 {
        let slice = &mmap[header.row_meta_offset..header.row_meta_offset + header.row_meta_len];
        // SAFETY: the slice offsets come from a header that `validate_v1` checked with
        // `rkyv::access` when the map was opened, and the mapping is read-only, so the
        // archive bytes have not changed.
        unsafe { rkyv::access_unchecked(slice) }
    }

    pub(super) fn rkyv_index_v2_unchecked<'a>(
        mmap: &'a Mmap,
        header: &ParsedMmapHeaderV2,
    ) -> &'a ArchivedMmapIdIndexV1 {
        let slice = &mmap[header.index_offset..header.index_offset + header.index_len];
        // SAFETY: the slice offsets come from a header that `validate_v2` checked with
        // `rkyv::access` when the map was opened, and the mapping is read-only, so the
        // archive bytes have not changed.
        unsafe { rkyv::access_unchecked(slice) }
    }

    pub(super) fn rkyv_row_meta_v2_unchecked<'a>(
        mmap: &'a Mmap,
        header: &ParsedMmapHeaderV2,
    ) -> &'a ArchivedMmapRowMetaTableV1 {
        let slice = &mmap[header.row_meta_offset..header.row_meta_offset + header.row_meta_len];
        // SAFETY: the slice offsets come from a header that `validate_v2` checked with
        // `rkyv::access` when the map was opened, and the mapping is read-only, so the
        // archive bytes have not changed.
        unsafe { rkyv::access_unchecked(slice) }
    }

    pub(super) fn vector_norm_v1(
        mmap: &Mmap,
        header: &ParsedMmapHeaderV1,
        row_index: usize,
    ) -> Result<f32, DripstoneError> {
        let chunk_count = usize::try_from(header.chunk_count)
            .map_err(|_| DripstoneError::storage("mmap v1 chunk count exceeds addressable size"))?;
        if row_index >= chunk_count {
            return Err(DripstoneError::storage(format!(
                "mmap v1 row index {row_index} out of bounds for {} rows",
                header.chunk_count
            )));
        }
        let byte_start = header
            .norms_offset
            .checked_add(
                row_index
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        DripstoneError::storage("mmap v1 norms row offset overflow".to_string())
                    })?,
            )
            .ok_or_else(|| {
                DripstoneError::storage("mmap v1 norms row offset overflow".to_string())
            })?;
        let byte_end = byte_start
            .checked_add(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                DripstoneError::storage("mmap v1 norms row length overflow".to_string())
            })?;
        let slab_end = header
            .norms_offset
            .checked_add(header.norms_len)
            .ok_or_else(|| {
                DripstoneError::storage("mmap v1 norms slab end overflow".to_string())
            })?;
        if byte_end > slab_end {
            return Err(DripstoneError::storage(
                "mmap v1 norms row exceeds the declared norms slab",
            ));
        }
        let bits = u32::from_le_bytes(
            mmap[byte_start..byte_end]
                .try_into()
                .map_err(|_| DripstoneError::storage("mmap v1 norms row read length mismatch"))?,
        );
        Ok(f32::from_bits(bits))
    }

    pub(super) fn vector_norm_v2(
        mmap: &Mmap,
        header: &ParsedMmapHeaderV2,
        row_index: usize,
    ) -> Result<f32, DripstoneError> {
        let chunk_count = usize::try_from(header.chunk_count)
            .map_err(|_| DripstoneError::storage("mmap v2 chunk count exceeds addressable size"))?;
        if row_index >= chunk_count {
            return Err(DripstoneError::storage(format!(
                "mmap v2 row index {row_index} out of bounds for {} rows",
                header.chunk_count
            )));
        }
        let byte_start = header
            .norms_offset
            .checked_add(
                row_index
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        DripstoneError::storage("mmap v2 norms row offset overflow".to_string())
                    })?,
            )
            .ok_or_else(|| {
                DripstoneError::storage("mmap v2 norms row offset overflow".to_string())
            })?;
        let byte_end = byte_start
            .checked_add(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                DripstoneError::storage("mmap v2 norms row length overflow".to_string())
            })?;
        let slab_end = header
            .norms_offset
            .checked_add(header.norms_len)
            .ok_or_else(|| {
                DripstoneError::storage("mmap v2 norms slab end overflow".to_string())
            })?;
        if byte_end > slab_end {
            return Err(DripstoneError::storage(
                "mmap v2 norms row exceeds the declared norms slab",
            ));
        }
        let bits = u32::from_le_bytes(
            mmap[byte_start..byte_end]
                .try_into()
                .map_err(|_| DripstoneError::storage("mmap v2 norms row read length mismatch"))?,
        );
        Ok(f32::from_bits(bits))
    }

    pub(super) fn find_row_index_v1(index: &ArchivedMmapIdIndexV1, id: &str) -> Option<usize> {
        let entries = index.entries.as_slice();
        entries
            .binary_search_by(|probe| probe.0.as_str().cmp(id))
            .ok()
            .map(|i| entries[i].1.to_native() as usize)
    }

    pub(super) fn not_ready() -> DripstoneError {
        DripstoneError::storage(
            "MmapBackend has no mapped index (open an existing Dripstone mmap file read-only, or use write_dripstone_mmap to build files)"
                .to_string(),
        )
    }

    pub(super) fn record_snapshot(&self, flushed: Option<u64>) {
        if let Some(len) = flushed {
            self.snapshot_bytes.store(len, Ordering::Relaxed);
        }
    }

    pub(super) fn namespace_matches_v1(
        row_meta: &ArchivedMmapRowMetaV1,
        namespace: Option<&str>,
    ) -> bool {
        namespace.is_none_or(|namespace| {
            row_meta
                .namespace
                .as_ref()
                .is_some_and(|value| value.as_str() == namespace)
        })
    }

    pub(super) fn metadata_from_row_meta_v1(
        row_meta: &ArchivedMmapRowMetaV1,
    ) -> Result<serde_json::Value, DripstoneError> {
        serde_json::from_str(row_meta.metadata_json.as_str())
            .map_err(|e| DripstoneError::storage(format!("Failed to parse chunk metadata: {e}")))
    }

    pub(super) fn get_v1(
        mmap: &Mmap,
        header: &ParsedMmapHeaderV1,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let index = Self::rkyv_index_v1_unchecked(mmap, header);
        let Some(row_index) = Self::find_row_index_v1(index, id) else {
            return Ok(None);
        };

        let row_meta_table = Self::rkyv_row_meta_v1_unchecked(mmap, header);
        let Some(row_meta) = row_meta_table.rows.get(row_index) else {
            return Ok(None);
        };
        if !Self::namespace_matches_v1(row_meta, namespace) {
            return Ok(None);
        }

        Self::materialize_v1_chunk(mmap, header, row_meta, row_index, None, true, true).map(Some)
    }

    pub(super) fn get_v2(
        mmap: &Mmap,
        header: &ParsedMmapHeaderV2,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let index = Self::rkyv_index_v2_unchecked(mmap, header);
        let Some(row_index) = Self::find_row_index_v1(index, id) else {
            return Ok(None);
        };

        let row_meta_table = Self::rkyv_row_meta_v2_unchecked(mmap, header);
        let Some(row_meta) = row_meta_table.rows.get(row_index) else {
            return Ok(None);
        };
        if !Self::namespace_matches_v1(row_meta, namespace) {
            return Ok(None);
        }

        Self::materialize_v2_chunk(mmap, header, row_meta, row_index, None, true, true).map(Some)
    }
}
