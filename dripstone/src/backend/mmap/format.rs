//! Versioned on-disk layout for `Dripstone` memory-mapped indexes.
//!
//! V1 splits hot vector bytes from colder row metadata so the search path can scan
//! a contiguous aligned vector slab without pointer chasing.
//! V2 adds int8 quantization parameters on top of the v1 slab layout.

use crate::errors::DripstoneError;
use rkyv::{Archive, Deserialize, Serialize};

/// Magic: `DRIPST01` (8 bytes).
pub const DRIPSTONE_MMAP_MAGIC_V1: &[u8; 8] = b"DRIPST01";
/// Magic: `DRIPST02` (8 bytes).
pub const DRIPSTONE_MMAP_MAGIC_V2: &[u8; 8] = b"DRIPST02";

/// Format version stored in the v1 header.
pub const DRIPSTONE_MMAP_FORMAT_VERSION_V1: u32 = 1;
/// Format version stored in the v2 header.
pub const DRIPSTONE_MMAP_FORMAT_VERSION_V2: u32 = 2;

/// Byte length of the v1 mmap file header on disk.
pub const MMAP_HEADER_V1_LEN: usize = 104;
/// Byte length of the v2 mmap file header on disk.
pub const MMAP_HEADER_V2_LEN: usize = 112;
/// Default section alignment for v1 mmap sections.
pub const MMAP_V1_DEFAULT_ALIGNMENT: u32 = 64;

/// Supported encoded vector layouts for mmap v1.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[rkyv(derive(Debug))]
pub enum VectorEncoding {
    F32 = 1,
    F16 = 2,
    Int8 = 3,
    ProductQuantized = 4,
}

impl VectorEncoding {
    #[must_use]
    pub const fn encoded_bytes_per_dimension(self) -> Option<usize> {
        match self {
            Self::F32 => Some(std::mem::size_of::<f32>()),
            Self::F16 => Some(std::mem::size_of::<u16>()),
            Self::Int8 => Some(std::mem::size_of::<i8>()),
            Self::ProductQuantized => None,
        }
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn encoded_vector_bytes(self, dimension: usize) -> Result<usize, DripstoneError> {
        let Some(bytes_per_dimension) = self.encoded_bytes_per_dimension() else {
            return Err(DripstoneError::storage(format!(
                "mmap v1 encoding {self:?} is reserved but not implemented"
            )));
        };
        dimension
            .checked_mul(bytes_per_dimension)
            .ok_or_else(|| DripstoneError::storage("mmap v1 vector byte size overflow".to_string()))
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn from_u32(raw: u32) -> Result<Self, DripstoneError> {
        match raw {
            1 => Ok(Self::F32),
            2 => Ok(Self::F16),
            3 => Ok(Self::Int8),
            4 => Ok(Self::ProductQuantized),
            _ => Err(DripstoneError::storage(format!(
                "unsupported mmap v1 vector encoding: {raw}"
            ))),
        }
    }
}

#[must_use]
pub const fn default_mmap_vector_encoding() -> VectorEncoding {
    #[cfg(target_arch = "aarch64")]
    {
        VectorEncoding::F32
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        VectorEncoding::F16
    }
}

/// Parsed, validated header for a v1 mmap file.
#[derive(Debug, Clone)]
pub struct ParsedMmapHeaderV1 {
    pub version: u32,
    pub dimension: u32,
    pub chunk_count: u64,
    pub vector_encoding: VectorEncoding,
    pub section_alignment: u32,
    pub vector_stride_bytes: usize,
    pub index_offset: usize,
    pub index_len: usize,
    pub row_meta_offset: usize,
    pub row_meta_len: usize,
    pub norms_offset: usize,
    pub norms_len: usize,
    pub vector_offset: usize,
    pub vector_len: usize,
}

/// Parsed, validated header for a v2 mmap file.
#[derive(Debug, Clone)]
pub struct ParsedMmapHeaderV2 {
    pub version: u32,
    pub dimension: u32,
    pub chunk_count: u64,
    pub vector_encoding: VectorEncoding,
    pub section_alignment: u32,
    pub vector_stride_bytes: usize,
    pub index_offset: usize,
    pub index_len: usize,
    pub row_meta_offset: usize,
    pub row_meta_len: usize,
    pub norms_offset: usize,
    pub norms_len: usize,
    pub vector_offset: usize,
    pub vector_len: usize,
    pub quant_scale: f32,
    pub quant_zero_point: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct MmapHeaderV1Spec {
    pub dimension: u32,
    pub chunk_count: u64,
    pub vector_encoding: VectorEncoding,
    pub section_alignment: u32,
    pub vector_stride_bytes: u64,
    pub index_offset: u64,
    pub index_len: u64,
    pub row_meta_offset: u64,
    pub row_meta_len: u64,
    pub norms_offset: u64,
    pub norms_len: u64,
    pub vector_offset: u64,
    pub vector_len: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct MmapHeaderV2Spec {
    pub dimension: u32,
    pub chunk_count: u64,
    pub vector_encoding: VectorEncoding,
    pub section_alignment: u32,
    pub vector_stride_bytes: u64,
    pub index_offset: u64,
    pub index_len: u64,
    pub row_meta_offset: u64,
    pub row_meta_len: u64,
    pub norms_offset: u64,
    pub norms_len: u64,
    pub vector_offset: u64,
    pub vector_len: u64,
    pub quant_scale: f32,
    pub quant_zero_point: i32,
}

/// Id -> row index in [`MmapRowMetaTableV1::rows`], sorted by `id` for binary search.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub struct MmapIdIndexV1 {
    pub entries: Vec<(String, u32)>,
}

/// Cold row metadata for mmap v1.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub struct MmapRowMetaV1 {
    pub id: String,
    pub doc_id: String,
    pub content: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub metadata_json: String,
    pub namespace: Option<String>,
    pub vector_row: u32,
}

/// All row metadata in row order for mmap v1.
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub struct MmapRowMetaTableV1 {
    pub rows: Vec<MmapRowMetaV1>,
}

fn read_u64_le(slice: &[u8], off: usize) -> u64 {
    // SAFETY invariant: callers validate that `slice` is large enough before
    // calling (e.g. the header-length check in `parse_mmap_header_*`).
    // We still avoid `expect` by using `get` + fallback to defend against
    // bugs in offset arithmetic; returning 0 on an out-of-bounds read is
    // detectable via the subsequent validation checks on the parsed values.
    slice
        .get(off..off + 8)
        .and_then(|b| <[u8; 8]>::try_from(b).ok())
        .map_or(0, u64::from_le_bytes)
}

fn usize_from_u64(value: u64, field: &str) -> Result<usize, DripstoneError> {
    usize::try_from(value)
        .map_err(|_| DripstoneError::storage(format!("mmap {field} exceeds addressable size")))
}

fn read_u32_le(slice: &[u8], off: usize) -> u32 {
    slice
        .get(off..off + 4)
        .and_then(|b| <[u8; 4]>::try_from(b).ok())
        .map_or(0, u32::from_le_bytes)
}

/// Returns `true` if `mmap` begins with the v1 magic.
pub fn is_rkyv_mmap_v1(mmap: &[u8]) -> bool {
    mmap.len() >= DRIPSTONE_MMAP_MAGIC_V1.len()
        && &mmap[..DRIPSTONE_MMAP_MAGIC_V1.len()] == DRIPSTONE_MMAP_MAGIC_V1.as_slice()
}

/// Returns `true` if `mmap` begins with the v2 magic.
pub fn is_rkyv_mmap_v2(mmap: &[u8]) -> bool {
    mmap.len() >= DRIPSTONE_MMAP_MAGIC_V2.len()
        && &mmap[..DRIPSTONE_MMAP_MAGIC_V2.len()] == DRIPSTONE_MMAP_MAGIC_V2.as_slice()
}

/// Parse and validate the v1 header; does not validate rkyv sections.
pub fn parse_mmap_header_v1(mmap: &[u8]) -> Result<ParsedMmapHeaderV1, DripstoneError> {
    if mmap.len() < MMAP_HEADER_V1_LEN {
        return Err(DripstoneError::storage(format!(
            "mmap file too short for v1 header: {} bytes",
            mmap.len()
        )));
    }
    if !is_rkyv_mmap_v1(mmap) {
        return Err(DripstoneError::storage(
            "mmap file missing DRIPST01 magic".to_string(),
        ));
    }

    let version = read_u32_le(mmap, 8);
    if version != DRIPSTONE_MMAP_FORMAT_VERSION_V1 {
        return Err(DripstoneError::storage(format!(
            "unsupported dripstone mmap format version: {version}"
        )));
    }

    let dimension = read_u32_le(mmap, 12);
    let chunk_count = read_u64_le(mmap, 16);
    let vector_encoding = VectorEncoding::from_u32(read_u32_le(mmap, 24))?;
    let section_alignment = read_u32_le(mmap, 28);
    let vector_stride_bytes = usize_from_u64(read_u64_le(mmap, 32), "v2 vector stride")?;
    let index_offset = usize_from_u64(read_u64_le(mmap, 40), "v2 index offset")?;
    let index_len = usize_from_u64(read_u64_le(mmap, 48), "v2 index length")?;
    let row_meta_offset = usize_from_u64(read_u64_le(mmap, 56), "v2 row metadata offset")?;
    let row_meta_len = usize_from_u64(read_u64_le(mmap, 64), "v2 row metadata length")?;
    let norms_offset = usize_from_u64(read_u64_le(mmap, 72), "v2 norms offset")?;
    let norms_len = usize_from_u64(read_u64_le(mmap, 80), "v2 norms length")?;
    let vector_offset = usize_from_u64(read_u64_le(mmap, 88), "v2 vector offset")?;
    let vector_len = usize_from_u64(read_u64_le(mmap, 96), "v2 vector length")?;

    let header = ParsedMmapHeaderV1 {
        version,
        dimension,
        chunk_count,
        vector_encoding,
        section_alignment,
        vector_stride_bytes,
        index_offset,
        index_len,
        row_meta_offset,
        row_meta_len,
        norms_offset,
        norms_len,
        vector_offset,
        vector_len,
    };
    validate_v1_header(mmap, &header)?;
    Ok(header)
}

fn validate_v1_header(mmap: &[u8], header: &ParsedMmapHeaderV1) -> Result<(), DripstoneError> {
    if header.dimension == 0 {
        return Err(DripstoneError::storage(
            "mmap v1 dimension must be greater than zero".to_string(),
        ));
    }
    if header.section_alignment == 0 || !header.section_alignment.is_power_of_two() {
        return Err(DripstoneError::storage(format!(
            "mmap v1 section alignment must be a non-zero power of two, got {}",
            header.section_alignment
        )));
    }
    validate_v1_section_ranges(mmap, header)?;
    validate_v1_alignment(header)?;
    validate_v1_payload_lengths(header)
}

fn validate_v1_section_ranges(
    mmap: &[u8],
    header: &ParsedMmapHeaderV1,
) -> Result<(), DripstoneError> {
    let end_index = header
        .index_offset
        .checked_add(header.index_len)
        .ok_or_else(|| DripstoneError::storage("mmap v1 index range overflow".to_string()))?;
    let end_row_meta = header
        .row_meta_offset
        .checked_add(header.row_meta_len)
        .ok_or_else(|| {
            DripstoneError::storage("mmap v1 row metadata range overflow".to_string())
        })?;
    let end_norms = header
        .norms_offset
        .checked_add(header.norms_len)
        .ok_or_else(|| DripstoneError::storage("mmap v1 norms range overflow".to_string()))?;
    let end_vector = header
        .vector_offset
        .checked_add(header.vector_len)
        .ok_or_else(|| DripstoneError::storage("mmap v1 vector range overflow".to_string()))?;
    if end_index > mmap.len()
        || end_row_meta > mmap.len()
        || end_norms > mmap.len()
        || end_vector > mmap.len()
    {
        return Err(DripstoneError::storage(format!(
            "mmap v1 sections out of bounds (file len {})",
            mmap.len()
        )));
    }
    Ok(())
}

fn validate_v1_alignment(header: &ParsedMmapHeaderV1) -> Result<(), DripstoneError> {
    let alignment = usize::try_from(header.section_alignment)
        .map_err(|_| DripstoneError::storage("mmap v1 alignment conversion failed".to_string()))?;
    if !header.index_offset.is_multiple_of(alignment)
        || !header.row_meta_offset.is_multiple_of(alignment)
        || !header.norms_offset.is_multiple_of(alignment)
        || !header.vector_offset.is_multiple_of(alignment)
    {
        return Err(DripstoneError::storage(
            "mmap v1 section offsets must satisfy the declared alignment".to_string(),
        ));
    }
    if header.vector_stride_bytes == 0 || !header.vector_stride_bytes.is_multiple_of(alignment) {
        return Err(DripstoneError::storage(
            "mmap v1 vector stride must be aligned and non-zero".to_string(),
        ));
    }
    Ok(())
}

fn validate_v1_payload_lengths(header: &ParsedMmapHeaderV1) -> Result<(), DripstoneError> {
    let dimension_usize = usize::try_from(header.dimension)
        .map_err(|_| DripstoneError::storage("mmap v1 dimension exceeds addressable size"))?;
    let logical_vector_bytes = header
        .vector_encoding
        .encoded_vector_bytes(dimension_usize)?;
    let f32_size = u64::try_from(std::mem::size_of::<f32>())
        .map_err(|_| DripstoneError::storage("mmap v1 f32 size conversion failed"))?;
    if header.vector_stride_bytes < logical_vector_bytes {
        return Err(DripstoneError::storage(
            "mmap v1 vector stride is smaller than the logical vector size".to_string(),
        ));
    }
    let expected_norms_len = header
        .chunk_count
        .checked_mul(f32_size)
        .ok_or_else(|| DripstoneError::storage("mmap v1 norms length overflow".to_string()))?;
    if usize_from_u64(expected_norms_len, "v2 expected norms length")? != header.norms_len {
        return Err(DripstoneError::storage(format!(
            "mmap v1 norms length mismatch: expected {expected_norms_len}, got {}",
            header.norms_len
        )));
    }
    let chunk_count_usize = usize_from_u64(header.chunk_count, "v2 chunk count")?;
    let expected_vector_len = header
        .vector_stride_bytes
        .checked_mul(chunk_count_usize)
        .ok_or_else(|| {
            DripstoneError::storage("mmap v1 vector slab length overflow".to_string())
        })?;
    if expected_vector_len != header.vector_len {
        return Err(DripstoneError::storage(format!(
            "mmap v1 vector slab length mismatch: expected {expected_vector_len}, got {}",
            header.vector_len
        )));
    }
    Ok(())
}

/// Parse and validate the v2 header; does not validate rkyv sections.
pub fn parse_mmap_header_v2(mmap: &[u8]) -> Result<ParsedMmapHeaderV2, DripstoneError> {
    if mmap.len() < MMAP_HEADER_V2_LEN {
        return Err(DripstoneError::storage(format!(
            "mmap file too short for v2 header: {} bytes",
            mmap.len()
        )));
    }
    if !is_rkyv_mmap_v2(mmap) {
        return Err(DripstoneError::storage(
            "mmap file missing DRIPST02 magic".to_string(),
        ));
    }

    let version = read_u32_le(mmap, 8);
    if version != DRIPSTONE_MMAP_FORMAT_VERSION_V2 {
        return Err(DripstoneError::storage(format!(
            "unsupported dripstone mmap format version: {version}"
        )));
    }

    let dimension = read_u32_le(mmap, 12);
    let chunk_count = read_u64_le(mmap, 16);
    let vector_encoding = VectorEncoding::from_u32(read_u32_le(mmap, 24))?;
    if !matches!(vector_encoding, VectorEncoding::Int8) {
        return Err(DripstoneError::storage(format!(
            "mmap v2 only supports Int8 vectors, got {vector_encoding:?}"
        )));
    }
    let section_alignment = read_u32_le(mmap, 28);
    let vector_stride_bytes = usize_from_u64(read_u64_le(mmap, 32), "v3 vector stride")?;
    let index_offset = usize_from_u64(read_u64_le(mmap, 40), "v3 index offset")?;
    let index_len = usize_from_u64(read_u64_le(mmap, 48), "v3 index length")?;
    let row_meta_offset = usize_from_u64(read_u64_le(mmap, 56), "v3 row metadata offset")?;
    let row_meta_len = usize_from_u64(read_u64_le(mmap, 64), "v3 row metadata length")?;
    let norms_offset = usize_from_u64(read_u64_le(mmap, 72), "v3 norms offset")?;
    let norms_len = usize_from_u64(read_u64_le(mmap, 80), "v3 norms length")?;
    let vector_offset = usize_from_u64(read_u64_le(mmap, 88), "v3 vector offset")?;
    let vector_len = usize_from_u64(read_u64_le(mmap, 96), "v3 vector length")?;
    let quant_scale = f32::from_bits(read_u32_le(mmap, 104));
    let quant_zero_point = i32::from_le_bytes(
        mmap[108..112]
            .try_into()
            .map_err(|_| DripstoneError::storage("mmap v2 quantization header length mismatch"))?,
    );

    let parsed_v2 = parse_common_aligned_header(
        mmap,
        "v2",
        ParsedMmapHeaderV1 {
            version,
            dimension,
            chunk_count,
            vector_encoding,
            section_alignment,
            vector_stride_bytes,
            index_offset,
            index_len,
            row_meta_offset,
            row_meta_len,
            norms_offset,
            norms_len,
            vector_offset,
            vector_len,
        },
    )?;

    if !quant_scale.is_finite() || quant_scale <= 0.0 {
        return Err(DripstoneError::storage(format!(
            "mmap v2 quantization scale must be positive and finite, got {quant_scale}"
        )));
    }
    if !(i32::from(i8::MIN)..=i32::from(i8::MAX)).contains(&quant_zero_point) {
        return Err(DripstoneError::storage(format!(
            "mmap v2 quantization zero point must fit in i8, got {quant_zero_point}"
        )));
    }

    Ok(ParsedMmapHeaderV2 {
        version,
        dimension: parsed_v2.dimension,
        chunk_count: parsed_v2.chunk_count,
        vector_encoding: parsed_v2.vector_encoding,
        section_alignment: parsed_v2.section_alignment,
        vector_stride_bytes: parsed_v2.vector_stride_bytes,
        index_offset: parsed_v2.index_offset,
        index_len: parsed_v2.index_len,
        row_meta_offset: parsed_v2.row_meta_offset,
        row_meta_len: parsed_v2.row_meta_len,
        norms_offset: parsed_v2.norms_offset,
        norms_len: parsed_v2.norms_len,
        vector_offset: parsed_v2.vector_offset,
        vector_len: parsed_v2.vector_len,
        quant_scale,
        quant_zero_point,
    })
}

pub fn build_header_bytes_v1(spec: MmapHeaderV1Spec) -> [u8; MMAP_HEADER_V1_LEN] {
    let mut out = [0u8; MMAP_HEADER_V1_LEN];
    out[0..8].copy_from_slice(DRIPSTONE_MMAP_MAGIC_V1.as_slice());
    out[8..12].copy_from_slice(&DRIPSTONE_MMAP_FORMAT_VERSION_V1.to_le_bytes());
    out[12..16].copy_from_slice(&spec.dimension.to_le_bytes());
    out[16..24].copy_from_slice(&spec.chunk_count.to_le_bytes());
    out[24..28].copy_from_slice(&(spec.vector_encoding as u32).to_le_bytes());
    out[28..32].copy_from_slice(&spec.section_alignment.to_le_bytes());
    out[32..40].copy_from_slice(&spec.vector_stride_bytes.to_le_bytes());
    out[40..48].copy_from_slice(&spec.index_offset.to_le_bytes());
    out[48..56].copy_from_slice(&spec.index_len.to_le_bytes());
    out[56..64].copy_from_slice(&spec.row_meta_offset.to_le_bytes());
    out[64..72].copy_from_slice(&spec.row_meta_len.to_le_bytes());
    out[72..80].copy_from_slice(&spec.norms_offset.to_le_bytes());
    out[80..88].copy_from_slice(&spec.norms_len.to_le_bytes());
    out[88..96].copy_from_slice(&spec.vector_offset.to_le_bytes());
    out[96..104].copy_from_slice(&spec.vector_len.to_le_bytes());
    out
}

pub fn build_header_bytes_v2(spec: MmapHeaderV2Spec) -> [u8; MMAP_HEADER_V2_LEN] {
    let mut out = [0u8; MMAP_HEADER_V2_LEN];
    out[0..8].copy_from_slice(DRIPSTONE_MMAP_MAGIC_V2.as_slice());
    out[8..12].copy_from_slice(&DRIPSTONE_MMAP_FORMAT_VERSION_V2.to_le_bytes());
    out[12..16].copy_from_slice(&spec.dimension.to_le_bytes());
    out[16..24].copy_from_slice(&spec.chunk_count.to_le_bytes());
    out[24..28].copy_from_slice(&(spec.vector_encoding as u32).to_le_bytes());
    out[28..32].copy_from_slice(&spec.section_alignment.to_le_bytes());
    out[32..40].copy_from_slice(&spec.vector_stride_bytes.to_le_bytes());
    out[40..48].copy_from_slice(&spec.index_offset.to_le_bytes());
    out[48..56].copy_from_slice(&spec.index_len.to_le_bytes());
    out[56..64].copy_from_slice(&spec.row_meta_offset.to_le_bytes());
    out[64..72].copy_from_slice(&spec.row_meta_len.to_le_bytes());
    out[72..80].copy_from_slice(&spec.norms_offset.to_le_bytes());
    out[80..88].copy_from_slice(&spec.norms_len.to_le_bytes());
    out[88..96].copy_from_slice(&spec.vector_offset.to_le_bytes());
    out[96..104].copy_from_slice(&spec.vector_len.to_le_bytes());
    out[104..108].copy_from_slice(&spec.quant_scale.to_bits().to_le_bytes());
    out[108..112].copy_from_slice(&spec.quant_zero_point.to_le_bytes());
    out
}

fn parse_common_aligned_header(
    mmap: &[u8],
    version_label: &str,
    header: ParsedMmapHeaderV1,
) -> Result<ParsedMmapHeaderV1, DripstoneError> {
    if header.dimension == 0 {
        return Err(DripstoneError::storage(format!(
            "mmap {version_label} dimension must be greater than zero"
        )));
    }
    if header.section_alignment == 0 || !header.section_alignment.is_power_of_two() {
        return Err(DripstoneError::storage(format!(
            "mmap {version_label} section alignment must be a non-zero power of two, got {}",
            header.section_alignment
        )));
    }

    validate_aligned_section_bounds(mmap, version_label, &header)?;
    validate_common_alignment(version_label, &header, header.section_alignment)?;
    validate_common_payload_lengths(version_label, &header)?;
    Ok(header)
}

fn validate_common_alignment(
    version_label: &str,
    header: &ParsedMmapHeaderV1,
    section_alignment: u32,
) -> Result<(), DripstoneError> {
    let alignment = usize::try_from(section_alignment).map_err(|_| {
        DripstoneError::storage(format!("mmap {version_label} alignment conversion failed"))
    })?;
    if !header.index_offset.is_multiple_of(alignment)
        || !header.row_meta_offset.is_multiple_of(alignment)
        || !header.norms_offset.is_multiple_of(alignment)
        || !header.vector_offset.is_multiple_of(alignment)
    {
        return Err(DripstoneError::storage(format!(
            "mmap {version_label} section offsets must satisfy the declared alignment"
        )));
    }
    if header.vector_stride_bytes == 0 || !header.vector_stride_bytes.is_multiple_of(alignment) {
        return Err(DripstoneError::storage(format!(
            "mmap {version_label} vector stride must be aligned and non-zero"
        )));
    }
    Ok(())
}

fn validate_common_payload_lengths(
    version_label: &str,
    header: &ParsedMmapHeaderV1,
) -> Result<(), DripstoneError> {
    let dimension_usize = usize::try_from(header.dimension).map_err(|_| {
        DripstoneError::storage(format!(
            "mmap {version_label} dimension exceeds addressable size"
        ))
    })?;
    let logical_vector_bytes = header
        .vector_encoding
        .encoded_vector_bytes(dimension_usize)?;
    if header.vector_stride_bytes < logical_vector_bytes {
        return Err(DripstoneError::storage(format!(
            "mmap {version_label} vector stride is smaller than the logical vector size"
        )));
    }
    let f32_size = u64::try_from(std::mem::size_of::<f32>()).map_err(|_| {
        DripstoneError::storage(format!("mmap {version_label} f32 size conversion failed"))
    })?;
    let expected_norms_len = header.chunk_count.checked_mul(f32_size).ok_or_else(|| {
        DripstoneError::storage(format!("mmap {version_label} norms length overflow"))
    })?;
    if usize_from_u64(
        expected_norms_len,
        &format!("{version_label} expected norms length"),
    )? != header.norms_len
    {
        return Err(DripstoneError::storage(format!(
            "mmap {version_label} norms length mismatch: expected {expected_norms_len}, got {}",
            header.norms_len
        )));
    }
    let chunk_count_usize =
        usize_from_u64(header.chunk_count, &format!("{version_label} chunk count"))?;
    let expected_vector_len = header
        .vector_stride_bytes
        .checked_mul(chunk_count_usize)
        .ok_or_else(|| {
            DripstoneError::storage(format!("mmap {version_label} vector slab length overflow"))
        })?;
    if expected_vector_len != header.vector_len {
        return Err(DripstoneError::storage(format!(
            "mmap {version_label} vector slab length mismatch: expected {expected_vector_len}, got {}",
            header.vector_len
        )));
    }
    Ok(())
}

fn validate_aligned_section_bounds(
    mmap: &[u8],
    version_label: &str,
    header: &ParsedMmapHeaderV1,
) -> Result<(), DripstoneError> {
    let ranges = [
        ("index", header.index_offset, header.index_len),
        ("row metadata", header.row_meta_offset, header.row_meta_len),
        ("norms", header.norms_offset, header.norms_len),
        ("vector", header.vector_offset, header.vector_len),
    ];
    for (name, offset, len) in ranges {
        let end = offset.checked_add(len).ok_or_else(|| {
            DripstoneError::storage(format!("mmap {version_label} {name} range overflow"))
        })?;
        if end > mmap.len() {
            return Err(DripstoneError::storage(format!(
                "mmap {version_label} {name} section out of bounds (file len {})",
                mmap.len()
            )));
        }
    }
    Ok(())
}

pub fn align_offset(offset: usize, alignment: usize) -> Result<usize, DripstoneError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(DripstoneError::storage(format!(
            "alignment must be a non-zero power of two, got {alignment}"
        )));
    }
    let remainder = offset % alignment;
    if remainder == 0 {
        return Ok(offset);
    }
    offset
        .checked_add(alignment - remainder)
        .ok_or_else(|| DripstoneError::storage("aligned offset overflow".to_string()))
}
