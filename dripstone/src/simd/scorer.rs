//! Query-scoped dense scoring helpers.

use crate::errors::DripstoneError;
use crate::simd::dispatch::{
    SimdDispatch, chebyshev_distance_with_dispatch, cosine_with_query_norm_dispatch,
    detect_dispatch, dot_product_with_dispatch, dot_products_batch_with_dispatch,
    ensure_batch_candidate_lengths, ensure_same_length, euclidean_distance_squared_with_dispatch,
    euclidean_distance_with_dispatch, euclidean_distances_batch_with_dispatch,
    magnitude_with_dispatch, manhattan_distance_with_dispatch, squared_sum_with_dispatch,
};
use half::f16;
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use crate::simd::avx2;
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use crate::simd::avx512;
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
use crate::simd::neon;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseMetric {
    Cosine,
    Dot,
    Euclidean,
    Manhattan,
    Chebyshev,
}

/// Query-scoped scorer that precomputes cosine norms and runtime dispatch once.
#[derive(Debug, Clone, Copy)]
pub struct PreparedQueryScorer<'a> {
    query: &'a [f32],
    metric: DenseMetric,
    dispatch: SimdDispatch,
    query_norm: Option<f32>,
}

impl<'a> PreparedQueryScorer<'a> {
    /// Creates a prepared scorer for a query vector.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn new(query: &'a [f32], metric: DenseMetric) -> Result<Self, DripstoneError> {
        let dispatch = detect_dispatch();
        let query_norm = if metric == DenseMetric::Cosine {
            Some(magnitude_with_dispatch(query, dispatch).unwrap_or(f32::INFINITY))
        } else {
            None
        };

        Ok(Self {
            query,
            metric,
            dispatch,
            query_norm,
        })
    }

    #[must_use]
    pub const fn metric(&self) -> DenseMetric {
        self.metric
    }

    /// Scores a borrowed dense slice.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn score_dense(&self, candidate: &[f32]) -> Result<f32, DripstoneError> {
        ensure_same_length(self.query, candidate)?;
        self.score_dense_unchecked(candidate)
    }

    /// Scores multiple borrowed dense slices with one prepared query.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn score_dense_batch(&self, candidates: &[&[f32]]) -> Result<Vec<f32>, DripstoneError> {
        ensure_batch_candidate_lengths(self.query, candidates)?;

        match self.metric {
            DenseMetric::Dot => {
                dot_products_batch_with_dispatch(self.query, candidates, self.dispatch)
            }
            DenseMetric::Euclidean => {
                euclidean_distances_batch_with_dispatch(self.query, candidates, self.dispatch)
            }
            DenseMetric::Cosine => {
                let query_norm = self.query_norm.ok_or_else(|| {
                    DripstoneError::storage("missing prepared query norm for cosine scoring")
                })?;
                let mut scores = Vec::with_capacity(candidates.len());
                for candidate in candidates {
                    scores.push(cosine_with_query_norm_dispatch(
                        self.query,
                        query_norm,
                        candidate,
                        self.dispatch,
                    )?);
                }
                Ok(scores)
            }
            DenseMetric::Manhattan => {
                let mut scores = Vec::with_capacity(candidates.len());
                for candidate in candidates {
                    scores.push(manhattan_distance_with_dispatch(
                        self.query,
                        candidate,
                        self.dispatch,
                    )?);
                }
                Ok(scores)
            }
            DenseMetric::Chebyshev => {
                let mut scores = Vec::with_capacity(candidates.len());
                for candidate in candidates {
                    scores.push(chebyshev_distance_with_dispatch(
                        self.query,
                        candidate,
                        self.dispatch,
                    )?);
                }
                Ok(scores)
            }
        }
    }

    /// Scores a little-endian `f32` byte slice without first materializing a `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn score_f32_le_bytes(&self, candidate_bytes: &[u8]) -> Result<f32, DripstoneError> {
        self.score_f32_le_bytes_with_candidate_norm(candidate_bytes, None)
    }

    /// Scores a little-endian `f32` byte slice with an optional precomputed candidate norm.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn score_f32_le_bytes_with_candidate_norm(
        &self,
        candidate_bytes: &[u8],
        candidate_norm: Option<f32>,
    ) -> Result<f32, DripstoneError> {
        let expected_len = self
            .query
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| DripstoneError::storage("candidate byte length overflow"))?;
        if candidate_bytes.len() != expected_len {
            return Err(DripstoneError::InvalidVectorLength {
                expected: expected_len,
                actual: candidate_bytes.len(),
            });
        }

        #[cfg(target_endian = "little")]
        {
            self.score_f32_le_little_endian(candidate_bytes, candidate_norm)
        }

        #[cfg(not(target_endian = "little"))]
        {
            self.score_f32_le_swapped_endian(candidate_bytes, candidate_norm)
        }
    }

    fn cosine_from_dot_and_norm(
        &self,
        dot: f32,
        candidate_norm: f32,
    ) -> Result<f32, DripstoneError> {
        let query_norm = self.query_norm.ok_or_else(|| {
            DripstoneError::storage("missing prepared query norm for cosine scoring")
        })?;
        Ok(super::cosine_or_zero(dot, query_norm, candidate_norm))
    }

    #[cfg(target_endian = "little")]
    fn score_f32_le_little_endian(
        &self,
        candidate_bytes: &[u8],
        candidate_norm: Option<f32>,
    ) -> Result<f32, DripstoneError> {
        let (prefix, candidate, suffix) = unsafe {
            // SAFETY: `align_to` only reinterprets the aligned middle
            // portion. Prefix/suffix checks below reject any unaligned or
            // partial representation before the typed slice is used.
            candidate_bytes.align_to::<f32>()
        };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(DripstoneError::storage(
                "unaligned mmap v1 f32 candidate bytes",
            ));
        }
        if self.metric == DenseMetric::Cosine
            && let Some(candidate_norm) = candidate_norm
        {
            let dot = dot_product_with_dispatch(self.query, candidate, self.dispatch)?;
            return self.cosine_from_dot_and_norm(dot, candidate_norm);
        }
        self.score_dense_unchecked(candidate)
    }

    #[cfg(not(target_endian = "little"))]
    fn score_f32_le_swapped_endian(
        &self,
        candidate_bytes: &[u8],
        candidate_norm: Option<f32>,
    ) -> Result<f32, DripstoneError> {
        let mut scratch = Vec::with_capacity(self.query.len());
        for chunk in candidate_bytes.chunks_exact(std::mem::size_of::<f32>()) {
            let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            scratch.push(f32::from_bits(bits));
        }
        if self.metric == DenseMetric::Cosine
            && let Some(candidate_norm) = candidate_norm
        {
            let dot = dot_product_with_dispatch(self.query, scratch.as_slice(), self.dispatch)?;
            return self.cosine_from_dot_and_norm(dot, candidate_norm);
        }
        self.score_dense(scratch.as_slice())
    }

    /// Scores a little-endian `f16` byte slice without building a full candidate vector.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn score_f16_le_bytes(&self, candidate_bytes: &[u8]) -> Result<f32, DripstoneError> {
        self.score_f16_le_bytes_with_candidate_norm(candidate_bytes, None)
    }

    /// Scores a little-endian `f16` byte slice with an optional precomputed candidate norm.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn score_f16_le_bytes_with_candidate_norm(
        &self,
        candidate_bytes: &[u8],
        candidate_norm: Option<f32>,
    ) -> Result<f32, DripstoneError> {
        let expected_len = self
            .query
            .len()
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| DripstoneError::storage("candidate byte length overflow"))?;
        if candidate_bytes.len() != expected_len {
            return Err(DripstoneError::InvalidVectorLength {
                expected: expected_len,
                actual: candidate_bytes.len(),
            });
        }

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        if matches!(self.dispatch, SimdDispatch::Avx512)
            && std::is_x86_feature_detected!("avx512fp16")
        {
            return unsafe {
                avx512::score_f16_bytes_avx512fp16(
                    self.query,
                    candidate_bytes,
                    self.metric,
                    self.query_norm,
                    candidate_norm,
                )
            };
        }

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        if matches!(self.dispatch, SimdDispatch::Avx512)
            && std::is_x86_feature_detected!("f16c")
            && std::is_x86_feature_detected!("fma")
        {
            return unsafe {
                avx512::score_f16_bytes_avx512_f16c(
                    self.query,
                    candidate_bytes,
                    self.metric,
                    self.query_norm,
                    candidate_norm,
                )
            };
        }

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        if matches!(self.dispatch, SimdDispatch::Avx2 | SimdDispatch::Avx512)
            && std::is_x86_feature_detected!("f16c")
        {
            return unsafe {
                avx2::score_f16_bytes_avx_f16c(
                    self.query,
                    candidate_bytes,
                    self.metric,
                    self.query_norm,
                    candidate_norm,
                )
            };
        }

        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        if matches!(self.dispatch, SimdDispatch::Neon) {
            return self.score_f16_le_bytes_blocked(candidate_bytes, candidate_norm);
        }

        self.score_f16_le_bytes_scalar(candidate_bytes, candidate_norm)
    }

    fn score_f16_le_bytes_scalar(
        &self,
        candidate_bytes: &[u8],
        candidate_norm: Option<f32>,
    ) -> Result<f32, DripstoneError> {
        let mut dot = 0.0_f32;
        let mut candidate_squared_sum = candidate_norm.map_or(0.0_f32, |norm| norm * norm);
        let mut euclidean_sum = 0.0_f32;
        let mut manhattan_sum = 0.0_f32;
        let mut chebyshev_max = 0.0_f32;

        for i in 0..self.query.len() {
            let query_value = self.query[i];
            let byte_offset = i * std::mem::size_of::<u16>();
            let encoded = u16::from_le_bytes([
                candidate_bytes[byte_offset],
                candidate_bytes[byte_offset + 1],
            ]);
            let candidate_value = f16::from_bits(encoded).to_f32();
            match self.metric {
                DenseMetric::Cosine => {
                    dot += query_value * candidate_value;
                    if candidate_norm.is_none() {
                        candidate_squared_sum += candidate_value * candidate_value;
                    }
                }
                DenseMetric::Dot => {
                    dot += query_value * candidate_value;
                }
                DenseMetric::Euclidean => {
                    let delta = query_value - candidate_value;
                    euclidean_sum += delta * delta;
                }
                DenseMetric::Manhattan => {
                    manhattan_sum += (query_value - candidate_value).abs();
                }
                DenseMetric::Chebyshev => {
                    chebyshev_max = chebyshev_max.max((query_value - candidate_value).abs());
                }
            }
        }

        match self.metric {
            DenseMetric::Cosine => {
                let query_norm = self.query_norm.ok_or_else(|| {
                    DripstoneError::storage("missing prepared query norm for cosine scoring")
                })?;
                let candidate_norm = candidate_squared_sum.sqrt();
                Ok(super::cosine_or_zero(dot, query_norm, candidate_norm))
            }
            DenseMetric::Dot => Ok(dot),
            DenseMetric::Euclidean => Ok(euclidean_sum.sqrt()),
            DenseMetric::Manhattan => Ok(manhattan_sum),
            DenseMetric::Chebyshev => Ok(chebyshev_max),
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    fn accumulate_decoded_block(
        &self,
        query_block: &[f32],
        candidate_block: &[f32],
        candidate_norm: Option<f32>,
        dot: &mut f32,
        candidate_squared_sum: &mut f32,
        euclidean_sum: &mut f32,
        manhattan_sum: &mut f32,
        chebyshev_max: &mut f32,
    ) -> Result<(), DripstoneError> {
        match self.metric {
            DenseMetric::Cosine => {
                *dot += dot_product_with_dispatch(query_block, candidate_block, self.dispatch)?;
                if candidate_norm.is_none() {
                    *candidate_squared_sum +=
                        squared_sum_with_dispatch(candidate_block, self.dispatch)?;
                }
            }
            DenseMetric::Dot => {
                *dot += dot_product_with_dispatch(query_block, candidate_block, self.dispatch)?;
            }
            DenseMetric::Euclidean => {
                *euclidean_sum += euclidean_distance_squared_with_dispatch(
                    query_block,
                    candidate_block,
                    self.dispatch,
                )?;
            }
            DenseMetric::Manhattan => {
                *manhattan_sum +=
                    manhattan_distance_with_dispatch(query_block, candidate_block, self.dispatch)?;
            }
            DenseMetric::Chebyshev => {
                *chebyshev_max = chebyshev_max.max(chebyshev_distance_with_dispatch(
                    query_block,
                    candidate_block,
                    self.dispatch,
                )?);
            }
        }
        Ok(())
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    fn score_f16_le_bytes_blocked(
        &self,
        candidate_bytes: &[u8],
        candidate_norm: Option<f32>,
    ) -> Result<f32, DripstoneError> {
        const BLOCK: usize = 64;

        let mut encoded = [0_u16; BLOCK];
        let mut decoded = [0.0_f32; BLOCK];
        let mut dot = 0.0_f32;
        let mut candidate_squared_sum = candidate_norm.map_or(0.0_f32, |norm| norm * norm);
        let mut euclidean_sum = 0.0_f32;
        let mut manhattan_sum = 0.0_f32;
        let mut chebyshev_max = 0.0_f32;

        for block_start in (0..self.query.len()).step_by(BLOCK) {
            let block_len = (self.query.len() - block_start).min(BLOCK);
            let byte_start = block_start * std::mem::size_of::<u16>();
            let byte_end = byte_start + block_len * std::mem::size_of::<u16>();
            let bytes = &candidate_bytes[byte_start..byte_end];
            for (slot, chunk) in encoded
                .iter_mut()
                .take(block_len)
                .zip(bytes.chunks_exact(std::mem::size_of::<u16>()))
            {
                *slot = u16::from_le_bytes([chunk[0], chunk[1]]);
            }
            let half_values: &[f16] = encoded[..block_len].reinterpret_cast();
            half_values.convert_to_f32_slice(&mut decoded[..block_len]);

            let query_block = &self.query[block_start..block_start + block_len];
            let candidate_block = &decoded[..block_len];
            self.accumulate_decoded_block(
                query_block,
                candidate_block,
                candidate_norm,
                &mut dot,
                &mut candidate_squared_sum,
                &mut euclidean_sum,
                &mut manhattan_sum,
                &mut chebyshev_max,
            )?;
        }

        super::accum::finish_scored(
            self.metric,
            super::accum::ScoreParts {
                dot,
                euclidean: euclidean_sum,
                manhattan: manhattan_sum,
                chebyshev: chebyshev_max,
            },
            self.query_norm,
            Some(candidate_squared_sum.sqrt()),
            "missing candidate norm for cosine scoring",
        )
    }

    /// Scores a little-endian `i8` byte slice using global affine quantization metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn score_i8_le_bytes_with_candidate_norm(
        &self,
        candidate_bytes: &[u8],
        candidate_norm: Option<f32>,
        quant_scale: f32,
        quant_zero_point: i8,
    ) -> Result<f32, DripstoneError> {
        if candidate_bytes.len() != self.query.len() {
            return Err(DripstoneError::InvalidVectorLength {
                expected: self.query.len(),
                actual: candidate_bytes.len(),
            });
        }
        if !quant_scale.is_finite() || quant_scale <= 0.0 {
            return Err(DripstoneError::storage(
                "int8 quantization scale must be positive and finite",
            ));
        }
        let candidate_norm = match candidate_norm {
            Some(norm) => Some(norm),
            None if self.metric == DenseMetric::Cosine => Some(Self::int8_candidate_norm(
                candidate_bytes,
                quant_scale,
                quant_zero_point,
            )),
            None => None,
        };

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        if matches!(self.dispatch, SimdDispatch::Avx2 | SimdDispatch::Avx512)
            && std::is_x86_feature_detected!("avx2")
        {
            return unsafe {
                avx2::score_i8_bytes_avx2(
                    self.query,
                    candidate_bytes,
                    self.metric,
                    self.query_norm,
                    candidate_norm,
                    quant_scale,
                    quant_zero_point,
                )
            };
        }

        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        if matches!(self.dispatch, SimdDispatch::Neon) {
            return unsafe {
                neon::score_i8_bytes_neon(
                    self.query,
                    candidate_bytes,
                    self.metric,
                    self.query_norm,
                    candidate_norm,
                    quant_scale,
                    quant_zero_point,
                )
            };
        }

        self.score_i8_scalar(
            candidate_bytes,
            candidate_norm,
            quant_scale,
            quant_zero_point,
        )
    }

    fn int8_candidate_norm(bytes: &[u8], quant_scale: f32, quant_zero_point: i8) -> f32 {
        let zero_point = f32::from(quant_zero_point);
        let mut squared = 0.0_f32;
        for byte in bytes {
            let value = (f32::from(i8::from_le_bytes([*byte])) - zero_point) * quant_scale;
            squared += value * value;
        }
        squared.sqrt()
    }

    fn score_i8_scalar(
        &self,
        candidate_bytes: &[u8],
        candidate_norm: Option<f32>,
        quant_scale: f32,
        quant_zero_point: i8,
    ) -> Result<f32, DripstoneError> {
        let zero_point = f32::from(quant_zero_point);
        let mut dot = 0.0_f32;
        let mut euclidean_sum = 0.0_f32;
        let mut manhattan_sum = 0.0_f32;
        let mut chebyshev_max = 0.0_f32;

        for (query_value, candidate_byte) in self.query.iter().zip(candidate_bytes.iter()) {
            let candidate =
                (f32::from(i8::from_le_bytes([*candidate_byte])) - zero_point) * quant_scale;
            match self.metric {
                DenseMetric::Cosine | DenseMetric::Dot => {
                    dot += query_value * candidate;
                }
                DenseMetric::Euclidean => {
                    let delta = query_value - candidate;
                    euclidean_sum += delta * delta;
                }
                DenseMetric::Manhattan => {
                    manhattan_sum += (query_value - candidate).abs();
                }
                DenseMetric::Chebyshev => {
                    chebyshev_max = chebyshev_max.max((query_value - candidate).abs());
                }
            }
        }

        match self.metric {
            DenseMetric::Cosine => {
                let query_norm = self.query_norm.ok_or_else(|| {
                    DripstoneError::storage("missing prepared query norm for cosine scoring")
                })?;
                let candidate_norm = candidate_norm.ok_or_else(|| {
                    DripstoneError::storage("missing candidate norm for int8 cosine scoring")
                })?;
                Ok(super::cosine_or_zero(dot, query_norm, candidate_norm))
            }
            DenseMetric::Dot => Ok(dot),
            DenseMetric::Euclidean => Ok(euclidean_sum.sqrt()),
            DenseMetric::Manhattan => Ok(manhattan_sum),
            DenseMetric::Chebyshev => Ok(chebyshev_max),
        }
    }

    /// Scores a candidate populated into a reusable scratch buffer.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn score_mapped_dense<I>(
        &self,
        candidate: I,
        scratch: &mut Vec<f32>,
    ) -> Result<f32, DripstoneError>
    where
        I: IntoIterator<Item = f32>,
    {
        scratch.clear();
        scratch.extend(candidate);
        self.score_dense(scratch.as_slice())
    }

    #[inline]
    fn score_dense_unchecked(&self, candidate: &[f32]) -> Result<f32, DripstoneError> {
        match self.metric {
            DenseMetric::Cosine => {
                let query_norm = self.query_norm.ok_or_else(|| {
                    DripstoneError::storage("missing prepared query norm for cosine scoring")
                })?;
                cosine_with_query_norm_dispatch(self.query, query_norm, candidate, self.dispatch)
            }
            DenseMetric::Dot => dot_product_with_dispatch(self.query, candidate, self.dispatch),
            DenseMetric::Euclidean => {
                euclidean_distance_with_dispatch(self.query, candidate, self.dispatch)
            }
            DenseMetric::Manhattan => {
                manhattan_distance_with_dispatch(self.query, candidate, self.dispatch)
            }
            DenseMetric::Chebyshev => {
                chebyshev_distance_with_dispatch(self.query, candidate, self.dispatch)
            }
        }
    }
}

/// Computes the dot product of two vectors using the best available implementation.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, DripstoneError> {
    dot_product_with_dispatch(a, b, detect_dispatch())
}

/// Computes cosine similarity using the best available implementation.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, DripstoneError> {
    let scorer = PreparedQueryScorer::new(a, DenseMetric::Cosine)?;
    scorer.score_dense(b)
}

/// Computes the sum of squared values using the best available implementation.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn squared_sum(values: &[f32]) -> Result<f32, DripstoneError> {
    squared_sum_with_dispatch(values, detect_dispatch())
}

/// Computes the magnitude (L2 norm) of a vector.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn magnitude(values: &[f32]) -> Result<f32, DripstoneError> {
    magnitude_with_dispatch(values, detect_dispatch())
}

/// Computes Euclidean distance using the best available implementation.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, DripstoneError> {
    euclidean_distance_with_dispatch(a, b, detect_dispatch())
}

/// Computes Manhattan distance (L1 norm) using the best available implementation.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> Result<f32, DripstoneError> {
    manhattan_distance_with_dispatch(a, b, detect_dispatch())
}

/// Computes Chebyshev distance (L∞ norm) using the best available implementation.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn chebyshev_distance(a: &[f32], b: &[f32]) -> Result<f32, DripstoneError> {
    chebyshev_distance_with_dispatch(a, b, detect_dispatch())
}

/// Computes Euclidean distance squared (no sqrt) using the best available implementation.
/// Faster than `euclidean_distance` when only comparing distances.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> Result<f32, DripstoneError> {
    euclidean_distance_squared_with_dispatch(a, b, detect_dispatch())
}
