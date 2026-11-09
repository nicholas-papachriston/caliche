//! Sparse vector types for hybrid search capabilities.
//!
//! Sparse vectors are represented as dimension-value pairs, efficient for
//! text-based retrieval methods like BM25 where most dimensions are zero.

use crate::errors::DripstoneError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A sparse vector represented as a `HashMap` of dimension indices to non-zero values.
///
/// This representation is memory-efficient for high-dimensional spaces where
/// most values are zero, common in text retrieval scenarios.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SparseVector {
    /// Map of dimension index to value for non-zero elements
    pub indices: HashMap<u32, f32>,
    /// Optional total dimensionality (if known)
    pub dimension: Option<usize>,
}

impl SparseVector {
    /// Creates a new empty sparse vector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            indices: HashMap::new(),
            dimension: None,
        }
    }

    /// Creates a sparse vector with a known dimensionality.
    #[must_use]
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            indices: HashMap::new(),
            dimension: Some(dimension),
        }
    }

    /// Creates a sparse vector from a dense vector, storing only non-zero values.
    #[must_use]
    pub fn from_dense(dense: &[f32]) -> Self {
        let mut indices = HashMap::new();
        for (idx, &value) in dense.iter().enumerate() {
            if value.abs() > f32::EPSILON {
                let Ok(index) = u32::try_from(idx) else {
                    break;
                };
                indices.insert(index, value);
            }
        }
        Self {
            indices,
            dimension: Some(dense.len()),
        }
    }

    /// Converts the sparse vector to a dense representation.
    ///
    /// Returns an error if dimensionality is not known.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn to_dense(&self) -> Result<Vec<f32>, DripstoneError> {
        let dimension = self.dimension.ok_or_else(|| {
            DripstoneError::storage("cannot convert sparse vector to dense without known dimension")
        })?;

        let mut dense = vec![0.0; dimension];
        for (&idx, &value) in &self.indices {
            if (idx as usize) < dimension {
                dense[idx as usize] = value;
            }
        }
        Ok(dense)
    }

    /// Sets a value at the given dimension index.
    pub fn set(&mut self, index: u32, value: f32) {
        if value.abs() > f32::EPSILON {
            self.indices.insert(index, value);
        } else {
            self.indices.remove(&index);
        }
    }

    /// Gets the value at the given dimension index (returns 0.0 if not set).
    #[must_use]
    pub fn get(&self, index: u32) -> f32 {
        self.indices.get(&index).copied().unwrap_or(0.0)
    }

    /// Returns the number of non-zero elements.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Computes the dot product with another sparse vector.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        let mut result = 0.0;
        // Iterate over the smaller vector for efficiency
        let (smaller, larger) = if self.indices.len() <= other.indices.len() {
            (&self.indices, &other.indices)
        } else {
            (&other.indices, &self.indices)
        };

        for (&idx, &value) in smaller {
            if let Some(&other_value) = larger.get(&idx) {
                result += value * other_value;
            }
        }
        result
    }

    /// Computes the L2 norm (magnitude) of the sparse vector.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        self.indices.values().map(|&v| v * v).sum::<f32>().sqrt()
    }

    /// Normalizes the sparse vector to unit length (L2 normalization).
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn normalize(&mut self) -> Result<(), DripstoneError> {
        let mag = self.magnitude();
        if !crate::simd::usable_cosine_norm(mag) {
            return Err(DripstoneError::storage(
                "cannot normalize zero-magnitude vector",
            ));
        }
        for value in self.indices.values_mut() {
            *value /= mag;
        }
        Ok(())
    }

    /// Returns a normalized copy of this sparse vector.
    ///
    /// Optimized to avoid unnecessary normalization if the vector is already normalized.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn normalized(&self) -> Result<Self, DripstoneError> {
        // Check if already normalized (magnitude ≈ 1.0) to avoid unnecessary work
        let mag = self.magnitude();
        if (mag - 1.0).abs() < f32::EPSILON {
            // Already normalized, clone is necessary for owned return
            return Ok(self.clone());
        }

        // Need to normalize, clone once and normalize in-place
        let mut copy = self.clone();
        copy.normalize()?;
        Ok(copy)
    }

    /// Computes cosine similarity with another sparse vector.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn cosine_similarity(&self, other: &Self) -> Result<f32, DripstoneError> {
        Ok(crate::simd::cosine_or_zero(
            self.dot(other),
            self.magnitude(),
            other.magnitude(),
        ))
    }
}

impl Default for SparseVector {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<(u32, f32)> for SparseVector {
    fn from_iter<T: IntoIterator<Item = (u32, f32)>>(iter: T) -> Self {
        let indices: HashMap<u32, f32> = iter
            .into_iter()
            .filter(|(_, v)| v.abs() > f32::EPSILON)
            .collect();
        Self {
            indices,
            dimension: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_creation() {
        let sparse = SparseVector::new();
        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.dimension, None);
    }

    #[test]
    fn test_from_dense() {
        let dense = vec![0.0, 1.5, 0.0, 2.3, 0.0];
        let sparse = SparseVector::from_dense(&dense);

        assert_eq!(sparse.nnz(), 2);
        assert!((sparse.get(1) - 1.5).abs() <= f32::EPSILON);
        assert!((sparse.get(3) - 2.3).abs() <= f32::EPSILON);
        assert!(sparse.get(0).abs() <= f32::EPSILON);
        assert_eq!(sparse.dimension, Some(5));
    }

    #[test]
    fn test_to_dense() {
        let mut sparse = SparseVector::with_dimension(5);
        sparse.set(1, 1.5);
        sparse.set(3, 2.3);

        let dense = sparse
            .to_dense()
            .expect("Failed to convert sparse vector to dense");
        assert_eq!(dense, vec![0.0, 1.5, 0.0, 2.3, 0.0]);
    }

    #[test]
    fn test_dot_product() {
        let mut v1 = SparseVector::new();
        v1.set(0, 1.0);
        v1.set(2, 2.0);
        v1.set(4, 3.0);

        let mut v2 = SparseVector::new();
        v2.set(0, 2.0);
        v2.set(2, 3.0);
        v2.set(5, 4.0);

        let dot = v1.dot(&v2);
        // 1*2 + 2*3 = 2 + 6 = 8
        assert!((dot - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_magnitude() {
        let mut sparse = SparseVector::new();
        sparse.set(0, 3.0);
        sparse.set(1, 4.0);

        let mag = sparse.magnitude();
        // sqrt(3^2 + 4^2) = sqrt(9 + 16) = 5
        assert!((mag - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut sparse = SparseVector::new();
        sparse.set(0, 3.0);
        sparse.set(1, 4.0);

        sparse
            .normalize()
            .expect("Failed to normalize sparse vector");

        let mag = sparse.magnitude();
        assert!((mag - 1.0).abs() < 1e-6);
        assert!((sparse.get(0) - 0.6).abs() < 1e-6);
        assert!((sparse.get(1) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let mut v1 = SparseVector::new();
        v1.set(0, 1.0);
        v1.set(1, 0.0);

        let mut v2 = SparseVector::new();
        v2.set(0, 1.0);
        v2.set(1, 0.0);

        let sim = v1
            .cosine_similarity(&v2)
            .expect("Failed to calculate cosine similarity");
        assert!((sim - 1.0).abs() < 1e-6);
    }
}
