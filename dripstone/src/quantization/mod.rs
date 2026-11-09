//! Product Quantization (PQ) for vector compression.
//!
//! Reduces memory footprint by compressing vectors into compact codes
//! while maintaining approximate similarity search capabilities.

use crate::errors::DripstoneError;
use crate::simd;
use crate::utils::numeric::usize_to_f32;

/// Product Quantizer for vector compression
pub struct ProductQuantizer {
    /// Number of sub-vectors (M parameter)
    m: usize,
    /// Number of centroids per sub-vector (K parameter, typically 256)
    k: usize,
    /// Sub-vector dimension (`original_dim` / m)
    sub_dim: usize,
    /// Codebooks: [M][K][`sub_dim`] - one codebook per sub-vector
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Original vector dimension
    original_dim: usize,
}

impl ProductQuantizer {
    /// Create a new Product Quantizer
    ///
    /// # Arguments
    /// * `m` - Number of sub-vectors (typically 8-16)
    /// * `k` - Number of centroids per sub-vector (typically 256)
    /// * `original_dim` - Original vector dimension
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn new(m: usize, k: usize, original_dim: usize) -> Result<Self, DripstoneError> {
        if !original_dim.is_multiple_of(m) {
            return Err(DripstoneError::storage(format!(
                "Original dimension {original_dim} must be divisible by m={m}"
            )));
        }

        let sub_dim = original_dim / m;

        Ok(Self {
            m,
            k,
            sub_dim,
            codebooks: Vec::new(),
            original_dim,
        })
    }

    /// Train the quantizer on a set of vectors using K-means clustering
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn train(
        &mut self,
        vectors: &[Vec<f32>],
        max_iterations: usize,
    ) -> Result<(), DripstoneError> {
        if vectors.is_empty() {
            return Err(DripstoneError::storage("Cannot train on empty vector set"));
        }

        // Verify all vectors have correct dimension
        for v in vectors {
            if v.len() != self.original_dim {
                return Err(DripstoneError::InvalidVectorLength {
                    expected: self.original_dim,
                    actual: v.len(),
                });
            }
        }

        self.codebooks.clear();

        // Train codebook for each sub-vector
        for sub_idx in 0..self.m {
            let start = sub_idx * self.sub_dim;
            let end = start + self.sub_dim;

            // Extract sub-vectors
            let sub_vectors: Vec<Vec<f32>> =
                vectors.iter().map(|v| v[start..end].to_vec()).collect();

            // Run K-means to get centroids
            let centroids = self.kmeans_clustering(&sub_vectors, max_iterations)?;
            self.codebooks.push(centroids);
        }

        Ok(())
    }

    /// K-means clustering for a set of sub-vectors
    fn kmeans_clustering(
        &self,
        vectors: &[Vec<f32>],
        max_iterations: usize,
    ) -> Result<Vec<Vec<f32>>, DripstoneError> {
        if vectors.is_empty() {
            return Err(DripstoneError::storage("Cannot cluster empty vector set"));
        }

        let k = self.k.min(vectors.len());
        let dim = vectors[0].len();

        // Initialize centroids using K-means++
        let mut centroids = Self::kmeans_plus_plus_init(vectors, k)?;

        for _iteration in 0..max_iterations {
            // Assign vectors to nearest centroids
            let assignments: Vec<usize> = vectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .enumerate()
                        .map(|(idx, c)| (idx, Self::euclidean_distance_squared(v, c)))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map_or(0, |(idx, _)| idx)
                })
                .collect();

            // Update centroids
            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut counts = vec![0_usize; k];

            for (v, &cluster_idx) in vectors.iter().zip(assignments.iter()) {
                for (i, &val) in v.iter().enumerate() {
                    new_centroids[cluster_idx][i] += val;
                }
                counts[cluster_idx] += 1;
            }

            // Average to get new centroids
            for (cluster_idx, (centroid, &count)) in
                new_centroids.iter_mut().zip(counts.iter()).enumerate()
            {
                if count > 0 {
                    for val in centroid.iter_mut() {
                        *val /= usize_to_f32(count);
                    }
                } else if !vectors.is_empty() {
                    // Re-seed empty clusters from a deterministic training sub-vector
                    // so empty centroids do not stay at the origin and keep
                    // collecting nothing on future iterations. Picking by
                    // `cluster_idx % vectors.len()` keeps the K-means run
                    // reproducible across invocations.
                    let pick = cluster_idx % vectors.len();
                    centroid.clone_from(&vectors[pick]);
                }
            }

            // Check convergence (simplified - check if centroids changed significantly)
            let mut converged = true;
            for (old, new) in centroids.iter().zip(new_centroids.iter()) {
                if Self::euclidean_distance_squared(old, new) > 1e-6 {
                    converged = false;
                    break;
                }
            }

            if converged {
                break;
            }

            centroids = new_centroids;
        }

        Ok(centroids)
    }

    /// K-means++ initialization
    fn kmeans_plus_plus_init(
        vectors: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<f32>>, DripstoneError> {
        if vectors.is_empty() {
            return Err(DripstoneError::storage(
                "Cannot initialize with empty vectors",
            ));
        }
        let mut centroids = Vec::with_capacity(k);

        // Choose first centroid randomly
        let first_idx = fastrand::usize(..vectors.len());
        centroids.push(vectors[first_idx].clone());

        // Choose remaining centroids using K-means++ algorithm
        for _ in 1..k {
            // Compute distances to nearest centroid
            let distances: Vec<f32> = vectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .map(|c| Self::euclidean_distance_squared(v, c))
                        .fold(f32::INFINITY, f32::min)
                })
                .collect();

            // Choose next centroid with probability proportional to distance squared
            let sum: f32 = distances.iter().sum();
            if sum <= f32::EPSILON {
                // All vectors are at centroids, choose randomly
                let idx = fastrand::usize(..vectors.len());
                centroids.push(vectors[idx].clone());
                continue;
            }

            let mut cumsum = 0.0;
            let random = fastrand::f32() * sum;
            let mut chosen_idx = 0;

            for (i, &dist) in distances.iter().enumerate() {
                cumsum += dist;
                if random <= cumsum {
                    chosen_idx = i;
                    break;
                }
            }

            centroids.push(vectors[chosen_idx].clone());
        }

        Ok(centroids)
    }

    /// Encode a vector into PQ codes
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>, DripstoneError> {
        if vector.len() != self.original_dim {
            return Err(DripstoneError::InvalidVectorLength {
                expected: self.original_dim,
                actual: vector.len(),
            });
        }

        if self.codebooks.is_empty() {
            return Err(DripstoneError::storage("Quantizer not trained"));
        }

        let mut codes = Vec::with_capacity(self.m);

        for sub_idx in 0..self.m {
            let start = sub_idx * self.sub_dim;
            let end = start + self.sub_dim;
            let sub_vector = &vector[start..end];

            // Find nearest centroid in this codebook
            let codebook = &self.codebooks[sub_idx];
            let nearest_idx = codebook
                .iter()
                .enumerate()
                .map(|(idx, centroid)| {
                    (idx, Self::euclidean_distance_squared(sub_vector, centroid))
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx);

            codes.push(u8::try_from(nearest_idx).map_err(|_| {
                DripstoneError::storage("product-quantization centroid index exceeds u8")
            })?);
        }

        Ok(codes)
    }

    /// Decode PQ codes back to approximate vector
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn decode(&self, codes: &[u8]) -> Result<Vec<f32>, DripstoneError> {
        if codes.len() != self.m {
            return Err(DripstoneError::storage(format!(
                "Expected {} codes, got {}",
                self.m,
                codes.len()
            )));
        }

        if self.codebooks.is_empty() {
            return Err(DripstoneError::storage("Quantizer not trained"));
        }

        let mut reconstructed = Vec::with_capacity(self.original_dim);

        for (sub_idx, &code) in codes.iter().enumerate() {
            let codebook = &self.codebooks[sub_idx];
            if code as usize >= codebook.len() {
                return Err(DripstoneError::storage(format!(
                    "Invalid code {code} for sub-vector {sub_idx}"
                )));
            }

            let centroid = &codebook[code as usize];
            reconstructed.extend_from_slice(centroid);
        }

        Ok(reconstructed)
    }

    /// Compute approximate distance between a query vector and encoded codes
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn approximate_distance(&self, query: &[f32], codes: &[u8]) -> Result<f32, DripstoneError> {
        if query.len() != self.original_dim {
            return Err(DripstoneError::InvalidVectorLength {
                expected: self.original_dim,
                actual: query.len(),
            });
        }

        if codes.len() != self.m {
            return Err(DripstoneError::storage(format!(
                "Expected {} codes, got {}",
                self.m,
                codes.len()
            )));
        }

        let mut total_distance_sq = 0.0;

        for (sub_idx, &code) in codes.iter().enumerate() {
            let start = sub_idx * self.sub_dim;
            let end = start + self.sub_dim;
            let query_sub = &query[start..end];

            let codebook = &self.codebooks[sub_idx];
            if code as usize >= codebook.len() {
                return Err(DripstoneError::storage("Invalid code"));
            }

            let centroid = &codebook[code as usize];
            total_distance_sq += Self::euclidean_distance_squared(query_sub, centroid);
        }

        Ok(total_distance_sq.sqrt())
    }

    /// Euclidean distance squared (faster than distance)
    /// Uses SIMD-accelerated implementation when available.
    fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
        simd::euclidean_distance_squared(a, b).unwrap_or_else(|_| {
            // Fallback to scalar if SIMD fails (shouldn't happen, but be safe)
            a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
        })
    }

    /// Get compression ratio (original bytes / compressed bytes)
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.original_dim * 4; // f32 = 4 bytes
        let compressed_bytes = self.m; // u8 per sub-vector
        usize_to_f32(original_bytes) / usize_to_f32(compressed_bytes)
    }

    /// Get the number of sub-vectors (M)
    #[must_use]
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Get the number of centroids per sub-vector (K)
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Get the original vector dimension
    #[must_use]
    pub const fn original_dim(&self) -> usize {
        self.original_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|_| {
                (0..dimension)
                    .map(|_| fastrand::f32().mul_add(2.0, -1.0))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_pq_creation() {
        let pq = ProductQuantizer::new(8, 256, 128).expect("Failed to create ProductQuantizer");
        assert_eq!(pq.m(), 8);
        assert_eq!(pq.k(), 256);
        assert_eq!(pq.original_dim(), 128);
        assert!((pq.compression_ratio() - 64.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn test_pq_train_encode_decode() {
        // Use smaller parameters for faster test execution while still validating correctness
        let mut pq = ProductQuantizer::new(4, 64, 128).expect("Failed to create ProductQuantizer");
        let vectors = create_test_vectors(100, 128);

        pq.train(&vectors, 3)
            .expect("Failed to train ProductQuantizer");

        let test_vector = &vectors[0];
        let codes = pq.encode(test_vector).expect("Failed to encode vector");
        assert_eq!(codes.len(), 4); // 4 sub-vectors = 4 codes

        let reconstructed = pq.decode(&codes).expect("Failed to decode codes");
        assert_eq!(reconstructed.len(), 128);

        // Check that reconstruction is approximate
        // Note: PQ introduces some error, so we check that it's reasonable
        // The exact threshold depends on the data distribution and PQ parameters
        let distance = pq
            .approximate_distance(test_vector, &codes)
            .expect("Failed to calculate approximate distance");
        // For well-trained PQ with 8 sub-vectors and 256 centroids, reconstruction error
        // should be relatively small. We use a more lenient threshold.
        assert!(
            distance < 10.0,
            "Reconstruction distance {distance} is too large"
        );
    }

    #[test]
    fn test_pq_invalid_dimension() {
        let result = ProductQuantizer::new(8, 256, 127);
        assert!(result.is_err());
    }
}
