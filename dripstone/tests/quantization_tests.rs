//! Tests for Product Quantization functionality

use dripstone::quantization::ProductQuantizer;

fn create_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            (0..dimension)
                .map(|j| ((i * dimension + j) as f32 / 1000.0).sin())
                .collect()
        })
        .collect()
}

#[test]
fn test_pq_creation() {
    let pq = ProductQuantizer::new(8, 256, 128).unwrap();
    assert_eq!(pq.m(), 8);
    assert_eq!(pq.k(), 256);
    assert_eq!(pq.original_dim(), 128);
    assert_eq!(pq.compression_ratio(), (128 * 4) as f32 / 8.0);
}

#[test]
fn test_pq_invalid_dimension() {
    let result = ProductQuantizer::new(8, 256, 127);
    assert!(result.is_err());
}

#[test]
fn test_pq_train() {
    let mut pq = ProductQuantizer::new(8, 256, 128).unwrap();
    let vectors = create_test_vectors(1000, 128);

    let result = pq.train(&vectors, 10);
    assert!(result.is_ok());
}

#[test]
fn test_pq_encode_decode() {
    let mut pq = ProductQuantizer::new(8, 256, 128).unwrap();
    let vectors = create_test_vectors(1000, 128);

    pq.train(&vectors, 10).unwrap();

    let test_vector = &vectors[0];
    let codes = pq.encode(test_vector).unwrap();
    assert_eq!(codes.len(), 8);

    let reconstructed = pq.decode(&codes).unwrap();
    assert_eq!(reconstructed.len(), 128);

    // Check that reconstruction is approximate
    let distance = pq.approximate_distance(test_vector, &codes).unwrap();
    assert!(distance < 1.0); // Should be reasonably close
}

#[test]
fn test_pq_multiple_vectors() {
    let mut pq = ProductQuantizer::new(8, 256, 128).unwrap();
    let vectors = create_test_vectors(1000, 128);

    pq.train(&vectors, 10).unwrap();

    for vector in &vectors[..10] {
        let codes = pq.encode(vector).unwrap();
        let reconstructed = pq.decode(&codes).unwrap();
        let distance = pq.approximate_distance(vector, &codes).unwrap();

        assert_eq!(reconstructed.len(), 128);
        assert!(distance < 2.0); // Should be reasonably close
    }
}

#[test]
fn test_pq_compression_ratio() {
    let pq = ProductQuantizer::new(8, 256, 128).unwrap();
    let ratio = pq.compression_ratio();

    // Original: 128 * 4 bytes = 512 bytes
    // Compressed: 8 bytes
    // Ratio: 512 / 8 = 64
    assert!((ratio - 64.0).abs() < 0.1);
}

#[test]
fn test_pq_empty_training_set() {
    let mut pq = ProductQuantizer::new(8, 256, 128).unwrap();
    let vectors: Vec<Vec<f32>> = Vec::new();

    let result = pq.train(&vectors, 10);
    assert!(result.is_err());
}

#[test]
fn test_pq_invalid_encode_dimension() {
    let mut pq = ProductQuantizer::new(8, 256, 128).unwrap();
    let vectors = create_test_vectors(100, 128);
    pq.train(&vectors, 10).unwrap();

    let wrong_vector = vec![0.0; 64]; // Wrong dimension
    let result = pq.encode(&wrong_vector);
    assert!(result.is_err());
}

#[test]
fn test_pq_invalid_decode_length() {
    let mut pq = ProductQuantizer::new(8, 256, 128).unwrap();
    let vectors = create_test_vectors(100, 128);
    pq.train(&vectors, 10).unwrap();

    let wrong_codes = vec![0u8; 4]; // Wrong length
    let result = pq.decode(&wrong_codes);
    assert!(result.is_err());
}
