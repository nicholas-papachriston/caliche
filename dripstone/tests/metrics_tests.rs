//! Tests for unified metrics interface

use dripstone::metrics::{NoOpMetrics, SimpleMetrics, VectorMetrics};
use std::time::Duration;

#[test]
fn test_noop_metrics() {
    let metrics = NoOpMetrics;

    // Should not panic
    metrics.record_search_latency(Duration::from_millis(10));
    metrics.record_index_build_latency(Duration::from_millis(100));
    metrics.record_upsert_latency(Duration::from_millis(5));
    metrics.record_index_size(1000);
    metrics.increment_cache_hit();
    metrics.increment_cache_miss();
    metrics.record_search_result_count(10);
    metrics.record_vector_dimension(128);
    metrics.record_error("TestError");
    metrics.record_batch_size(100);
    metrics.record_memory_usage(1024 * 1024);
    metrics.record_compression_ratio(4.0);
}

#[test]
fn test_simple_metrics_basic() {
    let metrics = SimpleMetrics::new();

    metrics.record_search_latency(Duration::from_millis(10));
    metrics.record_search_latency(Duration::from_millis(20));

    let avg = metrics.average_search_latency().unwrap();
    assert_eq!(avg, Duration::from_millis(15));
}

#[test]
fn test_simple_metrics_cache_hit_rate() {
    let metrics = SimpleMetrics::new();

    metrics.increment_cache_hit();
    metrics.increment_cache_hit();
    metrics.increment_cache_miss();

    let hit_rate = metrics.cache_hit_rate();
    assert!((hit_rate - 2.0 / 3.0).abs() < 0.01);
}

#[test]
fn test_simple_metrics_index_size() {
    let metrics = SimpleMetrics::new();

    metrics.record_index_size(100);
    metrics.record_index_size(200);
    metrics.record_index_size(300);

    let current_size = metrics.current_index_size().unwrap();
    assert_eq!(current_size, 300);
}

#[test]
fn test_simple_metrics_error_tracking() {
    let metrics = SimpleMetrics::new();

    metrics.record_error("InvalidVectorLength");
    metrics.record_error("InvalidVectorLength");
    metrics.record_error("StorageError");

    let errors = metrics.error_counts();
    assert_eq!(errors.get("InvalidVectorLength"), Some(&2));
    assert_eq!(errors.get("StorageError"), Some(&1));
}

#[test]
fn test_simple_metrics_search_result_count() {
    let metrics = SimpleMetrics::new();

    metrics.record_search_result_count(5);
    metrics.record_search_result_count(10);
    metrics.record_search_result_count(15);

    let avg = metrics.average_search_result_count().unwrap();
    assert!((avg - 10.0).abs() < 0.01);
}

#[test]
fn test_simple_metrics_empty() {
    let metrics = SimpleMetrics::new();

    assert!(metrics.average_search_latency().is_none());
    assert!(metrics.current_index_size().is_none());
    assert_eq!(metrics.cache_hit_rate(), 0.0);
    assert!(metrics.average_search_result_count().is_none());
}
