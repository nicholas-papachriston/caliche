//! Unit tests for [`VectorBackendConfig`].

use super::*;

#[test]
fn test_memory_config_creation() {
    let config = VectorBackendConfig::Memory {
        ttl_secs: 3600,
        max_entries: Some(1000),
    };

    match config {
        VectorBackendConfig::Memory {
            ttl_secs,
            max_entries,
        } => {
            assert_eq!(ttl_secs, 3600);
            assert_eq!(max_entries, Some(1000));
        }
        _ => panic!("Expected Memory variant"),
    }
}

#[cfg(feature = "vector-store-redis")]
#[test]
fn test_redis_config_creation() {
    let config = VectorBackendConfig::Redis {
        url: "redis://localhost:6379".to_string(),
        prefix: "test:".to_string(),
    };

    match config {
        VectorBackendConfig::Redis { url, prefix } => {
            assert_eq!(url, "redis://localhost:6379");
            assert_eq!(prefix, "test:");
        }
        _ => panic!("Expected Redis variant"),
    }
}

#[cfg(feature = "vector-store-redis")]
#[test]
fn test_valkey_config_creation() {
    let config = VectorBackendConfig::Valkey {
        url: "redis://localhost:6380".to_string(),
        prefix: "test:".to_string(),
    };

    match config {
        VectorBackendConfig::Valkey { url, prefix } => {
            assert_eq!(url, "redis://localhost:6380");
            assert_eq!(prefix, "test:");
        }
        _ => panic!("Expected Valkey variant"),
    }
}

#[tokio::test]
async fn test_build_memory_backend() {
    let config = VectorBackendConfig::Memory {
        ttl_secs: 3600,
        max_entries: Some(100),
    };

    let backend = config.build_backend().await;
    assert!(backend.is_ok());
}
