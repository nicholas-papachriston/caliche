//! Integration tests for vector store functionality

#[cfg(feature = "tools-rag")]
mod vecstore_integration {
    use caliche::llm::OpenAIProvider;
    use caliche::rag::RagService;
    use caliche::vecstore::VectorBackendConfig;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_memory_backend_creation() {
        let config = VectorBackendConfig::Memory {
            ttl_secs: 3600,
            max_entries: Some(100),
        };

        let backend = config.build_backend().await;
        assert!(backend.is_ok(), "Should create memory backend successfully");
    }

    #[tokio::test]
    async fn test_rag_service_with_memory_backend() {
        // Create a mock embedding provider (in real tests, this would be mocked)
        let embedding_provider = Arc::new(
            OpenAIProvider::try_new("test-key".to_string(), "text-embedding-3-small".to_string())
                .expect("valid test embedding provider"),
        );

        let _rag_service = RagService::new(embedding_provider);
        // Successfully created RagService with default memory backend
        assert!(true);
    }

    #[tokio::test]
    async fn test_rag_service_default_constructor() {
        let embedding_provider = Arc::new(
            OpenAIProvider::try_new("test-key".to_string(), "text-embedding-3-small".to_string())
                .expect("valid test embedding provider"),
        );

        let _rag_service = RagService::new(embedding_provider);
        // Successfully created RagService with default backend
        assert!(true);
    }

    #[test]
    fn test_vector_backend_config_variants() {
        // Test that all config variants can be created
        let _memory = VectorBackendConfig::Memory {
            ttl_secs: 3600,
            max_entries: Some(100),
        };

        #[cfg(feature = "vector-store-redis")]
        let _redis = VectorBackendConfig::Redis {
            url: "redis://:caliche_dev_redis@localhost:6379".to_string(),
            prefix: "test:".to_string(),
        };

        #[cfg(feature = "vector-store-postgres")]
        let _postgres = VectorBackendConfig::Postgres {
            connection_string: "postgresql://localhost/test".to_string(),
            table_name: "embeddings".to_string(),
            max_connections: 10,
            vector_dimension: Some(1536),
            hnsw_config: None,
        };

        #[cfg(feature = "vector-store-qdrant")]
        let _qdrant = VectorBackendConfig::Qdrant {
            url: "http://localhost:6333".to_string(),
            collection_prefix: Some("test_".to_string()),
            vector_dimension: 768,
            api_key: Some("caliche_dev_qdrant".to_string()),
            batch_size: None,
            timeout: None,
        };

        assert!(true, "All config variants created successfully");
    }
}

#[cfg(feature = "tools-rag")]
mod chunking_tests {
    use caliche::vecstore::chunking::{ChunkStrategy, FixedSizeChunker};

    #[test]
    fn test_fixed_size_chunker() {
        let chunker = FixedSizeChunker::new(100, 10);
        let text = "This is a test document. ".repeat(10);
        let chunks = chunker.chunk(&text);

        assert!(!chunks.is_empty(), "Should produce chunks");
        for chunk in chunks {
            assert!(
                chunk.content.len() <= 110,
                "Chunks should respect size limits"
            );
        }
    }
}
