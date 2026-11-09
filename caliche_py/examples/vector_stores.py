"""
Caliche Vector Backends Examples

Demonstrates how to configure different vector store backends for RAG operations.
Each backend has different characteristics suited for different use cases.

uv run examples/vector_backends_example.py
"""

import os
from caliche import AgentBuilder, ProviderType


def example_memory_backend():
    """Memory backend - default, perfect for development and testing."""
    print("=== Memory Backend (Default) ===\n")

    agent = (
        AgentBuilder()
        .provider(ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .api_key(os.getenv("OPENAI_API_KEY", ""))
        .set_vector_backend_memory(ttl_secs=86400, max_entries=100000)
        .enable_tool("semantic_search")
        .build()
    )

    print("Memory backend configured with:")
    print("- TTL: 86400 seconds (24 hours)")
    print("- Max entries: 100,000")
    print("- Use case: Development, testing, caching\n")


def example_redis_backend():
    """Redis backend for distributed caching with persistence."""
    print("=== Redis Backend ===\n")

    agent = (
        AgentBuilder()
        .provider(ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .api_key(os.getenv("OPENAI_API_KEY", ""))
        .set_vector_backend_redis(
            url="redis://localhost:6379",
            prefix="my_app:"
        )
        .enable_tool("semantic_search")
        .build()
    )

    print("Redis backend configured with:")
    print("- URL: redis://localhost:6379")
    print("- Prefix: my_app:")
    print("- Use case: Shared cache, real-time apps, distributed systems\n")


def example_postgres_backend():
    """PostgreSQL with pgvector for SQL-backed vectors."""
    print("=== PostgreSQL Backend ===\n")

    agent = (
        AgentBuilder()
        .provider(ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .api_key(os.getenv("OPENAI_API_KEY", ""))
        .set_vector_backend_postgres(
            connection_string="postgresql://user:pass@localhost/db",
            table_name="embeddings",
            vector_dimension=1536,
            max_connections=10
        )
        .enable_tool("semantic_search")
        .build()
    )

    print("PostgreSQL backend configured with:")
    print("- Connection: postgresql://user:pass@localhost/db")
    print("- Table: embeddings")
    print("- Vector dimension: 1536")
    print("- Max connections: 10")
    print("- Use case: SQL integration, persistence, production deployments\n")


def example_qdrant_backend():
    """Qdrant HTTP backend for dedicated vector search."""
    print("=== Qdrant Backend (HTTP) ===\n")

    agent = (
        AgentBuilder()
        .provider(ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .api_key(os.getenv("OPENAI_API_KEY", ""))
        .set_vector_backend_qdrant(
            url="http://localhost:6333",
            vector_dimension=1536,
            collection_prefix="kb_",
            api_key=os.getenv("QDRANT_API_KEY")
        )
        .enable_tool("semantic_search")
        .build()
    )

    print("Qdrant HTTP backend configured with:")
    print("- URL: http://localhost:6333")
    print("- Collection prefix: kb_")
    print("- Vector dimension: 1536")
    print("- Use case: Dedicated vector search, high-performance RAG\n")


def example_qdrant_grpc_backend():
    """Qdrant gRPC backend for high-performance vector operations."""
    print("=== Qdrant Backend (gRPC) ===\n")

    agent = (
        AgentBuilder()
        .provider(ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .api_key(os.getenv("OPENAI_API_KEY", ""))
        .set_vector_backend_qdrant_grpc(
            grpc_url="http://localhost:6334",
            vector_dimension=1536,
            collection_prefix="kb_",
            api_key=os.getenv("QDRANT_API_KEY"),
            batch_size=100
        )
        .enable_tool("semantic_search")
        .build()
    )

    print("Qdrant gRPC backend configured with:")
    print("- gRPC URL: http://localhost:6334")
    print("- Collection prefix: kb_")
    print("- Vector dimension: 1536")
    print("- Batch size: 100")
    print("- Use case: High-performance vector operations, lower latency\n")


def example_pinecone_backend():
    """Pinecone managed vector database service."""
    print("=== Pinecone Backend ===\n")

    agent = (
        AgentBuilder()
        .provider(ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .api_key(os.getenv("OPENAI_API_KEY", ""))
        .set_vector_backend_pinecone(
            base_url=os.getenv("PINECONE_INDEX_HOST", ""),
            api_key=os.getenv("PINECONE_API_KEY", ""),
            dimension=1536,
            timeout_secs=30
        )
        .enable_tool("semantic_search")
        .build()
    )

    print("Pinecone backend configured with:")
    print("- Index host: (from PINECONE_INDEX_HOST)")
    print("- API key: (from PINECONE_API_KEY)")
    print("- Dimension: 1536")
    print("- Timeout: 30 seconds")
    print("- Use case: Zero-ops vector database, managed SaaS\n")


def example_chroma_backend():
    """Chroma open-source embedding database."""
    print("=== Chroma Backend ===\n")

    agent = (
        AgentBuilder()
        .provider(ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .api_key(os.getenv("OPENAI_API_KEY", ""))
        .set_vector_backend_chroma(
            base_url="http://localhost:8000",
            collection_prefix="my_collection",
            expected_dimension=1536,
            timeout_secs=30
        )
        .enable_tool("semantic_search")
        .build()
    )

    print("Chroma backend configured with:")
    print("- Base URL: http://localhost:8000")
    print("- Collection prefix: my_collection")
    print("- Expected dimension: 1536")
    print("- Timeout: 30 seconds")
    print("- Use case: Python ecosystem integration, open-source deployment\n")


def example_s3_backend():
    """S3 backend for cloud object storage."""
    print("=== S3 Backend ===\n")

    agent = (
        AgentBuilder()
        .provider(ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .api_key(os.getenv("OPENAI_API_KEY", ""))
        .set_vector_backend_s3(
            bucket="my-embeddings-bucket",
            key_prefix="embeddings/",
            enable_compression=True,
            batch_size=100,
            cache_indices=True
        )
        .enable_tool("semantic_search")
        .build()
    )

    print("S3 backend configured with:")
    print("- Bucket: my-embeddings-bucket")
    print("- Key prefix: embeddings/")
    print("- Compression: enabled")
    print("- Batch size: 100")
    print("- Cache indices: enabled")
    print("- Use case: Cost-effective cold storage, large-scale deployments\n")


def example_coldforge_backend():
    """ColdForge S3-backed with SPFresh indexing."""
    print("=== ColdForge Backend ===\n")

    agent = (
        AgentBuilder()
        .provider(ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .api_key(os.getenv("OPENAI_API_KEY", ""))
        .set_vector_backend_coldforge(
            bucket="my-bucket",
            key_prefix="coldforge/",
            region="us-east-1",
            vector_dimension=1536,
            cache_size_mb=2048,
            num_centroids=256
        )
        .enable_tool("semantic_search")
        .build()
    )

    print("ColdForge backend configured with:")
    print("- Bucket: my-bucket")
    print("- Key prefix: coldforge/")
    print("- Region: us-east-1")
    print("- Cache size: 2048 MB")
    print("- Centroids: 256")
    print("- Vector dimension: 1536")
    print("- Use case: Large-scale cost-optimized storage with intelligent caching\n")


def example_tiered_backend():
    """Tiered (N-tier) vector backend: cold to hot, e.g. memory + redis."""
    print("=== Tiered Backend (N-tier: Memory + Redis) ===\n")

    # N-tier chain: cold (index 0) to hot (last). At least 2 tiers required.
    agent = (
        AgentBuilder()
        .provider(ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .api_key(os.getenv("OPENAI_API_KEY", ""))
        .set_vector_backend_tiered(
            [
                (
                    {"type": "memory", "ttl_secs": 3600, "max_entries": 10_000},
                    {"name": "hot", "max_vectors": 10_000, "promotion_threshold": 1},
                ),
                (
                    {"type": "redis", "url": "redis://localhost:6379", "prefix": "tiered:warm:"},
                    {"name": "warm", "max_vectors": 100_000, "promotion_threshold": 2},
                ),
            ],
            write_strategy="write_to_source_and_hot",
            background_sync=True,
            sync_interval_secs=60,
        )
        .enable_tool("semantic_search")
        .build()
    )

    print("Tiered vector backend configured:")
    print("- Tier 0 (cold): Memory, 10k vectors, promotion after 1 access")
    print("- Tier 1 (hot): Redis, 100k vectors, promotion after 2 accesses")
    print("- Write strategy: write_to_source_and_hot")
    print("- Background sync: every 60s")
    print("- Use case: Hot/warm caching, cost-effective scale\n")


def example_environment_config():
    """Auto-configure from environment variables."""
    print("=== Environment-Based Configuration ===\n")

    # Set: VECTOR_BACKEND=redis, REDIS_URL=redis://localhost:6379, etc.
    agent = (
        AgentBuilder()
        .provider(ProviderType.OpenAI)
        .model("gpt-4o-mini")
        .api_key(os.getenv("OPENAI_API_KEY", ""))
        .vector_backend_from_env()
        .enable_tool("semantic_search")
        .build()
    )

    print("Vector backend configured from environment variables:")
    print("- VECTOR_BACKEND: (determines backend type)")
    print("- Backend-specific vars: (REDIS_URL, POSTGRES_URL, etc.)")
    print("- Use case: 12-factor apps, deployment flexibility\n")


def main():
    try:
        example_memory_backend()
        example_redis_backend()
        example_postgres_backend()
        example_qdrant_backend()
        example_qdrant_grpc_backend()
        example_pinecone_backend()
        example_chroma_backend()
        example_s3_backend()
        example_coldforge_backend()
        example_tiered_backend()
        example_environment_config()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

