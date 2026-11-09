/**
 * Caliche Vector Backends Examples
 *
 * Demonstrates how to configure different vector store backends for RAG operations.
 * Each backend has different characteristics suited for different use cases.
 *
 * bun run examples/vector-backends.ts
 */

import { AgentBuilder } from "../src/index";

async function exampleMemoryBackend() {
  console.log("=== Memory Backend (Default) ===\n");

  // Memory backend is the default - perfect for development and testing
  await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .vectorBackendMemory(86400, 100000) // 24h TTL, max 100k entries
    .enableTool("semantic_search")
    .build();

  console.log("Memory backend configured with:");
  console.log("- TTL: 86400 seconds (24 hours)");
  console.log("- Max entries: 100,000");
  console.log("- Use case: Development, testing, caching\n");
}

async function exampleRedisBackend() {
  console.log("=== Redis Backend ===\n");

  // Redis backend for distributed caching with persistence
  await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .vectorBackendRedis("redis://localhost:6379", "my_app:")
    .enableTool("semantic_search")
    .build();

  console.log("Redis backend configured with:");
  console.log("- URL: redis://localhost:6379");
  console.log("- Prefix: my_app:");
  console.log("- Use case: Shared cache, real-time apps, distributed systems\n");
}

async function examplePostgresBackend() {
  console.log("=== PostgreSQL Backend ===\n");

  // PostgreSQL with pgvector for SQL-backed vectors
  await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .vectorBackendPostgres(
      "postgresql://user:pass@localhost/db",
      "embeddings",
      1536, // vector dimension
      10 // max connections
    )
    .enableTool("semantic_search")
    .build();

  console.log("PostgreSQL backend configured with:");
  console.log("- Connection: postgresql://user:pass@localhost/db");
  console.log("- Table: embeddings");
  console.log("- Vector dimension: 1536");
  console.log("- Max connections: 10");
  console.log("- Use case: SQL integration, persistence, production deployments\n");
}

async function exampleQdrantBackend() {
  console.log("=== Qdrant Backend (HTTP) ===\n");

  // Qdrant HTTP backend for dedicated vector search
  await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .vectorBackendQdrant(
      "http://localhost:6333",
      1536, // vector dimension
      "kb_", // collection prefix
      process.env.QDRANT_API_KEY
    )
    .enableTool("semantic_search")
    .build();

  console.log("Qdrant HTTP backend configured with:");
  console.log("- URL: http://localhost:6333");
  console.log("- Collection prefix: kb_");
  console.log("- Vector dimension: 1536");
  console.log("- Use case: Dedicated vector search, high-performance RAG\n");
}

async function exampleQdrantGrpcBackend() {
  console.log("=== Qdrant Backend (gRPC) ===\n");

  // Qdrant gRPC backend for high-performance vector operations
  await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .vectorBackendQdrantGrpc(
      "http://localhost:6334",
      1536,
      "kb_",
      process.env.QDRANT_API_KEY,
      100 // batch size
    )
    .enableTool("semantic_search")
    .build();

  console.log("Qdrant gRPC backend configured with:");
  console.log("- gRPC URL: http://localhost:6334");
  console.log("- Collection prefix: kb_");
  console.log("- Vector dimension: 1536");
  console.log("- Batch size: 100");
  console.log("- Use case: High-performance vector operations, lower latency\n");
}

async function examplePineconeBackend() {
  console.log("=== Pinecone Backend ===\n");

  // Pinecone managed vector database service
  await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .vectorBackendPinecone(
      process.env.PINECONE_INDEX_HOST || "",
      process.env.PINECONE_API_KEY || "",
      1536,
      30 // timeout seconds
    )
    .enableTool("semantic_search")
    .build();

  console.log("Pinecone backend configured with:");
  console.log("- Index host: (from PINECONE_INDEX_HOST)");
  console.log("- API key: (from PINECONE_API_KEY)");
  console.log("- Dimension: 1536");
  console.log("- Timeout: 30 seconds");
  console.log("- Use case: Zero-ops vector database, managed SaaS\n");
}

async function exampleChromaBackend() {
  console.log("=== Chroma Backend ===\n");

  // Chroma open-source embedding database
  await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .vectorBackendChroma("http://localhost:8000", "my_collection", 1536, 30)
    .enableTool("semantic_search")
    .build();

  console.log("Chroma backend configured with:");
  console.log("- Base URL: http://localhost:8000");
  console.log("- Collection prefix: my_collection");
  console.log("- Expected dimension: 1536");
  console.log("- Timeout: 30 seconds");
  console.log("- Use case: Python ecosystem integration, open-source deployment\n");
}

async function exampleS3Backend() {
  console.log("=== S3 Backend ===\n");

  // S3 backend for cloud object storage
  await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .vectorBackendS3(
      "my-embeddings-bucket",
      "embeddings/",
      true, // enable compression
      100, // batch size
      true // cache indices
    )
    .enableTool("semantic_search")
    .build();

  console.log("S3 backend configured with:");
  console.log("- Bucket: my-embeddings-bucket");
  console.log("- Key prefix: embeddings/");
  console.log("- Compression: enabled");
  console.log("- Batch size: 100");
  console.log("- Cache indices: enabled");
  console.log("- Use case: Cost-effective cold storage, large-scale deployments\n");
}

async function exampleColdforgeBackend() {
  console.log("=== ColdForge Backend ===\n");

  // ColdForge S3-backed with SPFresh indexing
  await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .vectorBackendColdforge(
      "my-bucket",
      "coldforge/",
      "us-east-1",
      1536, // vector dimension
      2048, // cache size MB
      256 // num centroids
    )
    .enableTool("semantic_search")
    .build();

  console.log("ColdForge backend configured with:");
  console.log("- Bucket: my-bucket");
  console.log("- Key prefix: coldforge/");
  console.log("- Region: us-east-1");
  console.log("- Cache size: 2048 MB");
  console.log("- Centroids: 256");
  console.log("- Vector dimension: 1536");
  console.log("- Use case: Large-scale cost-optimized storage with intelligent caching\n");
}

async function exampleDynamodbBackend() {
  console.log("=== DynamoDB Native Vector Backend ===\n");

  await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .vectorBackendDynamodb("my-vectors", "embeddings", "us-east-1", 1536, false)
    .enableTool("semantic_search")
    .build();

  console.log("DynamoDB native vector backend configured with:");
  console.log("- Table: my-vectors");
  console.log("- Index: embeddings");
  console.log("- Region: us-east-1");
  console.log("- Vector dimension: 1536");
  console.log("- Use case: Managed ANN via DynamoDB SearchVectors\n");
}

async function exampleTieredBackend() {
  console.log("=== Tiered Backend (N-tier: Memory + Redis) ===\n");

  // N-tier chain: cold (index 0) to hot (last). At least 2 tiers required.
  await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .vectorBackendTiered(
      [
        {
          backend: { type: "memory", ttlSecs: 3600, maxEntries: 10_000 },
          tierConfig: { name: "hot", maxVectors: 10_000, promotionThreshold: 1 },
        },
        {
          backend: { type: "redis", url: "redis://localhost:6379", prefix: "tiered:warm:" },
          tierConfig: { name: "warm", maxVectors: 100_000, promotionThreshold: 2 },
        },
      ],
      { writeStrategy: "write_to_source_and_hot", backgroundSync: true, syncIntervalSecs: 60 }
    )
    .enableTool("semantic_search")
    .build();

  console.log("Tiered vector backend configured:");
  console.log("- Tier 0 (cold): Memory, 10k vectors, promotion after 1 access");
  console.log("- Tier 1 (hot): Redis, 100k vectors, promotion after 2 accesses");
  console.log("- Write strategy: write_to_source_and_hot");
  console.log("- Background sync: every 60s");
  console.log("- Use case: Hot/warm caching, cost-effective scale\n");
}

async function exampleEnvironmentConfig() {
  console.log("=== Environment-Based Configuration ===\n");

  // Auto-configure from environment variables
  // Set: VECTOR_BACKEND=redis, REDIS_URL=redis://localhost:6379, etc.
  await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .vectorBackendFromEnv()
    .enableTool("semantic_search")
    .build();

  console.log("Vector backend configured from environment variables:");
  console.log("- VECTOR_BACKEND: (determines backend type)");
  console.log("- Backend-specific vars: (REDIS_URL, POSTGRES_URL, etc.)");
  console.log("- Use case: 12-factor apps, deployment flexibility\n");
}

async function main() {
  try {
    await exampleMemoryBackend();
    await exampleRedisBackend();
    await examplePostgresBackend();
    await exampleQdrantBackend();
    await exampleQdrantGrpcBackend();
    await examplePineconeBackend();
    await exampleChromaBackend();
    await exampleS3Backend();
    await exampleColdforgeBackend();
    await exampleDynamodbBackend();
    await exampleTieredBackend();
    await exampleEnvironmentConfig();
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}
