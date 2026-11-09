//! Backend construction dispatcher for [`VectorBackendConfig`].
//!
//! This file centralises the (fairly chunky) match that turns every backend
//! variant into a concrete `Dripstone` [`Backend`]. Splitting it out keeps the
//! config/module declarations tidy while preserving the original behaviour.

use super::VectorBackendConfig;
#[cfg(any(
    feature = "vector-store-s3-backend",
    feature = "vector-store-coldforge",
    feature = "vector-store-dynamodb"
))]
use super::env::EnvAwsCredentials;
use anyhow::Result;
#[cfg(feature = "vector-memory")]
use anyhow::anyhow;
use dripstone::Backend;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tracing::instrument;

#[cfg(feature = "vector-store-postgres")]
fn postgres_i32(value: usize, field: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| anyhow::anyhow!("{field} value {value} exceeds the PostgreSQL i32 range"))
}

impl VectorBackendConfig {
    /// Build a backend from this configuration.
    ///
    /// Returns an Arc-wrapped backend that implements Backend.
    /// Use this with `Dripstone` or other components that need a Backend.
    #[allow(clippy::type_complexity)]
    #[instrument(skip(self), fields(backend = ?self), err)]
    pub fn build_backend(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Arc<dyn Backend + Send + Sync>>> + Send + '_>> {
        Box::pin(async move { build_configured_backend(self).await })
    }
}

type BuiltBackend = Arc<dyn Backend + Send + Sync>;

async fn build_configured_backend(config: &VectorBackendConfig) -> Result<BuiltBackend> {
    match config {
        VectorBackendConfig::Memory {
            ttl_secs,
            max_entries,
        } => Ok(build_memory_backend(*ttl_secs, *max_entries)),
        #[cfg(feature = "vector-store-redis")]
        VectorBackendConfig::Redis { url, prefix }
        | VectorBackendConfig::Valkey { url, prefix } => build_redis_backend(url, prefix).await,
        #[cfg(feature = "vector-store-postgres")]
        VectorBackendConfig::Postgres {
            connection_string,
            table_name,
            vector_dimension,
            max_connections,
            hnsw_config,
        } => {
            build_postgres_backend(
                connection_string,
                table_name,
                *vector_dimension,
                *max_connections,
                hnsw_config.as_ref(),
            )
            .await
        }
        #[cfg(feature = "vector-store-qdrant-grpc")]
        VectorBackendConfig::QdrantGrpc {
            grpc_url,
            collection_prefix,
            vector_dimension,
            api_key,
            batch_size,
        } => {
            build_qdrant_grpc_backend(
                grpc_url,
                collection_prefix.as_deref(),
                *vector_dimension,
                api_key.as_deref(),
                *batch_size,
            )
            .await
        }
        #[cfg(feature = "vector-store-qdrant")]
        VectorBackendConfig::Qdrant {
            url,
            collection_prefix,
            vector_dimension,
            api_key,
            batch_size,
            timeout,
        } => {
            build_qdrant_http_backend(
                url,
                collection_prefix.as_deref(),
                *vector_dimension,
                api_key.as_deref(),
                *batch_size,
                *timeout,
            )
            .await
        }
        #[cfg(feature = "vector-store-pinecone")]
        VectorBackendConfig::Pinecone {
            api_key,
            base_url,
            dimension,
        } => build_pinecone_backend(api_key, base_url, *dimension).await,
        #[cfg(feature = "vector-store-chroma")]
        VectorBackendConfig::Chroma {
            url,
            collection_name,
            dimension,
        } => build_chroma_backend(url, collection_name, *dimension).await,
        #[cfg(feature = "vector-store-weaviate")]
        VectorBackendConfig::Weaviate {
            url,
            class_name,
            dimension,
            api_key,
        } => build_weaviate_backend(url, class_name, *dimension, api_key.as_deref()).await,
        #[cfg(feature = "vector-store-milvus")]
        VectorBackendConfig::Milvus {
            url,
            collection_name,
            dimension,
        } => build_milvus_backend(url, collection_name, *dimension).await,
        #[cfg(feature = "vector-store-s3-backend")]
        VectorBackendConfig::S3 {
            bucket,
            prefix,
            region,
            dimension: _,
        } => build_s3_backend(bucket, prefix, region.as_deref()).await,
        #[cfg(feature = "vector-store-coldforge")]
        VectorBackendConfig::ColdForge {
            bucket,
            prefix,
            region,
            dimension,
            compression,
        } => {
            build_coldforge_backend(bucket, prefix, region.as_deref(), *dimension, *compression)
                .await
        }
        #[cfg(feature = "vector-store-dynamodb")]
        VectorBackendConfig::DynamoDb {
            table_name,
            index_name,
            region,
            endpoint_url,
            dimension,
            ensure_table,
        } => {
            build_dynamodb_backend(
                table_name,
                index_name,
                region.as_deref(),
                endpoint_url.as_deref(),
                *dimension,
                *ensure_table,
            )
            .await
        }
        #[cfg(feature = "vector-store-mmap")]
        VectorBackendConfig::Mmap {
            file_path,
            dimension,
            read_write,
            flush_every_ops,
            vector_encoding,
        } => build_mmap_backend(
            file_path,
            *dimension,
            *read_write,
            *flush_every_ops,
            vector_encoding.as_ref(),
        ),
        #[cfg(feature = "vector-memory")]
        VectorBackendConfig::Indexed {
            inner,
            dimension,
            hnsw_config,
            config,
        } => build_indexed_backend(inner, *dimension, hnsw_config.as_ref(), config.as_ref()).await,
        #[cfg(feature = "vector-memory")]
        VectorBackendConfig::FullTextIndexed { inner, config } => {
            build_full_text_indexed_backend(inner, config.as_ref()).await
        }
        #[cfg(feature = "vector-memory")]
        VectorBackendConfig::Tiered {
            tiers,
            write_strategy,
            search_strategy,
            background_sync,
            sync_interval_secs,
        } => {
            build_tiered_backend(
                tiers,
                write_strategy.as_ref(),
                search_strategy.as_ref(),
                *background_sync,
                *sync_interval_secs,
            )
            .await
        }
    }
}

fn build_memory_backend(ttl_secs: u64, max_entries: Option<usize>) -> BuiltBackend {
    use dripstone::backend::memory::{InMemoryConfig, MemoryBackend};

    let config = InMemoryConfig {
        default_ttl_secs: Some(ttl_secs),
        max_entries,
        auto_cleanup: true,
    };
    Arc::new(MemoryBackend::with_config(config))
}

#[cfg(feature = "vector-store-redis")]
async fn build_redis_backend(url: &str, prefix: &str) -> Result<BuiltBackend> {
    use dripstone::backend::redis::RedisBackend;

    let backend = RedisBackend::with_prefix(url, prefix).await?;
    Ok(Arc::new(backend))
}

#[cfg(feature = "vector-store-qdrant-grpc")]
async fn build_qdrant_grpc_backend(
    grpc_url: &str,
    collection_prefix: Option<&str>,
    vector_dimension: usize,
    api_key: Option<&str>,
    batch_size: Option<usize>,
) -> Result<BuiltBackend> {
    use dripstone::backend::qdrant::{QdrantBackend, QdrantGrpcBackendOptions};

    let mut options = QdrantGrpcBackendOptions::new(grpc_url.to_owned(), vector_dimension);
    options.collection_prefix = collection_prefix.map(str::to_owned);
    options.api_key = api_key.map(str::to_owned);
    options.batch_size = batch_size;
    let backend = QdrantBackend::with_options(options).await?;
    Ok(Arc::new(backend))
}

#[cfg(feature = "vector-store-qdrant")]
async fn build_qdrant_http_backend(
    url: &str,
    collection_prefix: Option<&str>,
    vector_dimension: usize,
    api_key: Option<&str>,
    batch_size: Option<usize>,
    timeout: Option<std::time::Duration>,
) -> Result<BuiltBackend> {
    use dripstone::backend::qdrant::{QdrantBackendOptions, QdrantHttpBackend};

    let mut options = QdrantBackendOptions::new(url.to_owned(), vector_dimension);
    options.collection_prefix = collection_prefix.map(str::to_owned);
    options.api_key = api_key.map(str::to_owned);
    options.batch_size = batch_size;
    options.timeout = timeout;
    let backend = QdrantHttpBackend::with_options(options).await?;
    Ok(Arc::new(backend))
}

#[cfg(feature = "vector-store-pinecone")]
async fn build_pinecone_backend(
    api_key: &str,
    base_url: &str,
    dimension: usize,
) -> Result<BuiltBackend> {
    use dripstone::backend::pinecone::{PineconeBackend, PineconeBackendOptions};

    let options = PineconeBackendOptions {
        api_key: api_key.to_owned(),
        base_url: base_url.to_owned(),
        dimension,
        timeout: None,
        batch_size: None,
        retry_config: None,
    };
    let backend = PineconeBackend::with_options(options).await?;
    Ok(Arc::new(backend))
}

#[cfg(feature = "vector-store-chroma")]
async fn build_chroma_backend(
    url: &str,
    collection_name: &str,
    dimension: usize,
) -> Result<BuiltBackend> {
    use dripstone::backend::chroma::{ChromaBackend, ChromaBackendOptions};

    let mut options = ChromaBackendOptions::new(url.to_owned());
    options.collection_prefix = Some(collection_name.to_owned());
    options.expected_dimension = Some(dimension);
    let backend = ChromaBackend::with_options(options).await?;
    Ok(Arc::new(backend))
}

#[cfg(feature = "vector-store-weaviate")]
async fn build_weaviate_backend(
    url: &str,
    class_name: &str,
    dimension: usize,
    api_key: Option<&str>,
) -> Result<BuiltBackend> {
    use dripstone::backend::weaviate::{WeaviateBackend, WeaviateBackendOptions};

    let options = WeaviateBackendOptions {
        base_url: url.to_owned(),
        class_prefix: Some(class_name.to_owned()),
        vector_dimension: dimension,
        api_key: api_key.map(str::to_owned),
        timeout: None,
        batch_size: None,
        retry_config: None,
    };
    let backend = WeaviateBackend::with_options(options).await?;
    Ok(Arc::new(backend))
}

#[cfg(feature = "vector-store-milvus")]
async fn build_milvus_backend(
    url: &str,
    collection_name: &str,
    dimension: usize,
) -> Result<BuiltBackend> {
    use dripstone::backend::milvus::{MilvusBackend, MilvusBackendOptions};

    let options = MilvusBackendOptions {
        endpoint: url.to_owned(),
        collection_prefix: Some(collection_name.to_owned()),
        vector_dimension: dimension,
        index_type: None,
        metric_type: None,
        nlist: None,
        timeout: None,
        batch_size: None,
        retry_config: None,
        flush_policy: None,
    };
    let backend = MilvusBackend::with_options(options).await?;
    Ok(Arc::new(backend))
}

#[cfg(feature = "vector-store-s3-backend")]
async fn build_s3_backend(
    bucket: &str,
    prefix: &str,
    region: Option<&str>,
) -> Result<BuiltBackend> {
    use dripstone::backend::s3::{S3Backend, S3BackendConfig};

    let creds = EnvAwsCredentials::from_env();
    let config = S3BackendConfig {
        bucket: bucket.to_owned(),
        key_prefix: prefix.to_owned(),
        enable_compression: true,
        batch_size: 100,
        cache_indices: true,
        access_key_id: creds.access_key_id,
        secret_access_key: creds.secret_access_key,
        session_token: creds.session_token,
        region: region.map(str::to_owned),
    };
    let backend = S3Backend::with_config(config).await?;
    Ok(Arc::new(backend))
}

#[cfg(feature = "vector-store-coldforge")]
async fn build_coldforge_backend(
    bucket: &str,
    prefix: &str,
    region: Option<&str>,
    dimension: usize,
    compression: bool,
) -> Result<BuiltBackend> {
    use dripstone::backend::coldforge::{
        ColdForgeBackend, ColdForgeConfig, DEFAULT_BODY_PUT_CONCURRENCY,
        DEFAULT_WAL_REFRESH_INTERVAL_MS,
    };

    let creds = EnvAwsCredentials::from_env();
    let config = ColdForgeConfig {
        bucket: bucket.to_owned(),
        key_prefix: prefix.to_owned(),
        region: region.unwrap_or("us-east-1").to_owned(),
        cache_size_bytes: 100 * 1024 * 1024,
        num_centroids: 256,
        wal_batch_interval_ms: 1000,
        wal_max_batch_size: 256,
        enable_compression: compression,
        vector_dimension: dimension,
        hot_tier: None,
        refresh_shared_wal: None,
        wal_refresh_interval_ms: DEFAULT_WAL_REFRESH_INTERVAL_MS,
        access_key_id: creds.access_key_id,
        secret_access_key: creds.secret_access_key,
        session_token: creds.session_token,
        object_store_config: None,
        body_put_concurrency: DEFAULT_BODY_PUT_CONCURRENCY,
    };
    let backend = ColdForgeBackend::new(config).await?;
    Ok(Arc::new(backend))
}

#[cfg(feature = "vector-store-postgres")]
async fn build_postgres_backend(
    connection_string: &str,
    table_name: &str,
    vector_dimension: Option<usize>,
    max_connections: u32,
    hnsw_config: Option<&crate::data::storage::vector::HnswConfig>,
) -> Result<Arc<dyn Backend + Send + Sync>> {
    use dripstone::backend::postgres::{PostgresBackend, PostgresBackendOptions};

    let hnsw_index = hnsw_config
        .map(|cfg| {
            use dripstone::backend::postgres::HnswIndexOptions;
            Ok::<_, anyhow::Error>(HnswIndexOptions {
                m: Some(postgres_i32(cfg.m, "hnsw.m")?),
                ef_construction: Some(postgres_i32(cfg.ef_construction, "hnsw.ef_construction")?),
            })
        })
        .transpose()?;

    let options = PostgresBackendOptions {
        table_name: table_name.to_string(),
        ensure_schema: true,
        max_connections,
        vector_dimension: vector_dimension
            .map(|dimension| postgres_i32(dimension, "vector_dimension"))
            .transpose()?,
        hnsw_index,
    };

    let backend = PostgresBackend::with_options(connection_string, options).await?;
    Ok(Arc::new(backend) as Arc<dyn Backend + Send + Sync>)
}

#[cfg(feature = "vector-store-dynamodb")]
async fn build_dynamodb_backend(
    table_name: &str,
    index_name: &str,
    region: Option<&str>,
    endpoint_url: Option<&str>,
    dimension: usize,
    ensure_table: bool,
) -> Result<Arc<dyn Backend + Send + Sync>> {
    use dripstone::{DynamoDbVectorBackend, DynamoDbVectorConfig, Metric};

    let creds = EnvAwsCredentials::from_env();
    let mut config =
        DynamoDbVectorConfig::new(table_name.to_string(), index_name.to_string(), dimension)
            .with_distance_metric(Metric::Cosine)
            .with_ensure_table(ensure_table)
            .with_ensure_index(ensure_table);
    if let Some(region) = region {
        config = config.with_region(region);
    }
    if let Some(endpoint) = endpoint_url {
        config = config.with_endpoint_url(endpoint);
    }
    if let (Some(access_key), Some(secret_key)) = (creds.access_key_id, creds.secret_access_key) {
        config = config.with_credentials(access_key, secret_key, creds.session_token);
    }
    let backend = DynamoDbVectorBackend::new(config).await?;
    Ok(Arc::new(backend) as Arc<dyn Backend + Send + Sync>)
}

#[cfg(feature = "vector-store-mmap")]
fn build_mmap_backend(
    file_path: &str,
    dimension: usize,
    read_write: bool,
    flush_every_ops: usize,
    vector_encoding: Option<&dripstone::MmapVectorEncoding>,
) -> Result<Arc<dyn Backend + Send + Sync>> {
    use dripstone::{MmapBackend, MmapBackendConfig};

    let mut config = MmapBackendConfig::new(file_path.to_string(), dimension)
        .with_read_write(read_write)
        .with_flush_every(flush_every_ops);
    if let Some(vector_encoding) = vector_encoding {
        config = config.with_vector_encoding(*vector_encoding);
    }
    let backend = MmapBackend::new(config)?;
    Ok(Arc::new(backend) as Arc<dyn Backend + Send + Sync>)
}

#[cfg(feature = "vector-memory")]
async fn build_indexed_backend(
    inner: &VectorBackendConfig,
    dimension: usize,
    hnsw_config: Option<&crate::data::storage::vector::HnswConfig>,
    config: Option<&dripstone::IndexedBackendConfig>,
) -> Result<Arc<dyn Backend + Send + Sync>> {
    use dripstone::{HnswIndex, IndexedBackend};

    use crate::data::storage::vector::indexed::indexed_backend_config_from_env;

    let inner_backend = inner.build_backend().await?;
    let inner_backend: Arc<dyn Backend> = inner_backend;
    let index = Box::new(HnswIndex::new(
        hnsw_config.cloned().unwrap_or_default(),
        dimension,
    ));
    let backend = IndexedBackend::new(
        inner_backend,
        index,
        config
            .cloned()
            .unwrap_or_else(indexed_backend_config_from_env),
    );
    Ok(Arc::new(backend) as Arc<dyn Backend + Send + Sync>)
}

#[cfg(feature = "vector-memory")]
async fn build_full_text_indexed_backend(
    inner: &VectorBackendConfig,
    config: Option<&dripstone::FullTextIndexedBackendConfig>,
) -> Result<Arc<dyn Backend + Send + Sync>> {
    use dripstone::{FullTextIndexedBackend, FullTextIndexedBackendConfig, InMemoryFullTextIndex};

    let inner_backend = inner.build_backend().await?;
    let inner_backend: Arc<dyn Backend> = inner_backend;
    let backend = FullTextIndexedBackend::new(
        inner_backend,
        Arc::new(InMemoryFullTextIndex::new()),
        config
            .cloned()
            .unwrap_or_else(FullTextIndexedBackendConfig::default),
    );
    Ok(Arc::new(backend) as Arc<dyn Backend + Send + Sync>)
}

#[cfg(feature = "vector-memory")]
async fn build_tiered_backend(
    tiers: &[(VectorBackendConfig, dripstone::TierConfig)],
    write_strategy: Option<&dripstone::WriteStrategy>,
    search_strategy: Option<&dripstone::SearchStrategy>,
    background_sync: bool,
    sync_interval_secs: u64,
) -> Result<Arc<dyn Backend + Send + Sync>> {
    use dripstone::TieredChainBuilder;

    let mut builder = TieredChainBuilder::new();
    for (config, tier_config) in tiers {
        let backend = config.build_backend().await?;
        builder = builder.tier(backend, tier_config.clone());
    }
    if let Some(ws) = write_strategy {
        builder = builder.write_strategy(*ws);
    }
    if let Some(ss) = search_strategy {
        builder = builder.search_strategy(*ss);
    }
    builder = builder.background_sync(background_sync);
    builder = builder.sync_interval_secs(sync_interval_secs);
    let backend = builder
        .build()
        .map_err(|e| anyhow!("tiered backend build failed: {e}"))?;
    Ok(backend as Arc<dyn Backend + Send + Sync>)
}

#[cfg(all(test, feature = "vector-store-postgres"))]
mod tests {
    use super::postgres_i32;

    #[test]
    fn postgres_i32_accepts_boundary_and_rejects_overflow() {
        assert_eq!(postgres_i32(i32::MAX as usize, "field").unwrap(), i32::MAX);
        assert!(postgres_i32((i32::MAX as usize) + 1, "field").is_err());
    }
}
