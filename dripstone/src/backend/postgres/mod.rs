use async_trait::async_trait;
use pgvector::Vector;
use sqlx::postgres::{PgPoolOptions, PgRow};
use sqlx::{Pool, Postgres, Row};

use crate::errors::DripstoneError;

const DEFAULT_MAX_CONNECTIONS: u32 = 5;
const MAX_OVERSAMPLE: usize = 1024;

fn usize_to_i32(value: usize, field: &str) -> Result<i32, DripstoneError> {
    i32::try_from(value)
        .map_err(|_| DripstoneError::storage(format!("{field} exceeds PostgreSQL INTEGER range")))
}

fn i32_to_usize(value: i32, field: &str) -> Result<usize, DripstoneError> {
    usize::try_from(value)
        .map_err(|_| DripstoneError::storage(format!("{field} is negative or out of range")))
}

fn quote_table_name(raw: &str) -> Result<String, DripstoneError> {
    if raw.is_empty() {
        return Err(DripstoneError::Misconfiguration(
            "table name cannot be empty".to_string(),
        ));
    }

    let mut segments = Vec::new();
    for part in raw.split('.') {
        if part.is_empty() {
            return Err(DripstoneError::Misconfiguration(
                "table name contains empty identifier segments".to_string(),
            ));
        }
        let valid = part
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_');
        if !valid {
            return Err(DripstoneError::Misconfiguration(
                "table name may only contain alphanumeric characters and underscores".to_string(),
            ));
        }
        let escaped = part.replace('"', "\"\"");
        segments.push(format!("\"{escaped}\""));
    }

    Ok(segments.join("."))
}

fn identifier_basename(raw: &str) -> String {
    raw.rsplit('.').next().map_or_else(
        || raw.replace('"', "_"),
        |segment| segment.replace('"', "_"),
    )
}

pub struct PostgresBackendOptions {
    pub table_name: String,
    pub ensure_schema: bool,
    pub max_connections: u32,
    pub vector_dimension: Option<i32>,
    pub hnsw_index: Option<HnswIndexOptions>,
}

#[derive(Clone, Debug)]
pub struct HnswIndexOptions {
    pub m: Option<i32>,
    pub ef_construction: Option<i32>,
}

impl Default for HnswIndexOptions {
    fn default() -> Self {
        Self {
            m: Some(16),
            ef_construction: Some(64),
        }
    }
}

impl Default for PostgresBackendOptions {
    fn default() -> Self {
        Self {
            table_name: "embeddings".to_string(),
            ensure_schema: true,
            max_connections: DEFAULT_MAX_CONNECTIONS,
            vector_dimension: None,
            hnsw_index: None,
        }
    }
}

// =============================================================================
// Backend trait implementation
// =============================================================================

use crate::core::{Capabilities, Chunk, SearchQuery, SearchResult};
use crate::search::filter_to_sql::{FilterSql, filter_to_sql};
use crate::traits::Backend;

/// Postgres backend for the new `Dripstone` API.
#[derive(Debug)]
pub struct PostgresBackend {
    pool: Pool<Postgres>,
    table_sql: String,
    qualified_name: String,
}

impl PostgresBackend {
    /// Create a new Postgres backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn connect(
        connection_string: &str,
        table_name: &str,
    ) -> Result<Self, DripstoneError> {
        Self::with_options(
            connection_string,
            PostgresBackendOptions {
                table_name: table_name.to_string(),
                ..Default::default()
            },
        )
        .await
    }

    /// Create a new Postgres backend with custom options.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn with_options(
        connection_string: &str,
        options: PostgresBackendOptions,
    ) -> Result<Self, DripstoneError> {
        if connection_string.is_empty() {
            return Err(DripstoneError::configuration(
                "postgres connection string cannot be empty",
            ));
        }

        let table_sql = quote_table_name(&options.table_name)?;
        let qualified_name = options.table_name.clone();

        let max_connections = if options.max_connections == 0 {
            DEFAULT_MAX_CONNECTIONS
        } else {
            options.max_connections
        };

        let pool = PgPoolOptions::new()
            .max_connections(max_connections)
            .connect(connection_string)
            .await
            .map_err(DripstoneError::from)?;

        let backend = Self {
            pool,
            table_sql,
            qualified_name,
        };

        if options.ensure_schema {
            backend.ensure_schema(options.vector_dimension).await?;
            if let Some(hnsw_opts) = options.hnsw_index {
                backend.create_hnsw_index(hnsw_opts).await?;
            }
        }

        Ok(backend)
    }

    async fn ensure_schema(&self, vector_dimension: Option<i32>) -> Result<(), DripstoneError> {
        let vector_type = match vector_dimension {
            Some(dimension) if dimension > 0 => format!("vector({dimension})"),
            Some(_) => {
                return Err(DripstoneError::configuration(
                    "vector dimension must be a positive integer",
                ));
            }
            None => "vector".to_string(),
        };

        let create_extension = "CREATE EXTENSION IF NOT EXISTS vector";
        sqlx::query(create_extension)
            .execute(&self.pool)
            .await
            .map_err(DripstoneError::from)?;

        // Create table with content field for Chunk storage
        let create_table = format!(
            "CREATE TABLE IF NOT EXISTS {table} (
                namespace TEXT NOT NULL DEFAULT 'default',
                id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                content TEXT NOT NULL,
                vector {vector_type} NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (namespace, id)
            )",
            table = self.table_sql,
            vector_type = vector_type
        );

        sqlx::query(&create_table)
            .execute(&self.pool)
            .await
            .map_err(DripstoneError::from)?;

        // Create indexes
        let doc_index = format!(
            "CREATE INDEX IF NOT EXISTS {name}_doc_id_idx ON {table} (namespace, doc_id)",
            name = identifier_basename(&self.qualified_name),
            table = self.table_sql,
        );
        sqlx::query(&doc_index)
            .execute(&self.pool)
            .await
            .map_err(DripstoneError::from)?;

        Ok(())
    }

    async fn create_hnsw_index(&self, options: HnswIndexOptions) -> Result<(), DripstoneError> {
        let index_name = format!(
            "{}_vector_hnsw_idx",
            identifier_basename(&self.qualified_name)
        );

        let mut opts_parts = Vec::new();
        if let Some(m) = options.m {
            opts_parts.push(format!("m = {m}"));
        }
        if let Some(ef_construction) = options.ef_construction {
            opts_parts.push(format!("ef_construction = {ef_construction}"));
        }

        let opts_clause = if opts_parts.is_empty() {
            String::new()
        } else {
            format!(" WITH ({})", opts_parts.join(", "))
        };

        let create_index = format!(
            "CREATE INDEX IF NOT EXISTS {index_name}
             ON {table} USING hnsw (vector vector_cosine_ops){opts}",
            index_name = index_name,
            table = self.table_sql,
            opts = opts_clause
        );

        sqlx::query(&create_index)
            .execute(&self.pool)
            .await
            .map_err(DripstoneError::from)?;

        Ok(())
    }

    fn row_to_chunk(row: &PgRow) -> Result<Chunk, DripstoneError> {
        let id: String = row.get("id");
        let doc_id: String = row.get("doc_id");
        let content: String = row.get("content");
        let vector: Vector = row.get("vector");
        let start_offset: i32 = row.get("start_offset");
        let end_offset: i32 = row.get("end_offset");
        let chunk_index: i32 = row.get("chunk_index");
        let total_chunks: i32 = row.get("total_chunks");
        let metadata: Option<serde_json::Value> = row.get("metadata");
        let namespace: String = row.get("namespace");

        Ok(Chunk {
            id,
            doc_id,
            content,
            vector: vector.to_vec().into(),
            start_offset: i32_to_usize(start_offset, "start_offset")?,
            end_offset: i32_to_usize(end_offset, "end_offset")?,
            chunk_index: i32_to_usize(chunk_index, "chunk_index")?,
            total_chunks: i32_to_usize(total_chunks, "total_chunks")?,
            metadata: metadata.unwrap_or_else(|| serde_json::json!({})),
            namespace: if namespace == "default" {
                None
            } else {
                Some(namespace)
            },
            multi_vectors: None,
        })
    }
}

#[async_trait]
impl Backend for PostgresBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true,
            requires_exact_filter_pass: true,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: Some(2000), // pgvector limit
            max_batch: Some(500),
            supports_streaming: false,
        }
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        const BATCH_SIZE: usize = 500;
        if chunks.is_empty() {
            return Ok(());
        }

        let mut transaction = self.pool.begin().await.map_err(DripstoneError::from)?;

        for batch in chunks.chunks(BATCH_SIZE) {
            let mut namespaces = Vec::with_capacity(batch.len());
            let mut ids = Vec::with_capacity(batch.len());
            let mut doc_ids = Vec::with_capacity(batch.len());
            let mut contents = Vec::with_capacity(batch.len());
            let mut vectors = Vec::with_capacity(batch.len());
            let mut start_offsets = Vec::with_capacity(batch.len());
            let mut end_offsets = Vec::with_capacity(batch.len());
            let mut chunk_indexes = Vec::with_capacity(batch.len());
            let mut total_chunks_vec = Vec::with_capacity(batch.len());
            let mut metadatas = Vec::with_capacity(batch.len());

            for chunk in batch {
                namespaces.push(chunk.namespace.as_deref().unwrap_or("default").to_string());
                ids.push(chunk.id.clone());
                doc_ids.push(chunk.doc_id.clone());
                contents.push(chunk.content.clone());
                vectors.push(Vector::from(chunk.vector.to_vec()));
                start_offsets.push(usize_to_i32(chunk.start_offset, "start_offset")?);
                end_offsets.push(usize_to_i32(chunk.end_offset, "end_offset")?);
                chunk_indexes.push(usize_to_i32(chunk.chunk_index, "chunk_index")?);
                total_chunks_vec.push(usize_to_i32(chunk.total_chunks, "total_chunks")?);
                metadatas.push(chunk.metadata.clone());
            }

            let sql = format!(
                "INSERT INTO {table} (namespace, id, doc_id, content, vector, start_offset, end_offset, chunk_index, total_chunks, metadata)
                 SELECT * FROM UNNEST($1::text[], $2::text[], $3::text[], $4::text[], $5::vector[], $6::integer[], $7::integer[], $8::integer[], $9::integer[], $10::jsonb[])
                 ON CONFLICT (namespace, id) DO UPDATE SET
                    doc_id = EXCLUDED.doc_id,
                    content = EXCLUDED.content,
                    vector = EXCLUDED.vector,
                    start_offset = EXCLUDED.start_offset,
                    end_offset = EXCLUDED.end_offset,
                    chunk_index = EXCLUDED.chunk_index,
                    total_chunks = EXCLUDED.total_chunks,
                    metadata = EXCLUDED.metadata",
                table = self.table_sql
            );

            sqlx::query(&sql)
                .bind(&namespaces)
                .bind(&ids)
                .bind(&doc_ids)
                .bind(&contents)
                .bind(&vectors)
                .bind(&start_offsets)
                .bind(&end_offsets)
                .bind(&chunk_indexes)
                .bind(&total_chunks_vec)
                .bind(&metadatas)
                .execute(&mut *transaction)
                .await
                .map_err(DripstoneError::from)?;
        }

        transaction.commit().await.map_err(DripstoneError::from)?;
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let ns = namespace.unwrap_or("default");
        let sql = format!(
            "SELECT * FROM {table} WHERE namespace = $1 AND id = $2",
            table = self.table_sql
        );

        let row = sqlx::query(&sql)
            .bind(ns)
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(DripstoneError::from)?;

        match row {
            Some(r) => Ok(Some(Self::row_to_chunk(&r)?)),
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let ns = namespace.unwrap_or("default");
        let sql = format!(
            "DELETE FROM {table} WHERE namespace = $1 AND id = $2",
            table = self.table_sql
        );

        sqlx::query(&sql)
            .bind(ns)
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(DripstoneError::from)?;

        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let ns = namespace.unwrap_or("default");
        let sql = format!(
            "DELETE FROM {table} WHERE namespace = $1 AND doc_id = $2",
            table = self.table_sql
        );

        let result = sqlx::query(&sql)
            .bind(ns)
            .bind(doc_id)
            .execute(&self.pool)
            .await
            .map_err(DripstoneError::from)?;

        usize::try_from(result.rows_affected())
            .map_err(|_| DripstoneError::storage("deleted row count exceeds addressable size"))
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let ns = query.namespace.as_deref().unwrap_or("default");
        let vector = Vector::from(query.vector.to_vec());

        let (filter_clause, filter_params, use_oversample) = query.filter.as_ref().map_or_else(
            || (String::new(), vec![], true),
            |filter| match filter_to_sql(filter, 4) {
                Some(FilterSql { clause, params }) => (clause, params, false),
                None => (String::new(), vec![], true),
            },
        );

        let fetch_limit = if use_oversample {
            query
                .limit
                .saturating_mul(2)
                .min(MAX_OVERSAMPLE)
                .max(query.limit)
        } else {
            query.limit
        };

        let sql = format!(
            "SELECT *, vector <=> $2 as distance
             FROM {table}
             WHERE namespace = $1{filter_clause}
             ORDER BY distance ASC
             LIMIT $3",
            table = self.table_sql,
            filter_clause = filter_clause
        );

        let mut q =
            sqlx::query(&sql)
                .bind(ns)
                .bind(&vector)
                .bind(i64::try_from(fetch_limit).map_err(|_| {
                    DripstoneError::storage("PostgreSQL search limit exceeds BIGINT range")
                })?);

        for p in &filter_params {
            q = q.bind(p);
        }

        let rows = q
            .fetch_all(&self.pool)
            .await
            .map_err(DripstoneError::from)?;

        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            let chunk = Self::row_to_chunk(&row)?;

            if use_oversample
                && let Some(ref filter) = query.filter
                && !filter.matches(&chunk.metadata)
            {
                continue;
            }

            let score = query.metric.compute(&query.vector, &chunk.vector)?;
            results.push(SearchResult::new(chunk, score, results.len()));

            if results.len() >= query.limit {
                break;
            }
        }

        if query.metric.higher_is_better() {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            results.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }

        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let sql = format!(
            "SELECT DISTINCT namespace FROM {table} ORDER BY namespace ASC",
            table = self.table_sql
        );

        let rows = sqlx::query_scalar::<_, String>(&sql)
            .fetch_all(&self.pool)
            .await
            .map_err(DripstoneError::from)?;

        Ok(rows)
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let sql = format!(
            "DELETE FROM {table} WHERE namespace = $1",
            table = self.table_sql
        );

        sqlx::query(&sql)
            .bind(namespace)
            .execute(&self.pool)
            .await
            .map_err(DripstoneError::from)?;

        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let (sql, bind_ns) = namespace.map_or_else(
            || {
                (
                    format!("SELECT COUNT(*) FROM {table}", table = self.table_sql),
                    None,
                )
            },
            |ns| {
                (
                    format!(
                        "SELECT COUNT(*) FROM {table} WHERE namespace = $1",
                        table = self.table_sql
                    ),
                    Some(ns),
                )
            },
        );

        let count: i64 = match bind_ns {
            Some(ns) => {
                sqlx::query_scalar(&sql)
                    .bind(ns)
                    .fetch_one(&self.pool)
                    .await
            }
            None => sqlx::query_scalar(&sql).fetch_one(&self.pool).await,
        }
        .map_err(DripstoneError::from)?;

        usize::try_from(count)
            .map_err(|_| DripstoneError::storage("PostgreSQL count is negative or out of range"))
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let limit_clause = limit.map(|l| format!("LIMIT {l}")).unwrap_or_default();
        let (sql, bind_ns) = namespace.map_or_else(
            || {
                (
                    format!(
                        "SELECT id, doc_id, content, vector, start_offset, end_offset, chunk_index, total_chunks, metadata, namespace FROM {table} ORDER BY id {limit}",
                        table = self.table_sql,
                        limit = limit_clause
                    ),
                    None,
                )
            },
            |ns| {
                (
                format!(
                    "SELECT id, doc_id, content, vector, start_offset, end_offset, chunk_index, total_chunks, metadata, namespace FROM {table} WHERE namespace = $1 ORDER BY id {limit}",
                    table = self.table_sql,
                    limit = limit_clause
                ),
                Some(ns),
                )
            },
        );

        let rows = match bind_ns {
            Some(ns) => sqlx::query(&sql).bind(ns).fetch_all(&self.pool).await?,
            None => sqlx::query(&sql).fetch_all(&self.pool).await?,
        };

        let mut chunks = Vec::with_capacity(rows.len());
        for row in rows {
            let vector: Vector = row.get(3);
            chunks.push(Chunk {
                id: row.get(0),
                doc_id: row.get(1),
                content: row.get(2),
                vector: vector.to_vec().into(),
                start_offset: i32_to_usize(row.get(4), "start_offset")?,
                end_offset: i32_to_usize(row.get(5), "end_offset")?,
                chunk_index: i32_to_usize(row.get(6), "chunk_index")?,
                total_chunks: i32_to_usize(row.get(7), "total_chunks")?,
                metadata: row.get(8),
                namespace: row.get(9),
                multi_vectors: None,
            });
        }

        Ok(chunks)
    }
}
