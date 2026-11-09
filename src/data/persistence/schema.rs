#[cfg(feature = "tools-rag")]
use crate::data::storage::vector::Backend;
#[cfg(feature = "tools-rag")]
use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest};
#[cfg(any(feature = "sqlx", feature = "tools-rag"))]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "tools-rag")]
use anyhow::ensure;
use async_trait::async_trait;
#[cfg(feature = "tools-rag")]
use dripstone::Chunk;
use serde_json::{Value, json};
#[cfg(feature = "sqlx")]
use sqlx::{PgPool, Row};
#[cfg(feature = "sqlx")]
use std::collections::BTreeMap;
#[cfg(feature = "tools-rag")]
use std::collections::BTreeSet;
#[cfg(feature = "tools-rag")]
use std::sync::Arc;

/// Normalized representation of a database schema.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DatabaseSchema {
    pub database: String,
    pub tables: Vec<TableSchema>,
}

impl DatabaseSchema {
    #[must_use]
    pub const fn table_count(&self) -> usize {
        self.tables.len()
    }

    #[must_use]
    pub fn column_count(&self) -> usize {
        self.tables.iter().map(|table| table.columns.len()).sum()
    }

    #[must_use]
    pub fn to_documents(&self) -> Vec<SchemaDocument> {
        self.tables
            .iter()
            .map(|table| {
                let summary = table.render_summary(&self.database);
                SchemaDocument {
                    id: format!("{}::{}", self.database, table.qualified_name()),
                    text: summary.clone(),
                    metadata: json!({
                        "database": self.database,
                        "schema": table.schema,
                        "table": table.name,
                        "primary_key": table.primary_key,
                        "column_count": table.columns.len(),
                        "summary": summary,
                    }),
                }
            })
            .collect()
    }
}

/// Representation of a table with normalized metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TableSchema {
    pub schema: String,
    pub name: String,
    pub columns: Vec<ColumnSchema>,
    pub primary_key: Vec<String>,
    pub foreign_keys: Vec<ForeignKeySchema>,
    pub indexes: Vec<IndexSchema>,
    pub comment: Option<String>,
}

impl TableSchema {
    #[must_use]
    pub fn qualified_name(&self) -> String {
        format!("{}.{}", self.schema, self.name)
    }

    fn render_summary(&self, database: &str) -> String {
        let mut lines = Vec::new();

        lines.push(format!(
            "Database: {database}\nTable: {}\nDescription: {}",
            self.qualified_name(),
            self.comment.as_deref().map_or("N/A", str::trim)
        ));

        if self.primary_key.is_empty() {
            lines.push("Primary Key: None (consider identifying one)".to_string());
        } else {
            lines.push(format!(
                "Primary Key: {}",
                self.primary_key
                    .iter()
                    .map(|col| format!("\"{col}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        lines.push("Columns:".to_string());
        for column in &self.columns {
            lines.push(format!(
                "  - {name} {data_type}{nullability}{default}{comment}",
                name = column.name,
                data_type = column.data_type,
                nullability = if column.is_nullable { "" } else { " NOT NULL" },
                default = column
                    .default
                    .as_ref()
                    .map_or(String::new(), |def| format!(" DEFAULT {def}")),
                comment = column
                    .comment
                    .as_ref()
                    .map_or(String::new(), |c| format!(" -- {}", c.trim())),
            ));
        }

        if self.foreign_keys.is_empty() {
            lines.push("Foreign Keys: None".to_string());
        } else {
            lines.push("Foreign Keys:".to_string());
            for fk in &self.foreign_keys {
                let columns = fk
                    .columns
                    .iter()
                    .map(|c| format!("\"{c}\""))
                    .collect::<Vec<_>>()
                    .join(", ");
                let target_table = format!("{}.{}", fk.referenced_schema, fk.referenced_table);
                let target_columns = fk
                    .referenced_columns
                    .iter()
                    .map(|c| format!("\"{c}\""))
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(format!(
                    "  - {columns} -> {target_table}({target_columns}) ON UPDATE {on_update} ON DELETE {on_delete}",
                    columns = columns,
                    target_table = target_table,
                    target_columns = target_columns,
                    on_update = fk.on_update.as_deref().unwrap_or("NO ACTION"),
                    on_delete = fk.on_delete.as_deref().unwrap_or("NO ACTION"),
                ));
            }
        }

        if self.indexes.is_empty() {
            lines.push("Indexes: None".to_string());
        } else {
            lines.push("Indexes:".to_string());
            for index in &self.indexes {
                let columns = index
                    .columns
                    .iter()
                    .map(|c| format!("\"{c}\""))
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(format!(
                    "  - {name}: {columns}{}",
                    if index.is_unique { " (UNIQUE)" } else { "" },
                    name = index.name.as_deref().unwrap_or("unnamed"),
                    columns = columns,
                ));
            }
        }

        lines.join("\n")
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColumnSchema {
    pub name: String,
    pub data_type: String,
    pub is_nullable: bool,
    pub default: Option<String>,
    pub comment: Option<String>,
    /// Maximum length for bounded string/binary types (`VARCHAR(n)`, `CHAR(n)`,
    /// etc.). `None` when the backend does not expose it or the column has no
    /// intrinsic length limit (numeric types, `TEXT`, `SQLite`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<i64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ForeignKeySchema {
    pub constraint_name: Option<String>,
    pub columns: Vec<String>,
    pub referenced_schema: String,
    pub referenced_table: String,
    pub referenced_columns: Vec<String>,
    pub on_update: Option<String>,
    pub on_delete: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexSchema {
    pub name: Option<String>,
    pub columns: Vec<String>,
    pub is_unique: bool,
}

/// Simple struct used to load table summaries into a vector store.
#[derive(Debug, Clone)]
pub struct SchemaDocument {
    pub id: String,
    pub text: String,
    pub metadata: Value,
}

/// Extract schema metadata from a database backend.
#[async_trait]
pub trait DatabaseIntrospector: Send + Sync {
    async fn introspect(&self) -> Result<DatabaseSchema>;
}

/// Introspector implementation for `PostgreSQL`.
#[cfg(feature = "sqlx")]
pub struct PostgresSchemaIntrospector {
    pool: PgPool,
    database_name: String,
}

#[cfg(feature = "sqlx")]
impl PostgresSchemaIntrospector {
    pub fn new(pool: PgPool, database_name: impl Into<String>) -> Self {
        Self {
            pool,
            database_name: database_name.into(),
        }
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await
            .context("Failed to connect while building schema introspector")?;

        let db_name = sqlx::query_scalar::<_, String>("SELECT current_database()")
            .fetch_one(&pool)
            .await
            .context("Failed to resolve database name")?;

        Ok(Self::new(pool, db_name))
    }

    async fn populate_columns_and_primary_keys(
        &self,
        tables: &mut BTreeMap<(String, String), TableSchema>,
    ) -> Result<()> {
        let columns = sqlx::query(POSTGRES_COLUMNS_QUERY)
            .fetch_all(&self.pool)
            .await
            .context("Failed to fetch column metadata")?;
        for row in columns {
            let key = (
                row.try_get::<String, _>("table_schema")?.to_lowercase(),
                row.try_get::<String, _>("table_name")?.to_lowercase(),
            );
            if let Some(table) = tables.get_mut(&key) {
                table.columns.push(ColumnSchema {
                    name: row.try_get::<String, _>("column_name")?.to_lowercase(),
                    data_type: row.try_get::<String, _>("data_type")?.to_lowercase(),
                    is_nullable: row
                        .try_get::<String, _>("is_nullable")?
                        .eq_ignore_ascii_case("YES"),
                    default: row.try_get("column_default")?,
                    comment: row.try_get("comment")?,
                    max_length: row
                        .try_get::<Option<i32>, _>("character_maximum_length")?
                        .map(i64::from),
                });
            }
        }
        let primary_keys = sqlx::query(POSTGRES_PRIMARY_KEYS_QUERY)
            .fetch_all(&self.pool)
            .await
            .context("Failed to fetch primary keys")?;
        for row in primary_keys {
            let key = (
                row.try_get::<String, _>("table_schema")?.to_lowercase(),
                row.try_get::<String, _>("table_name")?.to_lowercase(),
            );
            if let Some(table) = tables.get_mut(&key) {
                table
                    .primary_key
                    .push(row.try_get::<String, _>("column_name")?.to_lowercase());
            }
        }
        Ok(())
    }

    async fn load_tables(&self) -> Result<BTreeMap<(String, String), TableSchema>> {
        let tables = sqlx::query(POSTGRES_TABLES_QUERY)
            .fetch_all(&self.pool)
            .await
            .context("Failed to fetch table list")?;
        let mut tables_map: BTreeMap<(String, String), TableSchema> = BTreeMap::new();
        for row in tables {
            Self::apply_table_row(&mut tables_map, &row)?;
        }
        Ok(tables_map)
    }

    fn apply_table_row(
        tables_map: &mut BTreeMap<(String, String), TableSchema>,
        row: &sqlx::postgres::PgRow,
    ) -> Result<()> {
        let schema: String = row.try_get("table_schema")?;
        let name: String = row.try_get("table_name")?;
        let comment: Option<String> = row.try_get("comment")?;
        let normalized_schema = schema.to_lowercase();
        let normalized_name = name.to_lowercase();
        tables_map.insert(
            (normalized_schema.clone(), normalized_name.clone()),
            TableSchema {
                schema: normalized_schema,
                name: normalized_name,
                columns: Vec::new(),
                primary_key: Vec::new(),
                foreign_keys: Vec::new(),
                indexes: Vec::new(),
                comment,
            },
        );
        Ok(())
    }

    async fn populate_foreign_keys(
        &self,
        tables_map: &mut BTreeMap<(String, String), TableSchema>,
    ) -> Result<()> {
        // Use `pg_catalog.pg_constraint.conkey` / `confkey` to pair
        // referencing and referenced columns positionally. The prior
        // `information_schema.constraint_column_usage` join produced a
        // Cartesian product on composite foreign keys because it has no
        // ordinal position column to join on.
        let foreign_keys = sqlx::query(POSTGRES_FOREIGN_KEYS_QUERY)
            .fetch_all(&self.pool)
            .await
            .context("Failed to fetch foreign keys")?;
        let mut fk_map: BTreeMap<(String, String, String), ForeignKeySchema> = BTreeMap::new();
        for row in foreign_keys {
            Self::collect_foreign_key_row(&mut fk_map, &row)?;
        }
        for ((schema, table_name, _), fk) in fk_map {
            if let Some(table) = tables_map.get_mut(&(schema, table_name)) {
                table.foreign_keys.push(fk);
            }
        }
        Ok(())
    }

    fn collect_foreign_key_row(
        fk_map: &mut BTreeMap<(String, String, String), ForeignKeySchema>,
        row: &sqlx::postgres::PgRow,
    ) -> Result<()> {
        let constraint_name: Option<String> = row.try_get("constraint_name")?;
        let schema: String = row.try_get("table_schema")?;
        let table_name: String = row.try_get("table_name")?;
        let column_name: String = row.try_get("column_name")?;
        let foreign_schema: String = row.try_get("foreign_schema")?;
        let foreign_table: String = row.try_get("foreign_table")?;
        let foreign_column: String = row.try_get("foreign_column")?;
        let update_rule: Option<String> = row.try_get("update_rule")?;
        let delete_rule: Option<String> = row.try_get("delete_rule")?;
        let schema_lc = schema.to_lowercase();
        let table_lc = table_name.to_lowercase();
        let fk_key_name = constraint_name
            .as_deref()
            .map_or_else(|| format!("{schema_lc}_{table_lc}_fk"), str::to_string);
        let key = (schema_lc, table_lc, fk_key_name);
        let entry = fk_map.entry(key).or_insert_with(|| ForeignKeySchema {
            constraint_name,
            columns: Vec::new(),
            referenced_schema: foreign_schema.to_lowercase(),
            referenced_table: foreign_table.to_lowercase(),
            referenced_columns: Vec::new(),
            on_update: update_rule.map(|r| r.to_uppercase()),
            on_delete: delete_rule.map(|r| r.to_uppercase()),
        });
        entry.columns.push(column_name.to_lowercase());
        entry.referenced_columns.push(foreign_column.to_lowercase());
        Ok(())
    }

    async fn populate_indexes(
        &self,
        tables_map: &mut BTreeMap<(String, String), TableSchema>,
    ) -> Result<()> {
        let indexes = sqlx::query(POSTGRES_INDEXES_QUERY)
            .fetch_all(&self.pool)
            .await
            .context("Failed to fetch indexes")?;
        for row in indexes {
            Self::apply_index_row(tables_map, &row)?;
        }
        Ok(())
    }

    fn apply_index_row(
        tables_map: &mut BTreeMap<(String, String), TableSchema>,
        row: &sqlx::postgres::PgRow,
    ) -> Result<()> {
        let schema: String = row.try_get("schemaname")?;
        let table_name: String = row.try_get("tablename")?;
        let index_name: Option<String> = row.try_get("indexname")?;
        let index_def: String = row.try_get("indexdef")?;
        if let Some(table) = tables_map.get_mut(&(schema.to_lowercase(), table_name.to_lowercase()))
        {
            table.indexes.push(IndexSchema {
                name: index_name.map(|name| name.to_lowercase()),
                columns: parse_index_columns(&index_def),
                is_unique: index_def.to_uppercase().contains("UNIQUE"),
            });
        }
        Ok(())
    }
}

#[cfg(feature = "sqlx")]
const POSTGRES_TABLES_QUERY: &str = r#"
            SELECT
                table_schema,
                table_name,
                obj_description(('"' || table_schema || '"."' || table_name || '"')::regclass) AS comment
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
            "#;

#[cfg(feature = "sqlx")]
const POSTGRES_COLUMNS_QUERY: &str = r"
            SELECT
                cols.table_schema,
                cols.table_name,
                cols.column_name,
                cols.data_type,
                cols.is_nullable,
                cols.column_default,
                cols.character_maximum_length,
                pgd.description AS comment
            FROM information_schema.columns cols
            LEFT JOIN pg_catalog.pg_namespace n ON n.nspname = cols.table_schema
            LEFT JOIN pg_catalog.pg_class c ON c.relname = cols.table_name AND c.relnamespace = n.oid
            LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attname = cols.column_name
            LEFT JOIN pg_catalog.pg_description pgd ON pgd.objoid = a.attrelid AND pgd.objsubid = a.attnum
            WHERE cols.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY cols.table_schema, cols.table_name, cols.ordinal_position
            ";

#[cfg(feature = "sqlx")]
const POSTGRES_PRIMARY_KEYS_QUERY: &str = r"
            SELECT
                kc.table_schema,
                kc.table_name,
                kc.column_name,
                kc.ordinal_position
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kc
                ON kc.constraint_name = tc.constraint_name
               AND kc.constraint_schema = tc.constraint_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND kc.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY kc.table_schema, kc.table_name, kc.ordinal_position
            ";

#[cfg(feature = "sqlx")]
const POSTGRES_FOREIGN_KEYS_QUERY: &str = r"
            SELECT
                con.conname                                   AS constraint_name,
                tc_ns.nspname                                 AS table_schema,
                tc_cls.relname                                AS table_name,
                tc_att.attname                                AS column_name,
                fc_ns.nspname                                 AS foreign_schema,
                fc_cls.relname                                AS foreign_table,
                fc_att.attname                                AS foreign_column,
                CASE con.confupdtype
                    WHEN 'a' THEN 'NO ACTION'
                    WHEN 'r' THEN 'RESTRICT'
                    WHEN 'c' THEN 'CASCADE'
                    WHEN 'n' THEN 'SET NULL'
                    WHEN 'd' THEN 'SET DEFAULT'
                    ELSE NULL
                END                                           AS update_rule,
                CASE con.confdeltype
                    WHEN 'a' THEN 'NO ACTION'
                    WHEN 'r' THEN 'RESTRICT'
                    WHEN 'c' THEN 'CASCADE'
                    WHEN 'n' THEN 'SET NULL'
                    WHEN 'd' THEN 'SET DEFAULT'
                    ELSE NULL
                END                                           AS delete_rule,
                u.ord                                         AS ordinal_position
            FROM pg_catalog.pg_constraint AS con
            JOIN pg_catalog.pg_class     AS tc_cls ON tc_cls.oid = con.conrelid
            JOIN pg_catalog.pg_namespace AS tc_ns  ON tc_ns.oid  = tc_cls.relnamespace
            JOIN pg_catalog.pg_class     AS fc_cls ON fc_cls.oid = con.confrelid
            JOIN pg_catalog.pg_namespace AS fc_ns  ON fc_ns.oid  = fc_cls.relnamespace
            JOIN LATERAL unnest(con.conkey, con.confkey) WITH ORDINALITY
                AS u(lkey, rkey, ord) ON TRUE
            JOIN pg_catalog.pg_attribute AS tc_att
              ON tc_att.attrelid = con.conrelid AND tc_att.attnum = u.lkey
            JOIN pg_catalog.pg_attribute AS fc_att
              ON fc_att.attrelid = con.confrelid AND fc_att.attnum = u.rkey
            WHERE con.contype = 'f'
              AND tc_ns.nspname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY tc_ns.nspname, tc_cls.relname, con.conname, u.ord
            ";

#[cfg(feature = "sqlx")]
const POSTGRES_INDEXES_QUERY: &str = r"
            SELECT
                schemaname,
                tablename,
                indexname,
                indexdef
            FROM pg_indexes
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY schemaname, tablename, indexname
            ";

#[cfg(feature = "sqlx")]
#[async_trait]
impl DatabaseIntrospector for PostgresSchemaIntrospector {
    async fn introspect(&self) -> Result<DatabaseSchema> {
        let mut tables_map = self.load_tables().await?;
        if tables_map.is_empty() {
            return Ok(DatabaseSchema {
                database: self.database_name.clone(),
                tables: Vec::new(),
            });
        }
        self.populate_columns_and_primary_keys(&mut tables_map)
            .await?;
        self.populate_foreign_keys(&mut tables_map).await?;
        self.populate_indexes(&mut tables_map).await?;
        Ok(DatabaseSchema {
            database: self.database_name.clone(),
            tables: tables_map.into_values().collect(),
        })
    }
}

#[cfg(feature = "sqlx")]
fn parse_index_columns(index_def: &str) -> Vec<String> {
    if let Some(start) = index_def.find('(')
        && let Some(end) = index_def.rfind(')')
        && end > start
    {
        let column_segment = &index_def[start + 1..end];
        return column_segment
            .split(',')
            .map(str::trim)
            .filter(|segment| !segment.is_empty())
            .map(|segment| segment.trim_matches('"').to_lowercase())
            .collect();
    }

    Vec::new()
}

/// Pipeline for embedding normalized schema into a vector store.
#[cfg(feature = "tools-rag")]
pub struct SchemaEmbeddingPipeline {
    introspector: Arc<dyn DatabaseIntrospector>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    backend: Arc<dyn Backend + Send + Sync>,
}

#[cfg(feature = "tools-rag")]
impl SchemaEmbeddingPipeline {
    pub fn new(
        introspector: Arc<dyn DatabaseIntrospector>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        backend: Arc<dyn Backend + Send + Sync>,
    ) -> Self {
        Self {
            introspector,
            embedding_provider,
            backend,
        }
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn sync(&self, namespace: &str) -> Result<SchemaSyncReport> {
        let schema = self.introspector.introspect().await?;
        let documents = schema.to_documents();

        if documents.is_empty() {
            return Ok(SchemaSyncReport {
                namespace: namespace.to_string(),
                table_count: 0,
                column_count: 0,
                embeddings_written: 0,
                token_usage: None,
            });
        }

        let texts: Vec<String> = documents.iter().map(|doc| doc.text.clone()).collect();
        let request = EmbeddingRequest { texts, model: None };

        let response = self.embedding_provider.embed(request).await?;
        let embeddings = response.embeddings;

        ensure!(
            embeddings.len() == documents.len(),
            "Embedding provider returned {} vectors for {} documents",
            embeddings.len(),
            documents.len()
        );

        let namespace_key = format!("schema::{namespace}");

        let mut chunks = Vec::new();
        for (doc, embedding) in documents.iter().zip(embeddings.iter()) {
            let chunk = Chunk::new(
                doc.id.clone(),
                0, // chunk_index
                doc.text.clone(),
                embedding.clone(),
                0,              // start_offset
                doc.text.len(), // end_offset
                1,              // total_chunks
                doc.metadata.clone(),
                Some(namespace_key.clone()),
            );
            chunks.push(chunk);
        }
        self.backend
            .upsert(&chunks)
            .await
            .with_context(|| "Failed to upsert schema embeddings")?;

        let mut seen_tables = BTreeSet::new();
        let mut column_total = 0usize;
        for doc in &documents {
            if let (Some(schema), Some(table)) = (
                doc.metadata.get("schema").and_then(Value::as_str),
                doc.metadata.get("table").and_then(Value::as_str),
            ) {
                seen_tables.insert(format!("{schema}.{table}"));
            }
            if let Some(count) = doc.metadata.get("column_count").and_then(Value::as_u64) {
                column_total =
                    column_total.saturating_add(usize::try_from(count).unwrap_or(usize::MAX));
            }
        }

        let token_usage = response.usage.map(|usage| crate::messages::TokenUsage {
            prompt_tokens: None,
            completion_tokens: None,
            embedding_tokens: usage.prompt_tokens.or(usage.total_tokens),
            context_window: None,
            total_tokens: usage.total_tokens,
            cache_creation_tokens: None,
            cache_read_tokens: None,
            last_prompt_tokens: None,
        });

        Ok(SchemaSyncReport {
            namespace: namespace_key,
            table_count: seen_tables.len(),
            column_count: column_total,
            embeddings_written: documents.len(),
            token_usage,
        })
    }
}

/// Report summarizing a synchronization run.
#[cfg(feature = "tools-rag")]
#[derive(Debug, Clone)]
pub struct SchemaSyncReport {
    pub namespace: String,
    pub table_count: usize,
    pub column_count: usize,
    pub embeddings_written: usize,
    pub token_usage: Option<crate::messages::TokenUsage>,
}
