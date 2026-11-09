use crate::data::persistence::schema::{
    ColumnSchema, DatabaseIntrospector, DatabaseSchema, ForeignKeySchema, IndexSchema, TableSchema,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use sqlx::{Row, SqlitePool};
use std::collections::BTreeMap;

/// Introspector implementation for `SQLite`.
///
/// Uses `SQLite` PRAGMAs to fetch table/column/index/foreign-key metadata.
#[derive(Debug)]
pub struct SqliteSchemaIntrospector {
    pool: SqlitePool,
    database_name: String,
}

impl SqliteSchemaIntrospector {
    pub fn new(pool: SqlitePool, database_name: impl Into<String>) -> Self {
        Self {
            pool,
            database_name: database_name.into(),
        }
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = crate::data::persistence::sqlite_pool::local_sqlite_pool()
            .connect(database_url)
            .await
            .context("Failed to connect while building SQLite schema introspector")?;

        Ok(Self::new(pool, "sqlite"))
    }

    async fn foreign_keys(&self, table_name: &str) -> Result<Vec<ForeignKeySchema>> {
        let query = format!("PRAGMA foreign_key_list({})", quote_ident(table_name));
        let rows = sqlx::query(&query)
            .fetch_all(&self.pool)
            .await
            .with_context(|| format!("Failed to fetch foreign keys for table '{table_name}'"))?;
        let mut grouped: BTreeMap<i64, ForeignKeyAccum> = BTreeMap::new();
        for row in rows {
            let id: i64 = row.try_get("id")?;
            let sequence: i64 = row.try_get("seq")?;
            let referenced_table: String = row.try_get("table")?;
            let column: String = row.try_get("from")?;
            let referenced_column: String = row.try_get("to")?;
            let on_update: Option<String> = row.try_get("on_update")?;
            let on_delete: Option<String> = row.try_get("on_delete")?;
            let entry = grouped.entry(id).or_insert_with(|| ForeignKeyAccum {
                referenced_table: referenced_table.to_lowercase(),
                on_update: on_update.clone(),
                on_delete: on_delete.clone(),
                ..ForeignKeyAccum::default()
            });
            entry.columns.push((sequence, column.to_lowercase()));
            entry
                .referenced_columns
                .push((sequence, referenced_column.to_lowercase()));
            entry.on_update = entry.on_update.take().or(on_update);
            entry.on_delete = entry.on_delete.take().or(on_delete);
        }
        Ok(grouped
            .into_values()
            .map(|mut foreign_key| {
                foreign_key.columns.sort_by_key(|(sequence, _)| *sequence);
                foreign_key
                    .referenced_columns
                    .sort_by_key(|(sequence, _)| *sequence);
                ForeignKeySchema {
                    constraint_name: None,
                    columns: foreign_key
                        .columns
                        .into_iter()
                        .map(|(_, column)| column)
                        .collect(),
                    referenced_schema: "main".to_owned(),
                    referenced_table: foreign_key.referenced_table,
                    referenced_columns: foreign_key
                        .referenced_columns
                        .into_iter()
                        .map(|(_, column)| column)
                        .collect(),
                    on_update: foreign_key.on_update,
                    on_delete: foreign_key.on_delete,
                }
            })
            .collect())
    }

    async fn indexes(&self, table_name: &str) -> Result<Vec<IndexSchema>> {
        let query = format!("PRAGMA index_list({})", quote_ident(table_name));
        let rows = sqlx::query(&query)
            .fetch_all(&self.pool)
            .await
            .with_context(|| format!("Failed to fetch indexes for table '{table_name}'"))?;
        let mut indexes = Vec::with_capacity(rows.len());
        for row in rows {
            let name: String = row.try_get("name")?;
            let is_unique: i64 = row.try_get("unique")?;
            let column_query = format!("PRAGMA index_info({})", quote_ident(&name));
            let column_rows = sqlx::query(&column_query)
                .fetch_all(&self.pool)
                .await
                .with_context(|| {
                    format!("Failed to fetch index columns for '{name}' on table '{table_name}'")
                })?;
            let mut columns = Vec::with_capacity(column_rows.len());
            for column in column_rows {
                columns.push((
                    column.try_get::<i64, _>("seqno")?,
                    column.try_get::<String, _>("name")?.to_lowercase(),
                ));
            }
            columns.sort_by_key(|(sequence, _)| *sequence);
            indexes.push(IndexSchema {
                name: Some(name.to_lowercase()),
                columns: columns.into_iter().map(|(_, column)| column).collect(),
                is_unique: is_unique != 0,
            });
        }
        Ok(indexes)
    }
}

fn quote_ident(ident: &str) -> String {
    // Double quotes are the SQLite identifier quoting mechanism.
    // Escape internal quotes by doubling them.
    format!("\"{}\"", ident.replace('"', "\"\""))
}

#[derive(Default)]
struct ForeignKeyAccum {
    referenced_table: String,
    columns: Vec<(i64, String)>,
    referenced_columns: Vec<(i64, String)>,
    on_update: Option<String>,
    on_delete: Option<String>,
}

#[async_trait]
impl DatabaseIntrospector for SqliteSchemaIntrospector {
    async fn introspect(&self) -> Result<DatabaseSchema> {
        let table_rows = sqlx::query(
            r"
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            ",
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch SQLite table list")?;

        let mut tables = Vec::with_capacity(table_rows.len());

        for row in table_rows {
            let table_name: String = row.try_get("name")?;
            let table_name_norm = table_name.to_lowercase();
            let schema_name = "main".to_owned();

            // Columns + primary key
            let pragma_table_info = format!("PRAGMA table_info({})", quote_ident(&table_name));
            let column_rows = sqlx::query(&pragma_table_info)
                .fetch_all(&self.pool)
                .await
                .with_context(|| format!("Failed to fetch columns for table '{table_name}'"))?;

            let mut columns = Vec::with_capacity(column_rows.len());
            let mut pk_columns: Vec<(i64, String)> = Vec::new();

            for c in column_rows {
                let name: String = c.try_get("name")?;
                let data_type: String = c.try_get("type")?;
                let notnull: i64 = c.try_get("notnull")?;
                let dflt_value: Option<String> = c.try_get("dflt_value")?;
                let pk: i64 = c.try_get("pk")?;

                if pk > 0 {
                    pk_columns.push((pk, name.to_lowercase()));
                }

                columns.push(ColumnSchema {
                    name: name.to_lowercase(),
                    data_type,
                    is_nullable: notnull == 0,
                    default: dflt_value,
                    comment: None,
                    // SQLite uses dynamic typing, so `VARCHAR(n)` is purely a
                    // hint stored in `type`. We intentionally do not parse the
                    // declared length because it has no runtime meaning.
                    max_length: None,
                });
            }

            pk_columns.sort_by_key(|(pos, _)| *pos);
            let primary_key = pk_columns.into_iter().map(|(_, n)| n).collect();
            let foreign_keys = self.foreign_keys(&table_name).await?;
            let indexes = self.indexes(&table_name).await?;

            tables.push(TableSchema {
                schema: schema_name,
                name: table_name_norm,
                columns,
                primary_key,
                foreign_keys,
                indexes,
                comment: None,
            });
        }

        Ok(DatabaseSchema {
            database: self.database_name.clone(),
            tables,
        })
    }
}
