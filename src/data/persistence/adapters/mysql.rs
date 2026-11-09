use anyhow::{Context, Result};
use async_trait::async_trait;
use sqlx::{MySqlPool, Row};
use std::collections::BTreeMap;

use crate::data::persistence::schema::{
    ColumnSchema, DatabaseIntrospector, DatabaseSchema, ForeignKeySchema, IndexSchema, TableSchema,
};

/// Introspector implementation for MySQL
#[derive(Debug)]
pub struct MySqlSchemaIntrospector {
    pool: MySqlPool,
    database_name: String,
}

impl MySqlSchemaIntrospector {
    pub fn new(pool: MySqlPool, database_name: impl Into<String>) -> Self {
        Self {
            pool,
            database_name: database_name.into(),
        }
    }

    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = sqlx::mysql::MySqlPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await
            .context("Failed to connect while building MySQL schema introspector")?;

        let db_name = sqlx::query_scalar::<_, String>("SELECT DATABASE()")
            .fetch_one(&pool)
            .await
            .context("Failed to resolve database name")?;

        Ok(Self::new(pool, db_name))
    }

    fn table_key(schema: &str, name: &str) -> (String, String) {
        (schema.to_lowercase(), name.to_lowercase())
    }

    fn apply_table_row(
        tables_map: &mut BTreeMap<(String, String), TableSchema>,
        row: &sqlx::mysql::MySqlRow,
    ) -> Result<()> {
        let schema: String = row.try_get("TABLE_SCHEMA")?;
        let name: String = row.try_get("TABLE_NAME")?;
        let comment: Option<String> = row.try_get("TABLE_COMMENT")?;
        let key = Self::table_key(&schema, &name);
        tables_map.insert(
            key.clone(),
            TableSchema {
                schema: key.0,
                name: key.1,
                columns: Vec::new(),
                primary_key: Vec::new(),
                foreign_keys: Vec::new(),
                indexes: Vec::new(),
                comment,
            },
        );
        Ok(())
    }

    fn apply_column_row(
        tables_map: &mut BTreeMap<(String, String), TableSchema>,
        row: &sqlx::mysql::MySqlRow,
    ) -> Result<()> {
        let schema: String = row.try_get("TABLE_SCHEMA")?;
        let name: String = row.try_get("TABLE_NAME")?;
        let column_name: String = row.try_get("COLUMN_NAME")?;
        let data_type: String = row.try_get("DATA_TYPE")?;
        let is_nullable: String = row.try_get("IS_NULLABLE")?;
        let default: Option<String> = row.try_get("COLUMN_DEFAULT")?;
        let max_length: Option<i64> = row.try_get("CHARACTER_MAXIMUM_LENGTH")?;
        let comment: Option<String> = row.try_get("COLUMN_COMMENT")?;
        if let Some(table) = tables_map.get_mut(&Self::table_key(&schema, &name)) {
            table.columns.push(ColumnSchema {
                name: column_name.to_lowercase(),
                data_type: data_type.to_lowercase(),
                is_nullable: is_nullable.eq_ignore_ascii_case("YES"),
                default,
                comment,
                max_length,
            });
        }
        Ok(())
    }

    fn apply_primary_key_row(
        tables_map: &mut BTreeMap<(String, String), TableSchema>,
        row: &sqlx::mysql::MySqlRow,
    ) -> Result<()> {
        let schema: String = row.try_get("TABLE_SCHEMA")?;
        let name: String = row.try_get("TABLE_NAME")?;
        let column_name: String = row.try_get("COLUMN_NAME")?;
        if let Some(table) = tables_map.get_mut(&Self::table_key(&schema, &name)) {
            table.primary_key.push(column_name.to_lowercase());
        }
        Ok(())
    }

    fn collect_foreign_key_row(
        fk_map: &mut BTreeMap<(String, String, String), ForeignKeySchema>,
        row: &sqlx::mysql::MySqlRow,
    ) -> Result<()> {
        let constraint_name: Option<String> = row.try_get("CONSTRAINT_NAME")?;
        let schema: String = row.try_get("TABLE_SCHEMA")?;
        let table_name: String = row.try_get("TABLE_NAME")?;
        let column_name: String = row.try_get("COLUMN_NAME")?;
        let foreign_schema: String = row.try_get("REFERENCED_TABLE_SCHEMA")?;
        let foreign_table: String = row.try_get("REFERENCED_TABLE_NAME")?;
        let foreign_column: String = row.try_get("REFERENCED_COLUMN_NAME")?;
        let on_update: Option<String> = row.try_get("UPDATE_RULE")?;
        let on_delete: Option<String> = row.try_get("DELETE_RULE")?;
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
            on_update,
            on_delete,
        });
        entry.columns.push(column_name.to_lowercase());
        entry.referenced_columns.push(foreign_column.to_lowercase());
        Ok(())
    }

    fn collect_index_row(
        index_map: &mut BTreeMap<(String, String, String), IndexSchema>,
        row: &sqlx::mysql::MySqlRow,
    ) -> Result<()> {
        let schema: String = row.try_get("TABLE_SCHEMA")?;
        let table_name: String = row.try_get("TABLE_NAME")?;
        let index_name: Option<String> = row.try_get("INDEX_NAME")?;
        let column_name: String = row.try_get("COLUMN_NAME")?;
        let non_unique: u8 = row.try_get("NON_UNIQUE")?;
        let schema_lc = schema.to_lowercase();
        let table_lc = table_name.to_lowercase();
        let index_key_name = index_name
            .as_deref()
            .map_or_else(|| "unnamed".to_owned(), str::to_string);
        let key = (schema_lc, table_lc, index_key_name);
        let entry = index_map.entry(key).or_insert_with(|| IndexSchema {
            name: index_name.map(|n| n.to_lowercase()),
            columns: Vec::new(),
            is_unique: non_unique == 0,
        });
        entry.columns.push(column_name.to_lowercase());
        Ok(())
    }

    async fn load_tables(&self) -> Result<BTreeMap<(String, String), TableSchema>> {
        let table_query = r#"
            SELECT 
                TABLE_SCHEMA,
                TABLE_NAME,
                TABLE_COMMENT
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = ?
              AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        "#;
        let rows = sqlx::query(table_query)
            .bind(&self.database_name)
            .fetch_all(&self.pool)
            .await
            .context("Failed to fetch table list")?;
        let mut tables_map = BTreeMap::new();
        for row in rows {
            Self::apply_table_row(&mut tables_map, &row)?;
        }
        Ok(tables_map)
    }

    async fn populate_columns(
        &self,
        tables_map: &mut BTreeMap<(String, String), TableSchema>,
    ) -> Result<()> {
        let column_query = r#"
            SELECT 
                TABLE_SCHEMA,
                TABLE_NAME,
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                CHARACTER_MAXIMUM_LENGTH,
                COLUMN_COMMENT
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = ?
            ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
        "#;
        let columns = sqlx::query(column_query)
            .bind(&self.database_name)
            .fetch_all(&self.pool)
            .await
            .context("Failed to fetch column metadata")?;
        for row in columns {
            Self::apply_column_row(tables_map, &row)?;
        }
        Ok(())
    }

    async fn populate_primary_keys(
        &self,
        tables_map: &mut BTreeMap<(String, String), TableSchema>,
    ) -> Result<()> {
        let primary_key_query = r#"
            SELECT 
                TABLE_SCHEMA,
                TABLE_NAME,
                COLUMN_NAME,
                ORDINAL_POSITION
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = ?
              AND CONSTRAINT_NAME = 'PRIMARY'
            ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
        "#;
        let primary_keys = sqlx::query(primary_key_query)
            .bind(&self.database_name)
            .fetch_all(&self.pool)
            .await
            .context("Failed to fetch primary keys")?;
        for row in primary_keys {
            Self::apply_primary_key_row(tables_map, &row)?;
        }
        Ok(())
    }

    async fn populate_foreign_keys(
        &self,
        tables_map: &mut BTreeMap<(String, String), TableSchema>,
    ) -> Result<()> {
        let foreign_key_query = r#"
            SELECT
                kcu.CONSTRAINT_NAME,
                kcu.TABLE_SCHEMA,
                kcu.TABLE_NAME,
                kcu.COLUMN_NAME,
                kcu.REFERENCED_TABLE_SCHEMA,
                kcu.REFERENCED_TABLE_NAME,
                kcu.REFERENCED_COLUMN_NAME,
                rc.UPDATE_RULE,
                rc.DELETE_RULE
            FROM information_schema.KEY_COLUMN_USAGE kcu
            LEFT JOIN information_schema.REFERENTIAL_CONSTRAINTS rc
              ON rc.CONSTRAINT_SCHEMA = kcu.TABLE_SCHEMA
             AND rc.CONSTRAINT_NAME   = kcu.CONSTRAINT_NAME
             AND rc.TABLE_NAME        = kcu.TABLE_NAME
            WHERE kcu.TABLE_SCHEMA = ?
              AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
            ORDER BY kcu.TABLE_SCHEMA, kcu.TABLE_NAME, kcu.CONSTRAINT_NAME, kcu.ORDINAL_POSITION
        "#;
        let foreign_keys = sqlx::query(foreign_key_query)
            .bind(&self.database_name)
            .fetch_all(&self.pool)
            .await
            .context("Failed to fetch foreign keys")?;
        let mut fk_map = BTreeMap::new();
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

    async fn populate_indexes(
        &self,
        tables_map: &mut BTreeMap<(String, String), TableSchema>,
    ) -> Result<()> {
        let index_query = r#"
            SELECT
                TABLE_SCHEMA,
                TABLE_NAME,
                INDEX_NAME,
                COLUMN_NAME,
                NON_UNIQUE
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = ?
            ORDER BY TABLE_SCHEMA, TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX
        "#;
        let indexes = sqlx::query(index_query)
            .bind(&self.database_name)
            .fetch_all(&self.pool)
            .await
            .context("Failed to fetch indexes")?;
        let mut index_map = BTreeMap::new();
        for row in indexes {
            Self::collect_index_row(&mut index_map, &row)?;
        }
        for ((schema, table_name, _), index) in index_map {
            if let Some(table) = tables_map.get_mut(&(schema, table_name)) {
                table.indexes.push(index);
            }
        }
        Ok(())
    }
}

#[async_trait]
impl DatabaseIntrospector for MySqlSchemaIntrospector {
    async fn introspect(&self) -> Result<DatabaseSchema> {
        let mut tables_map = self.load_tables().await?;
        if tables_map.is_empty() {
            return Ok(DatabaseSchema {
                database: self.database_name.clone(),
                tables: Vec::new(),
            });
        }
        self.populate_columns(&mut tables_map).await?;
        self.populate_primary_keys(&mut tables_map).await?;
        self.populate_foreign_keys(&mut tables_map).await?;
        self.populate_indexes(&mut tables_map).await?;
        Ok(DatabaseSchema {
            database: self.database_name.clone(),
            tables: tables_map.into_values().collect(),
        })
    }
}
