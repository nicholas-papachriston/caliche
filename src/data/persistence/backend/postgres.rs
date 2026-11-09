use super::{
    ConversationListQuery, ConversationRecord, ConversationStats, ConversationStoreBackend,
    MetadataQuery,
};
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use serde_json::Value;
use sqlx::{PgPool, Row, postgres::PgPoolOptions};
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

#[derive(Debug, Clone)]
struct Identifier {
    raw: String,
    quoted: String,
}

impl Identifier {
    fn new(raw: impl Into<String>, kind: &str) -> Result<Self> {
        let raw = raw.into();
        ensure!(
            Self::is_valid(&raw),
            "{kind} identifier '{raw}' must contain only ASCII letters, numbers, or underscores"
        );
        Ok(Self {
            quoted: Self::quote(&raw),
            raw,
        })
    }

    fn is_valid(value: &str) -> bool {
        !value.is_empty()
            && value
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    }

    fn quote(value: &str) -> String {
        format!("\"{}\"", value.replace('"', "\"\""))
    }

    fn raw(&self) -> &str {
        &self.raw
    }

    fn quoted(&self) -> &str {
        &self.quoted
    }
}

#[derive(Debug, Clone)]
pub struct ConversationTableConfig {
    table: Identifier,
    id: Identifier,
    agent_id: Identifier,
    messages: Identifier,
    metadata: Identifier,
    created_at: Identifier,
    updated_at: Identifier,
    expires_at: Identifier,
    manage_schema: bool,
    create_indexes: bool,
}

impl ConversationTableConfig {
    #[must_use]
    pub fn builder() -> ConversationTableConfigBuilder {
        ConversationTableConfigBuilder::default()
    }

    #[must_use]
    pub const fn manage_schema(&self) -> bool {
        self.manage_schema
    }

    #[must_use]
    pub const fn create_indexes(&self) -> bool {
        self.create_indexes
    }

    #[must_use]
    pub fn table_ident(&self) -> &str {
        self.table.quoted()
    }

    #[must_use]
    pub fn id_ident(&self) -> &str {
        self.id.quoted()
    }

    #[must_use]
    pub fn agent_id_ident(&self) -> &str {
        self.agent_id.quoted()
    }

    #[must_use]
    pub fn messages_ident(&self) -> &str {
        self.messages.quoted()
    }

    #[must_use]
    pub fn metadata_ident(&self) -> &str {
        self.metadata.quoted()
    }

    #[must_use]
    pub fn created_at_ident(&self) -> &str {
        self.created_at.quoted()
    }

    #[must_use]
    pub fn updated_at_ident(&self) -> &str {
        self.updated_at.quoted()
    }

    #[must_use]
    pub fn expires_at_ident(&self) -> &str {
        self.expires_at.quoted()
    }

    #[must_use]
    pub fn id_raw(&self) -> &str {
        self.id.raw()
    }

    #[must_use]
    pub fn agent_id_raw(&self) -> &str {
        self.agent_id.raw()
    }

    #[must_use]
    pub fn messages_raw(&self) -> &str {
        self.messages.raw()
    }

    #[must_use]
    pub fn metadata_raw(&self) -> &str {
        self.metadata.raw()
    }

    #[must_use]
    pub fn created_at_raw(&self) -> &str {
        self.created_at.raw()
    }

    #[must_use]
    pub fn updated_at_raw(&self) -> &str {
        self.updated_at.raw()
    }

    #[must_use]
    pub fn expires_at_raw(&self) -> &str {
        self.expires_at.raw()
    }

    #[must_use]
    pub fn select_list(&self) -> String {
        format!(
            "{id}, {agent_id}, {messages}, {metadata}, {created_at}, {updated_at}, {expires_at}",
            id = self.id_ident(),
            agent_id = self.agent_id_ident(),
            messages = self.messages_ident(),
            metadata = self.metadata_ident(),
            created_at = self.created_at_ident(),
            updated_at = self.updated_at_ident(),
            expires_at = self.expires_at_ident(),
        )
    }

    fn index_ident(&self, suffix: &str) -> String {
        let raw = format!("{}_{}", self.table.raw(), suffix);
        Identifier::quote(&raw)
    }
}

impl Default for ConversationTableConfig {
    fn default() -> Self {
        /// Create an `Identifier` from a known-good static name without fallible validation.
        fn ident(name: &str) -> Identifier {
            Identifier {
                quoted: format!("\"{name}\""),
                raw: name.to_string(),
            }
        }

        Self {
            table: ident("conversations"),
            id: ident("id"),
            agent_id: ident("agent_id"),
            messages: ident("messages"),
            metadata: ident("metadata"),
            created_at: ident("created_at"),
            updated_at: ident("updated_at"),
            expires_at: ident("expires_at"),
            manage_schema: true,
            create_indexes: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConversationTableConfigBuilder {
    table_name: String,
    id_column: String,
    agent_id_column: String,
    messages_column: String,
    metadata_column: String,
    created_at_column: String,
    updated_at_column: String,
    expires_at_column: String,
    manage_schema: bool,
    create_indexes: bool,
}

impl Default for ConversationTableConfigBuilder {
    fn default() -> Self {
        Self {
            table_name: "conversations".to_owned(),
            id_column: "id".to_owned(),
            agent_id_column: "agent_id".to_owned(),
            messages_column: "messages".to_owned(),
            metadata_column: "metadata".to_owned(),
            created_at_column: "created_at".to_owned(),
            updated_at_column: "updated_at".to_owned(),
            expires_at_column: "expires_at".to_owned(),
            manage_schema: true,
            create_indexes: true,
        }
    }
}

impl ConversationTableConfigBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn table_name(mut self, name: impl Into<String>) -> Self {
        self.table_name = name.into();
        self
    }

    #[must_use]
    pub fn id_column(mut self, name: impl Into<String>) -> Self {
        self.id_column = name.into();
        self
    }

    #[must_use]
    pub fn agent_id_column(mut self, name: impl Into<String>) -> Self {
        self.agent_id_column = name.into();
        self
    }

    #[must_use]
    pub fn messages_column(mut self, name: impl Into<String>) -> Self {
        self.messages_column = name.into();
        self
    }

    #[must_use]
    pub fn metadata_column(mut self, name: impl Into<String>) -> Self {
        self.metadata_column = name.into();
        self
    }

    #[must_use]
    pub fn created_at_column(mut self, name: impl Into<String>) -> Self {
        self.created_at_column = name.into();
        self
    }

    #[must_use]
    pub fn updated_at_column(mut self, name: impl Into<String>) -> Self {
        self.updated_at_column = name.into();
        self
    }

    #[must_use]
    pub fn expires_at_column(mut self, name: impl Into<String>) -> Self {
        self.expires_at_column = name.into();
        self
    }

    #[must_use]
    pub const fn manage_schema(mut self, manage: bool) -> Self {
        self.manage_schema = manage;
        self
    }

    #[must_use]
    pub const fn create_indexes(mut self, create: bool) -> Self {
        self.create_indexes = create;
        self
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub fn build(self) -> Result<ConversationTableConfig> {
        Ok(ConversationTableConfig {
            table: Identifier::new(self.table_name, "table")?,
            id: Identifier::new(self.id_column, "column")?,
            agent_id: Identifier::new(self.agent_id_column, "column")?,
            messages: Identifier::new(self.messages_column, "column")?,
            metadata: Identifier::new(self.metadata_column, "column")?,
            created_at: Identifier::new(self.created_at_column, "column")?,
            updated_at: Identifier::new(self.updated_at_column, "column")?,
            expires_at: Identifier::new(self.expires_at_column, "column")?,
            manage_schema: self.manage_schema,
            create_indexes: self.create_indexes,
        })
    }
}

pub struct PostgresConversationStore {
    pool: PgPool,
    table: Arc<ConversationTableConfig>,
}

impl PostgresConversationStore {
    /// Create a new PostgreSQL-backed conversation store and initialize schema.
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn new(database_url: &str) -> Result<Self> {
        Self::new_with_config(database_url, ConversationTableConfig::default()).await
    }

    /// Create a new PostgreSQL-backed conversation store with a custom table configuration.
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn new_with_config(
        database_url: &str,
        config: ConversationTableConfig,
    ) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await
            .context("Failed to connect to PostgreSQL")?;

        let store = Self {
            pool,
            table: Arc::new(config),
        };
        store.init().await?;
        Ok(store)
    }

    /// Construct a store from an existing connection pool using the default configuration.
    #[must_use]
    pub fn from_pool(pool: PgPool) -> Self {
        Self::from_pool_with_config(pool, ConversationTableConfig::default())
    }

    /// Construct a store from an existing connection pool and custom configuration.
    #[must_use]
    pub fn from_pool_with_config(pool: PgPool, config: ConversationTableConfig) -> Self {
        Self {
            pool,
            table: Arc::new(config),
        }
    }

    #[must_use]
    pub const fn pool(&self) -> &PgPool {
        &self.pool
    }

    #[must_use]
    pub fn config(&self) -> Arc<ConversationTableConfig> {
        Arc::clone(&self.table)
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn save(&self, conversation: &ConversationRecord) -> Result<()> {
        <Self as ConversationStoreBackend>::save(self, conversation).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn get(&self, id: &str) -> Result<Option<ConversationRecord>> {
        <Self as ConversationStoreBackend>::fetch(self, id).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn list_by_agent(
        &self,
        agent_id: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ConversationRecord>> {
        let query = ConversationListQuery::new(agent_id, limit, offset);
        <Self as ConversationStoreBackend>::list(self, &query).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn delete(&self, id: &str) -> Result<()> {
        <Self as ConversationStoreBackend>::delete(self, id).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn cleanup_expired(&self) -> Result<u64> {
        <Self as ConversationStoreBackend>::cleanup_expired(self).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn update_ttl(&self, id: &str, ttl_seconds: i64) -> Result<()> {
        <Self as ConversationStoreBackend>::update_ttl(self, id, ttl_seconds).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn search_by_metadata(
        &self,
        json_path: Option<&str>,
        filter: &Value,
        limit: i64,
    ) -> Result<Vec<ConversationRecord>> {
        let mut query = MetadataQuery::new(filter.clone(), limit);
        if let Some(path) = json_path.filter(|p| !p.is_empty()) {
            query = query.with_json_path(path);
        }
        <Self as ConversationStoreBackend>::search(self, &query).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn get_stats(&self) -> Result<ConversationStats> {
        <Self as ConversationStoreBackend>::stats(self).await
    }
}

#[async_trait]
impl ConversationStoreBackend for PostgresConversationStore {
    async fn init(&self) -> Result<()> {
        let table = self.table.as_ref();
        if !table.manage_schema() {
            return Ok(());
        }

        let create_table_sql = format!(
            r"
            CREATE TABLE IF NOT EXISTS {table} (
                {id} UUID PRIMARY KEY,
                {agent_id} VARCHAR(255) NOT NULL,
                {messages} JSONB NOT NULL,
                {metadata} JSONB NOT NULL DEFAULT '{{}}',
                {created_at} TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                {updated_at} TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                {expires_at} TIMESTAMPTZ
            )
            ",
            table = table.table_ident(),
            id = table.id_ident(),
            agent_id = table.agent_id_ident(),
            messages = table.messages_ident(),
            metadata = table.metadata_ident(),
            created_at = table.created_at_ident(),
            updated_at = table.updated_at_ident(),
            expires_at = table.expires_at_ident(),
        );

        sqlx::query(&create_table_sql)
            .execute(&self.pool)
            .await
            .context("Failed to create conversation table")?;

        if table.create_indexes() {
            let agent_index = format!(
                "CREATE INDEX IF NOT EXISTS {index} ON {table} ({agent_id})",
                index = table.index_ident("agent_id_idx"),
                table = table.table_ident(),
                agent_id = table.agent_id_ident(),
            );
            sqlx::query(&agent_index)
                .execute(&self.pool)
                .await
                .context("Failed to create agent_id index")?;

            let created_at_index = format!(
                "CREATE INDEX IF NOT EXISTS {index} ON {table} ({created_at})",
                index = table.index_ident("created_at_idx"),
                table = table.table_ident(),
                created_at = table.created_at_ident(),
            );
            sqlx::query(&created_at_index)
                .execute(&self.pool)
                .await
                .context("Failed to create created_at index")?;

            let expires_at_index = format!(
                "CREATE INDEX IF NOT EXISTS {index} ON {table} ({expires_at})",
                index = table.index_ident("expires_at_idx"),
                table = table.table_ident(),
                expires_at = table.expires_at_ident(),
            );
            sqlx::query(&expires_at_index)
                .execute(&self.pool)
                .await
                .context("Failed to create expires_at index")?;
        }

        Ok(())
    }

    #[instrument(skip(self, conversation), fields(conversation_id = %conversation.id, agent_id = %conversation.agent_id), err)]
    async fn save(&self, conversation: &ConversationRecord) -> Result<()> {
        let messages_json = serde_json::to_value(&conversation.messages)?;
        let table = self.table.as_ref();

        let sql = format!(
            r"
            INSERT INTO {table} ({id}, {agent_id}, {messages}, {metadata}, {created_at}, {updated_at}, {expires_at})
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT ({id}) DO UPDATE SET
                {messages} = EXCLUDED.{messages},
                {metadata} = EXCLUDED.{metadata},
                {updated_at} = EXCLUDED.{updated_at},
                {expires_at} = EXCLUDED.{expires_at}
            ",
            table = table.table_ident(),
            id = table.id_ident(),
            agent_id = table.agent_id_ident(),
            messages = table.messages_ident(),
            metadata = table.metadata_ident(),
            created_at = table.created_at_ident(),
            updated_at = table.updated_at_ident(),
            expires_at = table.expires_at_ident(),
        );

        sqlx::query(&sql)
            .bind(Uuid::parse_str(&conversation.id)?)
            .bind(&conversation.agent_id)
            .bind(messages_json)
            .bind(&conversation.metadata)
            .bind(conversation.created_at)
            .bind(conversation.updated_at)
            .bind(conversation.expires_at)
            .execute(&self.pool)
            .await
            .context("Failed to save conversation")?;

        Ok(())
    }

    #[instrument(skip(self), fields(conversation_id = %id), err)]
    async fn fetch(&self, id: &str) -> Result<Option<ConversationRecord>> {
        let uuid = Uuid::parse_str(id)?;
        let table = self.table.as_ref();

        let sql = format!(
            r"
            SELECT {select_columns}
            FROM {table}
            WHERE {id} = $1
              AND ({expires_at} IS NULL OR {expires_at} > NOW())
            ",
            select_columns = table.select_list(),
            table = table.table_ident(),
            id = table.id_ident(),
            expires_at = table.expires_at_ident(),
        );

        let row = sqlx::query(&sql)
            .bind(uuid)
            .fetch_optional(&self.pool)
            .await
            .context("Failed to fetch conversation")?;

        if let Some(row) = row {
            let messages_json: serde_json::Value = row.try_get(table.messages_raw())?;
            let messages = serde_json::from_value(messages_json)?;

            Ok(Some(ConversationRecord {
                id: row.try_get::<Uuid, _>(table.id_raw())?.to_string(),
                agent_id: row.try_get(table.agent_id_raw())?,
                messages,
                metadata: row.try_get(table.metadata_raw())?,
                created_at: row.try_get(table.created_at_raw())?,
                updated_at: row.try_get(table.updated_at_raw())?,
                expires_at: row.try_get(table.expires_at_raw())?,
            }))
        } else {
            Ok(None)
        }
    }

    #[instrument(skip(self, query), fields(agent_id = %query.agent_id, limit = query.limit, offset = query.offset), err)]
    async fn list(&self, query: &ConversationListQuery) -> Result<Vec<ConversationRecord>> {
        let table = self.table.as_ref();
        let sql = format!(
            r"
            SELECT {select_columns}
            FROM {table}
            WHERE {agent_id} = $1
              AND ({expires_at} IS NULL OR {expires_at} > NOW())
            ORDER BY {created_at} DESC
            LIMIT $2 OFFSET $3
            ",
            select_columns = table.select_list(),
            table = table.table_ident(),
            agent_id = table.agent_id_ident(),
            expires_at = table.expires_at_ident(),
            created_at = table.created_at_ident(),
        );

        let rows = sqlx::query(&sql)
            .bind(&query.agent_id)
            .bind(query.limit)
            .bind(query.offset)
            .fetch_all(&self.pool)
            .await
            .context("Failed to list conversations")?;

        let mut conversations = Vec::with_capacity(rows.len());
        for row in rows {
            let messages_json: serde_json::Value = row.try_get(table.messages_raw())?;
            let messages = serde_json::from_value(messages_json)?;

            conversations.push(ConversationRecord {
                id: row.try_get::<Uuid, _>(table.id_raw())?.to_string(),
                agent_id: row.try_get(table.agent_id_raw())?,
                messages,
                metadata: row.try_get(table.metadata_raw())?,
                created_at: row.try_get(table.created_at_raw())?,
                updated_at: row.try_get(table.updated_at_raw())?,
                expires_at: row.try_get(table.expires_at_raw())?,
            });
        }

        Ok(conversations)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let uuid = Uuid::parse_str(id)?;
        let table = self.table.as_ref();

        let sql = format!(
            "DELETE FROM {table} WHERE {id} = $1",
            table = table.table_ident(),
            id = table.id_ident(),
        );

        sqlx::query(&sql)
            .bind(uuid)
            .execute(&self.pool)
            .await
            .context("Failed to delete conversation")?;

        Ok(())
    }

    async fn cleanup_expired(&self) -> Result<u64> {
        let table = self.table.as_ref();
        let sql = format!(
            "DELETE FROM {table} WHERE {expires_at} IS NOT NULL AND {expires_at} < NOW()",
            table = table.table_ident(),
            expires_at = table.expires_at_ident(),
        );

        let result = sqlx::query(&sql)
            .execute(&self.pool)
            .await
            .context("Failed to cleanup expired conversations")?;

        Ok(result.rows_affected())
    }

    async fn update_ttl(&self, id: &str, ttl_seconds: i64) -> Result<()> {
        let uuid = Uuid::parse_str(id)?;
        let table = self.table.as_ref();

        let sql = format!(
            r"
            UPDATE {table}
            SET {expires_at} = NOW() + INTERVAL '1 second' * $2,
                {updated_at} = NOW()
            WHERE {id} = $1
            ",
            table = table.table_ident(),
            expires_at = table.expires_at_ident(),
            updated_at = table.updated_at_ident(),
            id = table.id_ident(),
        );

        sqlx::query(&sql)
            .bind(uuid)
            .bind(ttl_seconds)
            .execute(&self.pool)
            .await
            .context("Failed to update TTL")?;

        Ok(())
    }

    #[instrument(skip(self, query), fields(limit = query.limit, json_path = query.json_path.as_deref().unwrap_or("")), err)]
    async fn search(&self, query: &MetadataQuery) -> Result<Vec<ConversationRecord>> {
        let table = self.table.as_ref();
        let select_columns = table.select_list();
        let metadata_ident = table.metadata_ident();
        let metadata_condition = query
            .json_path
            .as_deref()
            .filter(|segment| !segment.is_empty())
            .map_or_else(
                || format!("{metadata_ident} @> $1"),
                |path| {
                    let pg_path = path
                        .split('.')
                        .filter(|segment| !segment.is_empty())
                        .map(|segment| format!("\"{}\"", segment.replace('"', "\"\"")))
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("{metadata_ident} #> '{{{pg_path}}}' = $1")
                },
            );
        let expires_at = table.expires_at_ident();
        let expiry = format!("({expires_at} IS NULL OR {expires_at} > NOW())");
        let sql = format!(
            r"
            SELECT {select_columns}
            FROM {table}
            WHERE {metadata_condition}
              AND {expiry}
            ORDER BY {created_at} DESC
            LIMIT $2
            ",
            select_columns = select_columns,
            table = table.table_ident(),
            metadata_condition = metadata_condition,
            expiry = expiry,
            created_at = table.created_at_ident(),
        );

        let rows = sqlx::query(&sql)
            .bind(query.filter.clone())
            .bind(query.limit)
            .fetch_all(&self.pool)
            .await
            .context("Failed to search conversations")?;

        let mut conversations = Vec::with_capacity(rows.len());
        for row in rows {
            let messages_json: serde_json::Value = row.try_get(table.messages_raw())?;
            let messages = serde_json::from_value(messages_json)?;

            conversations.push(ConversationRecord {
                id: row.try_get::<Uuid, _>(table.id_raw())?.to_string(),
                agent_id: row.try_get(table.agent_id_raw())?,
                messages,
                metadata: row.try_get(table.metadata_raw())?,
                created_at: row.try_get(table.created_at_raw())?,
                updated_at: row.try_get(table.updated_at_raw())?,
                expires_at: row.try_get(table.expires_at_raw())?,
            });
        }

        Ok(conversations)
    }

    #[instrument(skip(self), err)]
    async fn stats(&self) -> Result<ConversationStats> {
        let table = self.table.as_ref();
        let expires_at = table.expires_at_ident();
        let messages = table.messages_ident();
        let sql = format!(
            r"
            SELECT 
                COUNT(*) as total_conversations,
                COUNT(*) FILTER (WHERE {expires_at} IS NULL OR {expires_at} > NOW()) as active_conversations,
                COUNT(*) FILTER (WHERE {expires_at} IS NOT NULL AND {expires_at} <= NOW()) as expired_conversations,
                AVG(jsonb_array_length({messages})) as avg_message_count
            FROM {table}
            ",
            expires_at = expires_at,
            messages = messages,
            table = table.table_ident(),
        );

        let row = sqlx::query(&sql)
            .fetch_one(&self.pool)
            .await
            .context("Failed to get stats")?;

        Ok(ConversationStats {
            total_conversations: row.try_get("total_conversations")?,
            active_conversations: row.try_get("active_conversations")?,
            expired_conversations: row.try_get("expired_conversations")?,
            avg_message_count: row
                .try_get::<Option<f64>, _>("avg_message_count")?
                .unwrap_or(0.0),
        })
    }
}
