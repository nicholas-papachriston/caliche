use crate::data::persistence::backend::sql_session::status_from_db;
use crate::data::persistence::session_registry::TrackedSession;
use anyhow::Result;
#[cfg(any(feature = "sqlx", feature = "sqlite-store"))]
use sqlx::Row;
use uuid::Uuid;

#[cfg(feature = "sqlx")]
pub fn pg_row_to_session(row: &sqlx::postgres::PgRow) -> Result<TrackedSession> {
    let id: Uuid = row.try_get("id")?;
    let name_slug: Option<String> = row.try_get("name_slug")?;
    let agent_id: String = row.try_get("agent_id")?;
    let user_id: Option<String> = row.try_get("user_id")?;
    let conversation_id: Option<String> = row.try_get("conversation_id")?;
    let message_count: i32 = row.try_get("message_count")?;
    let status_str: String = row.try_get("status")?;
    let metadata: sqlx::types::Json<serde_json::Value> = row.try_get("metadata")?;
    let created_at: chrono::DateTime<chrono::Utc> = row.try_get("created_at")?;
    let updated_at: chrono::DateTime<chrono::Utc> = row.try_get("updated_at")?;
    let last_accessed: chrono::DateTime<chrono::Utc> = row.try_get("last_accessed")?;

    Ok(TrackedSession {
        id,
        name_slug,
        agent_id,
        user_id,
        conversation_id,
        created_at,
        updated_at,
        last_accessed,
        metadata: metadata.0,
        message_count: u32::try_from(message_count.max(0)).unwrap_or(0),
        status: status_from_db(&status_str)?,
    })
}

#[cfg(feature = "sqlite-store")]
pub(crate) fn sqlite_row_to_session(row: &sqlx::sqlite::SqliteRow) -> Result<TrackedSession> {
    use anyhow::Context;
    use chrono::{DateTime, Utc};

    let id_str: String = row.try_get("id")?;
    let id = Uuid::parse_str(&id_str).context("Failed to parse session id as UUID")?;

    let name_slug: Option<String> = row.try_get("name_slug")?;
    let agent_id: String = row.try_get("agent_id")?;
    let user_id: Option<String> = row.try_get("user_id")?;
    let conversation_id: Option<String> = row.try_get("conversation_id")?;
    let message_count: i64 = row.try_get("message_count")?;
    let status_str: String = row.try_get("status")?;
    let metadata_json: String = row.try_get("metadata")?;

    let created_at_str: String = row.try_get("created_at")?;
    let updated_at_str: String = row.try_get("updated_at")?;
    let last_accessed_str: String = row.try_get("last_accessed")?;

    let created_at = DateTime::parse_from_rfc3339(&created_at_str)
        .context("Failed to parse created_at as RFC3339")?
        .with_timezone(&Utc);
    let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
        .context("Failed to parse updated_at as RFC3339")?
        .with_timezone(&Utc);
    let last_accessed = DateTime::parse_from_rfc3339(&last_accessed_str)
        .context("Failed to parse last_accessed as RFC3339")?
        .with_timezone(&Utc);

    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_json).context("Failed to deserialize metadata")?;

    Ok(TrackedSession {
        id,
        name_slug,
        agent_id,
        user_id,
        conversation_id,
        created_at,
        updated_at,
        last_accessed,
        metadata,
        message_count: message_count.max(0) as u32,
        status: status_from_db(&status_str)?,
    })
}
