use super::types::{ForwardSessionRecord, GatewayRuntimeEntry, GatewayRuntimeRegistry};
use anyhow::{Context, Result};
use chrono::Utc;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub(super) fn normalize_cwd(cwd: Option<&Path>) -> Result<String> {
    let dir = match cwd {
        Some(p) => p.to_path_buf(),
        None => std::env::current_dir().context("failed to read current directory")?,
    };
    let canonical = dir.canonicalize().unwrap_or(dir);
    Ok(canonical.display().to_string())
}

fn runtime_db_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("failed to resolve home directory")?;
    Ok(home.join(".caliche").join("gateway-cluster.db"))
}

pub(super) fn logs_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("failed to resolve home directory")?;
    Ok(home.join(".caliche").join("logs").join("gateway"))
}

pub(super) fn next_log_path() -> Result<PathBuf> {
    let dir = logs_dir()?;
    std::fs::create_dir_all(&dir).context("failed to create gateway log directory")?;
    Ok(dir.join(format!(
        "gateway-{}.log",
        Utc::now().format("%Y%m%d-%H%M%S")
    )))
}

pub(super) async fn runtime_db_pool() -> Result<SqlitePool> {
    let db_path = runtime_db_path()?;
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).context("failed to create gateway db directory")?;
    }
    let dsn = format!("sqlite://{}", db_path.display());
    let options = SqliteConnectOptions::from_str(&dsn)
        .context("failed to build gateway sqlite connect options")?
        .create_if_missing(true);
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await
        .context("failed to open gateway sqlite database")?;
    ensure_runtime_tables(&pool).await?;
    Ok(pool)
}

async fn ensure_runtime_tables(pool: &SqlitePool) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS gateway_runtime_entries (
            cwd TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            pid INTEGER NOT NULL,
            port INTEGER NOT NULL,
            role TEXT NOT NULL,
            main_gateway_url TEXT,
            started_at TEXT NOT NULL,
            log_path TEXT NOT NULL,
            forward_provider TEXT,
            forward_tunnel_pid INTEGER,
            forward_started_at TEXT,
            forward_local_port INTEGER,
            forward_log_path TEXT,
            forward_public_url TEXT
        )
        "#,
    )
    .execute(pool)
    .await
    .context("failed to ensure gateway runtime table")?;
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS gateway_runtime_settings (
            cwd TEXT PRIMARY KEY,
            provider TEXT,
            model TEXT,
            updated_at TEXT NOT NULL
        )
        "#,
    )
    .execute(pool)
    .await
    .context("failed to ensure gateway runtime settings table")?;
    Ok(())
}

pub(super) async fn load_registry() -> Result<GatewayRuntimeRegistry> {
    let pool = runtime_db_pool().await?;
    let rows = sqlx::query(
        r#"
        SELECT
            cwd, agent_id, pid, port, role, main_gateway_url, started_at, log_path,
            forward_provider, forward_tunnel_pid, forward_started_at, forward_local_port,
            forward_log_path, forward_public_url
        FROM gateway_runtime_entries
        ORDER BY started_at ASC
        "#,
    )
    .fetch_all(&pool)
    .await
    .context("failed to query gateway runtime registry")?;
    let mut entries = Vec::new();
    for row in rows {
        let forwarding = match (
            row.get::<Option<String>, _>("forward_provider"),
            row.get::<Option<i64>, _>("forward_tunnel_pid"),
            row.get::<Option<String>, _>("forward_started_at"),
            row.get::<Option<i64>, _>("forward_local_port"),
            row.get::<Option<String>, _>("forward_log_path"),
        ) {
            (
                Some(provider),
                Some(tunnel_pid),
                Some(started_at),
                Some(local_port),
                Some(log_path),
            ) => {
                let tunnel_pid_u32 = u32::try_from(tunnel_pid)
                    .context("invalid stored forwarding pid in gateway runtime db")?;
                let local_port_u16 = u16::try_from(local_port)
                    .context("invalid stored forwarding port in gateway runtime db")?;
                Some(ForwardSessionRecord {
                    provider,
                    tunnel_pid: tunnel_pid_u32,
                    started_at,
                    local_port: local_port_u16,
                    log_path,
                    public_url: row.get::<Option<String>, _>("forward_public_url"),
                })
            }
            _ => None,
        };
        let pid_i64 = row.get::<i64, _>("pid");
        let port_i64 = row.get::<i64, _>("port");
        let pid = u32::try_from(pid_i64).context("invalid stored pid in gateway runtime db")?;
        let port = u16::try_from(port_i64).context("invalid stored port in gateway runtime db")?;
        entries.push(GatewayRuntimeEntry {
            agent_id: row.get::<String, _>("agent_id"),
            cwd: row.get::<String, _>("cwd"),
            pid,
            port,
            role: row.get::<String, _>("role"),
            main_gateway_url: row.get::<Option<String>, _>("main_gateway_url"),
            started_at: row.get::<String, _>("started_at"),
            log_path: row.get::<String, _>("log_path"),
            forwarding,
        });
    }
    Ok(GatewayRuntimeRegistry { entries })
}

pub(super) async fn save_registry(registry: &GatewayRuntimeRegistry) -> Result<()> {
    let pool = runtime_db_pool().await?;
    sqlx::query("DELETE FROM gateway_runtime_entries")
        .execute(&pool)
        .await
        .context("failed to clear gateway runtime registry table")?;
    for entry in &registry.entries {
        upsert_runtime_entry(&pool, entry).await?;
    }
    Ok(())
}

async fn upsert_runtime_entry(pool: &SqlitePool, entry: &GatewayRuntimeEntry) -> Result<()> {
    let (
        forward_provider,
        forward_tunnel_pid,
        forward_started_at,
        forward_local_port,
        forward_log_path,
        forward_public_url,
    ) = if let Some(forward) = &entry.forwarding {
        (
            Some(forward.provider.clone()),
            Some(i64::from(forward.tunnel_pid)),
            Some(forward.started_at.clone()),
            Some(i64::from(forward.local_port)),
            Some(forward.log_path.clone()),
            forward.public_url.clone(),
        )
    } else {
        (None, None, None, None, None, None)
    };
    sqlx::query(
        r#"
        INSERT INTO gateway_runtime_entries (
            cwd, agent_id, pid, port, role, main_gateway_url, started_at, log_path,
            forward_provider, forward_tunnel_pid, forward_started_at, forward_local_port,
            forward_log_path, forward_public_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cwd) DO UPDATE SET
            agent_id = excluded.agent_id,
            pid = excluded.pid,
            port = excluded.port,
            role = excluded.role,
            main_gateway_url = excluded.main_gateway_url,
            started_at = excluded.started_at,
            log_path = excluded.log_path,
            forward_provider = excluded.forward_provider,
            forward_tunnel_pid = excluded.forward_tunnel_pid,
            forward_started_at = excluded.forward_started_at,
            forward_local_port = excluded.forward_local_port,
            forward_log_path = excluded.forward_log_path,
            forward_public_url = excluded.forward_public_url
        "#,
    )
    .bind(&entry.cwd)
    .bind(&entry.agent_id)
    .bind(i64::from(entry.pid))
    .bind(i64::from(entry.port))
    .bind(&entry.role)
    .bind(&entry.main_gateway_url)
    .bind(&entry.started_at)
    .bind(&entry.log_path)
    .bind(forward_provider)
    .bind(forward_tunnel_pid)
    .bind(forward_started_at)
    .bind(forward_local_port)
    .bind(forward_log_path)
    .bind(forward_public_url)
    .execute(pool)
    .await
    .context("failed to upsert gateway runtime entry")?;
    Ok(())
}
