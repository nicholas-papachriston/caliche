//! SQLite-backed index metadata for incremental workspace indexing.
//!
//! We store file records in `indexing.sqlite` so unchanged files can be skipped
//! without trusting second-granularity mtimes alone.

use sha2::{Digest, Sha256};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqliteSynchronous};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, UNIX_EPOCH};

const INDEX_DB_FILENAME: &str = "indexing.sqlite";

/// In-memory view of persisted file mtimes.
#[derive(Debug, Default, Clone)]
pub struct IndexManifest {
    pub indexed: HashMap<String, IndexedFileRecord>,
    pub fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedFileRecord {
    pub mtime: u64,
    pub size: u64,
    pub checksum: String,
}

impl IndexManifest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_fingerprint(fingerprint: Option<String>) -> Self {
        Self {
            indexed: HashMap::new(),
            fingerprint,
        }
    }

    pub fn merge_from_indexed(
        &mut self,
        entries: impl IntoIterator<Item = (String, IndexedFileRecord)>,
    ) {
        for (p, record) in entries {
            self.indexed.insert(p, record);
        }
    }

    pub fn remove(&mut self, path: &str) {
        self.indexed.remove(path);
    }

    pub fn is_unchanged(&self, path: &str, record: &IndexedFileRecord) -> bool {
        self.indexed.get(path) == Some(record)
    }

    /// Load metadata from SQLite.
    pub async fn load(dir: &Path) -> anyhow::Result<Self> {
        Self::load_with_fingerprint(dir, None).await
    }

    /// Load metadata and validate the current index fingerprint.
    ///
    /// If the persisted fingerprint differs, the vector DB and file manifest are
    /// cleared so embeddings from incompatible providers/models are not mixed.
    pub async fn load_with_fingerprint(
        dir: &Path,
        fingerprint: Option<&str>,
    ) -> anyhow::Result<Self> {
        tokio::fs::create_dir_all(dir).await?;
        let db_path = Self::db_path(dir);
        let pool = Self::open_pool(&db_path).await?;
        Self::ensure_schema(&pool).await?;

        let persisted_fingerprint = Self::load_fingerprint(&pool).await?;
        if let Some(expected) = fingerprint {
            if persisted_fingerprint
                .as_deref()
                .is_some_and(|got| got != expected)
            {
                Self::clear_vector_artifacts(dir).await?;
                sqlx::query("DELETE FROM indexed_files")
                    .execute(&pool)
                    .await?;
                Self::save_fingerprint(&pool, expected).await?;
                return Ok(Self::with_fingerprint(Some(expected.to_string())));
            }
            if persisted_fingerprint.is_none() {
                Self::save_fingerprint(&pool, expected).await?;
            }
        }

        let rows: Vec<(String, i64, i64, Option<String>)> =
            sqlx::query_as("SELECT path, mtime, size, checksum FROM indexed_files")
                .fetch_all(&pool)
                .await?;
        let mut indexed = HashMap::with_capacity(rows.len());
        for (path, mtime, size, checksum) in rows {
            if let (Ok(mtime), Ok(size)) = (u64::try_from(mtime), u64::try_from(size)) {
                indexed.insert(
                    path,
                    IndexedFileRecord {
                        mtime,
                        size,
                        checksum: checksum.unwrap_or_default(),
                    },
                );
            }
        }

        Ok(Self {
            indexed,
            fingerprint: fingerprint.map(str::to_string).or(persisted_fingerprint),
        })
    }

    /// True when stored mtime and size both match, so the file can skip a content read.
    #[must_use]
    pub fn matches_stat(&self, path: &str, mtime: u64, size: u64) -> bool {
        self.indexed
            .get(path)
            .is_some_and(|record| record.mtime == mtime && record.size == size)
    }

    /// Save metadata into SQLite (`indexed_files` table).
    ///
    /// Uses upsert + delete-missing so unchanged rows are not wiped and rewritten.
    pub async fn save(&self, dir: &Path) -> anyhow::Result<()> {
        tokio::fs::create_dir_all(dir).await?;
        let db_path = Self::db_path(dir);
        let pool = Self::open_pool(&db_path).await?;
        Self::ensure_schema(&pool).await?;

        let mut tx = pool.begin().await?;
        sqlx::query(
            "CREATE TEMP TABLE IF NOT EXISTS indexed_files_keep (
                path TEXT PRIMARY KEY
            )",
        )
        .execute(&mut *tx)
        .await?;
        sqlx::query("DELETE FROM indexed_files_keep")
            .execute(&mut *tx)
            .await?;
        for path in self.indexed.keys() {
            sqlx::query("INSERT OR REPLACE INTO indexed_files_keep(path) VALUES (?)")
                .bind(path)
                .execute(&mut *tx)
                .await?;
        }
        sqlx::query(
            "DELETE FROM indexed_files
             WHERE path NOT IN (SELECT path FROM indexed_files_keep)",
        )
        .execute(&mut *tx)
        .await?;
        for (path, record) in &self.indexed {
            Self::upsert_file(&mut tx, path, record).await?;
        }
        if let Some(fingerprint) = &self.fingerprint {
            Self::save_fingerprint_in_tx(&mut tx, fingerprint).await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Persist only the given upserts and removals. Faster than [`save`] for incremental updates.
    pub async fn save_delta(
        &self,
        dir: &Path,
        upsert_paths: &[String],
        removed_paths: &[String],
    ) -> anyhow::Result<()> {
        if upsert_paths.is_empty() && removed_paths.is_empty() && self.fingerprint.is_none() {
            return Ok(());
        }
        tokio::fs::create_dir_all(dir).await?;
        let db_path = Self::db_path(dir);
        let pool = Self::open_pool(&db_path).await?;
        Self::ensure_schema(&pool).await?;

        let mut tx = pool.begin().await?;
        for path in removed_paths {
            sqlx::query("DELETE FROM indexed_files WHERE path = ?")
                .bind(path)
                .execute(&mut *tx)
                .await?;
        }
        for path in upsert_paths {
            let Some(record) = self.indexed.get(path) else {
                continue;
            };
            Self::upsert_file(&mut tx, path, record).await?;
        }
        if let Some(fingerprint) = &self.fingerprint {
            Self::save_fingerprint_in_tx(&mut tx, fingerprint).await?;
        }
        tx.commit().await?;
        Ok(())
    }

    async fn upsert_file(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        path: &str,
        record: &IndexedFileRecord,
    ) -> anyhow::Result<()> {
        let mtime_i64 = i64::try_from(record.mtime).unwrap_or(i64::MAX);
        let size_i64 = i64::try_from(record.size).unwrap_or(i64::MAX);
        sqlx::query(
            "INSERT INTO indexed_files(path, mtime, size, checksum, updated_at)
             VALUES(?, ?, ?, ?, strftime('%s','now'))
             ON CONFLICT(path) DO UPDATE SET
                mtime = excluded.mtime,
                size = excluded.size,
                checksum = excluded.checksum,
                updated_at = excluded.updated_at",
        )
        .bind(path)
        .bind(mtime_i64)
        .bind(size_i64)
        .bind(&record.checksum)
        .execute(&mut **tx)
        .await?;
        Ok(())
    }

    async fn save_fingerprint_in_tx(
        tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
        fingerprint: &str,
    ) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO index_metadata(key, value)
             VALUES('fingerprint', ?)",
        )
        .bind(fingerprint)
        .execute(&mut **tx)
        .await?;
        Ok(())
    }

    fn db_path(dir: &Path) -> PathBuf {
        dir.join(INDEX_DB_FILENAME)
    }

    async fn open_pool(db_path: &Path) -> anyhow::Result<sqlx::SqlitePool> {
        let options = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5))
            .synchronous(SqliteSynchronous::Normal);
        let pool = crate::data::persistence::sqlite_pool::local_sqlite_pool()
            .max_connections(1)
            .connect_with(options)
            .await?;
        Ok(pool)
    }

    async fn ensure_schema(pool: &sqlx::SqlitePool) -> anyhow::Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS indexed_files(
                path TEXT PRIMARY KEY,
                mtime INTEGER NOT NULL,
                size INTEGER NOT NULL DEFAULT 0,
                checksum TEXT NULL,
                updated_at INTEGER NOT NULL
            )",
        )
        .execute(pool)
        .await?;
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS index_metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
        )
        .execute(pool)
        .await?;
        Self::ensure_column(pool, "indexed_files", "size", "INTEGER NOT NULL DEFAULT 0").await?;
        Self::ensure_column(pool, "indexed_files", "checksum", "TEXT NULL").await?;
        Ok(())
    }

    async fn ensure_column(
        pool: &sqlx::SqlitePool,
        table: &str,
        column: &str,
        definition: &str,
    ) -> anyhow::Result<()> {
        let pragma = format!("PRAGMA table_info({table})");
        let rows: Vec<(i64, String, String, i64, Option<String>, i64)> =
            sqlx::query_as(&pragma).fetch_all(pool).await?;
        if rows.iter().any(|(_, name, _, _, _, _)| name == column) {
            return Ok(());
        }
        let alter = format!("ALTER TABLE {table} ADD COLUMN {column} {definition}");
        sqlx::query(&alter).execute(pool).await?;
        Ok(())
    }

    async fn load_fingerprint(pool: &sqlx::SqlitePool) -> anyhow::Result<Option<String>> {
        let value: Option<(String,)> =
            sqlx::query_as("SELECT value FROM index_metadata WHERE key = 'fingerprint'")
                .fetch_optional(pool)
                .await?;
        Ok(value.map(|(value,)| value))
    }

    async fn save_fingerprint(pool: &sqlx::SqlitePool, fingerprint: &str) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO index_metadata(key, value)
             VALUES('fingerprint', ?)",
        )
        .bind(fingerprint)
        .execute(pool)
        .await?;
        Ok(())
    }

    async fn clear_vector_artifacts(dir: &Path) -> anyhow::Result<()> {
        for file in ["vectors.db", "vectors.db-wal", "vectors.db-shm"] {
            let path = dir.join(file);
            if path.exists() {
                tokio::fs::remove_file(path).await?;
            }
        }
        Ok(())
    }
}

/// Stat a file as `(mtime_secs, size_bytes)` without reading content.
pub async fn file_stat(p: &Path) -> anyhow::Result<(u64, u64)> {
    let meta = tokio::fs::metadata(p).await?;
    let mtime = meta.modified()?;
    let secs = mtime
        .duration_since(UNIX_EPOCH)
        .map_err(|_| anyhow::anyhow!("mtime before epoch"))?
        .as_secs();
    Ok((secs, meta.len()))
}

/// Get mtime of path as seconds since Unix epoch.
pub async fn mtime_secs(p: &Path) -> anyhow::Result<u64> {
    Ok(file_stat(p).await?.0)
}

/// Build a manifest record for a file from its metadata and already-read content.
pub async fn file_record_with_content(
    p: &Path,
    content: &str,
) -> anyhow::Result<IndexedFileRecord> {
    let (secs, size) = file_stat(p).await?;
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let checksum = format!("{:x}", hasher.finalize());
    Ok(IndexedFileRecord {
        mtime: secs,
        size,
        checksum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn load_with_fingerprint_clears_stale_vector_artifacts_on_mismatch() {
        let dir = TempDir::new().expect("temp dir");
        tokio::fs::write(dir.path().join("vectors.db"), b"old vectors")
            .await
            .expect("write vectors");

        let mut manifest = IndexManifest::with_fingerprint(Some("old".to_string()));
        manifest.merge_from_indexed([(
            "src/lib.rs".to_string(),
            IndexedFileRecord {
                mtime: 1,
                size: 10,
                checksum: "abc".to_string(),
            },
        )]);
        manifest.save(dir.path()).await.expect("save manifest");

        let loaded = IndexManifest::load_with_fingerprint(dir.path(), Some("new"))
            .await
            .expect("load with new fingerprint");

        assert!(loaded.indexed.is_empty());
        assert_eq!(loaded.fingerprint.as_deref(), Some("new"));
        assert!(!dir.path().join("vectors.db").exists());
    }

    #[tokio::test]
    async fn save_and_load_round_trips_indexed_files() {
        let dir = TempDir::new().expect("temp dir");
        let mut manifest = IndexManifest::with_fingerprint(Some("fp".to_string()));
        manifest.merge_from_indexed([(
            "src/main.rs".to_string(),
            IndexedFileRecord {
                mtime: 123,
                size: 45,
                checksum: "abc".to_string(),
            },
        )]);
        manifest.save(dir.path()).await.expect("save");

        let loaded = IndexManifest::load(dir.path()).await.expect("load");
        assert_eq!(loaded.indexed, manifest.indexed);
        assert_eq!(loaded.fingerprint.as_deref(), Some("fp"));
    }

    #[tokio::test]
    async fn save_delta_upserts_and_removes_without_rewriting_other_rows() {
        let dir = TempDir::new().expect("temp dir");
        let mut manifest = IndexManifest::with_fingerprint(Some("fp".to_string()));
        manifest.merge_from_indexed([
            (
                "keep.rs".to_string(),
                IndexedFileRecord {
                    mtime: 1,
                    size: 2,
                    checksum: "keep".to_string(),
                },
            ),
            (
                "gone.rs".to_string(),
                IndexedFileRecord {
                    mtime: 3,
                    size: 4,
                    checksum: "gone".to_string(),
                },
            ),
        ]);
        manifest.save(dir.path()).await.expect("save");

        manifest.remove("gone.rs");
        manifest.merge_from_indexed([(
            "new.rs".to_string(),
            IndexedFileRecord {
                mtime: 5,
                size: 6,
                checksum: "new".to_string(),
            },
        )]);
        manifest
            .save_delta(
                dir.path(),
                &["new.rs".to_string()],
                &["gone.rs".to_string()],
            )
            .await
            .expect("delta");

        let loaded = IndexManifest::load(dir.path()).await.expect("load");
        assert!(loaded.indexed.contains_key("keep.rs"));
        assert!(loaded.indexed.contains_key("new.rs"));
        assert!(!loaded.indexed.contains_key("gone.rs"));
        assert_eq!(loaded.fingerprint.as_deref(), Some("fp"));
    }
}
