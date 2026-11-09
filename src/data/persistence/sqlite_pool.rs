//! Pool limits for a local SQLite file.
//!
//! Each SQLite connection owns one worker thread. One process opens several
//! of these files. Two connections cover one writer and one WAL reader.
//! Idle connections close so those threads do not stay up.

use sqlx::sqlite::SqlitePoolOptions;
use std::time::Duration;

/// Upper bound for one local SQLite file.
pub(crate) const LOCAL_SQLITE_MAX_CONNECTIONS: u32 = 2;

const LOCAL_SQLITE_IDLE: Duration = Duration::from_secs(30);

/// Pool options for a local SQLite file.
pub(crate) fn local_sqlite_pool() -> SqlitePoolOptions {
    SqlitePoolOptions::new()
        .max_connections(LOCAL_SQLITE_MAX_CONNECTIONS)
        .min_connections(0)
        .idle_timeout(LOCAL_SQLITE_IDLE)
}

#[cfg(test)]
mod tests {
    use super::{LOCAL_SQLITE_MAX_CONNECTIONS, local_sqlite_pool};

    #[test]
    fn local_sqlite_pool_caps_connections_at_two() {
        assert_eq!(LOCAL_SQLITE_MAX_CONNECTIONS, 2);
        assert_eq!(
            local_sqlite_pool().get_max_connections(),
            LOCAL_SQLITE_MAX_CONNECTIONS
        );
    }
}
