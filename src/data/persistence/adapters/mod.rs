#[cfg(feature = "mysql")]
pub mod mysql;

#[cfg(feature = "sqlx")]
pub mod sqlite;

#[cfg(feature = "mysql")]
pub use mysql::MySqlSchemaIntrospector;

#[cfg(feature = "sqlx")]
pub use sqlite::SqliteSchemaIntrospector;
