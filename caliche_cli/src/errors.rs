//! CLI error types for structured error handling.
//!
//! This module provides domain-specific error types that replace generic `anyhow::Error`
//! usage, enabling better error categorization, recovery, and user-facing messages.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// CLI-specific errors with structured categorization.
#[derive(Error, Debug)]
pub enum CliError {
    /// Configuration-related errors (invalid config, missing values, etc.)
    #[error("Configuration error: {0}")]
    Config(String),

    /// Credential management errors
    #[error("Credential error: {0}")]
    Credential(#[from] CredentialError),

    /// Command execution errors
    #[error("Command error: {0}")]
    Command(String),

    /// I/O errors (file operations, terminal, etc.)
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Agent/service errors from the core library
    #[error("Agent error: {0}")]
    Agent(String),

    /// Session management errors
    #[error("Session error: {0}")]
    Session(#[from] SessionError),

    /// Storage/database errors
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    /// Input parsing errors
    #[error("Input error: {0}")]
    Input(String),

    /// Provider-related errors
    #[error("Provider error: {0}")]
    Provider(String),

    /// Interactive mode errors
    #[error("Interactive error: {0}")]
    Interactive(String),

    /// User cancelled the operation
    #[error("Operation cancelled")]
    Cancelled,

    /// Stream/response errors
    #[error("Stream error: {0}")]
    Stream(#[from] StreamError),

    /// Generic error with source context
    #[error("{context}: {source}")]
    WithContext {
        context: String,
        #[source]
        source: Box<CliError>,
    },
}

/// Credential-specific errors.
#[derive(Error, Debug)]
pub enum CredentialError {
    /// Keyring access failed
    #[error("Keyring access failed: {0}")]
    KeyringAccess(String),

    /// Credential not found
    #[error("Credential not found for provider: {0}")]
    NotFound(String),

    /// Invalid credential format
    #[error("Invalid credential format: {0}")]
    InvalidFormat(String),

    /// Provider not supported
    #[error("Provider not supported: {0}")]
    UnsupportedProvider(String),

    /// Environment variable not set
    #[error("Environment variable '{var}' not set for provider '{provider}'")]
    EnvVarNotSet { provider: String, var: String },
}

/// Session management errors.
#[derive(Error, Debug)]
pub enum SessionError {
    /// Session not found
    #[error("Session not found: {0}")]
    NotFound(String),

    /// Session already exists
    #[error("Session already exists: {0}")]
    AlreadyExists(String),

    /// Invalid session state
    #[error("Invalid session state: {0}")]
    InvalidState(String),

    /// Session serialization failed
    #[error("Failed to serialize session: {0}")]
    Serialization(String),

    /// Session deserialization failed
    #[error("Failed to deserialize session: {0}")]
    Deserialization(String),

    /// Checkpoint error
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),
}

/// Storage/database errors.
#[derive(Error, Debug)]
pub enum StorageError {
    /// Database connection failed
    #[error("Database connection failed: {0}")]
    Connection(String),

    /// Query execution failed
    #[error("Query failed: {0}")]
    Query(String),

    /// File not found
    #[error("File not found: {}", .0.display())]
    FileNotFound(PathBuf),

    /// Permission denied
    #[error("Permission denied: {}", .0.display())]
    PermissionDenied(PathBuf),
}

/// Stream-related errors
#[derive(Error, Debug)]
pub enum StreamError {
    /// Stream completed with no content
    #[error("No response received from the model")]
    EmptyResponse,

    /// Stream was interrupted
    #[error("Stream interrupted: {0}")]
    Interrupted(String),

    /// Stream channel error
    #[error("Stream channel error: {0}")]
    Channel(String),

    /// Stream timeout
    #[error("Stream timed out after {0} seconds")]
    Timeout(u64),
}

/// Result type alias for CLI operations.
pub type CliResult<T> = Result<T, CliError>;

// Conversion implementations

impl From<anyhow::Error> for CliError {
    fn from(e: anyhow::Error) -> Self {
        // Try to downcast to known error types
        if let Some(io_err) = e.downcast_ref::<io::Error>() {
            return CliError::Io(io::Error::new(io_err.kind(), io_err.to_string()));
        }
        // Default to command error for generic anyhow errors
        CliError::Command(e.to_string())
    }
}

impl From<serde_json::Error> for CliError {
    fn from(e: serde_json::Error) -> Self {
        CliError::Input(format!("JSON error: {}", e))
    }
}

impl From<sqlx::Error> for StorageError {
    fn from(e: sqlx::Error) -> Self {
        match e {
            sqlx::Error::RowNotFound => StorageError::Query("Row not found".to_string()),
            sqlx::Error::Configuration(msg) => StorageError::Connection(msg.to_string()),
            other => StorageError::Query(other.to_string()),
        }
    }
}

impl From<sqlx::Error> for CliError {
    fn from(e: sqlx::Error) -> Self {
        CliError::Storage(StorageError::from(e))
    }
}

// Extension trait for adding context to errors
pub trait CliErrorExt<T> {
    /// Add context to an error.
    fn cli_context<C: Into<String>>(self, context: C) -> CliResult<T>;
}

impl<T, E: Into<CliError>> CliErrorExt<T> for Result<T, E> {
    fn cli_context<C: Into<String>>(self, context: C) -> CliResult<T> {
        self.map_err(|e| CliError::WithContext {
            context: context.into(),
            source: Box::new(e.into()),
        })
    }
}

impl<T> CliErrorExt<T> for Option<T> {
    fn cli_context<C: Into<String>>(self, context: C) -> CliResult<T> {
        self.ok_or_else(|| CliError::Command(context.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CliError::Config("missing api key".to_string());
        assert_eq!(err.to_string(), "Configuration error: missing api key");
    }

    #[test]
    fn test_credential_error_conversion() {
        let cred_err = CredentialError::NotFound("openai".to_string());
        let cli_err: CliError = cred_err.into();
        assert!(matches!(cli_err, CliError::Credential(_)));
    }

    #[test]
    fn test_error_context() {
        let result: Result<(), io::Error> =
            Err(io::Error::new(io::ErrorKind::NotFound, "file missing"));
        let with_context = result.cli_context("loading config");
        assert!(with_context.is_err());
        let err = with_context.unwrap_err();
        assert!(err.to_string().contains("loading config"));
    }
}
