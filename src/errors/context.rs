//! Error context helpers for richer error messages with request tracking.
//!
//! This module provides utilities for adding context to errors, making debugging
//! easier in production by including request IDs, timestamps, and other metadata.
//!
//! # Examples
//!
//! ```rust
//! use caliche::errors::{ErrorContext, ErrorContextExt};
//! use anyhow::Result;
//!
//! # fn example() -> Result<()> {
//! // Automatically add request ID and timestamp
//! let error = anyhow::anyhow!("LLM request failed");
//! let contextualized = error.with_context_auto("gpt-4", Some(1500));
//!
//! // Manual context creation
//! let ctx = ErrorContext::new("req_123")
//!     .with_model("claude-3")
//!     .with_tokens(1500)
//!     .with_metadata("endpoint", "/api/chat");
//!
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::SystemTime;

/// Context information to attach to errors for better debugging.
///
/// This struct captures important metadata about a request or operation
/// that can help diagnose issues in production.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Unique request identifier
    pub request_id: String,

    /// Timestamp when the error occurred
    pub timestamp: SystemTime,

    /// Model or service name
    pub model: Option<String>,

    /// Number of tokens processed (if applicable)
    pub tokens: Option<usize>,

    /// Additional arbitrary metadata
    pub metadata: HashMap<String, String>,
}

impl ErrorContext {
    /// Create a new error context with the given request ID.
    ///
    /// # Example
    ///
    /// ```rust
    /// use caliche::errors::ErrorContext;
    ///
    /// let ctx = ErrorContext::new("req_abc123");
    /// ```
    pub fn new(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            timestamp: SystemTime::now(),
            model: None,
            tokens: None,
            metadata: HashMap::new(),
        }
    }

    /// Generate a new error context with an auto-generated request ID.
    ///
    /// # Example
    ///
    /// ```rust
    /// use caliche::errors::ErrorContext;
    ///
    /// let ctx = ErrorContext::auto();
    /// println!("Request ID: {}", ctx.request_id);
    /// ```
    pub fn auto() -> Self {
        // UUID v7 is timestamp-ordered with embedded randomness, guaranteeing
        // uniqueness even under high concurrency.  The `uuid` crate with the
        // `v7` feature is already a workspace dependency.
        let request_id = format!("req_{}", uuid::Uuid::now_v7());
        Self::new(request_id)
    }

    /// Add model information to the context.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Add token count to the context.
    pub fn with_tokens(mut self, tokens: usize) -> Self {
        self.tokens = Some(tokens);
        self
    }

    /// Add arbitrary metadata to the context.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Format the context as a human-readable string.
    #[must_use]
    pub fn format(&self) -> String {
        let timestamp_str = self
            .timestamp
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let mut parts = vec![
            format!("request_id={}", self.request_id),
            format!("timestamp={}", timestamp_str),
        ];

        if let Some(ref model) = self.model {
            parts.push(format!("model={}", model));
        }

        if let Some(tokens) = self.tokens {
            parts.push(format!("tokens={}", tokens));
        }

        for (key, value) in &self.metadata {
            parts.push(format!("{}={}", key, value));
        }

        parts.join(", ")
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format())
    }
}

/// Extension trait for adding context to errors.
///
/// This trait provides convenient methods for enriching errors with
/// contextual information.
pub trait ErrorContextExt: Sized {
    /// Add error context to this error.
    ///
    /// # Example
    ///
    /// ```rust
    /// use caliche::errors::{ErrorContext, ErrorContextExt};
    /// use anyhow::Result;
    ///
    /// # fn example() -> Result<()> {
    /// let ctx = ErrorContext::auto()
    ///     .with_model("gpt-4")
    ///     .with_tokens(1000);
    ///
    /// let error = anyhow::anyhow!("Failed to generate response");
    /// let enriched = error.with_error_context(ctx);
    /// # Ok(())
    /// # }
    /// ```
    fn with_error_context(self, context: ErrorContext) -> Self;

    /// Add error context with auto-generated request ID.
    ///
    /// # Example
    ///
    /// ```rust
    /// use caliche::errors::ErrorContextExt;
    /// use anyhow::Result;
    ///
    /// # fn example() -> Result<()> {
    /// let error = anyhow::anyhow!("API call failed");
    /// let enriched = error.with_context_auto("claude-3", Some(2000));
    /// # Ok(())
    /// # }
    /// ```
    fn with_context_auto(self, model: &str, tokens: impl Into<Option<usize>>) -> Self;

    /// Add a simple request ID to the error.
    fn with_request_id(self, request_id: impl Into<String>) -> Self;
}

impl ErrorContextExt for anyhow::Error {
    fn with_error_context(self, context: ErrorContext) -> Self {
        self.context(format!("Error context: {}", context))
    }

    fn with_context_auto(self, model: &str, tokens: impl Into<Option<usize>>) -> Self {
        let mut ctx = ErrorContext::auto().with_model(model);
        if let Some(t) = tokens.into() {
            ctx = ctx.with_tokens(t);
        }
        self.with_error_context(ctx)
    }

    fn with_request_id(self, request_id: impl Into<String>) -> Self {
        let ctx = ErrorContext::new(request_id);
        self.with_error_context(ctx)
    }
}

impl<T> ErrorContextExt for Result<T, anyhow::Error> {
    fn with_error_context(self, context: ErrorContext) -> Self {
        self.map_err(|e| e.with_error_context(context))
    }

    fn with_context_auto(self, model: &str, tokens: impl Into<Option<usize>>) -> Self {
        self.map_err(|e| e.with_context_auto(model, tokens.into()))
    }

    fn with_request_id(self, request_id: impl Into<String>) -> Self {
        self.map_err(|e| e.with_request_id(request_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let ctx = ErrorContext::new("req_123")
            .with_model("gpt-4")
            .with_tokens(1500)
            .with_metadata("endpoint", "/api/chat");

        assert_eq!(ctx.request_id, "req_123");
        assert_eq!(ctx.model, Some("gpt-4".to_string()));
        assert_eq!(ctx.tokens, Some(1500));
        assert_eq!(ctx.metadata.get("endpoint"), Some(&"/api/chat".to_string()));
    }

    #[test]
    fn test_error_context_auto() {
        let ctx = ErrorContext::auto();
        assert!(ctx.request_id.starts_with("req_"));
        assert!(ctx.request_id.len() > 10); // At least "req_" + some ID
    }

    #[test]
    fn test_error_context_format() {
        let ctx = ErrorContext::new("req_123")
            .with_model("gpt-4")
            .with_tokens(1000);

        let formatted = ctx.format();
        assert!(formatted.contains("request_id=req_123"));
        assert!(formatted.contains("model=gpt-4"));
        assert!(formatted.contains("tokens=1000"));
    }

    #[test]
    fn test_error_context_ext() {
        let error = anyhow::anyhow!("Test error");
        let ctx = ErrorContext::new("req_123").with_model("gpt-4");
        let enriched = error.with_error_context(ctx);

        let error_str = format!("{}", enriched);
        assert!(error_str.contains("Error context"));
        assert!(error_str.contains("req_123"));
    }

    #[test]
    fn test_with_context_auto() {
        let error = anyhow::anyhow!("API failure");
        let enriched = error.with_context_auto("claude-3", Some(1500));

        let error_str = format!("{}", enriched);
        assert!(error_str.contains("Error context"));
        assert!(error_str.contains("model=claude-3"));
        assert!(error_str.contains("tokens=1500"));
    }
}
