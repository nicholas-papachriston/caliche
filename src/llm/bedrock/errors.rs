use std::fmt;

#[derive(Debug)]
pub(super) struct HeaderError(pub(super) String);

impl std::error::Error for HeaderError {}

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Header error: {}", self.0)
    }
}

/// Returns the Authorization header value for Bearer token auth.
/// Used by Bedrock when using bearer auth; the caller inserts it into the request.
pub(super) fn bearer_header_value(token: &str) -> Result<http::HeaderValue, HeaderError> {
    http::HeaderValue::from_str(&format!("Bearer {token}"))
        .map_err(|e| HeaderError(format!("Failed to create header: {e}")))
}
