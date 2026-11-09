//! Structured error types for JavaScript bindings

/// Binding error types for better error handling
#[derive(Debug)]
#[expect(dead_code, reason = "nothing returns this error type yet")]
pub enum BindingError {
    /// Conversion error between JavaScript and Rust types
    Conversion { message: String },
    /// Runtime error from agent operations
    Runtime { source: anyhow::Error },
    /// Invalid argument error
    InvalidArgument { name: String, reason: String },
}

impl std::fmt::Display for BindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conversion { message } => {
                write!(f, "Conversion error: {message}")
            }
            Self::Runtime { source } => {
                write!(f, "Runtime error: {source}")
            }
            Self::InvalidArgument { name, reason } => {
                write!(f, "Invalid argument '{name}': {reason}")
            }
        }
    }
}

impl std::error::Error for BindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime { source } => Some(source.as_ref()),
            _ => None,
        }
    }
}

/// Helper to convert `anyhow::Error` to `BindingError`
impl From<anyhow::Error> for BindingError {
    fn from(err: anyhow::Error) -> Self {
        Self::Runtime { source: err }
    }
}

/// Helper to convert String to `BindingError`
impl From<String> for BindingError {
    fn from(msg: String) -> Self {
        Self::Conversion { message: msg }
    }
}
