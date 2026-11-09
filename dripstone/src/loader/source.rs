//! Model source types for loading from local files or `HuggingFace` Hub.

use crate::errors::DripstoneError;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Unified model source - local path or `HuggingFace` Hub.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// Local file path on disk
    Local { path: PathBuf },
    /// `HuggingFace` Hub repository
    HuggingFace {
        repo: String,
        revision: Option<String>,
    },
}

impl ModelSource {
    /// Auto-detect model source from a string.
    ///
    /// If the string is an absolute path, starts with `./` or `../`, or points to an existing file,
    /// it's treated as a local path. Otherwise, it's treated as a `HuggingFace` repository ID.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.parse() {
            Ok(src) => src,
            Err(e) => match e {},
        }
    }

    /// Create a local model source from a path.
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::Local { path: path.into() }
    }

    /// Create a `HuggingFace` model source.
    pub fn huggingface(repo: impl Into<String>, revision: Option<String>) -> Self {
        Self::HuggingFace {
            repo: repo.into(),
            revision,
        }
    }

    /// Check if this is a local source.
    #[must_use]
    pub const fn is_local(&self) -> bool {
        matches!(self, Self::Local { .. })
    }

    /// Check if this is a `HuggingFace` source.
    #[must_use]
    pub const fn is_huggingface(&self) -> bool {
        matches!(self, Self::HuggingFace { .. })
    }

    /// Get the local path if this is a local source.
    #[must_use]
    pub fn local_path(&self) -> Option<&Path> {
        match self {
            Self::Local { path } => Some(path),
            Self::HuggingFace { .. } => None,
        }
    }

    /// Get the `HuggingFace` repo if this is a `HuggingFace` source.
    #[must_use]
    pub fn huggingface_repo(&self) -> Option<(&str, Option<&str>)> {
        match self {
            Self::HuggingFace { repo, revision } => Some((repo.as_str(), revision.as_deref())),
            Self::Local { .. } => None,
        }
    }

    /// Validate that a local path exists (file or directory).
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn validate_local(&self) -> Result<(), DripstoneError> {
        match self {
            Self::Local { path } => {
                if !path.exists() {
                    return Err(DripstoneError::storage(format!(
                        "Model path not found: '{}'",
                        path.display()
                    )));
                }
                // Allow both files and directories - inference backends load from directories
                Ok(())
            }
            Self::HuggingFace { .. } => Ok(()),
        }
    }
}

impl FromStr for ModelSource {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let path = Path::new(s);
        Ok(
            if path.is_absolute() || s.starts_with("./") || s.starts_with("../") || path.exists() {
                Self::Local {
                    path: path.to_path_buf(),
                }
            } else {
                Self::HuggingFace {
                    repo: s.to_string(),
                    revision: None,
                }
            },
        )
    }
}

impl From<PathBuf> for ModelSource {
    fn from(path: PathBuf) -> Self {
        Self::Local { path }
    }
}

impl From<&Path> for ModelSource {
    fn from(path: &Path) -> Self {
        Self::Local {
            path: path.to_path_buf(),
        }
    }
}

impl From<String> for ModelSource {
    fn from(s: String) -> Self {
        Self::from_str(&s)
    }
}

impl From<&str> for ModelSource {
    fn from(s: &str) -> Self {
        Self::from_str(s)
    }
}
