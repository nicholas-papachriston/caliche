//! Private temporary files for editor integrations.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

/// Create a private (0600 on Unix) temporary file in the system temp directory.
pub fn create_private_temp_file(prefix: &str, suffix: &str) -> Result<NamedTempFile> {
    let file = tempfile::Builder::new()
        .prefix(prefix)
        .suffix(suffix)
        .tempfile_in(std::env::temp_dir())
        .context("Failed to create temporary file")?;
    restrict_temp_file_permissions(file.path())?;
    Ok(file)
}

fn restrict_temp_file_permissions(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("Failed to chmod 0600 {}", path.display()))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

/// Stable path for a [`NamedTempFile`] while it remains in scope.
pub fn temp_file_path(file: &NamedTempFile) -> PathBuf {
    file.path().to_path_buf()
}
