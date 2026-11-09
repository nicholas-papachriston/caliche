//! Safe, resumable model artifact downloads with integrity checks.
//!
//! This module is intentionally lightweight (std + optional sha2/hex) so
//! consumers can provision model files without pulling Candle or ORT.

use crate::errors::DripstoneError;
use crate::loader::curl::run_curl;
use crate::loader::disk::{
    ensure_disk_space, probe_remote_content_length, reserve_free_bytes_from_env,
};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

/// Default upper bound for a single downloaded artifact (20 GiB).
pub const DEFAULT_MAX_DOWNLOAD_BYTES: u64 = 20 * 1024 * 1024 * 1024;

/// Options controlling download safety and performance.
#[derive(Debug, Clone)]
pub struct DownloadOptions {
    /// Reject downloads (and existing files) larger than this many bytes.
    /// `None` disables the limit. Defaults to [`DEFAULT_MAX_DOWNLOAD_BYTES`].
    pub max_bytes: Option<u64>,
    /// Optional lowercase hex SHA-256 digest that the final file must match.
    pub expected_sha256: Option<String>,
    /// Optional known size hint used when remote HEAD probing fails.
    pub expected_bytes: Option<u64>,
    /// Free bytes that must remain after the download completes.
    pub reserve_free_bytes: u64,
    /// Probe remote size with `curl -sIL` before downloading.
    pub check_remote_size: bool,
    /// Resume partial downloads with `curl -C -` when a `.partial` file exists.
    pub allow_resume: bool,
    /// Extra HTTP headers (for example `Authorization: Bearer …`).
    pub headers: Vec<(String, String)>,
    /// How long to wait for a peer holding the lockfile.
    pub lock_timeout: Duration,
    /// Poll interval while waiting on a lockfile.
    pub lock_poll: Duration,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            max_bytes: Some(DEFAULT_MAX_DOWNLOAD_BYTES),
            expected_sha256: None,
            expected_bytes: None,
            reserve_free_bytes: reserve_free_bytes_from_env(),
            check_remote_size: true,
            allow_resume: true,
            headers: Vec::new(),
            lock_timeout: Duration::from_secs(30 * 60),
            lock_poll: Duration::from_millis(200),
        }
    }
}

impl DownloadOptions {
    /// Builder: set max bytes (`None` = unlimited).
    #[must_use]
    pub fn with_max_bytes(mut self, max_bytes: Option<u64>) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Builder: require a SHA-256 hex digest.
    #[must_use]
    pub fn with_expected_sha256(mut self, digest: impl Into<String>) -> Self {
        self.expected_sha256 = Some(digest.into());
        self
    }

    /// Builder: provide a known size hint for disk preflight.
    #[must_use]
    pub fn with_expected_bytes(mut self, bytes: Option<u64>) -> Self {
        self.expected_bytes = bytes;
        self
    }

    /// Builder: set post-download free-space reserve.
    #[must_use]
    pub fn with_reserve_free_bytes(mut self, bytes: u64) -> Self {
        self.reserve_free_bytes = bytes;
        self
    }

    /// Builder: enable/disable remote size probing.
    #[must_use]
    pub fn with_check_remote_size(mut self, enabled: bool) -> Self {
        self.check_remote_size = enabled;
        self
    }

    /// Builder: add an HTTP header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Builder: attach a HuggingFace bearer token when present.
    #[must_use]
    pub fn with_huggingface_token(mut self, token: Option<&str>) -> Self {
        if let Some(token) = token.filter(|value| !value.is_empty()) {
            self.headers
                .push(("Authorization".to_string(), format!("Bearer {token}")));
        }
        self
    }
}

/// Build a HuggingFace resolve URL for `repo` / `revision` / `filename`.
#[must_use]
pub fn huggingface_resolve_url(repo: &str, revision: Option<&str>, filename: &str) -> String {
    let rev = revision.unwrap_or("main");
    format!("https://huggingface.co/{repo}/resolve/{rev}/{filename}")
}

/// Reject path traversal and absolute components in a repository-relative path.
///
/// # Errors
///
/// Returns an error when `relative` contains empty, `.`, `..`, or absolute
/// components, or when the resulting path would escape `root`.
pub fn sanitize_relative_path(root: &Path, relative: &str) -> Result<PathBuf, DripstoneError> {
    if relative.is_empty() {
        return Err(DripstoneError::storage("empty relative download path"));
    }
    if relative.starts_with('/') || relative.starts_with('\\') {
        return Err(DripstoneError::storage(format!(
            "absolute download path rejected: '{relative}'"
        )));
    }
    if relative.chars().nth(1) == Some(':') {
        return Err(DripstoneError::storage(format!(
            "absolute download path rejected: '{relative}'"
        )));
    }

    let mut path = PathBuf::new();
    for component in relative.split(['/', '\\']) {
        if component.is_empty() || component == "." {
            continue;
        }
        if component == ".." {
            return Err(DripstoneError::storage(format!(
                "path traversal rejected in download path '{relative}'"
            )));
        }
        if component.contains('\0') {
            return Err(DripstoneError::storage(format!(
                "NUL byte rejected in download path '{relative}'"
            )));
        }
        path.push(component);
    }

    if path.as_os_str().is_empty() {
        return Err(DripstoneError::storage(format!(
            "invalid relative download path '{relative}'"
        )));
    }

    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return Err(DripstoneError::storage(format!(
            "absolute or parent path rejected in download path '{relative}'"
        )));
    }

    let joined = root.join(&path);
    let root_canon = normalize_for_prefix(root)?;
    let joined_canon = normalize_for_prefix(&joined)?;
    if !joined_canon.starts_with(&root_canon) {
        return Err(DripstoneError::storage(format!(
            "download path '{relative}' escapes root '{}'",
            root.display()
        )));
    }

    Ok(joined)
}

/// Ensure `dest` contains the contents of `url`, downloading when needed.
///
/// Downloads write to `dest.partial`, then atomically rename into place.
/// Concurrent callers coordinating on the same `dest` wait on a `.lock` file.
///
/// # Errors
///
/// Returns an error when the path is invalid, the download fails, size/hash
/// checks fail, or the lock cannot be acquired before timeout.
pub fn ensure_downloaded(
    url: &str,
    dest: &Path,
    options: &DownloadOptions,
) -> Result<PathBuf, DripstoneError> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            DripstoneError::storage_with_source(
                format!("failed to create download directory '{}'", parent.display()),
                error,
            )
        })?;
    }

    if dest.is_file() {
        verify_existing_file(dest, options)?;
        return Ok(dest.to_path_buf());
    }

    match acquire_download_lock(dest, options)? {
        LockAcquire::Ready => {
            verify_existing_file(dest, options)?;
            return Ok(dest.to_path_buf());
        }
        LockAcquire::Acquired(_guard) => {
            if dest.is_file() {
                verify_existing_file(dest, options)?;
                return Ok(dest.to_path_buf());
            }

            let partial = partial_path(dest)?;
            preflight_disk_for_download(url, dest, &partial, options)?;
            download_with_curl(url, &partial, options)?;
            verify_existing_file(&partial, options)?;
            fs::rename(&partial, dest).map_err(|error| {
                DripstoneError::storage_with_source(
                    format!(
                        "failed to promote download from '{}' to '{}'",
                        partial.display(),
                        dest.display()
                    ),
                    error,
                )
            })?;

            Ok(dest.to_path_buf())
        }
    }
}

/// Copy `source` to `dest` atomically with optional size/hash verification.
///
/// # Errors
///
/// Returns an error when verification or filesystem operations fail.
pub fn atomic_copy(
    source: &Path,
    dest: &Path,
    options: &DownloadOptions,
) -> Result<(), DripstoneError> {
    verify_existing_file(source, options)?;

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            DripstoneError::storage_with_source(
                format!("failed to create directory '{}'", parent.display()),
                error,
            )
        })?;
    }

    if dest.is_file() {
        verify_existing_file(dest, options)?;
        return Ok(());
    }

    match acquire_download_lock(dest, options)? {
        LockAcquire::Ready => {
            verify_existing_file(dest, options)?;
            Ok(())
        }
        LockAcquire::Acquired(_guard) => {
            if dest.is_file() {
                verify_existing_file(dest, options)?;
                return Ok(());
            }

            let source_len = fs::metadata(source)
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            ensure_disk_space(dest, source_len, options.reserve_free_bytes)?;

            // Prefer a hard link when source and dest share a filesystem so model
            // materialization does not double disk usage.
            match fs::hard_link(source, dest) {
                Ok(()) => {
                    verify_existing_file(dest, options)?;
                    return Ok(());
                }
                Err(error) => {
                    tracing::debug!(
                        source = %source.display(),
                        dest = %dest.display(),
                        error = %error,
                        "hard_link failed; falling back to byte copy"
                    );
                }
            }

            let partial = partial_path(dest)?;
            fs::copy(source, &partial).map_err(|error| {
                DripstoneError::storage_with_source(
                    format!(
                        "failed to copy '{}' to '{}'",
                        source.display(),
                        partial.display()
                    ),
                    error,
                )
            })?;
            verify_existing_file(&partial, options)?;
            fs::rename(&partial, dest).map_err(|error| {
                DripstoneError::storage_with_source(
                    format!(
                        "failed to promote copy from '{}' to '{}'",
                        partial.display(),
                        dest.display()
                    ),
                    error,
                )
            })?;
            Ok(())
        }
    }
}

/// Verify size and optional SHA-256 of an on-disk artifact.
///
/// # Errors
///
/// Returns an error when the file is missing, oversized, or the digest mismatches.
pub fn verify_existing_file(path: &Path, options: &DownloadOptions) -> Result<(), DripstoneError> {
    let metadata = fs::metadata(path).map_err(|error| {
        DripstoneError::storage_with_source(
            format!("failed to stat downloaded file '{}'", path.display()),
            error,
        )
    })?;
    if !metadata.is_file() {
        return Err(DripstoneError::storage(format!(
            "expected a file at '{}'",
            path.display()
        )));
    }

    if let Some(max_bytes) = options.max_bytes
        && metadata.len() > max_bytes
    {
        return Err(DripstoneError::storage(format!(
            "file '{}' is {} bytes which exceeds max_bytes={max_bytes}",
            path.display(),
            metadata.len()
        )));
    }

    if let Some(expected) = options.expected_sha256.as_deref() {
        let actual = sha256_hex_file(path)?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(DripstoneError::storage(format!(
                "SHA-256 mismatch for '{}': expected {expected}, got {actual}",
                path.display()
            )));
        }
    }

    Ok(())
}

/// Compute the lowercase hex SHA-256 digest of a file.
///
/// # Errors
///
/// Returns an error when the file cannot be read.
#[cfg(feature = "model-download")]
pub fn sha256_hex_file(path: &Path) -> Result<String, DripstoneError> {
    use sha2::{Digest, Sha256};

    let mut file = File::open(path).map_err(|error| {
        DripstoneError::storage_with_source(
            format!("failed to open '{}' for hashing", path.display()),
            error,
        )
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 64];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            DripstoneError::storage_with_source(
                format!("failed to read '{}' while hashing", path.display()),
                error,
            )
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(not(feature = "model-download"))]
pub fn sha256_hex_file(_path: &Path) -> Result<String, DripstoneError> {
    Err(DripstoneError::storage(
        "SHA-256 verification requires the dripstone `model-download` feature",
    ))
}

fn preflight_disk_for_download(
    url: &str,
    dest: &Path,
    partial: &Path,
    options: &DownloadOptions,
) -> Result<(), DripstoneError> {
    let already = if partial.is_file() {
        fs::metadata(partial)
            .map(|metadata| metadata.len())
            .unwrap_or(0)
    } else {
        0
    };

    // Prefer a known size hint (catalog / caller) and skip the HEAD round-trip.
    let remote_total = if let Some(expected) = options.expected_bytes {
        Some(expected)
    } else if options.check_remote_size {
        probe_remote_content_length(url, &options.headers)?
    } else {
        None
    };

    if let Some(max_bytes) = options.max_bytes
        && let Some(total) = remote_total
        && total > max_bytes
    {
        return Err(DripstoneError::storage(format!(
            "remote object at {url} is {total} bytes which exceeds max_bytes={max_bytes}"
        )));
    }

    let needed = match remote_total {
        Some(total) => total.saturating_sub(already),
        // Unknown size: only enforce the reserve headroom so small downloads
        // are not blocked by a large max_bytes ceiling.
        None => 0,
    };
    ensure_disk_space(dest, needed, options.reserve_free_bytes)
}

fn download_with_curl(
    url: &str,
    partial: &Path,
    options: &DownloadOptions,
) -> Result<(), DripstoneError> {
    let mut args = vec!["-L", "-f", "-sS", "--retry", "3", "--retry-delay", "1"];
    let max_bytes_arg;
    if options.allow_resume {
        args.extend_from_slice(&["-C", "-"]);
    }
    if let Some(max_bytes) = options.max_bytes {
        max_bytes_arg = max_bytes.to_string();
        args.extend_from_slice(&["--max-filesize", max_bytes_arg.as_str()]);
    }
    let partial_arg = partial.to_string_lossy();
    args.extend_from_slice(&["-o", partial_arg.as_ref(), url]);

    let output = run_curl(&args, &options.headers).map_err(|error| {
        DripstoneError::storage_with_source(
            "failed to execute curl for model download (is curl installed?)",
            error,
        )
    })?;

    if output.status.success() {
        return Ok(());
    }

    let _ = fs::remove_file(partial);
    Err(DripstoneError::storage(format!(
        "curl exited with status {} for {url}",
        output.status
    )))
}

fn partial_path(dest: &Path) -> Result<PathBuf, DripstoneError> {
    let file_name = dest
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| DripstoneError::storage("download destination has invalid file name"))?;
    Ok(dest.with_file_name(format!("{file_name}.partial")))
}

fn lock_path(dest: &Path) -> Result<PathBuf, DripstoneError> {
    let file_name = dest
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| DripstoneError::storage("download destination has invalid file name"))?;
    Ok(dest.with_file_name(format!("{file_name}.lock")))
}

struct DownloadLockGuard {
    path: PathBuf,
}

impl Drop for DownloadLockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

enum LockAcquire {
    Acquired(DownloadLockGuard),
    Ready,
}

fn acquire_download_lock(
    dest: &Path,
    options: &DownloadOptions,
) -> Result<LockAcquire, DripstoneError> {
    let path = lock_path(dest)?;
    let started = Instant::now();
    loop {
        if dest.is_file() {
            return Ok(LockAcquire::Ready);
        }

        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                let payload = format!("pid={}\n", std::process::id());
                let _ = file.write_all(payload.as_bytes());
                return Ok(LockAcquire::Acquired(DownloadLockGuard { path }));
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if dest.is_file() {
                    return Ok(LockAcquire::Ready);
                }
                if started.elapsed() > options.lock_timeout {
                    return Err(DripstoneError::storage(format!(
                        "timed out waiting for download lock '{}'; delete it if the holder crashed",
                        path.display()
                    )));
                }
                // Sync download helper (not on the async runtime). Brief backoff
                // while another process holds the artifact lockfile.
                #[expect(
                    clippy::disallowed_methods,
                    reason = "sync download lock wait cannot use tokio::time::sleep"
                )]
                {
                    thread::sleep(options.lock_poll);
                }
            }
            Err(error) => {
                return Err(DripstoneError::storage_with_source(
                    format!("failed to create download lock '{}'", path.display()),
                    error,
                ));
            }
        }
    }
}

fn normalize_for_prefix(path: &Path) -> Result<PathBuf, DripstoneError> {
    if path.exists() {
        return path.canonicalize().map_err(|error| {
            DripstoneError::storage_with_source(
                format!("failed to canonicalize '{}'", path.display()),
                error,
            )
        });
    }

    // For not-yet-created destinations, canonicalize the deepest existing ancestor
    // and append the remaining components.
    let mut existing = path.to_path_buf();
    let mut missing = Vec::new();
    while !existing.exists() {
        let file_name = existing
            .file_name()
            .ok_or_else(|| {
                DripstoneError::storage(format!(
                    "cannot normalize non-existent path '{}'",
                    path.display()
                ))
            })?
            .to_os_string();
        missing.push(file_name);
        existing.pop();
        if existing.as_os_str().is_empty() {
            break;
        }
    }

    let mut normalized = if existing.as_os_str().is_empty() {
        PathBuf::from(".")
    } else if existing.exists() {
        existing.canonicalize().map_err(|error| {
            DripstoneError::storage_with_source(
                format!("failed to canonicalize '{}'", existing.display()),
                error,
            )
        })?
    } else {
        existing
    };

    for component in missing.into_iter().rev() {
        normalized.push(component);
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn sanitize_rejects_parent_and_absolute_components() {
        let root = PathBuf::from("/tmp/models");
        assert!(sanitize_relative_path(&root, "../escape.bin").is_err());
        assert!(sanitize_relative_path(&root, "/etc/passwd").is_err());
        assert!(sanitize_relative_path(&root, "weights/../../escape.bin").is_err());
        assert!(sanitize_relative_path(&root, "").is_err());
    }

    #[test]
    fn sanitize_accepts_nested_relative_paths() {
        let root = tempfile::tempdir().expect("tempdir");
        let resolved =
            sanitize_relative_path(root.path(), "onnx/model.onnx").expect("sanitize path");
        assert_eq!(resolved, root.path().join("onnx").join("model.onnx"));
    }

    #[test]
    fn max_bytes_rejects_oversized_existing_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("big.bin");
        fs::write(&path, vec![0_u8; 32]).expect("write");

        let options = DownloadOptions::default().with_max_bytes(Some(16));
        let error = verify_existing_file(&path, &options).expect_err("oversized");
        assert!(error.to_string().contains("max_bytes"));
    }

    #[cfg(feature = "model-download")]
    #[test]
    fn sha256_matches_known_digest() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("hello.txt");
        fs::write(&path, b"hello").expect("write");

        let digest = sha256_hex_file(&path).expect("hash");
        assert_eq!(
            digest,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );

        let options = DownloadOptions::default()
            .with_max_bytes(Some(1024))
            .with_expected_sha256(digest);
        verify_existing_file(&path, &options).expect("verify");
    }

    #[test]
    fn ensure_downloaded_uses_file_url_atomically() {
        let temp = tempfile::tempdir().expect("tempdir");
        let source = temp.path().join("source.bin");
        let dest = temp.path().join("nested").join("dest.bin");
        {
            let mut file = File::create(&source).expect("create source");
            file.write_all(b"dripstone-download").expect("write source");
        }

        let url = format!("file://{}", source.display());
        let options = DownloadOptions::default()
            .with_max_bytes(Some(1024))
            .with_check_remote_size(false)
            .with_expected_bytes(Some(17))
            .with_reserve_free_bytes(0);
        let path = ensure_downloaded(&url, &dest, &options).expect("download");
        assert_eq!(path, dest);
        assert_eq!(
            fs::read_to_string(&dest).expect("read"),
            "dripstone-download"
        );
        assert!(!partial_path(&dest).expect("partial").exists());
        assert!(!lock_path(&dest).expect("lock").exists());
    }

    #[test]
    fn ensure_downloaded_refuses_when_disk_is_too_full() {
        use crate::loader::disk::with_available_bytes_override;

        let temp = tempfile::tempdir().expect("tempdir");
        let source = temp.path().join("source.bin");
        let dest = temp.path().join("dest.bin");
        fs::write(&source, vec![0_u8; 64]).expect("write source");
        let url = format!("file://{}", source.display());
        let options = DownloadOptions::default()
            .with_max_bytes(Some(1024))
            .with_check_remote_size(false)
            .with_expected_bytes(Some(64))
            .with_reserve_free_bytes(100);

        with_available_bytes_override(Some(50), || {
            let err = ensure_downloaded(&url, &dest, &options).expect_err("disk full");
            assert!(err.to_string().contains("insufficient disk space"));
            assert!(!dest.exists());
        });
    }

    #[test]
    fn acquire_lock_is_exclusive() {
        let temp = tempfile::tempdir().expect("tempdir");
        let dest = temp.path().join("model.bin");
        let options = DownloadOptions::default();
        let lock = lock_path(&dest).expect("lock path");

        let first = acquire_download_lock(&dest, &options).expect("first lock");
        assert!(matches!(first, LockAcquire::Acquired(_)));
        assert!(lock.exists(), "lock file should exist while held");
        assert!(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock)
                .is_err(),
            "create_new must fail while lock is held"
        );
        drop(first);
        assert!(!lock.exists(), "lock file should be removed on drop");
        let third = acquire_download_lock(&dest, &options).expect("lock after release");
        assert!(matches!(third, LockAcquire::Acquired(_)));
    }
}
