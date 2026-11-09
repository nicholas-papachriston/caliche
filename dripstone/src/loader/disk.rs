//! Disk free-space checks for safe model provisioning.

use crate::errors::DripstoneError;
use crate::loader::curl::run_curl;
use std::path::{Path, PathBuf};

/// Default free-space headroom that must remain after a download (512 MiB).
pub const DEFAULT_RESERVE_FREE_BYTES: u64 = 512 * 1024 * 1024;

/// Environment override for download free-space reserve (`DRIPSTONE_DISK_RESERVE_BYTES`).
pub const DISK_RESERVE_ENV: &str = "DRIPSTONE_DISK_RESERVE_BYTES";

/// Environment override for the Dripstone artifact cache (`DRIPSTONE_CACHE_DIR`).
pub const CACHE_DIR_ENV: &str = "DRIPSTONE_CACHE_DIR";

#[cfg(test)]
thread_local! {
    static AVAILABLE_BYTES_OVERRIDE: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Return available bytes on the filesystem that will hold `path`.
///
/// Walks up to the nearest existing ancestor when `path` does not exist yet.
///
/// # Errors
///
/// Returns an error when the filesystem cannot be queried.
pub fn available_disk_bytes(path: &Path) -> Result<u64, DripstoneError> {
    #[cfg(test)]
    if let Some(overridden) = AVAILABLE_BYTES_OVERRIDE.get() {
        return Ok(overridden);
    }

    let probe = nearest_existing_ancestor(path);
    available_disk_bytes_os(&probe)
}

/// Ensure `needed_bytes + reserve_free_bytes` fit on the destination volume.
///
/// # Errors
///
/// Returns an error when free space is insufficient or cannot be measured.
pub fn ensure_disk_space(
    path: &Path,
    needed_bytes: u64,
    reserve_free_bytes: u64,
) -> Result<(), DripstoneError> {
    let available = available_disk_bytes(path)?;
    let required = needed_bytes.saturating_add(reserve_free_bytes);
    if available < required {
        return Err(DripstoneError::storage(format!(
            "insufficient disk space near '{}': need {required} bytes \
             ({needed_bytes} download + {reserve_free_bytes} reserve), have {available} bytes free",
            path.display()
        )));
    }
    Ok(())
}

/// Probe remote `Content-Length` / `X-Linked-Size` via `curl -sIL`.
///
/// # Errors
///
/// Returns an error when curl cannot be executed. Missing length is `Ok(None)`.
pub fn probe_remote_content_length(
    url: &str,
    headers: &[(String, String)],
) -> Result<Option<u64>, DripstoneError> {
    let output = run_curl(
        &["-sIL", "--max-redirs", "10", "--retry", "1", url],
        headers,
    )
    .map_err(|error| {
        DripstoneError::storage_with_source(
            "failed to execute curl for remote size probe (is curl installed?)",
            error,
        )
    })?;

    if !output.status.success() {
        return Ok(None);
    }

    let text = String::from_utf8_lossy(&output.stdout);
    Ok(parse_size_headers(&text))
}

fn parse_size_headers(headers: &str) -> Option<u64> {
    let mut linked_size = None;
    let mut content_length = None;
    for line in headers.lines() {
        let trimmed = line.trim();
        let Some((name, value)) = trimmed.split_once(':') else {
            continue;
        };
        let name = name.trim();
        let value = value.trim();
        if name.eq_ignore_ascii_case("x-linked-size") {
            if let Ok(parsed) = value.parse::<u64>() {
                linked_size = Some(parsed);
            }
        } else if name.eq_ignore_ascii_case("content-length")
            && let Ok(parsed) = value.parse::<u64>()
            && parsed > 0
        {
            // Keep the last non-zero Content-Length (final hop after redirects).
            content_length = Some(parsed);
        }
    }
    linked_size.or(content_length)
}

fn nearest_existing_ancestor(path: &Path) -> PathBuf {
    let mut current = path.to_path_buf();
    loop {
        if current.exists() {
            return current;
        }
        if !current.pop() {
            return PathBuf::from(".");
        }
    }
}

/// Resolve download free-space reserve, honoring [`DISK_RESERVE_ENV`] when set.
#[must_use]
pub fn reserve_free_bytes_from_env() -> u64 {
    std::env::var(DISK_RESERVE_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_RESERVE_FREE_BYTES)
}

/// Directory for Dripstone-managed model artifacts.
///
/// Order: `DRIPSTONE_CACHE_DIR`, then `$HF_HOME/dripstone`, then
/// `~/.cache/huggingface/dripstone`, then `$TMPDIR/dripstone`.
#[must_use]
pub fn cache_root() -> PathBuf {
    if let Some(dir) = std::env::var_os(CACHE_DIR_ENV) {
        return PathBuf::from(dir);
    }
    if let Some(hf_home) = std::env::var_os("HF_HOME") {
        return PathBuf::from(hf_home).join("dripstone");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("dripstone");
    }
    std::env::temp_dir().join("dripstone")
}

#[cfg(unix)]
fn available_disk_bytes_os(path: &Path) -> Result<u64, DripstoneError> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let c_path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        DripstoneError::storage(format!(
            "path '{}' contains interior NUL bytes",
            path.display()
        ))
    })?;

    // SAFETY: path is a valid CString; statvfs writes into a stack-allocated struct.
    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c_path.as_ptr(), &mut stat) != 0 {
            return Err(DripstoneError::storage(format!(
                "statvfs failed for '{}': {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }
        Ok(u64::from(stat.f_bavail).saturating_mul(stat.f_frsize))
    }
}

#[cfg(not(unix))]
fn available_disk_bytes_os(path: &Path) -> Result<u64, DripstoneError> {
    // Best-effort fallback via `df -Pk` for non-Unix hosts.
    let output = Command::new("df")
        .args(["-Pk"])
        .arg(path)
        .output()
        .map_err(|error| {
            DripstoneError::storage_with_source(
                format!("failed to query free disk space for '{}'", path.display()),
                error,
            )
        })?;
    if !output.status.success() {
        return Err(DripstoneError::storage(format!(
            "df failed for '{}'",
            path.display()
        )));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    parse_df_available_kib(&text).ok_or_else(|| {
        DripstoneError::storage(format!(
            "unable to parse df output for '{}'",
            path.display()
        ))
    })
}

#[cfg(not(unix))]
fn parse_df_available_kib(text: &str) -> Option<u64> {
    let line = text.lines().nth(1)?;
    let available_kib = line.split_whitespace().nth(3)?.parse::<u64>().ok()?;
    Some(available_kib.saturating_mul(1024))
}

#[cfg(test)]
pub(crate) fn with_available_bytes_override<T>(bytes: Option<u64>, f: impl FnOnce() -> T) -> T {
    AVAILABLE_BYTES_OVERRIDE.with(|cell| {
        let previous = cell.replace(bytes);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        cell.set(previous);
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_prefers_x_linked_size() {
        let headers = "\
HTTP/2 302\r\n\
x-linked-size: 19077344\r\n\
content-length: 996\r\n\
HTTP/2 200\r\n\
content-length: 19077344\r\n";
        assert_eq!(parse_size_headers(headers), Some(19_077_344));
    }

    #[test]
    fn parse_falls_back_to_final_content_length() {
        let headers = "\
HTTP/1.1 200 OK\r\n\
Content-Length: 4096\r\n";
        assert_eq!(parse_size_headers(headers), Some(4096));
    }

    #[test]
    fn ensure_disk_space_rejects_when_override_is_too_small() {
        let temp = tempfile::tempdir().expect("tempdir");
        with_available_bytes_override(Some(1_000), || {
            let err = ensure_disk_space(temp.path(), 2_000, 100).expect_err("should fail");
            assert!(err.to_string().contains("insufficient disk space"));
        });
    }

    #[test]
    fn ensure_disk_space_allows_when_override_is_large_enough() {
        let temp = tempfile::tempdir().expect("tempdir");
        with_available_bytes_override(Some(10_000), || {
            ensure_disk_space(temp.path(), 2_000, 100).expect("should pass");
        });
    }

    #[test]
    fn available_disk_bytes_reports_positive_for_temp() {
        let temp = tempfile::tempdir().expect("tempdir");
        let available = available_disk_bytes(temp.path()).expect("query free space");
        assert!(available > 0);
    }

    #[test]
    #[serial_test::serial(env)]
    fn reserve_free_bytes_from_env_honors_override() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set(DISK_RESERVE_ENV, "12345");
        assert_eq!(reserve_free_bytes_from_env(), 12_345);
        env.set(DISK_RESERVE_ENV, "not-a-number");
        assert_eq!(reserve_free_bytes_from_env(), DEFAULT_RESERVE_FREE_BYTES);
        env.remove(DISK_RESERVE_ENV);
        assert_eq!(reserve_free_bytes_from_env(), DEFAULT_RESERVE_FREE_BYTES);
    }

    #[test]
    #[serial_test::serial(env)]
    fn cache_root_uses_dripstone_and_env_override() {
        let temp = tempfile::tempdir().expect("tempdir");
        let override_path = temp.path().join("custom-cache");
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set(CACHE_DIR_ENV, &override_path);
        assert_eq!(cache_root(), override_path);
        let fallback = temp.path().join("fallback-cache");
        env.set(CACHE_DIR_ENV, &fallback);
        assert_eq!(cache_root(), fallback);
    }
}
