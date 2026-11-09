//! Variable substitution in config values.
//!
//! Supports {env:VAR_NAME} and {file:path} placeholders. `{file:}` is confined
//! to a caller-supplied root: absolute paths, `~`, `..`, and NUL are rejected.

use std::io::Read;
use std::path::{Path, PathBuf};

const MAX_CONFIG_FILE_SUBSTITUTION_BYTES: u64 = 64 * 1024;

/// Policy for `{env:}` / `{file:}` expansion.
#[derive(Debug, Clone)]
pub struct SubstitutionPolicy {
    pub allow_env: bool,
    pub allow_file: bool,
    pub file_root: Option<PathBuf>,
}

impl SubstitutionPolicy {
    /// Local trusted config: env plus files under `root`.
    #[must_use]
    pub fn local_trusted(root: impl Into<PathBuf>) -> Self {
        Self {
            allow_env: true,
            allow_file: true,
            file_root: Some(root.into()),
        }
    }

    /// Remote / untrusted config: no local I/O.
    #[must_use]
    pub fn untrusted() -> Self {
        Self {
            allow_env: false,
            allow_file: false,
            file_root: None,
        }
    }
}

/// Substitute `{env:VAR}` and `{file:path}` placeholders using the process cwd
/// as the `{file:}` root.
pub fn substitute_variables_in_string(value: &str) -> beef::Cow<'_, str> {
    let root = std::env::current_dir().ok();
    let policy = SubstitutionPolicy {
        allow_env: true,
        allow_file: root.is_some(),
        file_root: root,
    };
    substitute_variables_in_string_with(value, &policy)
}

/// Substitute placeholders under an explicit policy.
pub fn substitute_variables_in_string_with<'a>(
    value: &'a str,
    policy: &SubstitutionPolicy,
) -> beef::Cow<'a, str> {
    if !value.contains("{env:") && !value.contains("{file:") {
        return beef::Cow::borrowed(value);
    }
    let mut result = value.to_string();

    while let Some(start) = result.find("{env:") {
        if let Some(end) = result[start..].find('}') {
            let end = start + end;
            let var_name = &result[start + 5..end];
            let replacement = if policy.allow_env {
                caliche_contracts::credential_env::var(var_name).unwrap_or_default()
            } else {
                tracing::warn!(placeholder = %var_name, "skipping {{env:}} substitution from untrusted config");
                String::new()
            };
            result.replace_range(start..=end, &replacement);
        } else {
            break;
        }
    }

    while let Some(start) = result.find("{file:") {
        if let Some(end) = result[start..].find('}') {
            let end = start + end;
            let path_str = result[start + 6..end].trim().to_string();
            let replacement = if policy.allow_file {
                match resolve_path(&path_str, policy.file_root.as_deref()) {
                    Some(path) if path.exists() => read_config_substitution_file(&path),
                    _ => String::new(),
                }
            } else {
                tracing::warn!(path = %path_str, "skipping {{file:}} substitution from untrusted config");
                String::new()
            };
            result.replace_range(start..=end, &replacement);
        } else {
            break;
        }
    }

    beef::Cow::owned(result)
}

fn read_config_substitution_file(path: &Path) -> String {
    let metadata = match path.metadata() {
        Ok(metadata) => metadata,
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "failed to inspect config substitution file");
            return String::new();
        }
    };

    if !metadata.is_file() {
        tracing::warn!(path = %path.display(), "config substitution path is not a regular file");
        return String::new();
    }

    if metadata.len() > MAX_CONFIG_FILE_SUBSTITUTION_BYTES {
        tracing::warn!(
            path = %path.display(),
            size = metadata.len(),
            max_size = MAX_CONFIG_FILE_SUBSTITUTION_BYTES,
            "config substitution file is too large"
        );
        return String::new();
    }

    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "failed to open config substitution file");
            return String::new();
        }
    };

    let mut replacement = String::new();
    let mut reader = file.take(MAX_CONFIG_FILE_SUBSTITUTION_BYTES);
    if let Err(error) = reader.read_to_string(&mut replacement) {
        tracing::warn!(path = %path.display(), %error, "failed to read config substitution file");
        return String::new();
    }
    replacement
}

fn path_within_root(candidate: &Path, root: &Path) -> bool {
    candidate == root || candidate.starts_with(root)
}

fn resolve_path(s: &str, file_root: Option<&Path>) -> Option<PathBuf> {
    let s = s.trim();
    if s.contains('\0') {
        return None;
    }
    if Path::new(s)
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        tracing::warn!(path = s, "config substitution path contains '..'");
        return None;
    }
    if s.starts_with('~') {
        tracing::warn!(
            path = s,
            "config substitution path must not use home expansion"
        );
        return None;
    }
    let Some(root) = file_root else {
        tracing::warn!("config substitution has no file root");
        return None;
    };
    let requested = Path::new(s);
    if requested.is_absolute() {
        tracing::warn!(
            path = s,
            "config substitution path must be relative to the config root"
        );
        return None;
    }
    let joined = root.join(requested);
    if path_within_root(&joined, root) {
        Some(joined)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{SubstitutionPolicy, substitute_variables_in_string_with};
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn file_substitution_reads_regular_file() {
        let dir = TempDir::new().expect("tempdir");
        fs::write(dir.path().join("secret.txt"), "hello-from-file").expect("write");
        let policy = SubstitutionPolicy::local_trusted(dir.path());
        let got = substitute_variables_in_string_with("token={file:secret.txt}", &policy);
        assert_eq!(got.as_ref(), "token=hello-from-file");
    }

    #[test]
    fn file_substitution_rejects_parent_dir() {
        let dir = TempDir::new().expect("tempdir");
        let policy = SubstitutionPolicy::local_trusted(dir.path());
        let got = substitute_variables_in_string_with("x={file:../secret}", &policy);
        assert_eq!(got.as_ref(), "x=");
    }

    #[test]
    fn file_substitution_rejects_absolute_path() {
        let dir = TempDir::new().expect("tempdir");
        let policy = SubstitutionPolicy::local_trusted(dir.path());
        let got = substitute_variables_in_string_with("x={file:/etc/passwd}", &policy);
        assert_eq!(got.as_ref(), "x=");
    }

    #[test]
    fn file_substitution_rejects_home_expansion() {
        let dir = TempDir::new().expect("tempdir");
        let policy = SubstitutionPolicy::local_trusted(dir.path());
        let got = substitute_variables_in_string_with("x={file:~/.ssh/id_rsa}", &policy);
        assert_eq!(got.as_ref(), "x=");
    }

    #[test]
    #[serial_test::serial(env)]
    fn untrusted_policy_skips_file_and_env() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("CALICHE_SUBST_TEST_SECRET", "leaked");
        let got = substitute_variables_in_string_with(
            "a={file:/etc/passwd} b={env:CALICHE_SUBST_TEST_SECRET}",
            &SubstitutionPolicy::untrusted(),
        );
        assert_eq!(got.as_ref(), "a= b=");
    }
}
