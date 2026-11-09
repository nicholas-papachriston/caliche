//! Which host environment variables a spawned child process may receive.
//!
//! These children (shell commands, the validation hook, verification
//! commands, formatters, MCP stdio and LSP servers, plugin shell steps,
//! external agents, and session-start onboarding probes) start from an
//! empty environment plus these
//! variables, so provider keys and tokens in the parent never reach it. A
//! spawn path that needs a credential takes it from its own configured `env`,
//! applied after the scrub.
//!
//! Developer variables are not passed either: proxy settings
//! (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`), certificate bundles
//! (`SSL_CERT_FILE`, `NODE_EXTRA_CA_CERTS`), toolchain homes (`CARGO_HOME`,
//! `RUSTUP_HOME`, `JAVA_HOME`, `GOPATH`), `XDG_*`, `VIRTUAL_ENV`, and
//! `SSH_AUTH_SOCK`. Proxy URLs can carry credentials and the agent socket
//! signs as the user, so none of them is on the list. A shell command that
//! needs one reads a `CALICHE_BASH_`-prefixed copy the operator exports, for
//! example `CALICHE_BASH_CARGO_HOME`. `CALICHE_BASH_TIMEOUT_SECS` and
//! `CALICHE_BASH_DEDUPE` are host settings and are not forwarded.
//! `CALICHE_POLICY_PATH` and `CALICHE_REQUIRE_AUTH` are not forwarded.
//! `HOME` is still passed, so file-based
//! credentials such as `~/.aws` and `~/.netrc` stay readable to a child that
//! runs outside the sandbox.
//!
//! `PATH` reaches the child without its relative entries (`.`, an empty
//! entry, `./node_modules/.bin`), so a command the child looks up never
//! resolves to a file in the directory it runs in, which is usually the
//! repository. When no absolute entry is left, or the host has no `PATH`,
//! the child gets the platform default search path: an empty or missing `PATH`
//! makes libc search the current directory. A `PATH` in a spawn's own
//! configured `env` is filtered the same way ([`host_child_env_with`]).
//!
//! On Unix only `PATH` is the search path (`execvp` reads no other
//! spelling); on Windows, where names ignore case, `Path` is too.

/// The search path variable.
pub const PATH_VAR: &str = "PATH";

/// The Windows system directory variable.
pub const SYSTEMROOT: &str = "SYSTEMROOT";

/// Host variables a child needs to run: search path, home, user, locale,
/// terminal, temp dirs, and the Windows basics that process startup reads.
const ALLOWED: &[&str] = &[
    PATH_VAR,
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SHELL",
    "TERM",
    "TMPDIR",
    "PWD",
    "LOGNAME",
    "HOSTNAME",
    SYSTEMROOT,
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMDATA",
    "ALLUSERSPROFILE",
    "USERNAME",
    "COMPUTERNAME",
    "HOMEDRIVE",
    "HOMEPATH",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
];

/// Name fragments that mark a variable as a credential anywhere in its name.
const CREDENTIAL_MARKERS: &[&str] = &["KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL"];

/// Name suffixes that mark a variable as a credential.
const CREDENTIAL_SUFFIXES: &[&str] = &["_PASS", "_PAT"];

/// Whether the variable name `key` looks like it holds a credential.
///
/// Matches a name that contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`,
/// `PASSWD`, or `CREDENTIAL`, or ends in `_PASS` or `_PAT`. Names compare
/// case-insensitively, as on Windows. Child environments and transcript
/// redaction share this predicate.
#[must_use]
pub fn is_credential_name(key: &str) -> bool {
    let upper = key.to_ascii_uppercase();
    CREDENTIAL_MARKERS
        .iter()
        .any(|marker| upper.contains(marker))
        || CREDENTIAL_SUFFIXES
            .iter()
            .any(|suffix| upper.ends_with(suffix))
}

/// Operator copies a child may receive, for example `CALICHE_BASH_CARGO_HOME`.
const FORWARDED_OPERATOR_PREFIX: &str = "CALICHE_BASH_";

/// Host bash settings that share [`FORWARDED_OPERATOR_PREFIX`].
///
/// These names were already host-only. The prefix must not start forwarding them.
/// `CALICHE_POLICY_PATH` and `CALICHE_REQUIRE_AUTH` stay off the child too:
/// they do not use this prefix.
const HOST_ONLY_ENV: &[&str] = &["CALICHE_BASH_DEDUPE", "CALICHE_BASH_TIMEOUT_SECS"];

fn is_forwarded_operator_copy(upper: &str) -> bool {
    upper.starts_with(FORWARDED_OPERATOR_PREFIX) && !HOST_ONLY_ENV.contains(&upper)
}

/// Whether a child may receive the host variable `key`.
///
/// A name that [`is_credential_name`] matches is never passed, even with an
/// allowed prefix. Names compare case-insensitively, as on Windows.
#[must_use]
pub fn is_child_env_key_allowed(key: &str) -> bool {
    if is_credential_name(key) {
        return false;
    }
    let upper = key.to_ascii_uppercase();
    ALLOWED.contains(&upper.as_str())
        || upper.starts_with("LC_")
        || is_forwarded_operator_copy(&upper)
}

/// The search path a Unix child gets when the host's has no absolute entry.
pub const DEFAULT_SEARCH_PATH: &str = "/usr/bin:/bin";

/// The search path a child gets when the host's has no absolute entry:
/// [`DEFAULT_SEARCH_PATH`] on Unix; on Windows, `System32` and the Windows
/// directory under the validated absolute `system_root` (`C:\Windows` if absent).
#[must_use]
fn default_search_path(windows: bool, system_root: Option<&str>) -> String {
    if windows {
        windows_search_path(system_root)
    } else {
        DEFAULT_SEARCH_PATH.to_string()
    }
}

fn host_default_search_path() -> String {
    let root = std::env::var(SYSTEMROOT).ok();
    default_search_path(cfg!(windows), root.as_deref())
}

/// The Windows default under an absolute `system_root`, if available.
fn windows_search_path(system_root: Option<&str>) -> String {
    let root = system_root
        .filter(|root| {
            let bytes = root.as_bytes();
            matches!(bytes, [drive, b':', b'\\' | b'/', ..] if drive.is_ascii_alphabetic())
                || root.strip_prefix(r"\\").is_some_and(|unc| {
                    let mut parts = unc.split(['\\', '/']);
                    matches!((parts.next(), parts.next()), (Some(server), Some(share))
                        if !server.is_empty() && !share.is_empty())
                })
        })
        .unwrap_or(r"C:\Windows");
    format!(r"{root}\System32;{root}")
}

/// Whether `key` names the search path on this platform.
#[must_use]
pub fn is_search_path_key(key: &str) -> bool {
    same_name(key, PATH_VAR, cfg!(windows))
}

/// The current process environment, filtered by [`is_child_env_key_allowed`],
/// with `PATH` reduced to its absolute entries and always set.
///
/// Variables whose name or value is not valid Unicode are skipped.
#[must_use]
pub fn host_child_env() -> Vec<(String, String)> {
    child_env_from(std::env::vars_os(), cfg!(windows))
}

/// [`host_child_env`] for the variables `vars`.
fn child_env_from(
    vars: impl Iterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
    windows: bool,
) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = vars
        .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?)))
        .filter(|(key, _)| is_child_env_key_allowed(key))
        .map(|(key, value)| {
            let value = if same_name(&key, PATH_VAR, windows) {
                absolute_search_path(&value)
            } else {
                value
            };
            (key, value)
        })
        .collect();
    if !env.iter().any(|(key, _)| same_name(key, PATH_VAR, windows)) {
        env.push((PATH_VAR.to_string(), host_default_search_path()));
    }
    env
}

/// [`host_child_env`] with a spawn's configured `env` applied over it. A
/// configured search path is filtered like the host's, so `${PATH}` or an
/// unset reference cannot put a relative or empty entry back.
#[must_use]
pub fn host_child_env_with<K: AsRef<str>, V: AsRef<str>>(
    configured: impl IntoIterator<Item = (K, V)>,
) -> Vec<(String, String)> {
    with_configured(host_child_env(), configured)
}

/// `env` with each of `configured` replacing any variable of the same name.
fn with_configured<K: AsRef<str>, V: AsRef<str>>(
    mut env: Vec<(String, String)>,
    configured: impl IntoIterator<Item = (K, V)>,
) -> Vec<(String, String)> {
    for (key, value) in configured {
        let (key, value) = (key.as_ref(), value.as_ref());
        env.retain(|(existing, _)| !same_name(existing, key, cfg!(windows)));
        let value = if is_search_path_key(key) {
            absolute_search_path(value)
        } else {
            value.to_string()
        };
        env.push((key.to_string(), value));
    }
    env
}

/// Whether two variable names are the same variable on this platform.
fn same_name(a: &str, b: &str, windows: bool) -> bool {
    if windows {
        a.eq_ignore_ascii_case(b)
    } else {
        a == b
    }
}

/// The search path `value` with every relative entry removed, in order, or
/// the platform default search path when no absolute entry is left.
#[must_use]
pub fn absolute_search_path(value: &str) -> String {
    let absolute = std::env::split_paths(value).filter(|dir| dir.is_absolute());
    std::env::join_paths(absolute)
        .ok()
        .and_then(|joined| joined.into_string().ok())
        .filter(|joined| !joined.is_empty())
        .unwrap_or_else(host_default_search_path)
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::DEFAULT_SEARCH_PATH;
    use super::{PATH_VAR, SYSTEMROOT, is_child_env_key_allowed, is_credential_name};

    #[cfg(unix)]
    fn vars(
        pairs: &[(&str, &str)],
    ) -> impl Iterator<Item = (std::ffi::OsString, std::ffi::OsString)> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).into(), (*value).into()))
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[cfg(unix)]
    fn owned(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect()
    }

    #[cfg(unix)]
    #[test]
    fn relative_search_path_entries_are_dropped() {
        assert_eq!(
            super::absolute_search_path(".:/usr/bin::./node_modules/.bin:bin:/bin"),
            "/usr/bin:/bin"
        );
        for value in [".", "./bin", "node_modules/.bin", "::", ""] {
            assert_eq!(
                super::absolute_search_path(value),
                DEFAULT_SEARCH_PATH,
                "{value:?}"
            );
        }
    }

    /// `PATH` is filtered, a child gets the default when the host has none,
    /// and on Unix `Path` is not the search path, so it neither stands in
    /// for `PATH` nor is filtered.
    #[cfg(unix)]
    #[test]
    fn the_child_path_is_filtered_and_always_set() {
        let filtered =
            super::child_env_from(vars(&[(PATH_VAR, ".:/opt/bin"), ("HOME", "/h")]), false);
        let missing = super::child_env_from(vars(&[("HOME", "/h")]), false);
        let other_case = super::child_env_from(vars(&[("Path", ".:/opt/bin")]), false);

        assert_eq!(filtered, owned(&[(PATH_VAR, "/opt/bin"), ("HOME", "/h")]));
        assert_eq!(
            missing,
            owned(&[("HOME", "/h"), (PATH_VAR, DEFAULT_SEARCH_PATH)])
        );
        assert_eq!(
            other_case,
            owned(&[("Path", ".:/opt/bin"), (PATH_VAR, DEFAULT_SEARCH_PATH)])
        );
    }

    /// A configured `env` replaces the host variable of the same name, and a
    /// configured `PATH` loses its relative entries or, when none is left,
    /// becomes the default.
    #[cfg(unix)]
    #[test]
    fn a_configured_path_is_filtered_like_the_host_path() {
        let host = || owned(&[(PATH_VAR, "/usr/bin"), ("HOME", "/h")]);

        let relative = super::with_configured(host(), [(PATH_VAR, ".:/opt/bin"), ("TOKEN", "t")]);
        let empty = super::with_configured(host(), [(PATH_VAR, "")]);

        assert_eq!(
            relative,
            owned(&[("HOME", "/h"), (PATH_VAR, "/opt/bin"), ("TOKEN", "t")])
        );
        assert_eq!(
            empty,
            owned(&[("HOME", "/h"), (PATH_VAR, DEFAULT_SEARCH_PATH)])
        );
    }

    /// The Windows default follows `SYSTEMROOT`, and falls back to
    /// `C:\Windows` when no absolute root is supplied.
    #[test]
    fn the_windows_default_follows_systemroot() {
        assert_eq!(
            super::default_search_path(true, Some(r"D:\WINNT")),
            r"D:\WINNT\System32;D:\WINNT"
        );
        assert!(super::is_child_env_key_allowed(SYSTEMROOT));
        for root in [
            None,
            Some(""),
            Some("Windows"),
            Some(r"\Windows"),
            Some("D:"),
            Some(r"\\"),
            Some(r"\\server"),
        ] {
            assert_eq!(
                super::default_search_path(true, root),
                r"C:\Windows\System32;C:\Windows",
                "{root:?}"
            );
        }
        assert_eq!(
            super::default_search_path(true, Some(r"\\server\share\Windows")),
            r"\\server\share\Windows\System32;\\server\share\Windows"
        );
    }

    #[cfg(unix)]
    #[test]
    fn windows_child_path_matches_any_case() {
        for key in ["Path", "path", "pAtH"] {
            assert_eq!(
                super::child_env_from(
                    vars(&[(key, ".:/opt/bin"), (SYSTEMROOT, "D:\\WINNT")]),
                    true
                ),
                owned(&[(key, "/opt/bin"), (SYSTEMROOT, "D:\\WINNT")])
            );
        }
    }

    #[test]
    fn credentials_are_refused_even_with_an_allowed_prefix() {
        for key in [
            "OPENAI_API_KEY",
            "GITHUB_TOKEN",
            "CALICHE_BASH_TOKEN",
            "LC_SECRET",
            "path_key",
            "CALICHE_BASH_DB_PASSWORD",
            "CALICHE_BASH_SMTP_PASS",
            "CALICHE_BASH_GITHUB_PAT",
            "CALICHE_BASH_GCP_CREDENTIALS",
            "CALICHE_BASH_MYSQL_PASSWD",
        ] {
            assert!(is_credential_name(key), "{key}");
            assert!(!is_child_env_key_allowed(key), "{key}");
        }
    }

    #[test]
    fn shell_and_windows_basics_pass_in_any_case() {
        for key in [
            PATH_VAR,
            "Path",
            "HOME",
            "LC_MESSAGES",
            "CALICHE_BASH_FORWARD_PROBE",
            "CALICHE_BASH_CARGO_HOME",
            "caliche_bash_cargo_home",
            SYSTEMROOT,
            "SystemRoot",
            "ProgramFiles",
            "ProgramFiles(x86)",
            "ProgramData",
            "ALLUSERSPROFILE",
            "USERNAME",
            "COMPUTERNAME",
            "HOMEDRIVE",
            "HOMEPATH",
            "PROCESSOR_ARCHITECTURE",
            "NUMBER_OF_PROCESSORS",
        ] {
            assert!(is_child_env_key_allowed(key), "{key}");
        }
    }

    #[test]
    fn unlisted_developer_variables_are_dropped() {
        for key in [
            "SSH_AUTH_SOCK",
            "AWS_PROFILE",
            "DATABASE_URL",
            "HTTPS_PROXY",
            "CARGO_HOME",
            "CALICHE_POLICY_PATH",
            "CALICHE_REQUIRE_AUTH",
            "CALICHE_BASH_DEDUPE",
            "CALICHE_BASH_TIMEOUT_SECS",
            "caliche_bash_timeout_secs",
        ] {
            assert!(!is_child_env_key_allowed(key), "{key}");
        }
    }
}
