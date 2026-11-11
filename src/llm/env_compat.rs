//! Canonical `CALICHE_*` environment helpers for local-inference configuration.

/// Read a non-empty env var (trimmed).
#[must_use]
pub fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// True when `name` is a truthy flag (`1`/`true`/`yes`/`on`).
#[must_use]
pub fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Read an env var, treating whitespace-only values as missing.
pub fn env_var_nonempty(name: &str) -> Result<String, std::env::VarError> {
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        Ok(_) => Err(std::env::VarError::NotPresent),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial(env)]
    fn env_nonempty_trims_and_rejects_blank() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_TEST_ENV_NONEMPTY", "  padded  ");
        assert_eq!(
            env_nonempty("CALICHE_TEST_ENV_NONEMPTY").as_deref(),
            Some("padded")
        );
        env.set("CALICHE_TEST_ENV_NONEMPTY", "   ");
        assert_eq!(env_nonempty("CALICHE_TEST_ENV_NONEMPTY"), None);
    }

    #[test]
    #[serial_test::serial(env)]
    fn env_flag_accepts_truthy_aliases() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        for value in ["1", "true", "TRUE", " yes ", "On"] {
            env.set("CALICHE_TEST_FLAG_TRUTHY", value);
            assert!(
                env_flag("CALICHE_TEST_FLAG_TRUTHY"),
                "expected truthy for {value}"
            );
        }
        for value in ["0", "false", "no", "off", "", "maybe"] {
            env.set("CALICHE_TEST_FLAG_TRUTHY", value);
            assert!(
                !env_flag("CALICHE_TEST_FLAG_TRUTHY"),
                "expected falsey for {value}"
            );
        }
    }

    #[test]
    #[serial_test::serial(env)]
    fn env_var_nonempty_errors_on_whitespace() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_TEST_VAR_WS", " \t ");
        assert!(matches!(
            env_var_nonempty("CALICHE_TEST_VAR_WS"),
            Err(std::env::VarError::NotPresent)
        ));
        env.set("CALICHE_TEST_VAR_WS", "ok");
        assert_eq!(env_var_nonempty("CALICHE_TEST_VAR_WS").as_deref(), Ok("ok"));
    }
}
