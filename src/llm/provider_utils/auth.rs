//! Authentication utilities for LLM providers.

/// Helper function to read an API key from environment with a custom error message.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub fn read_api_key_from_env(env_var: &str) -> anyhow::Result<String> {
    let value = caliche_contracts::credential_env::var(env_var)
        .map_err(|e| anyhow::anyhow!("{env_var} environment variable not set: {e}"))?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        anyhow::bail!("{env_var} environment variable is empty");
    }
    Ok(trimmed.to_string())
}

/// Helper function to read a model from environment with fallback.
///
/// Empty/whitespace `provided_model` values are ignored so callers can fall
/// through to the environment or default.
#[must_use]
pub fn read_model_from_env(env_var: &str, default: &str, provided_model: Option<String>) -> String {
    provided_model
        .map(|model| model.trim().to_string())
        .filter(|model| !model.is_empty())
        .or_else(|| {
            caliche_contracts::credential_env::var(env_var)
                .ok()
                .map(|model| model.trim().to_string())
                .filter(|model| !model.is_empty())
        })
        .unwrap_or_else(|| default.to_string())
}

/// Helper function to read an optional base URL from environment with fallback.
///
/// Empty/whitespace env values are treated as unset so callers do not build
/// broken URLs like `/chat/completions`.
#[must_use]
pub fn read_base_url_from_env(env_var: &str, default: &str) -> String {
    caliche_contracts::credential_env::var(env_var)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// Resolve API key from explicit value or environment variable.
///
/// Empty/whitespace provided keys are ignored so the environment fallback can
/// still succeed.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub fn resolve_api_key(
    provided_key: Option<String>,
    env_var: &str,
    missing_message: &str,
) -> anyhow::Result<String> {
    provided_key
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
        .or_else(|| read_api_key_from_env(env_var).ok())
        .ok_or_else(|| anyhow::anyhow!("{missing_message}"))
}

/// Resolve model from explicit value or environment with fallback.
#[must_use]
pub fn resolve_model(provided_model: Option<String>, env_var: &str, default: &str) -> String {
    read_model_from_env(env_var, default, provided_model)
}

/// OpenRouter issues keys with the `sk-or-` prefix. Those fail against api.openai.com.
#[must_use]
pub fn looks_like_openrouter_api_key(key: &str) -> bool {
    key.trim().starts_with("sk-or-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial(env)]
    fn resolve_api_key_rejects_empty_provided_and_falls_back_to_env() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_TEST_RESOLVE_API_KEY", "from-env");
        let key = resolve_api_key(
            Some("   ".into()),
            "CALICHE_TEST_RESOLVE_API_KEY",
            "missing api key",
        )
        .expect("should fall back to env");
        assert_eq!(key, "from-env");
    }

    #[test]
    #[serial_test::serial(env)]
    fn resolve_api_key_errors_when_provided_and_env_are_empty() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_TEST_RESOLVE_API_KEY_EMPTY", "  ");
        let err = resolve_api_key(
            Some(String::new()),
            "CALICHE_TEST_RESOLVE_API_KEY_EMPTY",
            "missing api key",
        )
        .expect_err("empty values should fail");
        assert!(err.to_string().contains("missing api key"));
    }

    #[test]
    #[serial_test::serial(env)]
    fn read_base_url_ignores_empty_env() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_TEST_BASE_URL_EMPTY", "");
        assert_eq!(
            read_base_url_from_env("CALICHE_TEST_BASE_URL_EMPTY", "http://localhost:11434"),
            "http://localhost:11434"
        );
    }

    #[test]
    #[serial_test::serial(env)]
    fn read_model_from_env_ignores_whitespace_provided() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.remove("CALICHE_TEST_MODEL_ENV");
        assert_eq!(
            read_model_from_env(
                "CALICHE_TEST_MODEL_ENV",
                "fallback-model",
                Some("  ".into())
            ),
            "fallback-model"
        );
    }

    #[test]
    fn openrouter_key_prefix_is_detected() {
        assert!(looks_like_openrouter_api_key("sk-or-v1-abc"));
        assert!(looks_like_openrouter_api_key("  sk-or-v1-abc  "));
        assert!(!looks_like_openrouter_api_key("sk-proj-abc"));
        assert!(!looks_like_openrouter_api_key(""));
    }
}
