use agent_client_protocol_schema::v1::ErrorCode;
use caliche_contracts::acp::AuthMethod;

use crate::error::AcpClientError;

/// Refuse peer auth methods that are outside the hint allowlist.
///
/// An empty list is accepted. A non-empty list is accepted only when every
/// method id is exactly `opencode-login` or `hermes-setup`. Any other id,
/// including a provider id beside `hermes-setup`, stops the client before
/// `session/new`. This function does not start a terminal, does not call
/// `authenticate`, and does not read secrets.
///
/// # Errors
///
/// Returns [`AcpClientError::AuthRejected`] when any method id is outside
/// that allowlist.
///
/// # Examples
///
/// ```
/// use caliche_acp_client::reject_peer_auth;
/// use caliche_contracts::acp::{AuthMethod, AuthMethodAgent};
///
/// let cursor = AuthMethod::Agent(
///     AuthMethodAgent::new("cursor-account", "Cursor account")
///         .description("Sign in with your Cursor subscription"),
/// );
/// let error = reject_peer_auth(std::slice::from_ref(&cursor)).unwrap_err();
/// assert!(error.to_string().contains("cursor-account"));
/// assert!(reject_peer_auth(&[]).is_ok());
/// ```
pub fn reject_peer_auth(methods: &[AuthMethod]) -> Result<(), AcpClientError> {
    if methods.is_empty() || methods.iter().all(is_allowlisted_hint) {
        return Ok(());
    }
    Err(AcpClientError::AuthRejected {
        message: allowlist_message(&blocked_labels(methods)),
    })
}

/// Map an ACP `auth_required` JSON-RPC error to out-of-band BYO guidance.
///
/// Other errors pass through. This does not call `authenticate`.
pub fn map_auth_required(err: AcpClientError) -> AcpClientError {
    match err {
        AcpClientError::Rpc { code, message } if is_auth_required(code, &message) => {
            AcpClientError::AuthRequired {
                message: auth_required_guidance(&message),
            }
        }
        other => other,
    }
}

fn allowlist_message(labels: &str) -> String {
    format!(
        "peer initialize advertised auth methods ({labels}) outside the hint allowlist. Caliche continues only when authMethods is empty, or every method id is exactly `opencode-login` or `hermes-setup`. It does not call `authenticate` and does not run terminal login. Supply a provider API key or a local model through the child environment."
    )
}

fn auth_required_guidance(peer_message: &str) -> String {
    format!(
        "peer returned auth_required ({peer_message}). Caliche does not call `authenticate` and does not run `opencode auth login`, `hermes setup`, or `hermes acp --setup`. Put a provider API key or a local model on the peer outside Caliche. OpenCode: set OPENAI_API_KEY or ANTHROPIC_API_KEY in the child environment, or point opencode.json at a local OpenAI-compatible base URL. Hermes: run `hermes config set` for that key outside Caliche and set model.provider to an API-key or custom local endpoint. Leave ChatGPT, Claude, Codex, and Nous subscriptions unused."
    )
}

fn is_auth_required(code: i32, message: &str) -> bool {
    code == i32::from(ErrorCode::AuthRequired) || text_is_auth_required(message)
}

fn text_is_auth_required(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("auth_required") || lower.contains("authentication required")
}

fn blocked_labels(methods: &[AuthMethod]) -> String {
    let mut labels = String::new();
    for method in methods {
        if is_allowlisted_hint(method) {
            continue;
        }
        if !labels.is_empty() {
            labels.push_str(", ");
        }
        labels.push_str(method_label(method));
    }
    labels
}

fn method_label(method: &AuthMethod) -> &str {
    let id = method.id().0.as_ref();
    if !id.is_empty() {
        id
    } else if !method.name().is_empty() {
        method.name()
    } else {
        "unnamed"
    }
}

fn is_allowlisted_hint(method: &AuthMethod) -> bool {
    let id = method.id().0.as_ref();
    id == "opencode-login" || id == "hermes-setup"
}

#[cfg(test)]
mod tests {
    use super::{map_auth_required, reject_peer_auth};
    use crate::error::AcpClientError;
    use caliche_contracts::acp::{
        AuthMethod, AuthMethodAgent, AuthMethodId, AuthMethodTerminal, InitializeResponse,
    };

    fn method(id: &str, name: &str, description: Option<&str>, kind: Option<&str>) -> AuthMethod {
        let description = description.map(ToOwned::to_owned);
        let id = AuthMethodId::new(id);
        if kind.is_some_and(|kind| kind.eq_ignore_ascii_case("terminal")) {
            return AuthMethod::Terminal(
                AuthMethodTerminal::new(id, name).description(description),
            );
        }
        AuthMethod::Agent(AuthMethodAgent::new(id, name).description(description))
    }

    #[test]
    fn empty_auth_is_allowed() {
        assert!(reject_peer_auth(&[]).is_ok());
    }

    #[test]
    fn login_hints_are_informational() {
        assert!(
            reject_peer_auth(&[method(
                "hermes-setup",
                "Terminal setup",
                Some("Run hermes --setup"),
                Some("terminal"),
            )])
            .is_ok()
        );
        assert!(
            reject_peer_auth(&[method(
                "opencode-login",
                "Login with opencode",
                Some("Run `opencode auth login` in the terminal"),
                None,
            )])
            .is_ok()
        );
    }

    #[test]
    fn cursor_chatgpt_and_claude_subscription_are_refused() {
        let samples = [
            method(
                "cursor-account",
                "Cursor",
                Some("Cursor account"),
                Some("agent"),
            ),
            method("chatgpt", "ChatGPT", Some("Sign in to ChatGPT"), None),
            method(
                "claude",
                "Claude",
                Some("Claude subscription"),
                Some("agent"),
            ),
        ];
        for sample in samples {
            let error = reject_peer_auth(std::slice::from_ref(&sample)).unwrap_err();
            let text = error.to_string();
            assert!(text.contains(sample.id().0.as_ref()), "{text}");
            assert!(text.contains("opencode-login"), "{text}");
        }
    }

    #[test]
    fn opencode_login_payload_proceeds() {
        let raw = r#"{"protocolVersion":1,"authMethods":[{"description":"Run `opencode auth login` in the terminal","name":"Login with opencode","id":"opencode-login"}]}"#;
        let parsed: InitializeResponse =
            serde_json::from_str(raw).expect("decode opencode initialize");
        assert_eq!(parsed.auth_methods.len(), 1, "auth method must decode");
        assert!(reject_peer_auth(&parsed.auth_methods).is_ok());
    }

    #[test]
    fn hermes_setup_beside_openrouter_is_refused() {
        let raw = r#"{"protocolVersion":1,"authMethods":[{"id":"openrouter","name":"openrouter runtime credentials","description":"Authenticate Hermes using the currently configured openrouter runtime credentials."},{"type":"terminal","id":"hermes-setup","name":"Configure Hermes provider","args":["--setup"],"description":"Open Hermes' interactive model/provider setup in a terminal."}]}"#;
        let parsed: InitializeResponse =
            serde_json::from_str(raw).expect("decode hermes initialize");
        assert_eq!(parsed.auth_methods.len(), 2, "both methods must decode");
        let error = reject_peer_auth(&parsed.auth_methods).expect_err("mixed list");
        let text = error.to_string();
        assert!(text.contains("openrouter"), "{text}");
        assert!(text.contains("authenticate"), "{text}");
    }

    #[test]
    fn hint_ids_together_are_allowed() {
        assert!(
            reject_peer_auth(&[
                method("opencode-login", "Login with opencode", None, None),
                method("hermes-setup", "Configure Hermes", None, Some("terminal")),
            ])
            .is_ok()
        );
    }

    #[test]
    fn custom_terminal_openrouter_and_custom_login_are_refused() {
        let samples = [
            method(
                "custom-terminal",
                "Custom terminal",
                Some("Run a terminal login"),
                Some("terminal"),
            ),
            method(
                "openrouter",
                "openrouter",
                Some("runtime credentials"),
                None,
            ),
            method(
                "custom-login",
                "Sign in",
                Some("Complete authenticate"),
                None,
            ),
        ];
        for sample in samples {
            let error = reject_peer_auth(std::slice::from_ref(&sample)).unwrap_err();
            assert!(
                error.to_string().contains(sample.id().0.as_ref()),
                "{error}"
            );
        }
    }

    #[test]
    fn codex_beside_hermes_setup_is_refused() {
        let raw = r#"{"protocolVersion":1,"authMethods":[{"id":"openai-codex","name":"openai-codex runtime credentials","description":"Authenticate Hermes using the currently configured openai-codex runtime credentials."},{"type":"terminal","id":"hermes-setup","name":"Configure Hermes provider","args":["--setup"],"description":"Open Hermes' interactive model/provider setup in a terminal."}]}"#;
        let parsed: InitializeResponse =
            serde_json::from_str(raw).expect("decode codex initialize");
        let error = reject_peer_auth(&parsed.auth_methods).expect_err("refuse codex");
        let text = error.to_string();
        assert!(text.contains("openai-codex"), "{text}");
        assert!(text.contains("authenticate"), "{text}");
    }

    #[test]
    fn other_auth_methods_are_refused() {
        let error = reject_peer_auth(&[method(
            "custom-login",
            "Sign in",
            Some("Complete authenticate before using the agent"),
            None,
        )])
        .unwrap_err();
        let text = error.to_string();
        assert!(text.contains("custom-login"), "{text}");
        assert!(text.contains("hermes-setup"), "{text}");
    }

    #[test]
    fn auth_required_names_out_of_band_byo() {
        let err = map_auth_required(AcpClientError::Rpc {
            code: -32000,
            message: "Authentication required".to_owned(),
        });
        let AcpClientError::AuthRequired { message } = err else {
            panic!("expected auth_required");
        };
        assert!(message.contains("auth_required"), "{message}");
        assert!(message.contains("hermes config set"), "{message}");
        assert!(message.contains("OPENAI_API_KEY"), "{message}");
        assert!(message.contains("does not run"), "{message}");
    }

    #[test]
    fn ordinary_rpc_errors_stay_rpc_errors() {
        let err = map_auth_required(AcpClientError::Rpc {
            code: -32603,
            message: "provider overloaded".to_owned(),
        });
        assert!(matches!(err, AcpClientError::Rpc { code: -32603, .. }));
    }
}
