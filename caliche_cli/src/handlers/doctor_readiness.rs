//! Machine-readable doctor lines for readiness checks.
//!
//! Flags are opt-in truth. `build_id` is tip literacy. `listen_bind` is the
//! CLI `--bind` / `--port` pair `caliche --server` would use. None of
//! these lines claim the process is closed, secured, or authenticated.

use std::io::{self, Write};
use std::sync::OnceLock;

use caliche_server::security::{
    OpenBindPosture, join_listen_host_port, open_unspecified_bind_posture,
};

/// Default `caliche --server` bind. Matches CLI `--bind`.
pub(crate) const DEFAULT_BIND: &str = "127.0.0.1";

/// Default `caliche --server` port. Matches CLI `--port`.
pub(crate) const DEFAULT_PORT: u16 = 3000;

const BUILD_ID_NOTE: &str = "\
build_id is the git commit embedded at compile time, for comparing a PATH binary with tip. It is not a security property. \
unknown means git was missing at compile time and CALICHE_BUILD_ID was unset or not a 40-hex sha. \
unknown is not a commit and does not match git rev-parse HEAD.";

const JWT_NOTE: &str = "\
jwt_enabled is the JWT_ENABLED opt-in flag. Default is false.";

const REQUIRE_AUTH_NOTE: &str = "\
require_auth is the CALICHE_REQUIRE_AUTH opt-in flag. Default is false.";

const LISTEN_NOTE: &str = "\
listen_bind is the address `caliche --server` uses from CLI --bind and --port. \
Default bind is 127.0.0.1. That default only sets the socket address. \
Callers still need their own authentication. API_HOST does not set the CLI listen address.";

const WARN_NOTE: &str = "\
open_bind_warn is the non-fatal unspecified-bind warning for listen_bind. \
Startup still proceeds. Fail-closed applies only when the production profile is on \
and the JWT_ENABLED plus CALICHE_REQUIRE_AUTH matrix is incomplete.";

const PROVIDER_NOTE: &str = "\
default_provider_source is the provider caliche --server boots. \
flag is CLI --provider. \
last is settings.json beside mcp.json when that file has a provider and model. SQLite last_provider and last_model are the fallback when the file is absent. last is used when --provider and --model are omitted, and last wins over AGENT_PROVIDER and DEFAULT_PROVIDER. \
--provider selects the provider and still keeps that record's model when the record names the same provider. --model replaces the record's model. --temperature replaces the record's temperature. \
env is AGENT_PROVIDER, else DEFAULT_PROVIDER, when last does not apply. \
default is built-in openrouter. \
Plain --server does not probe for a local runtime. Local auto-detect runs only when --provider is unset and either --prefer-local is set or the process is opening the REPL. Doctor does not probe. It is not a missing-key switch. \
Project caliche.toml is not this line. \
Interactive REPL and one-shot can still fill a provider from merged config after this choice. \
Missing provider credentials are not authentication and not an isolation boundary. \
An egress-blocked provider URL does not prevent listen. A turn reports that provider's egress error. It is not authentication and not a provider switch.";

const VECTOR_NOTE: &str = "\
vector_backend is the VECTOR_BACKEND kind for this process. An empty value means \
the variable is unset. Unset is unconfigured, not an implicit memory tenant. \
configured means this binary can construct that name because the feature is \
compiled in. It does not mean a remote store is reachable or healthy. \
unavailable is an unknown name or a feature that is off. Dripstone is optional \
retrieval. It is not an isolation boundary or an authorization layer.";

struct RecordedListen {
    bind: String,
    port: u16,
}

struct RecordedBootProvider {
    explicit_provider: Option<String>,
    explicit_model: bool,
}

static CLI_LISTEN: OnceLock<RecordedListen> = OnceLock::new();
static CLI_BOOT_PROVIDER: OnceLock<RecordedBootProvider> = OnceLock::new();

/// Remember parsed `--bind` / `--port` for later doctor reports in this process.
pub(crate) fn remember_cli_listen(bind: &str, port: u16) {
    let _ = CLI_LISTEN.set(RecordedListen {
        bind: bind.to_owned(),
        port,
    });
}

/// Bind recorded at process start, if [`remember_cli_listen`] ran.
pub(crate) fn remembered_bind() -> Option<(&'static str, u16)> {
    CLI_LISTEN
        .get()
        .map(|listen| (listen.bind.as_str(), listen.port))
}

/// Remember `--provider` and whether `--model` was set, for later doctor reports.
pub(crate) fn remember_boot_provider(explicit_provider: Option<&str>, explicit_model: bool) {
    let explicit_provider = explicit_provider
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    let _ = CLI_BOOT_PROVIDER.set(RecordedBootProvider {
        explicit_provider,
        explicit_model,
    });
}

/// Provider flags recorded at process start.
pub(crate) fn remembered_boot_provider() -> (Option<&'static str>, bool) {
    CLI_BOOT_PROVIDER.get().map_or((None, false), |boot| {
        (boot.explicit_provider.as_deref(), boot.explicit_model)
    })
}

/// `JWT_ENABLED` the way API startup parses it (`true` / `false` only).
///
/// This matches `JwtAuthConfig::from_env`: missing, empty, `1`, and `yes` are off.
#[must_use]
pub(crate) fn jwt_enabled_from_raw(value: Option<&str>) -> bool {
    value.is_some_and(|raw| raw.parse::<bool>().unwrap_or(false))
}

/// `CALICHE_REQUIRE_AUTH` the way `caliche::security::env_truthy` reads it.
///
/// Truthy values are `1`, `true`, `yes`, and `on` after trim and ASCII lowercasing.
#[must_use]
pub(crate) fn require_auth_from_raw(value: Option<&str>) -> bool {
    value.is_some_and(|raw| {
        matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// Opt-in flags and the CLI listen address for one doctor report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReadinessLines {
    pub build_id: String,
    pub jwt_enabled: bool,
    pub require_auth: bool,
    pub listen_bind: String,
    pub api_host_is_cli_listen: bool,
    pub open_bind_warn: bool,
    pub default_provider: String,
    pub default_provider_source: String,
    pub vector_backend: Option<String>,
    pub vector_backend_state: String,
}

/// Where the default provider comes from when `--provider` is omitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReportedDefaultProvider {
    pub provider: String,
    pub source: &'static str,
}

/// Resolve the reported `--server` provider. `last` wins over env. Config is not consulted.
#[must_use]
pub(crate) fn reported_default_provider(
    agent_provider: Option<&str>,
    default_provider_env: Option<&str>,
    last_provider: Option<&str>,
    last_model: Option<&str>,
) -> ReportedDefaultProvider {
    let choice = crate::boot_provider::select_server_boot_provider(
        crate::boot_provider::ServerBootProviderInput {
            explicit_provider: None,
            explicit_model: false,
            agent_provider,
            default_provider_env,
            last_provider,
            last_model,
        },
    );
    ReportedDefaultProvider {
        provider: choice.provider,
        source: choice.source,
    }
}

impl ReadinessLines {
    /// Build lines from already-parsed flags. `production` is the production profile.
    #[must_use]
    pub(crate) fn from_parts(
        build_id: &str,
        bind_host: &str,
        port: u16,
        jwt_enabled: bool,
        require_auth: bool,
        production: bool,
    ) -> Self {
        let open_bind_warn = open_unspecified_bind_posture(bind_host, production, require_auth)
            == OpenBindPosture::Warn;
        Self {
            build_id: build_id.to_owned(),
            jwt_enabled,
            require_auth,
            // Same host:port string `start_server` parses, including bare IPv6.
            listen_bind: join_listen_host_port(bind_host, port),
            api_host_is_cli_listen: false,
            open_bind_warn,
            default_provider: caliche::llm::ProviderType::OpenRouter.to_string(),
            default_provider_source: "default".to_owned(),
            vector_backend: None,
            vector_backend_state: caliche::capabilities::VECTOR_RETRIEVAL_UNCONFIGURED.to_owned(),
        }
    }

    /// Read opt-in flags from the process environment. `bind_host` is CLI `--bind`.
    #[must_use]
    pub(crate) fn from_process(bind_host: &str, port: u16) -> Self {
        let jwt = std::env::var("JWT_ENABLED").ok();
        let require_auth = std::env::var("CALICHE_REQUIRE_AUTH").ok();
        let mut lines = Self::from_parts(
            crate::build_info::BUILD_ID,
            bind_host,
            port,
            jwt_enabled_from_raw(jwt.as_deref()),
            require_auth_from_raw(require_auth.as_deref()),
            caliche::security::is_production_profile(),
        );
        let agent_provider = std::env::var("AGENT_PROVIDER").ok();
        let default_provider_env = std::env::var("DEFAULT_PROVIDER").ok();
        let reported = reported_default_provider(
            agent_provider.as_deref(),
            default_provider_env.as_deref(),
            None,
            None,
        );
        lines.default_provider = reported.provider;
        lines.default_provider_source = reported.source.to_owned();
        let retrieval = caliche::capabilities::vector_retrieval_status();
        lines.vector_backend = retrieval.backend;
        lines.vector_backend_state = retrieval.state.to_owned();
        lines
    }

    /// Write stable `key=value` lines, then the short human notes.
    pub(crate) fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        writeln!(w, "build_id={}", self.build_id)?;
        writeln!(w, "jwt_enabled={}", self.jwt_enabled)?;
        writeln!(w, "require_auth={}", self.require_auth)?;
        writeln!(w, "listen_bind={}", self.listen_bind)?;
        writeln!(w, "api_host_is_cli_listen={}", self.api_host_is_cli_listen)?;
        writeln!(w, "open_bind_warn={}", self.open_bind_warn)?;
        writeln!(w, "default_provider={}", self.default_provider)?;
        writeln!(
            w,
            "default_provider_source={}",
            self.default_provider_source
        )?;
        writeln!(
            w,
            "vector_backend={}",
            self.vector_backend.as_deref().unwrap_or("")
        )?;
        writeln!(w, "vector_backend_state={}", self.vector_backend_state)?;
        writeln!(w)?;
        writeln!(w, "{BUILD_ID_NOTE}")?;
        writeln!(w, "{JWT_NOTE}")?;
        writeln!(w, "{REQUIRE_AUTH_NOTE}")?;
        writeln!(w, "{LISTEN_NOTE}")?;
        writeln!(w, "{WARN_NOTE}")?;
        writeln!(w, "{PROVIDER_NOTE}")?;
        writeln!(w, "{VECTOR_NOTE}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BUILD_ID_NOTE, DEFAULT_BIND, DEFAULT_PORT, JWT_NOTE, LISTEN_NOTE, REQUIRE_AUTH_NOTE,
        ReadinessLines, VECTOR_NOTE, WARN_NOTE, jwt_enabled_from_raw, reported_default_provider,
        require_auth_from_raw,
    };

    fn render(
        bind: &str,
        port: u16,
        jwt_enabled: bool,
        require_auth: bool,
        production: bool,
    ) -> String {
        let lines = ReadinessLines::from_parts(
            "0123456789abcdef0123456789abcdef01234567",
            bind,
            port,
            jwt_enabled,
            require_auth,
            production,
        );
        let mut buf = Vec::new();
        lines.write(&mut buf).expect("write readiness");
        String::from_utf8(buf).expect("utf8 readiness")
    }

    fn value<'a>(text: &'a str, key: &str) -> &'a str {
        let prefix = format!("{key}=");
        text.lines()
            .find_map(|line| line.strip_prefix(prefix.as_str()))
            .unwrap_or_else(|| panic!("missing {key}"))
    }

    #[test]
    fn default_listen_is_loopback_and_quiet() {
        assert_eq!(DEFAULT_BIND, "127.0.0.1");
        assert_eq!(DEFAULT_PORT, 3000);
        let text = render(DEFAULT_BIND, DEFAULT_PORT, false, false, false);
        assert_eq!(value(&text, "listen_bind"), "127.0.0.1:3000");
        assert_eq!(value(&text, "api_host_is_cli_listen"), "false");
        assert_eq!(value(&text, "open_bind_warn"), "false");
        assert!(text.contains("Default bind is 127.0.0.1"));
        assert!(text.contains("only sets the socket address"));
        assert!(text.contains("Callers still need their own authentication"));
        assert!(text.contains("API_HOST does not set"));
        assert!(text.contains("Startup still proceeds."));
    }

    #[test]
    fn open_bind_lines_are_opt_in_truth() {
        let text = render("0.0.0.0", 3000, false, false, false);
        assert_eq!(
            value(&text, "build_id"),
            "0123456789abcdef0123456789abcdef01234567"
        );
        assert_eq!(value(&text, "jwt_enabled"), "false");
        assert_eq!(value(&text, "require_auth"), "false");
        assert_eq!(value(&text, "listen_bind"), "0.0.0.0:3000");
        assert_eq!(value(&text, "api_host_is_cli_listen"), "false");
        assert_eq!(value(&text, "open_bind_warn"), "true");
        assert_eq!(value(&text, "default_provider"), "openrouter");
        assert_eq!(value(&text, "default_provider_source"), "default");
        assert_eq!(value(&text, "vector_backend"), "");
        assert_eq!(value(&text, "vector_backend_state"), "unconfigured");
        assert!(text.contains(BUILD_ID_NOTE));
        assert!(text.contains(JWT_NOTE));
        assert!(text.contains(REQUIRE_AUTH_NOTE));
        assert!(text.contains(LISTEN_NOTE));
        assert!(text.contains(WARN_NOTE));
    }

    #[test]
    fn custom_cli_bind_is_the_listen_line() {
        let text = render("127.0.0.1", 8080, true, true, false);
        assert_eq!(value(&text, "listen_bind"), "127.0.0.1:8080");
        assert_eq!(value(&text, "jwt_enabled"), "true");
        assert_eq!(value(&text, "require_auth"), "true");
        assert_eq!(value(&text, "api_host_is_cli_listen"), "false");
        assert_eq!(value(&text, "open_bind_warn"), "false");
        assert!(!text.contains("API_HOST="));
    }

    #[test]
    fn unspecified_hosts_warn_only_when_profile_and_require_auth_are_off() {
        assert_eq!(
            value(&render("::", 3000, false, false, false), "open_bind_warn"),
            "true"
        );
        assert_eq!(
            value(&render("::", 3000, false, false, false), "listen_bind"),
            "[::]:3000"
        );
        assert_eq!(
            value(&render("[::]", 3000, false, false, false), "open_bind_warn"),
            "true"
        );
        assert_eq!(
            value(&render("[::]", 3000, false, false, false), "listen_bind"),
            "[::]:3000"
        );
        assert_eq!(
            value(
                &render("0.0.0.0", 3000, false, false, true),
                "open_bind_warn"
            ),
            "false"
        );
        assert_eq!(
            value(
                &render("0.0.0.0", 3000, false, true, false),
                "open_bind_warn"
            ),
            "false"
        );
        assert_eq!(
            value(
                &render("localhost", 3000, false, false, false),
                "open_bind_warn"
            ),
            "false"
        );
    }

    #[test]
    fn wording_does_not_claim_secured_or_fail_closed_warning() {
        let text = render("0.0.0.0", 3000, false, false, false);
        let lower = text.to_ascii_lowercase();
        assert!(!lower.contains("secured"));
        assert!(!lower.contains("safe"));
        assert!(!lower.contains("authenticated by default"));
        let warn_line = text
            .lines()
            .find(|line| line.starts_with("open_bind_warn="))
            .expect("warn line");
        assert!(!warn_line.contains("fail-closed"));
        assert!(text.contains("Startup still proceeds."));
        assert!(text.contains("Fail-closed applies only when the production profile is on"));
        assert!(text.contains("It is not a security property."));
        assert!(text.contains("unknown is not a commit"));
        assert!(text.contains("does not match git rev-parse HEAD"));
        assert!(text.contains("last wins over"));
        assert!(text.contains("Plain --server does not probe"));
        assert!(!text.contains("can still auto-select"));
        assert!(text.contains("not authentication and not an isolation boundary"));
        assert!(text.contains("egress-blocked provider URL"));
        assert!(text.contains("does not prevent listen"));
        assert!(text.contains(VECTOR_NOTE));
        assert!(text.contains("It is not an isolation boundary"));
        assert!(!lower.contains("auth-by-default"));
        assert!(text.contains("Callers still need their own authentication"));
        assert!(!lower.contains("job object"));
    }

    #[test]
    fn provider_source_precedence_is_last_then_env_then_default() {
        let from_last = reported_default_provider(
            Some("openai"),
            Some("anthropic"),
            Some("google"),
            Some("gemini-2.5-flash"),
        );
        assert_eq!(from_last.source, "last");
        assert_eq!(from_last.provider, "google");

        let from_agent = reported_default_provider(Some("openai"), Some("anthropic"), None, None);
        assert_eq!(from_agent.source, "env");
        assert_eq!(from_agent.provider, "openai");

        let from_default_env = reported_default_provider(
            None,
            Some(" anthropic "),
            None,
            Some("ignored-without-provider"),
        );
        assert_eq!(from_default_env.source, "env");
        assert_eq!(from_default_env.provider, "anthropic");

        let builtin = reported_default_provider(None, None, None, None);
        assert_eq!(builtin.source, "default");
        assert_eq!(builtin.provider, "openrouter");
    }

    #[test]
    fn jwt_flag_matches_api_bool_parse() {
        assert!(!jwt_enabled_from_raw(None));
        assert!(!jwt_enabled_from_raw(Some("")));
        assert!(!jwt_enabled_from_raw(Some("false")));
        assert!(!jwt_enabled_from_raw(Some("TRUE")));
        assert!(!jwt_enabled_from_raw(Some("1")));
        assert!(!jwt_enabled_from_raw(Some("yes")));
        assert!(!jwt_enabled_from_raw(Some(" true")));
        assert!(jwt_enabled_from_raw(Some("true")));
    }

    #[test]
    fn require_auth_flag_matches_env_truthy() {
        assert!(!require_auth_from_raw(None));
        assert!(!require_auth_from_raw(Some("")));
        assert!(!require_auth_from_raw(Some("false")));
        assert!(!require_auth_from_raw(Some("0")));
        assert!(!require_auth_from_raw(Some("no")));
        assert!(require_auth_from_raw(Some("true")));
        assert!(require_auth_from_raw(Some("TRUE")));
        assert!(require_auth_from_raw(Some("1")));
        assert!(require_auth_from_raw(Some("yes")));
        assert!(require_auth_from_raw(Some("on")));
        assert!(require_auth_from_raw(Some("  true  ")));
    }

    #[test]
    fn from_process_uses_cli_bind_and_flag_parsers() {
        let jwt = std::env::var("JWT_ENABLED").ok();
        let require_auth = std::env::var("CALICHE_REQUIRE_AUTH").ok();
        let lines = ReadinessLines::from_process("10.1.2.3", 4040);
        let mut expected = ReadinessLines::from_parts(
            crate::build_info::BUILD_ID,
            "10.1.2.3",
            4040,
            jwt_enabled_from_raw(jwt.as_deref()),
            require_auth_from_raw(require_auth.as_deref()),
            caliche::security::is_production_profile(),
        );
        let agent_provider = std::env::var("AGENT_PROVIDER").ok();
        let default_provider_env = std::env::var("DEFAULT_PROVIDER").ok();
        let reported = reported_default_provider(
            agent_provider.as_deref(),
            default_provider_env.as_deref(),
            None,
            None,
        );
        expected.default_provider = reported.provider;
        expected.default_provider_source = reported.source.to_owned();
        let retrieval = caliche::capabilities::vector_retrieval_status();
        expected.vector_backend = retrieval.backend;
        expected.vector_backend_state = retrieval.state.to_owned();
        assert_eq!(lines, expected);
        assert_eq!(lines.listen_bind, "10.1.2.3:4040");
        assert!(!lines.api_host_is_cli_listen);
    }
}
