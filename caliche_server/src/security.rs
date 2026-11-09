//! Startup security policy: fail-closed JWT initialization and auth mode flags.

use crate::middleware::jwt_auth::{JwtAuthConfig, JwtAuthState};
use anyhow::Context;
use axum::http::Method;
use ipnetwork::IpNetwork;
use std::net::{IpAddr, SocketAddr};

/// Resolved authentication posture for the API process.
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// `JWT_ENABLED` was true and JWT validation initialized successfully.
    pub jwt_enabled: bool,
    /// `CALICHE_REQUIRE_AUTH` — most `/api/*` routes require a valid JWT.
    pub require_auth: bool,
    /// Initialized JWT state when [`Self::jwt_enabled`] is true.
    pub jwt_state: Option<JwtAuthState>,
    /// Honor `X-Forwarded-For` / `X-Real-IP` for rate limiting when true.
    pub trusted_proxy: bool,
    /// CIDRs whose peer addresses may supply `X-Forwarded-For`.
    pub trusted_proxy_cidrs: Vec<IpNetwork>,
    /// Bearer token for `/metrics` when [`Self::require_auth`] is set.
    pub metrics_token: Option<String>,
    /// `CALICHE_ALLOW_UNAUTH_ORCHESTRATION` — local-only opt-in that lets the
    /// orchestration routes spawn work without a token. Ignored in a production profile.
    pub allow_unauth_orchestration: bool,
    /// `CALICHE_PUBLIC_CHAT` — keep chat stream routes public even when
    /// [`Self::require_auth`] is set.
    pub public_chat: bool,
}

/// Route prefixes that start multi-agent or workflow work.
///
/// These routes spawn agents, spend tokens, and run tools, so they are refused without an
/// authenticated principal even when the rest of the API runs open in development.
const ORCHESTRATION_PREFIXES: &[&str] = &["/api/orchestration", "/api/workflows"];

/// Whether a path starts orchestration or workflow work.
pub fn is_orchestration_path(path: &str) -> bool {
    ORCHESTRATION_PREFIXES
        .iter()
        .any(|prefix| path.starts_with(prefix))
}

/// Path prefixes that require authenticated users when JWT auth is enabled.
const REQUIRED_AUTH_PREFIXES: &[&str] = &[
    "/api/gateway/",
    "/api/mcp/config",
    "/api/plugins",
    "/api/orchestration",
    "/api/endpoints",
    "/api/workflows",
    "/api/agent/sessions",
    "/api/agent/",
    "/api/cli/config",
    "/api/cli/vault",
];

const PUBLIC_GATEWAY_INGRESS: &[&str] = &[
    "/api/gateway/ingress/slack",
    "/api/gateway/ingress/telegram",
    "/api/gateway/ingress/discord",
    "/api/gateway/ingress/twilio",
    "/api/gateway/ingress/teams",
];

fn is_gateway_webhook_ingress(path: &str) -> bool {
    PUBLIC_GATEWAY_INGRESS.iter().any(|prefix| {
        path == *prefix
            || path
                .strip_prefix(prefix)
                .is_some_and(|rest| rest.starts_with('/'))
    })
}

fn is_chat_stream_path(path: &str) -> bool {
    matches!(path, "/api/chat/stream" | "/api/chat/stream/multipart")
}

impl SecurityPolicy {
    /// Load and validate security configuration from the environment.
    ///
    /// Fails closed when JWT is enabled but cannot be initialized, or when auth is
    /// required without a working JWT stack.
    pub fn from_env() -> anyhow::Result<Self> {
        let require_auth = env_bool("CALICHE_REQUIRE_AUTH");
        let trusted_proxy = env_bool("CALICHE_TRUSTED_PROXY");
        let public_chat = env_bool("CALICHE_PUBLIC_CHAT");
        let trusted_proxy_cidrs = parse_trusted_proxy_cidrs()?;
        let metrics_token = std::env::var("CALICHE_METRICS_TOKEN")
            .ok()
            .filter(|value| !value.trim().is_empty());
        let jwt_config = JwtAuthConfig::from_env()?;
        let insecure_ok = allow_insecure_production();
        warn_configured_open_binds();

        let mut allow_unauth_orchestration = env_bool("CALICHE_ALLOW_UNAUTH_ORCHESTRATION");
        if allow_unauth_orchestration && is_production_profile() {
            tracing::error!(
                target: "caliche_server::security",
                "CALICHE_ALLOW_UNAUTH_ORCHESTRATION is ignored when CALICHE_ENV=production; \
                 orchestration routes continue to require authentication"
            );
            allow_unauth_orchestration = false;
        }
        if allow_unauth_orchestration {
            tracing::warn!(
                target: "caliche_server::security",
                "CALICHE_ALLOW_UNAUTH_ORCHESTRATION is set; orchestration routes accept \
                 unauthenticated requests — use this for local development only"
            );
        }

        if require_auth && !jwt_config.enabled {
            anyhow::bail!(
                "CALICHE_REQUIRE_AUTH is set but JWT_ENABLED is false; \
                 enable JWT or unset CALICHE_REQUIRE_AUTH for development"
            );
        }

        if trusted_proxy && trusted_proxy_cidrs.is_empty() {
            anyhow::bail!(
                "CALICHE_TRUSTED_PROXY is set but CALICHE_TRUSTED_PROXY_CIDRS is empty; \
                 list proxy CIDRs or unset CALICHE_TRUSTED_PROXY"
            );
        }

        if is_production_profile() && !insecure_ok && (!jwt_config.enabled || !require_auth) {
            anyhow::bail!(
                "CALICHE_ENV=production requires JWT_ENABLED and CALICHE_REQUIRE_AUTH; \
                 set CALICHE_ALLOW_INSECURE_PRODUCTION=1 to override"
            );
        }

        let jwt_state = if jwt_config.enabled {
            let state = JwtAuthState::new(jwt_config).map_err(|err| {
                anyhow::anyhow!(
                    "JWT_ENABLED is true but JWT authentication failed to initialize: {err}"
                )
            })?;
            tracing::info!(
                target: "caliche_server::security",
                require_auth,
                trusted_proxy,
                "JWT authentication initialized (fail-closed)"
            );
            Some(state)
        } else {
            if require_auth {
                anyhow::bail!("CALICHE_REQUIRE_AUTH requires JWT_ENABLED");
            }
            None
        };

        Ok(Self {
            jwt_enabled: jwt_state.is_some(),
            require_auth,
            jwt_state,
            trusted_proxy,
            trusted_proxy_cidrs,
            metrics_token,
            allow_unauth_orchestration,
            public_chat,
        })
    }

    /// Whether an orchestration route must refuse an unauthenticated request.
    ///
    /// Orchestration is default-deny: without a JWT stack the route is refused unless the
    /// operator sets `CALICHE_ALLOW_UNAUTH_ORCHESTRATION` for a local run.
    pub fn orchestration_requires_auth(&self, path: &str) -> bool {
        is_orchestration_path(path) && !self.allow_unauth_orchestration
    }

    /// Whether session ownership checks should be enforced (JWT or global require-auth).
    pub fn session_auth_required(&self) -> bool {
        self.jwt_enabled || self.require_auth
    }

    /// Whether webhook signature verification must be configured for enabled adapters.
    pub fn webhook_verification_required(&self) -> bool {
        caliche::gateway::webhook_verification_required()
    }

    /// Whether the request path requires a valid authenticated principal.
    pub fn path_requires_auth(&self, path: &str, method: &Method) -> bool {
        if self.is_public_api_route(path, method) {
            return false;
        }

        if path == "/metrics" && self.require_auth {
            return true;
        }

        if path == "/rpc" || path == "/ws" {
            return self.require_auth || self.jwt_enabled;
        }

        if self.require_auth {
            return path.starts_with("/api/") || path == "/rpc" || path == "/ws";
        }

        let needs_prefix = REQUIRED_AUTH_PREFIXES
            .iter()
            .any(|prefix| path.starts_with(prefix));
        // PUT writes operator config. The policy preset matches settings:
        // authenticated when JWT is on, including when REQUIRE_AUTH is unset.
        let is_protected_put =
            (path == "/api/settings" || path == "/api/policy") && method == Method::PUT;
        needs_prefix || is_protected_put
    }

    /// Whether forwarded client IPs may be trusted from this peer.
    #[must_use]
    pub fn peer_is_trusted_proxy(&self, ip: IpAddr) -> bool {
        self.trusted_proxy && self.trusted_proxy_cidrs.iter().any(|net| net.contains(ip))
    }

    fn is_public_api_route(&self, path: &str, method: &Method) -> bool {
        if is_gateway_webhook_ingress(path) {
            return true;
        }
        if method == Method::POST && is_chat_stream_path(path) {
            return self.public_chat || !self.require_auth;
        }
        false
    }

    /// Whether the presented bearer token authorizes `/metrics` access.
    pub fn metrics_bearer_authorized(&self, authorization: Option<&str>) -> bool {
        let Some(expected) = self.metrics_token.as_deref() else {
            return false;
        };
        let Some(token) = authorization
            .and_then(|value| value.strip_prefix("Bearer "))
            .map(str::trim)
        else {
            return false;
        };
        use subtle::ConstantTimeEq;
        let e = expected.as_bytes();
        let p = token.as_bytes();
        e.len() == p.len() && bool::from(e.ct_eq(p))
    }
}

/// Parse common truthy environment variable values (`1`, `true`, `yes`, `on`).
pub fn env_bool(name: &str) -> bool {
    caliche::security::env_truthy(name)
}

/// Whether the process is running with a production environment profile.
pub fn is_production_profile() -> bool {
    caliche::security::is_production_profile()
}

/// Explicit opt-out for production fail-closed checks.
pub fn allow_insecure_production() -> bool {
    caliche::security::allow_insecure_production()
}

/// Env vars checked at router startup for an unspecified listen host.
const CONFIGURED_BIND_ENV: &[&str] = &["API_HOST", "BIND_ADDRESS"];

/// How a configured listen host is exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindHostKind {
    /// `0.0.0.0`, `::`, or `[::]`.
    Unspecified,
    /// Any other host, including loopback.
    Specific,
}

/// Whether an unspecified bind with open auth should log a startup warning.
///
/// The warning does not refuse startup. Production fail-closed remains the
/// `CALICHE_ENV=production` path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenBindPosture {
    /// Log once per call. The process still starts.
    Warn,
    /// No unspecified-bind warning.
    Quiet,
}

/// Classify a bind host string.
#[must_use]
pub fn classify_bind_host(host: &str) -> BindHostKind {
    match host.trim() {
        "0.0.0.0" | "::" | "[::]" => BindHostKind::Unspecified,
        _ => BindHostKind::Specific,
    }
}

/// Decide whether `host` should log an unspecified-bind warning.
///
/// `Warn` means every-interface listen, production profile off, and
/// require-auth off. The caller still starts.
#[must_use]
pub fn open_unspecified_bind_posture(
    host: &str,
    production: bool,
    require_auth: bool,
) -> OpenBindPosture {
    match classify_bind_host(host) {
        BindHostKind::Unspecified if !production && !require_auth => OpenBindPosture::Warn,
        BindHostKind::Unspecified | BindHostKind::Specific => OpenBindPosture::Quiet,
    }
}

/// Join `host` and `port` into a string [`SocketAddr`] can parse.
///
/// Bare IPv6 literals contain colons. `format!("{host}:{port}")` turns `::`
/// into `:::3000`, which is an `AddrParseError` before listen. Those hosts
/// are bracketed (`[::]:3000`). An already-bracketed host (`[::]`), an IPv4
/// address, and a hostname stay `{host}:{port}`.
#[must_use]
pub fn join_listen_host_port(host: &str, port: u16) -> String {
    let host = host.trim();
    if host.contains(':') && !(host.starts_with('[') && host.ends_with(']')) {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

/// Parse the listen [`SocketAddr`] for `host` and `port`.
///
/// # Errors
///
/// Returns an error when the joined address is not a valid socket address.
pub fn listen_socket_addr(host: &str, port: u16) -> anyhow::Result<SocketAddr> {
    let joined = join_listen_host_port(host, port);
    joined
        .parse::<SocketAddr>()
        .map_err(|err| anyhow::anyhow!("Invalid bind address: {err}"))
}

/// Log a non-fatal warning when `host` is an open unspecified bind.
///
/// Callers still start the server. Local loopback (`127.0.0.1`, `::1`,
/// `localhost`) stays quiet so intentional open-API development is unchanged.
pub fn warn_if_open_unspecified_bind(host: &str) {
    if open_unspecified_bind_posture(
        host,
        is_production_profile(),
        env_bool("CALICHE_REQUIRE_AUTH"),
    ) != OpenBindPosture::Warn
    {
        return;
    }
    tracing::warn!(
        target: "caliche_server::security",
        bind_host = host.trim(),
        "unspecified bind (0.0.0.0 or ::) while CALICHE_ENV is not production and \
         CALICHE_REQUIRE_AUTH is off; the HTTP API can accept unauthenticated \
         requests on that listen address. Startup continues. Set CALICHE_ENV=production \
         with JWT_ENABLED and CALICHE_REQUIRE_AUTH before a public listen. \
         See docs/ops/production-auth-checklist.md"
    );
}

fn warn_configured_open_binds() {
    for key in CONFIGURED_BIND_ENV {
        let Ok(host) = std::env::var(key) else {
            continue;
        };
        warn_if_open_unspecified_bind(&host);
    }
}

/// Whether `CORS_PERMISSIVE` may be enabled for this process.
///
/// # Errors
///
/// Returns an error when permissive CORS is requested in production without
/// `CALICHE_ALLOW_INSECURE_PRODUCTION`.
pub fn allow_cors_permissive(requested: bool) -> anyhow::Result<bool> {
    if requested && is_production_profile() && !allow_insecure_production() {
        anyhow::bail!(
            "CORS_PERMISSIVE is not allowed when CALICHE_ENV=production; \
             set CALICHE_ALLOW_INSECURE_PRODUCTION=1 to override"
        );
    }
    Ok(requested)
}

/// Whether `CORS_PERMISSIVE` is set in the environment and allowed.
///
/// # Errors
///
/// Returns an error when permissive CORS is requested in production without
/// `CALICHE_ALLOW_INSECURE_PRODUCTION`.
pub fn cors_permissive_from_env() -> anyhow::Result<bool> {
    allow_cors_permissive(env_bool("CORS_PERMISSIVE"))
}

fn parse_trusted_proxy_cidrs() -> anyhow::Result<Vec<IpNetwork>> {
    let Ok(raw) = std::env::var("CALICHE_TRUSTED_PROXY_CIDRS") else {
        return Ok(Vec::new());
    };
    raw.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|cidr| {
            cidr.parse::<IpNetwork>()
                .with_context(|| format!("invalid CALICHE_TRUSTED_PROXY_CIDRS entry '{cidr}'"))
        })
        .collect()
}

#[cfg(test)]
mod listen_addr_tests {
    use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};

    use super::{join_listen_host_port, listen_socket_addr};

    #[test]
    fn bare_ipv6_unspecified_joins_to_a_socket_addr() {
        assert_eq!(join_listen_host_port("::", 3000), "[::]:3000");
        assert_eq!(join_listen_host_port(" :: ", 3000), "[::]:3000");
        let addr = listen_socket_addr("::", 3000).expect("bare :: parses");
        assert_eq!(addr, SocketAddr::from((Ipv6Addr::UNSPECIFIED, 3000)));
        assert!(addr.ip().is_unspecified());
    }

    #[test]
    fn bracketed_ipv6_unspecified_is_not_double_wrapped() {
        assert_eq!(join_listen_host_port("[::]", 3000), "[::]:3000");
        let addr = listen_socket_addr("[::]", 3000).expect("bracketed :: parses");
        assert_eq!(addr, SocketAddr::from((Ipv6Addr::UNSPECIFIED, 3000)));
    }

    #[test]
    fn bare_ipv6_loopback_is_bracketed() {
        assert_eq!(join_listen_host_port("::1", 3000), "[::1]:3000");
        let addr = listen_socket_addr("::1", 3000).expect("::1 parses");
        assert!(addr.ip().is_loopback());
        assert_eq!(addr.port(), 3000);
    }

    #[test]
    fn ipv4_hosts_stay_unbracketed() {
        assert_eq!(join_listen_host_port("0.0.0.0", 3000), "0.0.0.0:3000");
        assert_eq!(join_listen_host_port("127.0.0.1", 3000), "127.0.0.1:3000");
        let open = listen_socket_addr("0.0.0.0", 3000).expect("ipv4 unspecified");
        assert_eq!(open, SocketAddr::from((Ipv4Addr::UNSPECIFIED, 3000)));
        let loopback = listen_socket_addr("127.0.0.1", 9).expect("loopback");
        assert!(loopback.ip().is_loopback());
        assert_eq!(loopback.port(), 9);
    }
}
