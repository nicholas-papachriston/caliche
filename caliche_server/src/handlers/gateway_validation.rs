//! Validation helpers for gateway cluster registration.
//!
//! These helpers keep the gateway handler focused on HTTP concerns while
//! centralising SSRF / auth defenses for cluster node registration and
//! forwarding. Behaviour is driven by env vars documented in
//! [`caliche_server/README.md`](../README.md).

use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use axum::http::{HeaderMap, StatusCode};
use ipnetwork::IpNetwork;
use subtle::ConstantTimeEq;

use crate::middleware::context::UserContext;

/// HTTP header used to pass the shared cluster-registration token.
pub const CLUSTER_REGISTER_HEADER: &str = "x-cluster-register-token";

/// Role that grants cluster-admin permissions when JWT auth is configured.
pub const CLUSTER_ADMIN_ROLE: &str = "cluster-admin";

/// Hard cap on the forwarded-response body (1 MiB) to prevent oversized
/// responses from exhausting server memory when fronting untrusted workers.
pub const MAX_WORKER_BODY_BYTES: usize = 1024 * 1024;

fn read_env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn host_allowlist() -> Vec<String> {
    std::env::var("CALICHE_CLUSTER_HOST_ALLOWLIST")
        .ok()
        .map(|raw| {
            raw.split(',')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn is_private_or_reserved(ip: &IpAddr) -> bool {
    // RFC1918 + loopback + link-local + unique local + unspecified + CGNAT
    let cidrs = [
        "10.0.0.0/8",
        "127.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "100.64.0.0/10",
        "0.0.0.0/32",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    ];
    for cidr in cidrs {
        if let Ok(net) = cidr.parse::<IpNetwork>()
            && net.contains(*ip)
        {
            return true;
        }
    }
    false
}

/// Validate a cluster node `base_url`. Returns a parsed [`url::Url`] or a
/// descriptive reason string. Rejects non-http(s) schemes, hosts outside the
/// allowlist, and private/loopback/link-local IPs unless explicitly enabled
/// via env. See README for the full flag list.
pub fn validate_cluster_base_url(raw: &str) -> Result<url::Url> {
    let url = url::Url::parse(raw).context("invalid URL")?;
    let scheme = url.scheme();
    if scheme != "https" {
        if scheme == "http" {
            if !read_env_flag("CALICHE_CLUSTER_ALLOW_HTTP") {
                bail!("http scheme requires CALICHE_CLUSTER_ALLOW_HTTP=true");
            }
        } else {
            bail!("unsupported scheme: {scheme}");
        }
    }

    let host_str = url.host_str().context("missing host")?.to_ascii_lowercase();

    let allowlist = host_allowlist();
    if allowlist.is_empty() {
        bail!("CALICHE_CLUSTER_HOST_ALLOWLIST is unset; refusing registration");
    }
    if !allowlist.iter().any(|h| h == &host_str) {
        bail!("host {host_str} is not in the cluster allowlist");
    }

    if let Ok(ip) = host_str.parse::<IpAddr>() {
        if is_private_or_reserved(&ip) && !read_env_flag("CALICHE_CLUSTER_ALLOW_PRIVATE") {
            bail!(
                "host {ip} is a private/reserved address; set CALICHE_CLUSTER_ALLOW_PRIVATE=true to allow"
            );
        }
    }

    Ok(url)
}

/// Authorize a cluster-registration request. Accepts either:
/// - a shared secret via `X-Cluster-Register-Token` matching
///   `CALICHE_CLUSTER_REGISTER_TOKEN` (constant-time compare), or
/// - a valid [`UserContext`] with the [`CLUSTER_ADMIN_ROLE`].
///
/// When neither is configured or presented, returns `401 Unauthorized` so
/// operators do not accidentally expose cluster registration.
pub fn authorize_cluster_register(
    headers: &HeaderMap,
    user: Option<&UserContext>,
) -> Result<(), (StatusCode, &'static str)> {
    if let Some(ctx) = user
        && ctx.roles.iter().any(|r| r == CLUSTER_ADMIN_ROLE)
    {
        return Ok(());
    }

    let expected = std::env::var("CALICHE_CLUSTER_REGISTER_TOKEN").ok();
    let presented = headers
        .get(CLUSTER_REGISTER_HEADER)
        .and_then(|v| v.to_str().ok());

    match (expected.as_deref(), presented) {
        (Some(expected), Some(presented)) if !expected.is_empty() => {
            let e = expected.as_bytes();
            let p = presented.as_bytes();
            if e.len() == p.len() && bool::from(e.ct_eq(p)) {
                Ok(())
            } else {
                Err((StatusCode::UNAUTHORIZED, "invalid cluster register token"))
            }
        }
        _ => Err((
            StatusCode::UNAUTHORIZED,
            "cluster registration requires CALICHE_CLUSTER_REGISTER_TOKEN or cluster-admin JWT",
        )),
    }
}

fn lookup_host_ips(host: &str) -> Result<Vec<IpAddr>> {
    (host, 0u16)
        .to_socket_addrs()
        .map(|addrs| {
            let mut ips: Vec<IpAddr> = addrs.map(|addr| addr.ip()).collect();
            ips.sort_unstable();
            ips.dedup();
            ips
        })
        .with_context(|| format!("cluster host did not resolve: {host}"))
}

fn deny_private_cluster_ips(ips: &[IpAddr]) -> Result<()> {
    if read_env_flag("CALICHE_CLUSTER_ALLOW_PRIVATE") {
        return Ok(());
    }
    for ip in ips {
        if is_private_or_reserved(ip) {
            bail!(
                "host {ip} is a private/reserved address; set CALICHE_CLUSTER_ALLOW_PRIVATE=true to allow"
            );
        }
    }
    Ok(())
}

/// DNS-pin a no-redirect client to `raw` after re-running allowlist checks.
///
/// # Errors
///
/// Returns an error when the URL is blocked, cannot be resolved, or the
/// client cannot be built. Construction failure is fail-closed (no redirecting
/// fallback client).
pub fn cluster_forward_client(raw: &str) -> Result<(reqwest::Client, url::Url)> {
    cluster_forward_client_with_lookup(raw, lookup_host_ips)
}

fn cluster_forward_client_with_lookup(
    raw: &str,
    lookup: impl Fn(&str) -> Result<Vec<IpAddr>>,
) -> Result<(reqwest::Client, url::Url)> {
    let url = validate_cluster_base_url(raw)?;
    let host = url.host_str().context("missing host")?.to_string();
    let ips = match url.host() {
        Some(url::Host::Ipv4(v4)) => vec![IpAddr::V4(v4)],
        Some(url::Host::Ipv6(v6)) => vec![IpAddr::V6(v6)],
        Some(url::Host::Domain(name)) => {
            let ips = lookup(name)?;
            if ips.is_empty() {
                bail!("cluster host did not resolve: {name}");
            }
            ips
        }
        None => bail!("missing host"),
    };
    deny_private_cluster_ips(&ips)?;

    let mut builder = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(10))
        .connect_timeout(Duration::from_secs(5))
        .pool_max_idle_per_host(4);
    if host.parse::<IpAddr>().is_err()
        && let Some(port) = url.port_or_known_default()
    {
        let addrs: Vec<SocketAddr> = ips
            .iter()
            .copied()
            .map(|ip| SocketAddr::new(ip, port))
            .collect();
        builder = builder.resolve_to_addrs(&host, &addrs);
    }
    let client = builder.build().context("cluster forward client")?;
    Ok((client, url))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear_env(env: &mut caliche_test_env::EnvGuard) {
        env.remove("CALICHE_CLUSTER_HOST_ALLOWLIST");
        env.remove("CALICHE_CLUSTER_ALLOW_HTTP");
        env.remove("CALICHE_CLUSTER_ALLOW_PRIVATE");
        env.remove("CALICHE_CLUSTER_REGISTER_TOKEN");
    }

    #[test]
    #[serial_test::serial]
    fn rejects_unset_allowlist() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        let err =
            validate_cluster_base_url("https://example.com/").expect_err("should require list");
        assert!(err.to_string().contains("HOST_ALLOWLIST"));
    }

    #[test]
    #[serial_test::serial]
    fn rejects_disallowed_host() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        env.set("CALICHE_CLUSTER_HOST_ALLOWLIST", "trusted.example");
        let err = validate_cluster_base_url("https://evil.example/")
            .expect_err("should reject unknown host");
        assert!(err.to_string().contains("allowlist"));
    }

    #[test]
    #[serial_test::serial]
    fn rejects_http_without_opt_in() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        env.set("CALICHE_CLUSTER_HOST_ALLOWLIST", "trusted.example");
        let err =
            validate_cluster_base_url("http://trusted.example/").expect_err("should reject http");
        assert!(err.to_string().contains("CALICHE_CLUSTER_ALLOW_HTTP"));
    }

    #[test]
    #[serial_test::serial]
    fn rejects_private_ip() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        env.set("CALICHE_CLUSTER_HOST_ALLOWLIST", "10.0.0.5");
        env.set("CALICHE_CLUSTER_ALLOW_HTTP", "true");
        let err =
            validate_cluster_base_url("http://10.0.0.5/").expect_err("should reject private IP");
        assert!(err.to_string().contains("CALICHE_CLUSTER_ALLOW_PRIVATE"));
    }

    #[test]
    #[serial_test::serial]
    fn accepts_allowed_https_host() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        env.set("CALICHE_CLUSTER_HOST_ALLOWLIST", "trusted.example");
        validate_cluster_base_url("https://trusted.example:8443/path")
            .expect("should accept allowed host");
    }

    #[test]
    #[serial_test::serial]
    fn accepts_private_ip_with_opt_in() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        env.set("CALICHE_CLUSTER_HOST_ALLOWLIST", "127.0.0.1");
        env.set("CALICHE_CLUSTER_ALLOW_HTTP", "true");
        env.set("CALICHE_CLUSTER_ALLOW_PRIVATE", "true");
        validate_cluster_base_url("http://127.0.0.1:8080/")
            .expect("should accept private IP when opted in");
    }

    #[test]
    #[serial_test::serial]
    fn authorize_requires_token_or_role() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        let headers = HeaderMap::new();
        assert!(authorize_cluster_register(&headers, None).is_err());
    }

    #[test]
    #[serial_test::serial]
    fn authorize_accepts_matching_token() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        env.set("CALICHE_CLUSTER_REGISTER_TOKEN", "s3cret");
        let mut headers = HeaderMap::new();
        headers.insert(
            CLUSTER_REGISTER_HEADER,
            axum::http::HeaderValue::from_static("s3cret"),
        );
        authorize_cluster_register(&headers, None).expect("valid token should authorize");
    }

    #[test]
    #[serial_test::serial]
    fn authorize_rejects_wrong_token() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        env.set("CALICHE_CLUSTER_REGISTER_TOKEN", "s3cret");
        let mut headers = HeaderMap::new();
        headers.insert(
            CLUSTER_REGISTER_HEADER,
            axum::http::HeaderValue::from_static("wrong"),
        );
        assert!(authorize_cluster_register(&headers, None).is_err());
    }

    #[test]
    #[serial_test::serial]
    fn authorize_accepts_cluster_admin_role() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        let user = UserContext {
            user_id: "u".into(),
            tenant_id: None,
            email: None,
            roles: vec![CLUSTER_ADMIN_ROLE.to_string()],
            custom_claims: Default::default(),
        };
        let headers = HeaderMap::new();
        authorize_cluster_register(&headers, Some(&user))
            .expect("cluster-admin role should authorize");
    }

    #[test]
    #[serial_test::serial]
    fn forward_rejects_allowlisted_host_resolved_to_metadata() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        env.set("CALICHE_CLUSTER_HOST_ALLOWLIST", "trusted.example");
        let err = cluster_forward_client_with_lookup("https://trusted.example/agent", |_| {
            Ok(vec![IpAddr::from([169, 254, 169, 254])])
        })
        .expect_err("rebinding to metadata");
        assert!(
            err.to_string().contains("private") || err.to_string().contains("reserved"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[serial_test::serial]
    fn forward_pins_public_resolution() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        env.set("CALICHE_CLUSTER_HOST_ALLOWLIST", "trusted.example");
        cluster_forward_client_with_lookup("https://trusted.example/agent", |_| {
            Ok(vec![IpAddr::from([93, 184, 216, 34])])
        })
        .expect("public pin");
    }

    #[test]
    #[serial_test::serial]
    fn forward_rejects_loopback_literal_without_opt_in() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        clear_env(&mut env);
        env.set("CALICHE_CLUSTER_HOST_ALLOWLIST", "127.0.0.1");
        env.set("CALICHE_CLUSTER_ALLOW_HTTP", "true");
        cluster_forward_client_with_lookup("http://127.0.0.1:8080/", |_| unreachable!("literal"))
            .expect_err("loopback");
    }
}
