//! HTTP egress policy (SSRF mitigation).
//!
//! Shared by plugin steps, gateway webhooks, the HTTP tool, and scheduled
//! delivery. Hostnames are allowlisted first (when configured), then
//! DNS-resolved so a public name cannot point at loopback, link-local, or
//! RFC1918/CGNAT space.

use anyhow::{Context, Result, bail};
#[cfg(feature = "reqwest")]
use std::net::SocketAddr;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
#[cfg(feature = "reqwest")]
use std::time::Duration;

/// URL that passed allowlist and DNS-time IP checks.
///
/// Call [`Self::pinned_client`] so connect uses these IPs instead of
/// re-resolving (DNS rebinding). Redirects are disabled on that client so a
/// public host cannot 302 into link-local or loopback space.
#[derive(Debug, Clone)]
pub struct ValidatedHttpUrl {
    pub url: url::Url,
    pub host: String,
    pub ips: Vec<IpAddr>,
}

#[cfg(feature = "reqwest")]
impl ValidatedHttpUrl {
    /// Client builder that dials the validated IPs while keeping TLS SNI/Host.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP client cannot be constructed.
    pub fn pinned_client_builder(&self) -> Result<reqwest::ClientBuilder> {
        let mut builder = reqwest::Client::builder();
        if self.host.parse::<IpAddr>().is_err()
            && !self.ips.is_empty()
            && let Some(port) = self.url.port_or_known_default()
        {
            let addrs: Vec<SocketAddr> = self
                .ips
                .iter()
                .copied()
                .map(|ip| SocketAddr::new(ip, port))
                .collect();
            builder = builder.resolve_to_addrs(&self.host, &addrs);
        }
        Ok(builder)
    }

    /// Build a client that dials the validated IPs and does not follow redirects.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP client cannot be constructed.
    pub fn pinned_client(&self) -> Result<reqwest::Client> {
        self.pinned_client_builder()?
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .context("pin http client")
    }
}

/// HTTP egress policy: domain allowlist and blocked private/metadata targets.
#[derive(Debug, Clone)]
pub struct HttpPolicy {
    domain_allowlist: Option<Vec<String>>,
    /// When true, loopback and RFC1918 are allowed (local MCP / user
    /// customization). Link-local, metadata hostnames, and CGNAT stay blocked.
    allow_loopback_and_private: bool,
}

impl HttpPolicy {
    pub fn new(domain_allowlist: Option<Vec<String>>) -> Self {
        Self {
            domain_allowlist,
            allow_loopback_and_private: false,
        }
    }

    /// Permit loopback and private RFC1918 targets for operator-local
    /// customization (MCP HTTP, local sidecars). Metadata and link-local
    /// addresses remain blocked.
    #[must_use]
    pub fn allowing_loopback_and_private(mut self) -> Self {
        self.allow_loopback_and_private = true;
        self
    }

    /// Parse comma-separated hostnames from `PLUGIN_HTTP_ALLOWLIST`.
    pub fn allowlist_from_env() -> Option<Vec<String>> {
        std::env::var("PLUGIN_HTTP_ALLOWLIST")
            .ok()
            .map(|raw| {
                raw.split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
            .filter(|list: &Vec<String>| !list.is_empty())
    }

    pub fn domain_allowlist(&self) -> Option<&[String]> {
        self.domain_allowlist.as_deref()
    }

    /// Load allowlist from `PLUGIN_HTTP_ALLOWLIST`.
    ///
    /// Production requires a non-empty allowlist unless
    /// `CALICHE_ALLOW_INSECURE_PRODUCTION` is set. A lone `*` is an
    /// explicit allow-any-public-host opt-out (private/loopback IPs stay blocked).
    ///
    /// # Errors
    ///
    /// Returns an error when production has no allowlist and the insecure
    /// override is unset.
    pub fn from_env() -> Result<Self> {
        let allowlist = Self::allowlist_from_env();
        if allowlist.is_none()
            && super::is_production_profile()
            && !super::allow_insecure_production()
        {
            bail!(
                "PLUGIN_HTTP_ALLOWLIST is required when CALICHE_ENV=production; \
                 set a hostname list or '*' to allow any public host"
            );
        }
        Ok(Self::new(allowlist))
    }

    /// Validate a URL before issuing an outbound request.
    ///
    /// Hostnames are allowlisted first, then DNS-resolved so a public name
    /// cannot point at loopback, link-local, or RFC1918/CGNAT space.
    ///
    /// # Errors
    ///
    /// Returns an error when the URL is blocked or cannot be resolved.
    pub fn validate_resolved_url(&self, url: &str) -> Result<ValidatedHttpUrl> {
        self.validate_url(url, lookup_host_ips)
    }

    /// Scheme, hostname, IP-literal, and allowlist checks without DNS.
    ///
    /// Use at plugin load so a public `base_url` does not require network.
    /// Runtime callers must still use [`Self::validate_resolved_url`].
    ///
    /// # Errors
    ///
    /// Returns an error when the URL is syntactically blocked.
    pub fn validate_static_url(&self, url: &str) -> Result<()> {
        self.inspect_url(url).map(|_| ())
    }

    /// Redirect policy that re-validates each hop against this policy.
    ///
    /// Used by the HTTP tool when following redirects. Plugin and webhook
    /// clients should keep [`ValidatedHttpUrl::pinned_client`] (no redirects).
    #[cfg(feature = "reqwest")]
    #[must_use]
    pub fn ssrf_safe_redirect_policy(&self) -> reqwest::redirect::Policy {
        let policy = self.clone();
        reqwest::redirect::Policy::custom(move |attempt| {
            if attempt.previous().len() >= 10 {
                return attempt.error("too many redirects");
            }
            match policy.validate_resolved_url(attempt.url().as_str()) {
                Ok(_) => attempt.follow(),
                Err(err) => attempt.error(err),
            }
        })
    }

    /// Validate `url` and return a DNS-pinned, no-redirect client.
    ///
    /// # Errors
    ///
    /// Returns an error when the URL is blocked or the client cannot be built.
    #[cfg(feature = "reqwest")]
    pub fn egress_client(&self, url: &str) -> Result<(reqwest::Client, url::Url)> {
        let validated = self.validate_resolved_url(url)?;
        Ok((validated.pinned_client()?, validated.url))
    }

    fn validate_url(
        &self,
        url: &str,
        lookup: impl Fn(&str) -> Result<Vec<IpAddr>>,
    ) -> Result<ValidatedHttpUrl> {
        let inspected = self.inspect_url(url)?;
        let ips = match inspected.literal_ips {
            Some(ips) => ips,
            None => {
                let ips = lookup(&inspected.host)?;
                if ips.is_empty() {
                    bail!("plugin http url host did not resolve: {}", inspected.host);
                }
                for ip in &ips {
                    deny_blocked_ip(*ip, self.allow_loopback_and_private)?;
                }
                ips
            }
        };
        Ok(ValidatedHttpUrl {
            url: inspected.parsed,
            host: inspected.host,
            ips,
        })
    }

    fn inspect_url(&self, url: &str) -> Result<InspectedHttpUrl> {
        let parsed =
            url::Url::parse(url).with_context(|| format!("invalid plugin http url: {url}"))?;

        match parsed.scheme() {
            "http" | "https" => {}
            other => bail!("plugin http url scheme not allowed: {other}"),
        }

        let (host, literal_ips) = match parsed.host() {
            Some(url::Host::Ipv4(v4)) => {
                let ip = IpAddr::V4(v4);
                deny_blocked_ip(ip, self.allow_loopback_and_private)?;
                let host = v4.to_string();
                self.deny_unlisted_host(&host)?;
                (host, Some(vec![ip]))
            }
            Some(url::Host::Ipv6(v6)) => {
                let ip = IpAddr::V6(v6);
                deny_blocked_ip(ip, self.allow_loopback_and_private)?;
                let host = v6.to_string();
                self.deny_unlisted_host(&host)?;
                (host, Some(vec![ip]))
            }
            Some(url::Host::Domain(host)) => {
                if is_metadata_hostname(host)
                    || (!self.allow_loopback_and_private && is_loopback_hostname(host))
                {
                    bail!("plugin http url targets a blocked host: {host}");
                }
                self.deny_unlisted_host(host)?;
                (host.to_string(), None)
            }
            None => bail!("plugin http url missing host"),
        };

        Ok(InspectedHttpUrl {
            parsed,
            host,
            literal_ips,
        })
    }

    fn deny_unlisted_host(&self, host: &str) -> Result<()> {
        let Some(ref allowlist) = self.domain_allowlist else {
            return Ok(());
        };
        let allowed = allowlist
            .iter()
            .any(|allowed| host_matches_allowlist(host, allowed));
        if allowed {
            return Ok(());
        }
        bail!("egress host not in allowlist: {host}");
    }
}

struct InspectedHttpUrl {
    parsed: url::Url,
    host: String,
    literal_ips: Option<Vec<IpAddr>>,
}

/// Hop-by-hop and request-smuggling header names that outbound callers must
/// not set from untrusted job/plugin input.
#[must_use]
pub fn is_forbidden_egress_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "host"
            | "content-length"
            | "transfer-encoding"
            | "connection"
            | "upgrade"
            | "te"
            | "trailer"
            | "keep-alive"
            | "cookie"
            | "set-cookie"
    )
}

fn deny_blocked_ip(ip: IpAddr, allow_loopback_and_private: bool) -> Result<()> {
    let blocked = if allow_loopback_and_private {
        is_customization_blocked_ip(ip)
    } else {
        is_blocked_ip(ip)
    };
    if blocked {
        bail!("plugin http url targets a blocked host: {ip}");
    }
    Ok(())
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
        .with_context(|| format!("plugin http url host did not resolve: {host}"))
}

pub(crate) fn host_matches_allowlist(host: &str, allowed: &str) -> bool {
    if allowed == "*" {
        return true;
    }
    host.eq_ignore_ascii_case(allowed)
        || host
            .len()
            .checked_sub(allowed.len() + 1)
            .is_some_and(|idx| {
                host.as_bytes().get(idx) == Some(&b'.')
                    && host[idx + 1..].eq_ignore_ascii_case(allowed)
            })
}

fn is_loopback_hostname(host: &str) -> bool {
    matches!(
        host.to_ascii_lowercase().as_str(),
        "localhost" | "localhost.localdomain"
    )
}

fn is_metadata_hostname(host: &str) -> bool {
    matches!(
        host.to_ascii_lowercase().as_str(),
        "metadata.google.internal" | "metadata" | "instance-data"
    )
}

fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_ipv4(v4),
        IpAddr::V6(v6) => is_blocked_ipv6(v6),
    }
}

/// IPs that stay blocked even when local customization (MCP, sidecars) is allowed.
fn is_customization_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast()
                || v4.is_documentation()
                || is_cgnat(v4)
                || is_benchmarking_ipv4(v4)
                || v4.octets()[0] == 0
        }
        IpAddr::V6(v6) => {
            if v6.is_unspecified()
                || v6.is_multicast()
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                || is_aws_imds_ipv6(v6)
            {
                return true;
            }
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_customization_blocked_ip(IpAddr::V4(mapped));
            }
            false
        }
    }
}

fn is_blocked_ipv4(v4: Ipv4Addr) -> bool {
    v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_unspecified()
        || v4.is_broadcast()
        || v4.is_multicast()
        || v4.is_documentation()
        || is_cgnat(v4)
        || is_benchmarking_ipv4(v4)
        || v4.octets()[0] == 0
}

/// RFC 6598 shared address space `100.64.0.0/10` (CGNAT).
fn is_cgnat(v4: Ipv4Addr) -> bool {
    v4.octets()[0] == 100 && (v4.octets()[1] & 0xc0) == 64
}

/// RFC 2544 benchmarking range `198.18.0.0/15`.
fn is_benchmarking_ipv4(v4: Ipv4Addr) -> bool {
    v4.octets()[0] == 198 && (v4.octets()[1] & 0xfe) == 18
}

/// AWS IMDS IPv6 (`fd00:ec2::254`). Unique-local is otherwise allowed so
/// operator IPv6 MCP sidecars still work.
fn is_aws_imds_ipv6(v6: Ipv6Addr) -> bool {
    v6 == Ipv6Addr::new(0xfd00, 0x0ec2, 0, 0, 0, 0, 0, 0x0254)
}

fn is_blocked_ipv6(v6: Ipv6Addr) -> bool {
    if v6.is_loopback()
        || v6.is_unspecified()
        || v6.is_unique_local()
        || v6.is_multicast()
        || (v6.segments()[0] & 0xffc0) == 0xfe80
    {
        return true;
    }
    if let Some(mapped) = v6.to_ipv4_mapped() {
        return is_blocked_ipv4(mapped);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn allowlist_matches_subdomains_not_suffix_spoof() {
        assert!(host_matches_allowlist("cdn.example.com", "example.com"));
        assert!(host_matches_allowlist("EXAMPLE.COM", "example.com"));
        assert!(!host_matches_allowlist("evil-example.com", "example.com"));
        assert!(!host_matches_allowlist(
            "example.com.evil.test",
            "example.com"
        ));
    }

    #[test]
    fn mapped_ipv6_loopback_is_blocked() {
        let mapped: IpAddr = "::ffff:127.0.0.1".parse().expect("mapped loopback");
        assert!(is_blocked_ip(mapped));
        let metadata: IpAddr = "::ffff:169.254.169.254".parse().expect("mapped metadata");
        assert!(is_blocked_ip(metadata));
    }

    #[test]
    fn cgnat_and_docs_ipv4_are_blocked() {
        assert!(is_blocked_ipv4(Ipv4Addr::new(100, 64, 0, 1)));
        assert!(is_blocked_ipv4(Ipv4Addr::new(192, 0, 2, 1)));
        assert!(is_blocked_ipv4(Ipv4Addr::new(198, 18, 0, 1)));
        assert!(is_blocked_ipv4(Ipv4Addr::new(224, 0, 0, 1)));
    }

    #[test]
    fn dns_to_loopback_is_blocked() {
        let policy = HttpPolicy::new(None);
        let err = policy
            .validate_url("https://evil.example/ssrf", |_| {
                Ok(vec![IpAddr::V4(Ipv4Addr::LOCALHOST)])
            })
            .expect_err("resolved loopback");
        assert!(err.to_string().contains("blocked"));
    }

    #[test]
    fn dns_failure_is_fail_closed() {
        let policy = HttpPolicy::new(None);
        let err = policy
            .validate_url("https://does-not-resolve.invalid/", |_| {
                anyhow::bail!("nxdomain")
            })
            .expect_err("lookup failure");
        assert!(err.to_string().contains("nxdomain"));
    }

    #[test]
    fn validate_url_returns_looked_up_ips_for_pinning() {
        let policy = HttpPolicy::new(None);
        let public = IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34));
        let validated = policy
            .validate_url("https://example.com/health", |_| Ok(vec![public]))
            .expect("public ip");
        assert_eq!(validated.host, "example.com");
        assert_eq!(validated.ips, vec![public]);
    }

    #[test]
    fn redirect_to_loopback_is_rejected() {
        let policy = HttpPolicy::new(None);
        let err = policy
            .validate_url("https://127.0.0.1/redirect-ssrf", |_| {
                Ok(vec![IpAddr::V4(Ipv4Addr::LOCALHOST)])
            })
            .expect_err("literal loopback redirect hop");
        assert!(err.to_string().contains("blocked"));
    }

    #[test]
    fn static_validation_skips_dns_and_blocks_literals() {
        let policy = HttpPolicy::new(None);
        policy
            .validate_static_url("https://api.example.invalid/v1")
            .expect("public hostname does not need DNS at load");
        let loopback = policy
            .validate_static_url("http://127.0.0.1/v1")
            .expect_err("loopback literal");
        assert!(loopback.to_string().contains("blocked"));
        let localhost = policy
            .validate_static_url("http://localhost/v1")
            .expect_err("localhost hostname");
        assert!(localhost.to_string().contains("blocked"));
        let ipv6 = policy
            .validate_static_url("http://[::1]/v1")
            .expect_err("ipv6 loopback");
        assert!(ipv6.to_string().contains("blocked"));
    }

    #[test]
    fn forbidden_egress_headers_include_host_and_te() {
        assert!(is_forbidden_egress_header("Host"));
        assert!(is_forbidden_egress_header("Transfer-Encoding"));
        assert!(is_forbidden_egress_header("content-length"));
        assert!(!is_forbidden_egress_header("Authorization"));
        assert!(!is_forbidden_egress_header("X-Caliche-Job"));
    }

    #[test]
    fn customization_policy_allows_loopback_and_rfc1918_not_metadata() {
        let policy = HttpPolicy::new(None).allowing_loopback_and_private();
        policy
            .validate_static_url("http://127.0.0.1:3000/mcp")
            .expect("loopback MCP");
        policy
            .validate_static_url("http://localhost:3000/mcp")
            .expect("localhost MCP");
        policy
            .validate_static_url("http://10.0.0.5:8080/mcp")
            .expect("rfc1918 MCP");
        let metadata = policy
            .validate_static_url("http://169.254.169.254/latest/meta-data/")
            .expect_err("imds");
        assert!(metadata.to_string().contains("blocked"));
        let meta_host = policy
            .validate_static_url("http://metadata.google.internal/")
            .expect_err("gcp metadata");
        assert!(meta_host.to_string().contains("blocked"));
        let aws_imds_v6 = policy
            .validate_static_url("http://[fd00:ec2::254]/latest/meta-data/")
            .expect_err("aws imds v6");
        assert!(aws_imds_v6.to_string().contains("blocked"));
    }
}
