//! Floor path and host classifiers.
//!
//! These are fact checks. They do not return effects; [`crate::CompiledPolicy`]
//! turns a match into deny.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Class of blocked egress target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SsrfClass {
    /// Cloud metadata hostname or the instance metadata address.
    Metadata,
    /// Link-local range, including `169.254.0.0/16` other than the metadata address.
    LinkLocal,
    /// Loopback.
    Loopback,
    /// RFC1918 or IPv6 unique-local.
    Private,
    /// Carrier-grade NAT `100.64.0.0/10`.
    Cgnat,
}

const VENDOR_AUTH_SUFFIXES: &[&str] = &[
    "/.codex/auth.json",
    "/.claude.json",
    "/.claude/credentials.json",
    "/.claude/.credentials.json",
    "/.config/claude/credentials.json",
    "/.config/gh/hosts.yml",
    "/.config/gh/hosts.yaml",
];

/// Whether `path` is a vendor auth file the floor refuses to read.
///
/// Covers Codex `auth.json`, Claude credential files, and `gh` host tokens.
/// Comparison is case-insensitive and accepts `\` separators.
#[must_use]
pub fn is_vendor_auth_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/").to_ascii_lowercase();
    VENDOR_AUTH_SUFFIXES
        .iter()
        .any(|suffix| normalized.ends_with(suffix) || normalized == suffix.trim_start_matches('/'))
}

/// Classify a hostname or IP literal. DNS resolution is not performed here.
///
/// A name that does not parse as an IP and is not a known metadata hostname
/// returns `None`. This check does not resolve DNS. The engine HTTP policy
/// still owns the resolved-address check.
#[must_use]
pub fn classify_ssrf_host(host: &str) -> Option<SsrfClass> {
    let host = host
        .trim()
        .trim_matches(|ch| ch == '[' || ch == ']')
        .trim_end_matches('.');
    let lower = host.to_ascii_lowercase();
    match lower.as_str() {
        "metadata.google.internal" | "metadata" | "instance-data" => {
            return Some(SsrfClass::Metadata);
        }
        "localhost" => return Some(SsrfClass::Loopback),
        _ => {}
    }
    let ip: IpAddr = host.parse().ok()?;
    classify_ip(ip)
}

fn classify_ip(ip: IpAddr) -> Option<SsrfClass> {
    match ip {
        IpAddr::V4(v4) => classify_v4(v4),
        IpAddr::V6(v6) => classify_v6(v6),
    }
}

fn classify_v4(v4: Ipv4Addr) -> Option<SsrfClass> {
    if v4 == Ipv4Addr::new(169, 254, 169, 254) {
        return Some(SsrfClass::Metadata);
    }
    if v4.is_loopback() || v4.is_unspecified() {
        return Some(SsrfClass::Loopback);
    }
    if v4.is_link_local() {
        return Some(SsrfClass::LinkLocal);
    }
    if v4.is_private() {
        return Some(SsrfClass::Private);
    }
    if is_cgnat(v4) {
        return Some(SsrfClass::Cgnat);
    }
    None
}

/// The IPv4 address an IPv6 address carries, which a router or a
/// translator would deliver to: IPv4-mapped `::ffff:a.b.c.d`,
/// IPv4-compatible `::a.b.c.d` (not `::` or `::1`), NAT64
/// `64:ff9b::a.b.c.d`, and 6to4 `2002:aabb:ccdd::/48`.
#[must_use]
pub fn embedded_ipv4(v6: Ipv6Addr) -> Option<Ipv4Addr> {
    if let Some(v4) = v6.to_ipv4_mapped() {
        return Some(v4);
    }
    let octets = v6.octets();
    let segments = v6.segments();
    let tail = Ipv4Addr::new(octets[12], octets[13], octets[14], octets[15]);
    if segments[..6] == [0; 6] && !v6.is_loopback() && !v6.is_unspecified() {
        return Some(tail);
    }
    if segments[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        return Some(tail);
    }
    if segments[0] == 0x2002 {
        return Some(Ipv4Addr::new(octets[2], octets[3], octets[4], octets[5]));
    }
    None
}

fn classify_v6(v6: Ipv6Addr) -> Option<SsrfClass> {
    if let Some(v4) = embedded_ipv4(v6) {
        return classify_v4(v4);
    }
    if v6.is_loopback() || v6.is_unspecified() {
        return Some(SsrfClass::Loopback);
    }
    if v6.is_unicast_link_local() {
        return Some(SsrfClass::LinkLocal);
    }
    let first = v6.octets()[0];
    if first & 0xfe == 0xfc {
        return Some(SsrfClass::Private);
    }
    None
}

fn is_cgnat(v4: Ipv4Addr) -> bool {
    let octets = v4.octets();
    octets[0] == 100 && (64..=127).contains(&octets[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ipv4_target_inside_an_ipv6_address_is_classified() {
        let hosts = [
            "::7f00:1",
            "64:ff9b::a9fe:a9fe",
            "2002:a00:1::",
            "[::ffff:192.168.0.1]",
            "64:ff9b::808:808",
            "2002:808:808::1",
            "::1",
        ];

        assert_eq!(
            hosts.map(classify_ssrf_host),
            [
                Some(SsrfClass::Loopback),
                Some(SsrfClass::Metadata),
                Some(SsrfClass::Private),
                Some(SsrfClass::Private),
                None,
                None,
                Some(SsrfClass::Loopback),
            ]
        );
    }
}
