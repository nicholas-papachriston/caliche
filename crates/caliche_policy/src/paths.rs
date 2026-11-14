//! Floor path and host classifiers.
//!
//! These are fact checks. They do not return effects; [`crate::CompiledPolicy`]
//! turns a match into deny.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};

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
/// `64:ff9b::a.b.c.d`, 6to4 `2002:aabb:ccdd::/48`, and Teredo
/// `2001:0::/32`, whose client address is the last 32 bits inverted.
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
    if segments[..2] == [0x2001, 0] {
        return Some(Ipv4Addr::from(!u32::from(tail)));
    }
    None
}

/// Whether `v6` is in the local-use NAT64 prefix `64:ff9b:1::/48`, where a
/// network's own translator may carry an IPv4 address at any of several
/// offsets, so the target cannot be read off the address.
fn is_local_use_nat64(v6: Ipv6Addr) -> bool {
    v6.segments()[..3] == [0x64, 0xff9b, 1]
}

fn classify_v6(v6: Ipv6Addr) -> Option<SsrfClass> {
    if let Some(v4) = embedded_ipv4(v6) {
        return classify_v4(v4);
    }
    if v6.is_loopback() || v6.is_unspecified() {
        return Some(SsrfClass::Loopback);
    }
    if is_local_use_nat64(v6) {
        return Some(SsrfClass::Private);
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

/// The workspace root an SDK caller names for a tool call: it must be an
/// absolute path to an existing directory, and is returned canonical, so a
/// `..` through a link cannot widen it.
///
/// # Errors
///
/// Returns the reason when `path` is relative, missing or not a directory.
pub fn caller_workspace_root(path: &str) -> Result<PathBuf, String> {
    let given = Path::new(path);
    if !given.is_absolute() {
        return Err(format!("workspace must be an absolute path: {path}"));
    }
    match given.canonicalize() {
        Ok(root) if root.is_dir() => Ok(root),
        Ok(_) => Err(format!("workspace is not a directory: {path}")),
        Err(err) => Err(format!("workspace cannot be resolved: {path}: {err}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_caller_workspace_root_is_an_existing_absolute_directory() {
        let base = std::env::temp_dir().join(format!("policy-ws-root-{}", std::process::id()));
        std::fs::create_dir_all(&base).expect("dir");
        let root = base.canonicalize().expect("canonical");
        std::fs::create_dir(root.join("real")).expect("real");
        std::fs::write(root.join("file"), "x").expect("file");
        #[cfg(unix)]
        std::os::unix::fs::symlink(root.join("real"), root.join("lnk")).expect("link");
        let through_link = format!("{}/lnk/..", root.display());

        assert_eq!(
            [
                caller_workspace_root("relative/ws"),
                caller_workspace_root(&format!("{}/file", root.display())),
                caller_workspace_root(&format!("{}/real", root.display())),
            ],
            [
                Err("workspace must be an absolute path: relative/ws".to_string()),
                Err(format!(
                    "workspace is not a directory: {}/file",
                    root.display()
                )),
                Ok(root.join("real")),
            ]
        );
        #[cfg(unix)]
        let linked = caller_workspace_root(&through_link);
        std::fs::remove_dir_all(&root).expect("cleanup");
        #[cfg(unix)]
        assert_eq!(linked, Ok(root));
    }

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
            "64:ff9b:1::7f00:1",
            "64:ff9b:1:808:8:800::",
            "2001:0:4136:e378:8000:63bf:80ff:fffe",
            "2001:0:4136:e378:8000:63bf:f7f7:f7f7",
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
                Some(SsrfClass::Private),
                Some(SsrfClass::Private),
                Some(SsrfClass::Loopback),
                None,
            ]
        );
    }
}
