//! Decide whether a Landlock ruleset should deny TCP.
//!
//! The decision is pure so tests run on every OS. The Linux backend applies
//! it only when the `landlock-net` feature is on and the kernel ABI can
//! handle bind and connect.

use crate::config::NetworkPolicy;

/// What to do with TCP bind and connect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcpPlan {
    /// Leave network rights unchanged.
    Unrestricted,
    /// Handle bind and connect and add no allow rule.
    DenyAll,
    /// The policy asks for a network shape Landlock cannot express.
    Unsupported,
}

/// Choose the TCP plan.
///
/// `feature_on` is the `landlock-net` Cargo feature. `abi_ok` is a successful
/// kernel probe for TCP rights. `NetworkPolicy::None` is the read-only and
/// workspace-write posture. Landlock's network isolation is TCP bind and
/// connect only.
///
/// `NetworkPolicy::HostOnly` and allowlists are unsupported even when
/// `feature_on` is false. The Linux backend turns that into a hard error, so
/// a host-only policy does not run with an open network.
#[must_use]
pub fn tcp_plan(network: &NetworkPolicy, feature_on: bool, abi_ok: bool) -> TcpPlan {
    match network {
        NetworkPolicy::Full => TcpPlan::Unrestricted,
        NetworkPolicy::None if feature_on && abi_ok => TcpPlan::DenyAll,
        NetworkPolicy::None => TcpPlan::Unrestricted,
        NetworkPolicy::HostOnly | NetworkPolicy::Restricted { .. } => TcpPlan::Unsupported,
    }
}

/// True when this plan actually confines TCP.
#[must_use]
pub const fn tcp_isolation_claimed(plan: TcpPlan) -> bool {
    matches!(plan, TcpPlan::DenyAll)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_write_denies_tcp_only_when_the_feature_and_abi_agree() {
        let network = NetworkPolicy::None;
        assert_eq!(tcp_plan(&network, true, true), TcpPlan::DenyAll);
        assert_eq!(tcp_plan(&network, true, false), TcpPlan::Unrestricted);
        assert_eq!(tcp_plan(&network, false, true), TcpPlan::Unrestricted);
        assert!(tcp_isolation_claimed(TcpPlan::DenyAll));
        assert!(!tcp_isolation_claimed(TcpPlan::Unrestricted));
    }

    #[test]
    fn full_access_does_not_deny_tcp() {
        assert_eq!(
            tcp_plan(&NetworkPolicy::Full, true, true),
            TcpPlan::Unrestricted
        );
    }

    #[test]
    fn allowlists_are_unsupported() {
        assert_eq!(
            tcp_plan(&NetworkPolicy::HostOnly, true, true),
            TcpPlan::Unsupported
        );
        // HostOnly stays refused when the feature is off, on every ABI.
        assert_eq!(
            tcp_plan(&NetworkPolicy::HostOnly, false, false),
            TcpPlan::Unsupported
        );
        assert_eq!(
            tcp_plan(&NetworkPolicy::HostOnly, false, true),
            TcpPlan::Unsupported
        );
        assert_eq!(
            tcp_plan(
                &NetworkPolicy::Restricted {
                    allowed_hosts: vec!["example.com".to_owned()],
                },
                true,
                true
            ),
            TcpPlan::Unsupported
        );
    }

    #[test]
    fn this_process_reports_network_only_on_a_linux_landlock_net_build() {
        if cfg!(all(feature = "landlock-net", target_os = "linux")) {
            return;
        }
        assert!(!crate::landlock_tcp_enforceable());
    }
}
