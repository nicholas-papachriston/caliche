//! Shadow observation beside the live hook path.
//!
//! `CALICHE_POLICY_SHADOW=1` (also `true`, `yes`, `on`) evaluates the shadow
//! snapshot and logs diffs. The `Result` from the hook path is returned unchanged.
//! When the variable is unset or off, the policy decider is not called.

use beef::lean::Cow;
use caliche_policy::{Effect, SHADOW_ENV, observe_shadow, shadow_enabled_value};
use std::sync::OnceLock;

fn shadow_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(SHADOW_ENV).is_ok_and(|raw| shadow_enabled_value(&raw)))
}

/// Log a shadow comparison for a finished hook decision.
///
/// `Ok` maps to [`Effect::Allow`] and `Err` maps to [`Effect::Deny`]. The
/// caller still returns `live`.
pub(crate) fn observe_live_outcome(
    tool: &str,
    args: &serde_json::Value,
    live: &Result<(), Cow<'static, str>>,
) {
    if !shadow_enabled() {
        return;
    }
    let effect = if live.is_ok() {
        Effect::Allow
    } else {
        Effect::Deny
    };
    observe_shadow(tool, args, effect);
}
