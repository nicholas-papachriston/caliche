//! Strict parsers. Unknown strings are errors, not [`crate::ModeOverlay::Default`].

use crate::effect::Effect;
use crate::error::PolicyError;
use crate::posture::{ModeOverlay, Preset};

/// Parse a permission-mode string into a session overlay.
///
/// `None`, empty, and `default` are the explicit default overlay. Any other
/// unknown token is an error. There is no silent fallback.
///
/// The spellings are `default`, `accept-edits`, `accept-all`, `deny-all`,
/// and `interactive`.
///
/// # Errors
///
/// Returns [`PolicyError::UnknownPermissionMode`] for an unrecognized token.
pub fn parse_mode_overlay(value: Option<&str>) -> Result<ModeOverlay, PolicyError> {
    let Some(raw) = value else {
        return Ok(ModeOverlay::Default);
    };
    let token = raw.trim().to_ascii_lowercase();
    if token.is_empty() || token == "default" {
        return Ok(ModeOverlay::Default);
    }
    match token.as_str() {
        "accept-edits" => Ok(ModeOverlay::AcceptEdits),
        "accept-all" => Ok(ModeOverlay::AcceptAll),
        "deny-all" => Ok(ModeOverlay::DenyAll),
        "interactive" => Ok(ModeOverlay::Interactive),
        _ => Err(PolicyError::UnknownPermissionMode(token)),
    }
}

/// Parse a preset name.
///
/// # Errors
///
/// Returns [`PolicyError::UnknownPreset`] when `value` is not
/// `locked`, `review`, `standard`, `trusted`, or `danger`.
pub fn parse_preset(value: &str) -> Result<Preset, PolicyError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "locked" => Ok(Preset::Locked),
        "review" => Ok(Preset::Review),
        "standard" => Ok(Preset::Standard),
        "trusted" => Ok(Preset::Trusted),
        "danger" => Ok(Preset::Danger),
        other => Err(PolicyError::UnknownPreset(other.to_owned())),
    }
}

/// Parse an effect token.
///
/// # Errors
///
/// Returns [`PolicyError::UnknownEffect`] for anything other than
/// `allow`, `ask`, or `deny`.
pub fn parse_effect(value: &str) -> Result<Effect, PolicyError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "allow" => Ok(Effect::Allow),
        "ask" => Ok(Effect::Ask),
        "deny" => Ok(Effect::Deny),
        other => Err(PolicyError::UnknownEffect(other.to_owned())),
    }
}
