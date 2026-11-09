//! Webhook signature verification — thin re-export of the root SDK helpers.
//!
//! Canonical implementation lives in `caliche::gateway::webhook_verify`.

pub use caliche::gateway::webhook_verify::{
    PublicUrlOptions, discord_signing_payload, inbound_public_url, inbound_public_url_with,
    resolve_webhook_secret, signatures_equal, slack_signature_v0, slack_timestamp_is_fresh,
    teams_verification_required, twilio_expected_signature, verify_teams_hmac_bearer,
    verify_teams_signature, webhook_verification_required,
};
