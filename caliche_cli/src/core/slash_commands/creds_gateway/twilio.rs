use super::super::{SlashCommandExecutor, SlashCommandResult};
use crate::storage::SecureCredentialManager;
use anyhow::Result;
use caliche_contracts::credential_env;
use std::path::Path;

async fn load_twilio_credentials(
    dir: Option<&Path>,
) -> Result<SecureCredentialManager, SlashCommandResult> {
    let credential_manager = SecureCredentialManager::new_with_scope(dir)
        .await
        .map_err(|err| SlashCommandResult::Error(format!("Credential manager error: {err}")))?;
    if let Some(account_sid) = credential_manager.get_api_key("twilio_account_sid").await
        && !account_sid.trim().is_empty()
    {
        credential_env::set("TWILIO_ACCOUNT_SID", account_sid);
    }
    if let Some(auth_token) = credential_manager.get_api_key("twilio_auth_token").await
        && !auth_token.trim().is_empty()
    {
        credential_env::set("TWILIO_AUTH_TOKEN", auth_token);
    }
    if let Some(from_number) = credential_manager.get_api_key("twilio_from").await
        && !from_number.trim().is_empty()
    {
        credential_env::set("TWILIO_FROM", from_number.clone());
        credential_env::set("TWILIO_PHONE_NUMBER", from_number);
    }
    Ok(credential_manager)
}

fn twilio_action_and_options_start<'a>(scope: &str, args: &'a [String]) -> (&'a str, usize) {
    match scope {
        "webhook" => (args.get(3).map(String::as_str).unwrap_or("sync"), 4),
        "sync" => ("sync", 3),
        "status" => ("status", 3),
        _ => ("", 3),
    }
}

fn format_twilio_status(status: crate::gateway_runtime::TwilioWebhookStatus) -> String {
    let mut lines = vec![
        "Twilio Webhook Status".to_string(),
        String::new(),
        format!("Phone Number: {}", status.phone_number),
        format!("Incoming SID: {}", status.incoming_phone_number_sid),
        format!(
            "Configured SmsUrl: {}",
            status.sms_url.as_deref().unwrap_or("<unset>")
        ),
        format!(
            "Configured SmsMethod: {}",
            status.sms_method.as_deref().unwrap_or("<unset>")
        ),
    ];
    if let Some(expected) = &status.expected_sms_url {
        lines.push(format!("Expected SmsUrl: {}", expected));
    }
    if let Some(matches) = status.matches_expected {
        lines.push(format!("Matches Expected: {}", matches));
    }
    lines.join("\n")
}

async fn sync_twilio(dir: Option<&Path>, base_url: Option<&str>) -> Result<SlashCommandResult> {
    match crate::gateway_runtime::sync_twilio_webhook(dir, base_url).await {
        Ok(result) => Ok(SlashCommandResult::StatusMessage(format!(
            "Twilio webhook synced for {} (sid {}). SmsUrl: {}",
            result.phone_number, result.incoming_phone_number_sid, result.sms_url
        ))),
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to sync Twilio webhook: {err}"
        ))),
    }
}

async fn status_twilio(dir: Option<&Path>, base_url: Option<&str>) -> Result<SlashCommandResult> {
    match crate::gateway_runtime::twilio_webhook_status(dir, base_url).await {
        Ok(status) => Ok(SlashCommandResult::ShowPopup {
            title: "Gateway Twilio".to_string(),
            content: format_twilio_status(status),
        }),
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to read Twilio webhook status: {err}"
        ))),
    }
}

impl SlashCommandExecutor {
    pub(crate) async fn handle_gateway_twilio(args: &[String]) -> Result<SlashCommandResult> {
        let twilio_scope = args.get(2).map(String::as_str).unwrap_or("webhook");
        let (twilio_action, option_start_idx) = twilio_action_and_options_start(twilio_scope, args);
        let (twilio_dir, base_url_override) =
            Self::parse_gateway_twilio_options(args.get(option_start_idx..).unwrap_or(&[]));
        if let Err(error) = load_twilio_credentials(twilio_dir.as_deref()).await {
            return Ok(error);
        }

        match (twilio_scope, twilio_action) {
            ("webhook", "sync") | ("sync", "sync") => {
                sync_twilio(twilio_dir.as_deref(), base_url_override.as_deref()).await
            }
            ("webhook", "status") | ("status", "status") => {
                status_twilio(twilio_dir.as_deref(), base_url_override.as_deref()).await
            }
            _ => Ok(SlashCommandResult::Error(
                "Unknown /gateway twilio subcommand.\n\nUsage:\n  /gateway twilio webhook sync [--url <public_base_url>] [--dir <path>]\n  /gateway twilio webhook status [--url <public_base_url>] [--dir <path>]".to_string(),
            )),
        }
    }
}
