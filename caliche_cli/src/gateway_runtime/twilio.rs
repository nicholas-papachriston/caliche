use super::api::reqwest_client;
use super::forward::{status_gateway_forwarding, wait_for_forwarding_public_url};
use crate::storage::credentials::SecureCredentialManager;
use anyhow::{Context, Result, anyhow};
use std::path::Path;

pub async fn sync_twilio_webhook(
    cwd: Option<&Path>,
    base_url_override: Option<&str>,
) -> Result<super::types::TwilioWebhookSyncResult> {
    let credential_manager = SecureCredentialManager::new_with_scope(cwd).await.ok();
    let account_sid = gateway_credential_or_env(
        credential_manager.as_ref(),
        "twilio_account_sid",
        "TWILIO_ACCOUNT_SID",
    )
    .await
    .ok_or_else(|| anyhow!("TWILIO_ACCOUNT_SID is required"))?;
    let auth_token = gateway_credential_or_env(
        credential_manager.as_ref(),
        "twilio_auth_token",
        "TWILIO_AUTH_TOKEN",
    )
    .await
    .ok_or_else(|| anyhow!("TWILIO_AUTH_TOKEN is required"))?;
    let phone_number =
        gateway_credential_or_env(credential_manager.as_ref(), "twilio_from", "TWILIO_FROM")
            .await
            .ok_or_else(|| anyhow!("TWILIO_FROM (or TWILIO_PHONE_NUMBER) is required"))?;

    let base_url = match base_url_override {
        Some(v) if !v.trim().is_empty() => v.trim().to_string(),
        _ => wait_for_forwarding_public_url(cwd).await?,
    };
    let sms_url = format!(
        "{}/api/gateway/ingress/twilio",
        base_url.trim_end_matches('/')
    );

    let incoming_sid =
        twilio_lookup_incoming_phone_number_sid(&account_sid, &auth_token, &phone_number).await?;
    twilio_update_sms_webhook(&account_sid, &auth_token, &incoming_sid, &sms_url).await?;
    let verify =
        twilio_fetch_incoming_phone_number_details(&account_sid, &auth_token, &incoming_sid)
            .await?;
    let configured_url = verify
        .get("sms_url")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .unwrap_or_default()
        .to_string();
    if configured_url.trim_end_matches('/') != sms_url.trim_end_matches('/') {
        return Err(anyhow!(
            "Twilio webhook update did not persist expected SmsUrl. expected='{}' actual='{}'",
            sms_url,
            configured_url
        ));
    }

    Ok(super::types::TwilioWebhookSyncResult {
        phone_number,
        incoming_phone_number_sid: incoming_sid,
        sms_url,
    })
}

pub async fn twilio_webhook_status(
    cwd: Option<&Path>,
    base_url_override: Option<&str>,
) -> Result<super::types::TwilioWebhookStatus> {
    let credential_manager = SecureCredentialManager::new_with_scope(cwd).await.ok();
    let account_sid = gateway_credential_or_env(
        credential_manager.as_ref(),
        "twilio_account_sid",
        "TWILIO_ACCOUNT_SID",
    )
    .await
    .ok_or_else(|| anyhow!("TWILIO_ACCOUNT_SID is required"))?;
    let auth_token = gateway_credential_or_env(
        credential_manager.as_ref(),
        "twilio_auth_token",
        "TWILIO_AUTH_TOKEN",
    )
    .await
    .ok_or_else(|| anyhow!("TWILIO_AUTH_TOKEN is required"))?;
    let phone_number =
        gateway_credential_or_env(credential_manager.as_ref(), "twilio_from", "TWILIO_FROM")
            .await
            .ok_or_else(|| anyhow!("TWILIO_FROM (or TWILIO_PHONE_NUMBER) is required"))?;
    let incoming_sid =
        twilio_lookup_incoming_phone_number_sid(&account_sid, &auth_token, &phone_number).await?;
    let details =
        twilio_fetch_incoming_phone_number_details(&account_sid, &auth_token, &incoming_sid)
            .await?;

    let expected_sms_url = match base_url_override {
        Some(v) if !v.trim().is_empty() => Some(format!(
            "{}/api/gateway/ingress/twilio",
            v.trim().trim_end_matches('/')
        )),
        _ => status_gateway_forwarding(cwd)
            .await?
            .and_then(|forward| forward.public_url)
            .map(|url| format!("{}/api/gateway/ingress/twilio", url.trim_end_matches('/'))),
    };
    let configured_sms_url = details
        .get("sms_url")
        .and_then(|v| v.as_str())
        .map(std::string::ToString::to_string);
    let sms_method = details
        .get("sms_method")
        .and_then(|v| v.as_str())
        .map(std::string::ToString::to_string);
    let matches_expected = match (&configured_sms_url, &expected_sms_url) {
        (Some(configured), Some(expected)) => Some(configured == expected),
        _ => None,
    };

    Ok(super::types::TwilioWebhookStatus {
        phone_number,
        incoming_phone_number_sid: incoming_sid,
        sms_url: configured_sms_url,
        sms_method,
        expected_sms_url,
        matches_expected,
    })
}

async fn gateway_credential_or_env(
    credential_manager: Option<&SecureCredentialManager>,
    credential_name: &str,
    env_var: &str,
) -> Option<String> {
    if let Some(manager) = credential_manager
        && let Some(value) = manager.get_api_key(credential_name).await
        && !value.trim().is_empty()
    {
        return Some(value);
    }

    if let Ok(value) = caliche_contracts::credential_env::var(env_var)
        && !value.trim().is_empty()
    {
        return Some(value);
    }

    if credential_name == "twilio_from" {
        return caliche_contracts::credential_env::var("TWILIO_PHONE_NUMBER")
            .ok()
            .filter(|value| !value.trim().is_empty());
    }
    None
}

async fn twilio_lookup_incoming_phone_number_sid(
    account_sid: &str,
    auth_token: &str,
    phone_number: &str,
) -> Result<String> {
    let endpoint = format!(
        "https://api.twilio.com/2010-04-01/Accounts/{}/IncomingPhoneNumbers.json",
        account_sid
    );
    let client = reqwest_client()?;
    let response = client
        .get(&endpoint)
        .basic_auth(account_sid, Some(auth_token))
        .query(&[("PhoneNumber", phone_number)])
        .send()
        .await
        .context("failed to query Twilio incoming phone numbers")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "Twilio lookup failed: HTTP {} {}",
            status,
            body.trim()
        ));
    }

    let payload: serde_json::Value = response
        .json()
        .await
        .context("failed to parse Twilio lookup response")?;
    let Some(numbers) = payload
        .get("incoming_phone_numbers")
        .and_then(|v| v.as_array())
    else {
        return Err(anyhow!("Twilio response missing incoming_phone_numbers"));
    };
    let Some(first) = numbers.first() else {
        return Err(anyhow!(
            "no Twilio IncomingPhoneNumber found for '{}'",
            phone_number
        ));
    };
    first
        .get("sid")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string())
        .ok_or_else(|| anyhow!("Twilio IncomingPhoneNumber missing sid"))
}

async fn twilio_update_sms_webhook(
    account_sid: &str,
    auth_token: &str,
    incoming_sid: &str,
    sms_url: &str,
) -> Result<()> {
    let endpoint = format!(
        "https://api.twilio.com/2010-04-01/Accounts/{}/IncomingPhoneNumbers/{}.json",
        account_sid, incoming_sid
    );
    let client = reqwest_client()?;
    let form = [("SmsUrl", sms_url), ("SmsMethod", "POST")];
    let response = client
        .post(&endpoint)
        .basic_auth(account_sid, Some(auth_token))
        .form(&form)
        .send()
        .await
        .context("failed to update Twilio SMS webhook")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "Twilio webhook update failed: HTTP {} {}",
            status,
            body.trim()
        ));
    }
    Ok(())
}

async fn twilio_fetch_incoming_phone_number_details(
    account_sid: &str,
    auth_token: &str,
    incoming_sid: &str,
) -> Result<serde_json::Value> {
    let endpoint = format!(
        "https://api.twilio.com/2010-04-01/Accounts/{}/IncomingPhoneNumbers/{}.json",
        account_sid, incoming_sid
    );
    let client = reqwest_client()?;
    let response = client
        .get(&endpoint)
        .basic_auth(account_sid, Some(auth_token))
        .send()
        .await
        .context("failed to fetch Twilio incoming phone number details")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "Twilio details lookup failed: HTTP {} {}",
            status,
            body.trim()
        ));
    }
    response
        .json()
        .await
        .context("failed to parse Twilio details response")
}
