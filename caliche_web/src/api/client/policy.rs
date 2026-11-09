//! Permissions page HTTP calls.

use anyhow::{Context, Result};

use super::ApiClient;
use super::http_error_text;
use crate::api::types::{EffectivePolicy, PolicyDocument};

impl ApiClient {
    /// GET `/api/policy/effective`.
    pub async fn get_policy(&self) -> Result<EffectivePolicy> {
        let response = self
            .client
            .get(format!("{}/api/policy/effective", self.base_url))
            .send()
            .await
            .context("failed to load policy")?;
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("{}", http_error_text(status, &body));
        }
        response.json().await.context("failed to decode policy")
    }

    /// PUT `/api/policy`.
    pub async fn put_policy(&self, document: &PolicyDocument) -> Result<EffectivePolicy> {
        let response = self
            .client
            .put(format!("{}/api/policy", self.base_url))
            .json(document)
            .send()
            .await
            .context("failed to save policy")?;
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("{}", http_error_text(status, &body));
        }
        response
            .json()
            .await
            .context("failed to decode saved policy")
    }
}
