use super::auth::{AzureAuth, resolve_azure_auth_header};
use crate::llm::config::CommonProviderOptions;
use crate::llm::openai_compat::{AuthHeadersFuture, OpenAICompatConfig};
use std::sync::Arc;

pub const DEFAULT_API_VERSION: &str = "2024-10-21";

fn deployment_base_url(resource_name: &str, deployment_id: &str) -> String {
    format!("https://{resource_name}.openai.azure.com/openai/deployments/{deployment_id}")
}

fn chat_completions_url(resource_name: &str, deployment_id: &str, api_version: &str) -> String {
    format!(
        "{}/chat/completions?api-version={api_version}",
        deployment_base_url(resource_name, deployment_id)
    )
}

fn embeddings_url(resource_name: &str, deployment_id: &str, api_version: &str) -> String {
    format!(
        "{}/embeddings?api-version={api_version}",
        deployment_base_url(resource_name, deployment_id)
    )
}

fn uses_max_completion_tokens_for_model(model: &str) -> bool {
    model.starts_with("gpt-4o")
        || model.starts_with("gpt-4.1")
        || model.starts_with("gpt-4.5")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.starts_with("gpt-5")
        || model.starts_with("chatgpt-4o")
}

fn requires_default_temperature_for_model(model: &str) -> bool {
    model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.starts_with("gpt-5")
}

/// Configuration for [`super::AzureOpenAIProvider`].
#[derive(Debug, Clone)]
pub struct AzureOpenAIProviderConfig {
    /// Azure resource name (e.g., "my-resource").
    pub resource_name: String,
    /// Deployment ID/name.
    pub deployment_id: String,
    /// API version (defaults to "2024-10-21").
    pub api_version: String,
    /// Authentication method.
    pub auth: AzureAuth,
    /// Common provider options.
    pub common: CommonProviderOptions,
}

impl AzureOpenAIProviderConfig {
    /// Create a new configuration with API key authentication.
    pub fn new_api_key(
        resource_name: impl Into<String>,
        deployment_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            resource_name: resource_name.into(),
            deployment_id: deployment_id.into(),
            api_version: DEFAULT_API_VERSION.to_owned(),
            auth: AzureAuth::ApiKey(api_key.into()),
            common: CommonProviderOptions::default(),
        }
    }

    /// Create a builder for [`AzureOpenAIProviderConfig`].
    #[must_use]
    pub fn builder() -> AzureOpenAIProviderConfigBuilder {
        AzureOpenAIProviderConfigBuilder::default()
    }
}

crate::impl_provider_config!(AzureOpenAIProviderConfig);

/// Builder for [`AzureOpenAIProviderConfig`].
#[derive(Debug, Default)]
pub struct AzureOpenAIProviderConfigBuilder {
    resource_name: Option<String>,
    deployment_id: Option<String>,
    api_version: Option<String>,
    api_key: Option<String>,
    auth: Option<AzureAuth>,
    common: CommonProviderOptions,
}

impl AzureOpenAIProviderConfigBuilder {
    /// Set the resource name.
    #[must_use]
    pub fn resource_name(mut self, resource_name: impl Into<String>) -> Self {
        self.resource_name = Some(resource_name.into());
        self
    }

    /// Set the deployment ID.
    #[must_use]
    pub fn deployment_id(mut self, deployment_id: impl Into<String>) -> Self {
        self.deployment_id = Some(deployment_id.into());
        self
    }

    /// Set the API version.
    #[must_use]
    pub fn api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = Some(api_version.into());
        self
    }

    /// Set the API key (for API key authentication).
    #[must_use]
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set Azure AD authentication.
    #[cfg(feature = "azure-openai")]
    #[must_use]
    pub fn azure_ad(
        mut self,
        credential: Box<dyn azure_core::auth::TokenCredential + Send + Sync>,
    ) -> Self {
        self.auth = Some(AzureAuth::AzureAD(Arc::new(tokio::sync::Mutex::new(
            credential,
        ))));
        self
    }

    /// Set Managed Identity authentication.
    ///
    /// Returns an error if the `DefaultAzureCredential` cannot be created
    /// (e.g. missing Azure environment configuration).
    #[cfg(feature = "azure-openai")]
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn managed_identity(mut self) -> anyhow::Result<Self> {
        let credential = azure_identity::DefaultAzureCredential::create(
            azure_identity::TokenCredentialOptions::default(),
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to create DefaultAzureCredential — ensure Azure credentials \
                 are configured (AZURE_CLIENT_ID, AZURE_TENANT_ID, etc.): {e}"
            )
        })?;
        self.auth = Some(AzureAuth::ManagedIdentity(Arc::new(
            tokio::sync::Mutex::new(credential),
        )));
        Ok(self)
    }

    /// Build the configuration.
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn build(self) -> anyhow::Result<AzureOpenAIProviderConfig> {
        let resource_name = self
            .resource_name
            .or_else(|| {
                caliche_contracts::credential_env::var("AZURE_OPENAI_RESOURCE_NAME")
                    .or_else(|_| caliche_contracts::credential_env::var("AZURE_RESOURCE_NAME"))
                    .ok()
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "resource_name is required for AzureOpenAIProviderConfig or AZURE_OPENAI_RESOURCE_NAME must be set"
                )
            })?;

        let deployment_id = self
            .deployment_id
            .or_else(|| {
                caliche_contracts::credential_env::var("AZURE_OPENAI_DEPLOYMENT_ID")
                    .or_else(|_| caliche_contracts::credential_env::var("AZURE_DEPLOYMENT_ID"))
                    .ok()
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "deployment_id is required for AzureOpenAIProviderConfig or AZURE_OPENAI_DEPLOYMENT_ID must be set"
                )
            })?;

        let api_version = self
            .api_version
            .unwrap_or_else(|| DEFAULT_API_VERSION.to_owned());

        let auth = if let Some(auth) = self.auth {
            auth
        } else if let Some(api_key) = self.api_key {
            AzureAuth::ApiKey(api_key)
        } else if let Ok(api_key) = caliche_contracts::credential_env::var("AZURE_OPENAI_API_KEY")
            .or_else(|_| caliche_contracts::credential_env::var("AZURE_OPENAI_KEY"))
        {
            AzureAuth::ApiKey(api_key)
        } else {
            #[cfg(feature = "azure-openai")]
            {
                AzureAuth::ManagedIdentity(Arc::new(tokio::sync::Mutex::new(
                    azure_identity::DefaultAzureCredential::create(
                        azure_identity::TokenCredentialOptions::default(),
                    )
                    .map_err(|e| anyhow::anyhow!("Failed to create DefaultAzureCredential: {e}"))?,
                )))
            }
            #[cfg(not(feature = "azure-openai"))]
            {
                return Err(anyhow::anyhow!(
                    "api_key is required for AzureOpenAIProviderConfig or AZURE_OPENAI_API_KEY must be set"
                ));
            }
        };

        Ok(AzureOpenAIProviderConfig {
            resource_name,
            deployment_id,
            api_version,
            auth,
            common: self.common,
        })
    }
}

crate::impl_common_builder_methods!(AzureOpenAIProviderConfigBuilder);

/// OpenAI-compat configuration for Azure `OpenAI` deployments.
#[derive(Debug, Clone)]
pub struct AzureOpenAICompatConfig {
    deployment_id: String,
    auth: AzureAuth,
    deployment_base_url: String,
    chat_completions_url: String,
    embeddings_url: String,
}

impl AzureOpenAICompatConfig {
    pub(crate) fn from_provider_config_with_common(
        config: AzureOpenAIProviderConfig,
    ) -> (Self, CommonProviderOptions) {
        let compat = Self::from_provider_config(&config);
        (compat, config.common)
    }

    pub fn from_provider_config(config: &AzureOpenAIProviderConfig) -> Self {
        let deployment_base_url = deployment_base_url(&config.resource_name, &config.deployment_id);
        let chat_completions_url = chat_completions_url(
            &config.resource_name,
            &config.deployment_id,
            &config.api_version,
        );
        let embeddings_url = embeddings_url(
            &config.resource_name,
            &config.deployment_id,
            &config.api_version,
        );

        Self {
            deployment_id: config.deployment_id.clone(),
            auth: config.auth.clone(),
            deployment_base_url,
            chat_completions_url,
            embeddings_url,
        }
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }
}

impl OpenAICompatConfig for AzureOpenAICompatConfig {
    fn provider_name(&self) -> &'static str {
        "azure-openai"
    }

    fn base_url(&self) -> &str {
        &self.deployment_base_url
    }

    fn chat_completions_url(&self) -> Option<&str> {
        Some(&self.chat_completions_url)
    }

    fn embeddings_url(&self) -> Option<&str> {
        Some(&self.embeddings_url)
    }

    fn auth_headers(&self) -> Vec<(&'static str, String)> {
        if let AzureAuth::ApiKey(key) = &self.auth {
            vec![("Authorization", format!("Bearer {key}"))]
        } else {
            vec![]
        }
    }

    fn resolve_auth_headers(&self) -> AuthHeadersFuture<'_> {
        let auth = self.auth.clone();
        Box::pin(async move {
            if let AzureAuth::ApiKey(key) = &auth {
                return vec![("Authorization", format!("Bearer {key}"))];
            }
            match resolve_azure_auth_header(&auth).await {
                Ok(header) => vec![("Authorization", header)],
                Err(error) => {
                    tracing::warn!("Azure auth header resolution failed: {error}");
                    vec![]
                }
            }
        })
    }

    fn model(&self) -> &str {
        &self.deployment_id
    }

    fn uses_max_completion_tokens(&self) -> bool {
        uses_max_completion_tokens_for_model(&self.deployment_id)
    }

    fn requires_default_temperature(&self) -> bool {
        requires_default_temperature_for_model(&self.deployment_id)
    }

    fn include_usage_in_stream(&self) -> bool {
        true
    }

    fn supports_embeddings(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        true
    }

    fn supports_audio_input(&self) -> bool {
        let deployment = self.deployment_id.to_lowercase();
        deployment.contains("gpt-4o")
            || deployment.contains("gpt-4.1")
            || deployment.contains("audio")
            || deployment.contains("realtime")
    }

    fn supported_media_types(&self) -> Vec<&str> {
        vec![
            "image/png",
            "image/jpeg",
            "image/webp",
            "audio/wav",
            "audio/mpeg",
        ]
    }
}
