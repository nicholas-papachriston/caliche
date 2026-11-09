use std::sync::Arc;

/// Azure authentication methods.
#[derive(Debug, Clone)]
pub enum AzureAuth {
    /// API Key authentication.
    ApiKey(String),
    /// Azure AD token credential (for service principal or user authentication).
    #[cfg(feature = "azure-openai")]
    AzureAD(Arc<tokio::sync::Mutex<Box<dyn azure_core::auth::TokenCredential + Send + Sync>>>),
    /// Managed Identity (automatic credential chain).
    #[cfg(feature = "azure-openai")]
    ManagedIdentity(Arc<tokio::sync::Mutex<azure_identity::DefaultAzureCredential>>),
    /// Placeholder for when azure-openai feature is disabled.
    #[cfg(not(feature = "azure-openai"))]
    _Placeholder,
}

/// Resolve the `Authorization` header value for an Azure `OpenAI` request.
pub async fn resolve_azure_auth_header(auth: &AzureAuth) -> anyhow::Result<String> {
    match auth {
        AzureAuth::ApiKey(key) => Ok(format!("Bearer {key}")),
        #[cfg(feature = "azure-openai")]
        AzureAuth::AzureAD(credential) => {
            #[allow(unused_imports)]
            use azure_core::auth::TokenCredential;
            let cred = credential.lock().await;
            let token = cred
                .get_token(&["https://cognitiveservices.azure.com/.default"])
                .await
                .map_err(|e| anyhow::anyhow!("Failed to get Azure AD token: {e}"))?;
            let secret = token.token.secret().to_owned();
            drop(cred);
            Ok(format!("Bearer {secret}"))
        }
        #[cfg(feature = "azure-openai")]
        AzureAuth::ManagedIdentity(credential) => {
            #[allow(unused_imports)]
            use azure_core::auth::TokenCredential;
            let cred = credential.lock().await;
            let token = cred
                .get_token(&["https://cognitiveservices.azure.com/.default"])
                .await
                .map_err(|e| anyhow::anyhow!("Failed to get Managed Identity token: {e}"))?;
            let secret = token.token.secret().to_owned();
            drop(cred);
            Ok(format!("Bearer {secret}"))
        }
        #[cfg(not(feature = "azure-openai"))]
        AzureAuth::_Placeholder => Err(anyhow::anyhow!(
            "Azure authentication requires azure-openai feature"
        )),
    }
}
