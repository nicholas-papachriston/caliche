use crate::llm::{Provider, ProviderType};
use crate::traits::credentials::CredentialProvider;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

pub type ProviderFactoryFuture =
    Pin<Box<dyn Future<Output = anyhow::Result<Arc<dyn Provider>>> + Send>>;

pub type ProviderFactoryFn = Arc<
    dyn Fn(
            ProviderType,
            String,
            Arc<dyn CredentialProvider>,
            Option<String>,
            Option<Duration>,
            Option<usize>,
        ) -> ProviderFactoryFuture
        + Send
        + Sync,
>;

#[must_use]
pub fn default_provider_factory() -> ProviderFactoryFn {
    Arc::new(
        move |provider, model, credentials, base_url, timeout, max_retries| {
            Box::pin(async move {
                crate::agent::service::ProviderFactory::create(
                    provider,
                    model,
                    credentials,
                    base_url.as_deref(),
                    timeout,
                    max_retries,
                    #[cfg(feature = "reqwest")]
                    None,
                )
                .await
            })
        },
    )
}
