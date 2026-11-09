use crate::llm::provider::Provider;
use crate::messages::{CompletionRequest, CompletionResponse, StreamEvent};
use anyhow::Result;
use async_stream::stream;
use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;
use std::sync::Arc;

#[async_trait]
pub trait ProviderMiddleware: Send + Sync + std::fmt::Debug {
    async fn on_request(&self, _request: &mut CompletionRequest) -> Result<()> {
        Ok(())
    }

    async fn on_response(
        &self,
        _request: &CompletionRequest,
        _response: &mut CompletionResponse,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_stream_event(
        &self,
        _request: &CompletionRequest,
        _event: &mut StreamEvent,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_error(&self, _request: &CompletionRequest, _error: &anyhow::Error) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str;
}

pub struct MiddlewareProvider {
    inner: Arc<dyn Provider>,
    middleware: Vec<Arc<dyn ProviderMiddleware>>,
}

impl std::fmt::Debug for MiddlewareProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiddlewareProvider")
            .field("inner", &self.inner.name())
            .field("middleware_count", &self.middleware.len())
            .finish()
    }
}

impl MiddlewareProvider {
    pub fn new(inner: Arc<dyn Provider>, middleware: Vec<Arc<dyn ProviderMiddleware>>) -> Self {
        Self { inner, middleware }
    }
}

#[async_trait]
impl Provider for MiddlewareProvider {
    async fn complete(&self, mut request: CompletionRequest) -> Result<CompletionResponse> {
        for layer in &self.middleware {
            layer.on_request(&mut request).await?;
        }

        let request_for_response = request.clone();
        let mut response = match self.inner.complete(request).await {
            Ok(response) => response,
            Err(error) => {
                for layer in &self.middleware {
                    layer.on_error(&request_for_response, &error).await?;
                }
                return Err(error);
            }
        };

        for layer in &self.middleware {
            layer
                .on_response(&request_for_response, &mut response)
                .await?;
        }

        Ok(response)
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send + '_>> {
        let inner = Arc::clone(&self.inner);
        let middleware = self.middleware.clone();

        Box::pin(stream! {
            let mut request_for_middleware = request.clone();
            for layer in &middleware {
                if let Err(err) = layer.on_request(&mut request_for_middleware).await {
                    yield Err(err);
                    return;
                }
            }

            let mut inner_stream = inner.stream(request_for_middleware.clone());
            while let Some(item) = futures_util::StreamExt::next(&mut inner_stream).await {
                match item {
                    Ok(mut event) => {
                        let mut hook_error = None;
                        for layer in &middleware {
                            if let Err(err) = layer
                                .on_stream_event(&request_for_middleware, &mut event)
                                .await
                            {
                                hook_error = Some(err);
                                break;
                            }
                        }

                        if let Some(err) = hook_error {
                            yield Err(err);
                            return;
                        }

                        yield Ok(event);
                    }
                    Err(err) => {
                        for layer in &middleware {
                            if let Err(hook_error) =
                                layer.on_error(&request_for_middleware, &err).await
                            {
                                yield Err(hook_error);
                                return;
                            }
                        }
                        yield Err(err);
                        return;
                    }
                }
            }
        })
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    crate::delegate_provider_capabilities!(inner);
}
