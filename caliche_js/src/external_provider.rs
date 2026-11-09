//! JavaScript bindings for `ExternalProvider`
//!
//! This module provides bridges between JavaScript functions and Rust's `ExternalProvider`,
//! enabling `LangChain` and other JavaScript LLM libraries to be used with Caliche.

use async_trait::async_trait;
use caliche::llm::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use caliche::llm::external::{
    ExternalCompletionCallback, ExternalEmbeddingCallback, ExternalEmbeddingProvider,
    ExternalProvider,
};
use caliche::llm::provider::Provider;
use caliche::messages::{CompletionRequest, CompletionResponse, StreamEvent};
use futures_util::Stream;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use crate::call_js_sync_bridge;

/// JavaScript wrapper for `ExternalCompletionCallback`
///
/// This struct bridges JavaScript async functions to Rust's `ExternalCompletionCallback` trait.
/// Uses `ThreadsafeFunction`<Value> with sync bridge to avoid thread safety issues.
pub struct JsExternalCompletionCallback {
    complete_fn: Arc<ThreadsafeFunction<Value>>,
    stream_fn: Arc<ThreadsafeFunction<Value>>,
}

impl JsExternalCompletionCallback {
    pub fn new(
        complete_fn: ThreadsafeFunction<Value>,
        stream_fn: ThreadsafeFunction<Value>,
    ) -> Self {
        Self {
            complete_fn: Arc::new(complete_fn),
            stream_fn: Arc::new(stream_fn),
        }
    }
}

#[async_trait]
impl ExternalCompletionCallback for JsExternalCompletionCallback {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        // Convert CompletionRequest to JSON Value
        let request_value = serde_json::to_value(&request)
            .map_err(|e| anyhow::anyhow!("Failed to serialize request: {e}"))?;

        // Use sync bridge to call JavaScript function
        // This avoids thread safety issues by using blocking calls with oneshot channels
        let json_string = call_js_sync_bridge(&self.complete_fn, request_value)
            .await
            .map_err(|e| anyhow::anyhow!("JavaScript callback error: {e}"))?;

        // Parse JSON string as CompletionResponse
        let response: CompletionResponse = serde_json::from_str(&json_string)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize response: {e}"))?;

        Ok(response)
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        // Clone Arc to share ThreadsafeFunction
        let stream_fn = Arc::clone(&self.stream_fn);

        // Convert request to JSON Value
        let request_value = match serde_json::to_value(&request) {
            Ok(v) => v,
            Err(e) => {
                return Box::pin(futures_util::stream::once(async move {
                    Err(anyhow::anyhow!("Failed to serialize request: {e}"))
                }));
            }
        };

        // Create a channel to forward stream events
        let (tx, rx) = mpsc::unbounded_channel();

        // Clone for the async task
        let tx_clone = tx;

        // Spawn async task to handle JavaScript callback using sync bridge
        tokio::spawn(async move {
            let (tx_result, rx_result) = oneshot::channel();

            // Call JavaScript function synchronously in blocking mode
            let _ = stream_fn.call_with_return_value(
                Ok(request_value),
                ThreadsafeFunctionCallMode::Blocking,
                move |result, _env: Env| {
                    let response = match result {
                        Ok(unknown) => {
                            // Convert on JS thread (safe here, no async boundary)
                            String::from_unknown(unknown)
                                .map_err(|e| format!("Conversion error: {e}"))
                        }
                        Err(e) => Err(format!("JS error: {e:?}")),
                    };
                    let _ = tx_result.send(response);
                    Ok(())
                },
            );

            // Await the result from the channel
            match rx_result.await {
                Ok(Ok(json_string)) => {
                    // Parse JSON string - it could be:
                    // 1. A JSON array of StreamEvent objects (all events at once)
                    // 2. A single StreamEvent object
                    let result_value: Value = match serde_json::from_str(&json_string) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = tx_clone
                                .send(Err(anyhow::anyhow!("Failed to parse JSON string: {e}")));
                            return;
                        }
                    };
                    match result_value {
                        Value::Array(events) => {
                            // Array of events - stream them one by one
                            for event_value in events {
                                match serde_json::from_value::<StreamEvent>(event_value) {
                                    Ok(event) => {
                                        if tx_clone.send(Ok(event)).is_err() {
                                            break; // Receiver dropped
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx_clone.send(Err(anyhow::anyhow!(
                                            "Failed to parse stream event: {e}"
                                        )));
                                        break;
                                    }
                                }
                            }
                            // Send done event
                            let _ = tx_clone.send(Ok(StreamEvent::Done));
                        }
                        Value::Object(_) => {
                            // Single event object
                            match serde_json::from_value::<StreamEvent>(result_value) {
                                Ok(event) => {
                                    let _ = tx_clone.send(Ok(event));
                                    let _ = tx_clone.send(Ok(StreamEvent::Done));
                                }
                                Err(e) => {
                                    let _ = tx_clone.send(Err(anyhow::anyhow!(
                                        "Failed to parse stream event: {e}"
                                    )));
                                }
                            }
                        }
                        _ => {
                            let _ = tx_clone.send(Err(anyhow::anyhow!(
                                "Stream function returned unexpected format"
                            )));
                        }
                    }
                }
                Err(e) => {
                    let _ =
                        tx_clone.send(Err(anyhow::anyhow!("Failed to call stream function: {e}")));
                }
                Ok(Err(e)) => {
                    let _ = tx_clone.send(Err(anyhow::anyhow!("JavaScript callback error: {e}")));
                }
            }
        });

        // Create a stream from the receiver
        let receiver_stream = futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Box::pin(receiver_stream)
    }
}

/// JavaScript wrapper for `ExternalEmbeddingCallback`
/// Uses `ThreadsafeFunction`<Value> with sync bridge to avoid thread safety issues.
pub struct JsExternalEmbeddingCallback {
    embed_fn: Arc<ThreadsafeFunction<Value>>,
}

impl JsExternalEmbeddingCallback {
    pub fn new(embed_fn: ThreadsafeFunction<Value>) -> Self {
        Self {
            embed_fn: Arc::new(embed_fn),
        }
    }
}

#[async_trait]
impl ExternalEmbeddingCallback for JsExternalEmbeddingCallback {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        // Convert EmbeddingRequest to JSON Value
        let request_value = serde_json::to_value(&request)
            .map_err(|e| anyhow::anyhow!("Failed to serialize request: {e}"))?;

        // Use sync bridge to call JavaScript function
        // This avoids thread safety issues by using blocking calls with oneshot channels
        let json_string = call_js_sync_bridge(&self.embed_fn, request_value)
            .await
            .map_err(|e| anyhow::anyhow!("JavaScript callback error: {e}"))?;

        // Parse JSON string as EmbeddingResponse
        let response: EmbeddingResponse = serde_json::from_str(&json_string)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize response: {e}"))?;

        Ok(response)
    }
}

/// Create an `ExternalProvider` from JavaScript functions
///
/// The JavaScript functions should return JSON strings (e.g., JSON.stringify(result))
/// The `ThreadsafeFunction` callback converts the return value to String on the JS thread,
/// avoiding thread safety issues with `from_unknown`.
///
/// NOTE: This is an internal helper function, not exposed to JavaScript.
/// External providers are created via `AgentBuilder::set_external_provider`.
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn create_external_provider(
    name: String,
    complete_fn: ThreadsafeFunction<Value>,
    stream_fn: ThreadsafeFunction<Value>,
    supports_native_tools: bool,
    supports_streaming: bool,
) -> Arc<dyn Provider> {
    let callback = Arc::new(JsExternalCompletionCallback::new(complete_fn, stream_fn));
    Arc::new(ExternalProvider::new(
        name,
        callback,
        supports_native_tools,
        supports_streaming,
    ))
}

/// Create an `ExternalEmbeddingProvider` from JavaScript `ThreadsafeFunction`
///
/// The JavaScript function should return JSON strings (e.g., JSON.stringify(result))
/// The sync bridge handles conversion to avoid thread safety issues with `from_unknown`.
///
/// NOTE: This is an internal helper function, not exposed to JavaScript.
/// External embedding providers are created via `AgentBuilder::set_external_embedding_provider`.
///
/// # Errors
///
/// Returns a NAPI error when validation, synchronization, serialization,
/// or an underlying runtime operation fails.
pub fn create_external_embedding_provider(
    name: String,
    embed_fn: ThreadsafeFunction<Value>,
) -> Arc<dyn EmbeddingProvider> {
    let callback = Arc::new(JsExternalEmbeddingCallback::new(embed_fn));
    Arc::new(ExternalEmbeddingProvider::new(name, callback))
}
