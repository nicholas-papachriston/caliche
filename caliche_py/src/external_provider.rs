//! Python bindings for `ExternalProvider`
//!
//! This module provides bridges between Python callables and Rust's `ExternalProvider`,
//! enabling `LangChain` and other Python LLM libraries to be used with Caliche.

use crate::python_async::call_python_maybe_async;
use async_trait::async_trait;
use caliche::llm::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use caliche::llm::external::{
    ExternalCompletionCallback, ExternalEmbeddingCallback, ExternalEmbeddingProvider,
    ExternalProvider,
};
use caliche::llm::provider::Provider;
use caliche::messages::{CompletionRequest, CompletionResponse, StreamEvent};
use futures::Stream;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pythonize::{depythonize, pythonize};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;

/// Python wrapper for `ExternalCompletionCallback`
///
/// This struct bridges Python async functions to Rust's `ExternalCompletionCallback` trait.
/// The Python callable should accept a `CompletionRequest` dict and return a `CompletionResponse` dict.
pub struct PyExternalCompletionCallback {
    complete_fn: Py<PyAny>,
    stream_fn: Py<PyAny>,
}

impl PyExternalCompletionCallback {
    pub const fn new(complete_fn: Py<PyAny>, stream_fn: Py<PyAny>) -> Self {
        Self {
            complete_fn,
            stream_fn,
        }
    }
}

type StreamSender = tokio::sync::mpsc::UnboundedSender<anyhow::Result<StreamEvent>>;

async fn resolve_stream_iterator(
    stream_fn: Py<PyAny>,
    request_value: Value,
    tx: &StreamSender,
) -> Option<Py<PyAny>> {
    let py_result = Python::attach(|py| -> PyResult<Py<PyAny>> {
        let stream_fn = stream_fn.bind(py);
        let py_request = pythonize(py, &request_value)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert request: {e}")))?;
        stream_fn.call1((py_request,)).map(Bound::unbind)
    });
    let object = match py_result {
        Ok(object) => object,
        Err(error) => {
            let _ = tx.send(Err(anyhow::anyhow!(
                "Failed to call Python stream function: {error}"
            )));
            return None;
        }
    };
    match call_python_maybe_async(object).await {
        Ok(iterator) => Some(iterator),
        Err(error) => {
            let _ = tx.send(Err(anyhow::anyhow!(
                "Failed to execute Python stream function: {error}"
            )));
            None
        }
    }
}

async fn forward_async_iterator(iterator: Py<PyAny>, tx: &StreamSender) {
    loop {
        let next_result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let iterator = iterator.bind(py);
            let aiter = iterator.getattr("__aiter__")?.call0()?;
            aiter.getattr("__anext__")?.call0().map(Bound::unbind)
        });
        let coroutine = match next_result {
            Ok(coroutine) => coroutine,
            Err(error) => {
                let message = error.to_string();
                if message.contains("StopAsyncIteration") || message.contains("StopIteration") {
                    let _ = tx.send(Ok(StreamEvent::Done));
                } else {
                    let _ = tx.send(Err(anyhow::anyhow!(
                        "Failed to iterate async iterator: {error}"
                    )));
                }
                break;
            }
        };
        match call_python_maybe_async(coroutine).await {
            Ok(value) => {
                let event = Python::attach(|py| -> anyhow::Result<StreamEvent> {
                    let event_value: Value = depythonize(value.bind(py))
                        .map_err(|e| anyhow::anyhow!("Failed to convert event: {e}"))?;
                    serde_json::from_value(event_value)
                        .map_err(|e| anyhow::anyhow!("Failed to parse event: {e}"))
                });
                match event {
                    Ok(event) => {
                        if tx.send(Ok(event)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = tx.send(Err(error));
                        break;
                    }
                }
            }
            Err(error) => {
                let message = error.to_string();
                if message.contains("StopAsyncIteration") || message.contains("StopIteration") {
                    let _ = tx.send(Ok(StreamEvent::Done));
                } else {
                    let _ = tx.send(Err(anyhow::anyhow!(
                        "Failed to get next value from async iterator: {error}"
                    )));
                }
                break;
            }
        }
    }
}

fn forward_sync_result(iterator: &Py<PyAny>, tx: &StreamSender) {
    let result = Python::attach(|py| -> anyhow::Result<Value> {
        depythonize(iterator.bind(py)).map_err(|e| anyhow::anyhow!("Failed to convert result: {e}"))
    });
    let event_value = match result {
        Ok(value) => value,
        Err(error) => {
            let _ = tx.send(Err(error));
            return;
        }
    };
    let events = match event_value {
        Value::Array(events) => events,
        event => vec![event],
    };
    for event_value in events {
        match serde_json::from_value::<StreamEvent>(event_value) {
            Ok(event) => {
                if tx.send(Ok(event)).is_err() {
                    return;
                }
            }
            Err(error) => {
                let _ = tx.send(Err(anyhow::anyhow!("Failed to parse event: {error}")));
                return;
            }
        }
    }
    let _ = tx.send(Ok(StreamEvent::Done));
}

#[async_trait]
impl ExternalCompletionCallback for PyExternalCompletionCallback {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        // Convert CompletionRequest to Python dict using pythonize
        let request_value: Value = serde_json::to_value(&request)
            .map_err(|e| anyhow::anyhow!("Failed to serialize request: {e}"))?;

        let complete_fn = Python::attach(|py| self.complete_fn.clone_ref(py));
        let py_result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let complete_fn = complete_fn.bind(py);
            let py_request = pythonize(py, &request_value)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert request: {e}")))?;
            let result = complete_fn.call1((py_request,))?;
            Ok(result.unbind())
        })?;

        // Handle both sync and async Python functions
        let response_obj = call_python_maybe_async(py_result)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to execute Python callback: {e}"))?;

        // Convert Python dict back to CompletionResponse
        Python::attach(|py| -> anyhow::Result<CompletionResponse> {
            let bound = response_obj.bind(py);

            // Use depythonize to convert Python dict to JSON Value
            let response_value: Value = depythonize(bound)
                .map_err(|e| anyhow::anyhow!("Failed to convert response: {e}"))?;

            // Deserialize to CompletionResponse
            let response: CompletionResponse = serde_json::from_value(response_value)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize response: {e}"))?;

            Ok(response)
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        // Convert request to Python dict
        let request_value = match serde_json::to_value(&request) {
            Ok(v) => v,
            Err(e) => {
                return Box::pin(futures::stream::once(async move {
                    Err(anyhow::anyhow!("Failed to serialize request: {e}"))
                }));
            }
        };

        // Create a channel to forward stream events
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream_fn = Python::attach(|py| self.stream_fn.clone_ref(py));
        let tx_clone = tx;

        // Spawn async task to handle Python async iterator
        let rt = match crate::get_runtime() {
            Ok(rt) => rt,
            Err(e) => {
                return Box::pin(futures::stream::once(async move {
                    Err(anyhow::anyhow!("Failed to get Tokio runtime: {e}"))
                }));
            }
        };
        rt.spawn(async move {
            let Some(iterator) = resolve_stream_iterator(stream_fn, request_value, &tx_clone).await
            else {
                return;
            };
            let is_async_iter =
                Python::attach(|py| iterator.bind(py).hasattr("__aiter__")).unwrap_or(false);
            if is_async_iter {
                forward_async_iterator(iterator, &tx_clone).await;
            } else {
                forward_sync_result(&iterator, &tx_clone);
            }
        });

        // Create a stream from the receiver
        let receiver_stream = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Box::pin(receiver_stream)
    }
}

/// Python wrapper for `ExternalEmbeddingCallback`
pub struct PyExternalEmbeddingCallback {
    embed_fn: Py<PyAny>,
}

impl PyExternalEmbeddingCallback {
    pub const fn new(embed_fn: Py<PyAny>) -> Self {
        Self { embed_fn }
    }
}

#[async_trait]
impl ExternalEmbeddingCallback for PyExternalEmbeddingCallback {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        // Convert EmbeddingRequest to Python dict using pythonize
        let request_value: Value = serde_json::to_value(&request)
            .map_err(|e| anyhow::anyhow!("Failed to serialize request: {e}"))?;

        let embed_fn = Python::attach(|py| self.embed_fn.clone_ref(py));
        let py_result = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let embed_fn = embed_fn.bind(py);
            let py_request = pythonize(py, &request_value)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert request: {e}")))?;
            let result = embed_fn.call1((py_request,))?;
            Ok(result.unbind())
        })?;

        // Handle both sync and async Python functions
        let response_obj = call_python_maybe_async(py_result)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to execute Python callback: {e}"))?;

        // Convert Python dict back to EmbeddingResponse
        Python::attach(|py| -> anyhow::Result<EmbeddingResponse> {
            let bound = response_obj.bind(py);

            // Use depythonize to convert Python dict to JSON Value
            let response_value: Value = depythonize(bound)
                .map_err(|e| anyhow::anyhow!("Failed to convert response: {e}"))?;

            // Deserialize to EmbeddingResponse
            let response: EmbeddingResponse = serde_json::from_value(response_value)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize response: {e}"))?;

            Ok(response)
        })
    }
}

/// Create an `ExternalProvider` from Python callables
pub fn create_external_provider(
    name: String,
    _model_name: String,
    complete_fn: Py<PyAny>,
    stream_fn: Py<PyAny>,
    supports_native_tools: bool,
    supports_streaming: bool,
) -> Arc<dyn Provider> {
    let callback = Arc::new(PyExternalCompletionCallback::new(complete_fn, stream_fn));
    Arc::new(ExternalProvider::new(
        name,
        callback,
        supports_native_tools,
        supports_streaming,
    ))
}

/// Create an `ExternalEmbeddingProvider` from Python callable
pub fn create_external_embedding_provider(
    name: String,
    embed_fn: Py<PyAny>,
) -> Arc<dyn EmbeddingProvider> {
    let callback = Arc::new(PyExternalEmbeddingCallback::new(embed_fn));
    Arc::new(ExternalEmbeddingProvider::new(name, callback))
}
