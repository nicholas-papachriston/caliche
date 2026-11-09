//! Python bindings for `LlmClient`

use crate::block_on_detached;
use crate::conversions::python_to_json_value;
use crate::get_runtime;
use crate::providers::ProviderType;
use crate::streaming::StreamIterator;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use caliche::agent::builder::AwsCredentials;
use caliche::llm::ProviderType as RustProviderType;
use caliche::llm::{LlmClient as RustLlmClient, LlmClientBuilder as RustLlmClientBuilder};
use caliche::messages::{
    Attachment, AttachmentType, Message, MessageRole, StreamEvent, ToolChoice as RustToolChoice,
    ToolSchema,
};
use futures::StreamExt;
use futures::stream::unfold;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

#[pyclass]
pub struct LlmClientBuilder {
    inner: RustLlmClientBuilder,
}

#[pymethods]
impl LlmClientBuilder {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RustLlmClientBuilder::new(),
        }
    }

    /// Set the provider type
    pub fn provider(mut slf: PyRefMut<Self>, provider: ProviderType) -> PyRefMut<Self> {
        let rust_provider: RustProviderType = provider.into();
        slf.inner = slf.inner.clone().provider(rust_provider);
        slf
    }

    /// Set the model name
    pub fn model(mut slf: PyRefMut<Self>, model: String) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().model(model);
        slf
    }

    /// Set the API key for authentication
    pub fn api_key(mut slf: PyRefMut<Self>, key: String) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().api_key(key);
        slf
    }

    /// Set AWS credentials for Bedrock provider
    pub fn aws_credentials(
        mut slf: PyRefMut<Self>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
        region: Option<String>,
    ) -> PyRefMut<Self> {
        let aws_creds = AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token,
            region,
            bearer_token: None,
        };
        slf.inner = slf.inner.clone().aws_credentials(aws_creds);
        slf
    }

    /// Set base URL (for `LMStudio` or custom endpoints)
    pub fn base_url(mut slf: PyRefMut<Self>, url: String) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().base_url(url);
        slf
    }

    /// Set default temperature for completions
    pub fn temperature(mut slf: PyRefMut<Self>, temp: f32) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().temperature(temp);
        slf
    }

    /// Set default max tokens for completions
    pub fn max_tokens(mut slf: PyRefMut<Self>, tokens: u32) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().max_tokens(tokens);
        slf
    }

    /// Set request timeout in seconds
    pub fn timeout(mut slf: PyRefMut<Self>, seconds: f64) -> PyRefMut<Self> {
        let timeout = Duration::from_secs_f64(seconds);
        slf.inner = slf.inner.clone().timeout(timeout);
        slf
    }

    /// Set maximum retry attempts
    pub fn max_retries(mut slf: PyRefMut<Self>, retries: usize) -> PyRefMut<Self> {
        slf.inner = slf.inner.clone().max_retries(retries);
        slf
    }

    /// Set default tools for completions
    ///
    /// Tools are optional and can be used for function calling.
    /// If not set, no tools will be sent to the LLM.
    ///
    /// Args:
    ///     tools: List of tool dictionaries with 'name', 'description', and 'parameters' keys.
    ///            Parameters should be a dict with 'type', 'properties', and 'required' keys.
    pub fn tools<'py>(
        mut slf: PyRefMut<'py, Self>,
        tools: &Bound<'py, PyList>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut rust_tools = Vec::new();
        for item in tools.iter() {
            let tool_dict = item.cast::<PyDict>()?;
            let name = tool_dict
                .get_item("name")?
                .and_then(|v| v.extract::<String>().ok())
                .ok_or_else(|| PyRuntimeError::new_err("Tool must have 'name' field"))?;

            let description = tool_dict
                .get_item("description")?
                .and_then(|v| v.extract::<String>().ok())
                .ok_or_else(|| PyRuntimeError::new_err("Tool must have 'description' field"))?;

            let params_value = tool_dict
                .get_item("parameters")?
                .ok_or_else(|| PyRuntimeError::new_err("Tool must have 'parameters' field"))?;
            let params_dict = params_value
                .cast::<PyDict>()
                .map_err(|_| PyRuntimeError::new_err("Tool 'parameters' must be a dict"))?;

            let schema_type = params_dict
                .get_item("type")?
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_else(|| "object".to_string());

            let properties = params_dict
                .get_item("properties")?
                .map(|v| python_to_json_value(&v))
                .transpose()?
                .unwrap_or_else(|| serde_json::json!({}));

            let required = params_dict
                .get_item("required")?
                .and_then(|v| v.extract::<Vec<String>>().ok())
                .unwrap_or_default();

            rust_tools.push(ToolSchema {
                name: beef::lean::Cow::owned(name),
                description: beef::lean::Cow::owned(description),
                parameters: caliche::messages::ParametersSchema {
                    schema_type: beef::lean::Cow::owned(schema_type),
                    properties,
                    required: required.into_iter().map(beef::lean::Cow::owned).collect(),
                },
            });
        }
        slf.inner = slf.inner.clone().tools(rust_tools);
        Ok(slf)
    }

    /// Set default tool choice for completions
    ///
    /// Controls how the LLM uses tools:
    /// - "auto": LLM decides whether to use tools (default)
    /// - "none": Disable tool usage
    /// - "required": Force tool usage
    /// - Specific tool name: Force use of a specific tool
    pub fn tool_choice(mut slf: PyRefMut<Self>, choice: String) -> PyRefMut<Self> {
        let rust_choice = match choice.to_lowercase().as_str() {
            "auto" => RustToolChoice::Auto,
            "none" => RustToolChoice::None,
            "required" => RustToolChoice::Required,
            _ => RustToolChoice::Specific(beef::lean::Cow::owned(choice)),
        };
        slf.inner = slf.inner.clone().tool_choice(rust_choice);
        slf
    }

    /// Build the `LlmClient`
    pub fn build(&self, py: Python<'_>) -> PyResult<Py<LlmClient>> {
        let builder = self.inner.clone();
        let client = block_on_detached(py, builder.build())?
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build LlmClient: {e}")))?;
        Py::new(
            py,
            LlmClient {
                inner: Arc::new(client),
            },
        )
    }
}

#[pyclass]
pub struct LlmClient {
    inner: Arc<RustLlmClient>,
}

#[pymethods]
impl LlmClient {
    /// Create a new builder for constructing an `LlmClient`
    #[staticmethod]
    pub fn builder() -> LlmClientBuilder {
        LlmClientBuilder::new()
    }

    /// Complete a conversation with messages (non-streaming)
    ///
    /// Returns the full completion response including content, usage, and finish reason.
    pub fn complete(&self, py: Python<'_>, messages: Vec<PyMessage>) -> PyResult<Py<PyAny>> {
        let rust_messages: Vec<Message> =
            messages.into_iter().map(std::convert::Into::into).collect();

        let client = Arc::clone(&self.inner);
        let response = block_on_detached(py, async move { client.complete(rust_messages).await })?
            .map_err(|e| PyRuntimeError::new_err(format!("Completion failed: {e}")))?;

        // Convert to Python dict
        let dict = PyDict::new(py);
        dict.set_item("content", response.content.extract_text())?;
        if let Some(finish_reason) = response.finish_reason {
            dict.set_item("finish_reason", finish_reason.as_ref())?;
        }
        if let Some(usage) = response.usage {
            let usage_dict = PyDict::new(py);
            if let Some(prompt) = usage.prompt_tokens {
                usage_dict.set_item("prompt_tokens", prompt)?;
            }
            if let Some(completion) = usage.completion_tokens {
                usage_dict.set_item("completion_tokens", completion)?;
            }
            if let Some(total) = usage.total_tokens {
                usage_dict.set_item("total_tokens", total)?;
            }
            dict.set_item("usage", usage_dict)?;
        }
        Ok(dict.into())
    }

    /// Stream a conversation with messages
    ///
    /// Returns a `StreamIterator` for iterating over stream events.
    pub fn stream(&self, py: Python<'_>, messages: Vec<PyMessage>) -> PyResult<Py<PyAny>> {
        let rust_messages: Vec<Message> =
            messages.into_iter().map(std::convert::Into::into).collect();

        // Create a channel for forwarding stream events
        let (tx, rx) = mpsc::unbounded_channel();
        let client_clone = Arc::clone(&self.inner);

        // Spawn a task to consume the stream and forward events
        let rt = get_runtime()?;
        rt.spawn(async move {
            let mut stream = client_clone.stream(rust_messages);
            while let Some(item) = stream.next().await {
                if tx.send(item).is_err() {
                    break; // Receiver dropped
                }
            }
        });

        // Create a stream from the receiver
        let receiver_stream = unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        let boxed_stream: std::pin::Pin<
            Box<dyn futures::Stream<Item = anyhow::Result<StreamEvent>> + Send>,
        > = Box::pin(receiver_stream);

        let stream_iter = StreamIterator::new(boxed_stream, None);
        Ok(Py::new(py, stream_iter)?.into_any())
    }

    /// Complete a single text prompt (convenience method)
    ///
    /// Creates a user message from the prompt and returns just the text content.
    pub fn complete_text(&self, py: Python<'_>, prompt: String) -> PyResult<String> {
        let client = Arc::clone(&self.inner);
        block_on_detached(py, async move { client.complete_text(&prompt).await })?
            .map_err(|e| PyRuntimeError::new_err(format!("Completion failed: {e}")))
    }

    /// Complete with multimodal attachments (base64 payloads).
    pub fn complete_with_attachments(
        &self,
        py: Python<'_>,
        message: &str,
        attachments: Vec<PyAttachment>,
    ) -> PyResult<Py<PyAny>> {
        let mut rust_attachments = Vec::with_capacity(attachments.len());
        for attachment in attachments {
            let data = BASE64
                .decode(attachment.data_base64.as_bytes())
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Invalid base64 attachment payload: {e}"))
                })?;
            let attachment_type = match attachment.attachment_type.as_str() {
                "image" => AttachmentType::Image,
                "audio" => AttachmentType::Audio,
                "video" => AttachmentType::Video,
                "document" => AttachmentType::Document,
                other => {
                    return Err(PyRuntimeError::new_err(format!(
                        "Unsupported attachment_type '{other}', expected one of: image,audio,video,document"
                    )));
                }
            };
            rust_attachments.push(Attachment {
                data,
                media_type: attachment.media_type,
                filename: attachment.filename,
                attachment_type,
            });
        }

        let rust_messages = vec![Message::user_with_attachments(message, rust_attachments)];
        let client = Arc::clone(&self.inner);
        let response = block_on_detached(py, async move { client.complete(rust_messages).await })?
            .map_err(|e| PyRuntimeError::new_err(format!("Completion failed: {e}")))?;

        let dict = PyDict::new(py);
        dict.set_item("content", response.content.extract_text())?;
        if let Some(finish_reason) = response.finish_reason {
            dict.set_item("finish_reason", finish_reason.as_ref())?;
        }
        if let Some(usage) = response.usage {
            let usage_dict = PyDict::new(py);
            if let Some(prompt) = usage.prompt_tokens {
                usage_dict.set_item("prompt_tokens", prompt)?;
            }
            if let Some(completion) = usage.completion_tokens {
                usage_dict.set_item("completion_tokens", completion)?;
            }
            if let Some(total) = usage.total_tokens {
                usage_dict.set_item("total_tokens", total)?;
            }
            dict.set_item("usage", usage_dict)?;
        }
        Ok(dict.into())
    }

    /// Stream a single text prompt (convenience method)
    ///
    /// Creates a user message from the prompt and returns a stream.
    pub fn stream_text(&self, py: Python<'_>, prompt: String) -> PyResult<Py<PyAny>> {
        // Create a channel for forwarding stream events
        let (tx, rx) = mpsc::unbounded_channel();
        let client_clone = Arc::clone(&self.inner);
        let prompt_clone = prompt;

        // Spawn a task to consume the stream and forward events
        let rt = get_runtime()?;
        rt.spawn(async move {
            let mut stream = client_clone.stream_text(&prompt_clone);
            while let Some(item) = stream.next().await {
                if tx.send(item).is_err() {
                    break; // Receiver dropped
                }
            }
        });

        // Create a stream from the receiver
        let receiver_stream = unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        let boxed_stream: std::pin::Pin<
            Box<dyn futures::Stream<Item = anyhow::Result<StreamEvent>> + Send>,
        > = Box::pin(receiver_stream);

        let stream_iter = StreamIterator::new(boxed_stream, None);
        Ok(Py::new(py, stream_iter)?.into_any())
    }

    /// Get the provider name
    pub fn provider_name(&self) -> String {
        self.inner.provider_name().to_string()
    }

    /// Get the model name
    pub fn model(&self) -> String {
        self.inner.model().to_string()
    }
}

/// Python wrapper for Message
#[pyclass]
#[derive(Clone)]
pub struct PyMessage {
    role: String,
    content: String,
}

/// Python wrapper for multimodal attachments.
#[pyclass]
#[derive(Clone)]
pub struct PyAttachment {
    #[pyo3(get, set)]
    attachment_type: String,
    #[pyo3(get, set)]
    data_base64: String,
    #[pyo3(get, set)]
    media_type: String,
    #[pyo3(get, set)]
    filename: Option<String>,
}

#[pymethods]
impl PyAttachment {
    #[new]
    pub const fn new(
        attachment_type: String,
        data_base64: String,
        media_type: String,
        filename: Option<String>,
    ) -> Self {
        Self {
            attachment_type,
            data_base64,
            media_type,
            filename,
        }
    }
}

#[pymethods]
impl PyMessage {
    #[new]
    pub const fn new(role: String, content: String) -> Self {
        Self { role, content }
    }

    /// Create a user message
    #[staticmethod]
    pub fn user(content: String) -> Self {
        Self {
            role: "user".to_string(),
            content,
        }
    }

    /// Create a system message
    #[staticmethod]
    pub fn system(content: String) -> Self {
        Self {
            role: "system".to_string(),
            content,
        }
    }

    /// Create an assistant message
    #[staticmethod]
    pub fn assistant(content: String) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
        }
    }
}

impl From<PyMessage> for Message {
    fn from(py_msg: PyMessage) -> Self {
        let role = match py_msg.role.as_str() {
            "system" => MessageRole::System,
            "assistant" => MessageRole::Assistant,
            "tool" => MessageRole::Tool,
            _ => MessageRole::User,
        };
        Self::new(role, py_msg.content)
    }
}
