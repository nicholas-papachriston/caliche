//! Python bindings for `ProviderType`

use caliche::llm::ProviderType as RustProviderType;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Copy)]
pub struct ProviderType {
    inner: RustProviderType,
}

#[pymethods]
#[allow(non_upper_case_globals)]
impl ProviderType {
    #[classattr]
    pub const OpenAI: Self = Self {
        inner: RustProviderType::OpenAI,
    };

    #[classattr]
    pub const Anthropic: Self = Self {
        inner: RustProviderType::Anthropic,
    };

    #[classattr]
    pub const Bedrock: Self = Self {
        inner: RustProviderType::Bedrock,
    };

    #[classattr]
    pub const OpenRouter: Self = Self {
        inner: RustProviderType::OpenRouter,
    };

    #[classattr]
    pub const CloudflareAIGateway: Self = Self {
        inner: RustProviderType::CloudflareAIGateway,
    };

    #[classattr]
    pub const Perplexity: Self = Self {
        inner: RustProviderType::Perplexity,
    };

    #[classattr]
    pub const LMStudio: Self = Self {
        inner: RustProviderType::LMStudio,
    };

    #[classattr]
    pub const Google: Self = Self {
        inner: RustProviderType::Google,
    };

    #[classattr]
    pub const Cohere: Self = Self {
        inner: RustProviderType::Cohere,
    };

    #[classattr]
    pub const AzureOpenAI: Self = Self {
        inner: RustProviderType::AzureOpenAI,
    };

    #[classattr]
    pub const Ollama: Self = Self {
        inner: RustProviderType::Ollama,
    };

    #[classattr]
    pub const External: Self = Self {
        inner: RustProviderType::External,
    };

    #[cfg(any(
        feature = "local-inference",
        feature = "local-inference-candle",
        feature = "local-inference-ort",
        feature = "local-inference-llamacpp"
    ))]
    #[classattr]
    pub const LocalInference: Self = Self {
        inner: RustProviderType::LocalInference,
    };
}

impl From<ProviderType> for RustProviderType {
    fn from(py_provider: ProviderType) -> Self {
        py_provider.inner
    }
}

impl From<RustProviderType> for ProviderType {
    fn from(rust_provider: RustProviderType) -> Self {
        Self {
            inner: rust_provider,
        }
    }
}
