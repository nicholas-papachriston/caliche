//! Configuration for local model inference.

use crate::loader::ModelSource;
use crate::runtime::{ResolvedRuntime, RuntimeConfig};
use serde::{Deserialize, Serialize};

/// Inference backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceBackendType {
    /// Candle backend (safetensors support)
    Candle,
    /// ONNX Runtime backend (ONNX format)
    Ort,
    /// MLX backend (Apple Silicon, feature-gated)
    Mlx,
    /// llama.cpp backend (GGUF format, feature-gated)
    LlamaCpp,
}

/// Device configuration for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DeviceConfig {
    /// CPU inference
    Cpu,
    /// CUDA GPU inference (for NVIDIA GPUs)
    Cuda {
        /// GPU device ID (0 for first GPU, 1 for second, etc.)
        device_id: usize,
    },
    /// Metal GPU inference (for Apple Silicon)
    Metal,
    /// MLX runtime (Apple Silicon only, feature-gated)
    Mlx,
    /// Auto-detect best available device (default; Metal on Apple, CUDA elsewhere).
    #[default]
    Auto,
}

impl DeviceConfig {
    /// Convert to shared runtime config for backend selection.
    #[must_use]
    pub const fn to_runtime_config(self) -> RuntimeConfig {
        match self {
            Self::Cpu => RuntimeConfig::Cpu,
            Self::Cuda { device_id } => RuntimeConfig::Cuda { device_id },
            Self::Metal => RuntimeConfig::Metal,
            Self::Mlx => RuntimeConfig::Mlx,
            Self::Auto => RuntimeConfig::Auto,
        }
    }

    /// Resolve device to a concrete runtime (for Auto, uses platform/feature policy).
    #[must_use]
    pub fn resolve(self) -> ResolvedRuntime {
        self.to_runtime_config().resolve_auto()
    }
}

/// Generation parameters for text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use]
pub struct GenerationParams {
    /// Sampling temperature (0.0 = deterministic, >1.0 = more random)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold (0.0-1.0)
    pub top_p: Option<f32>,
    /// Top-k sampling (number of top tokens to consider)
    pub top_k: Option<usize>,
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Stop sequences that will stop generation when encountered
    pub stop_sequences: Vec<String>,
    /// Repetition penalty (1.0 = no penalty, >1.0 = reduce repetition)
    pub repetition_penalty: Option<f32>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: None,
            top_k: None,
            max_tokens: 512,
            stop_sequences: Vec::new(),
            repetition_penalty: None,
        }
    }
}

impl GenerationParams {
    /// Create new generation parameters with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the temperature.
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-p sampling.
    pub const fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set top-k sampling.
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set maximum tokens.
    pub const fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Add a stop sequence.
    pub fn with_stop_sequence(mut self, sequence: impl Into<String>) -> Self {
        self.stop_sequences.push(sequence.into());
        self
    }

    /// Set repetition penalty.
    pub const fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = Some(penalty);
        self
    }
}

/// Information about a loaded model.
#[derive(Debug, Clone)]
#[must_use]
pub struct ModelInfo {
    /// Model name/identifier
    pub name: String,
    /// Model architecture (e.g., "llama", "gpt2", "bert")
    pub architecture: Option<String>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Context length / maximum sequence length
    pub context_length: Option<usize>,
    /// Number of parameters (if known)
    pub num_parameters: Option<usize>,
}

impl ModelInfo {
    /// Create new model info with just a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            architecture: None,
            vocab_size: None,
            context_length: None,
            num_parameters: None,
        }
    }

    /// Set the architecture.
    pub fn with_architecture(mut self, arch: impl Into<String>) -> Self {
        self.architecture = Some(arch.into());
        self
    }

    /// Set the vocabulary size.
    pub const fn with_vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = Some(size);
        self
    }

    /// Set the context length.
    pub const fn with_context_length(mut self, length: usize) -> Self {
        self.context_length = Some(length);
        self
    }

    /// Set the number of parameters.
    pub const fn with_num_parameters(mut self, num: usize) -> Self {
        self.num_parameters = Some(num);
        self
    }
}

/// Configuration for local model inference.
#[derive(Debug, Clone)]
#[must_use]
pub struct InferenceConfig {
    /// Model source (local path or `HuggingFace` repo)
    pub source: ModelSource,
    /// Backend type to use
    pub backend: InferenceBackendType,
    /// Device configuration
    pub device: DeviceConfig,
    /// Architecture hints (optional, for auto-detection)
    pub architecture_hint: Option<String>,
    /// Default generation parameters
    pub default_params: GenerationParams,
}

impl InferenceConfig {
    /// Create a new inference configuration.
    pub fn new(source: ModelSource, backend: InferenceBackendType) -> Self {
        Self {
            source,
            backend,
            device: DeviceConfig::default(),
            architecture_hint: None,
            default_params: GenerationParams::default(),
        }
    }

    /// Set device configuration.
    pub const fn with_device(mut self, device: DeviceConfig) -> Self {
        self.device = device;
        self
    }

    /// Set an architecture hint.
    pub fn with_architecture_hint(mut self, hint: impl Into<String>) -> Self {
        self.architecture_hint = Some(hint.into());
        self
    }

    /// Set default generation parameters.
    pub fn with_default_params(mut self, params: GenerationParams) -> Self {
        self.default_params = params;
        self
    }
}
