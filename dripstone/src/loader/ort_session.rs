//! ONNX Runtime session builder utilities.
//!
//! Execution provider selection is policy-driven: [`execution_providers_for_device`]
//! centralizes the ordered list of providers. Extend it to add `CoreML` (Apple),
//! `TensorRT`, or `DirectML` by enabling the corresponding `ort` Cargo features and
//! appending to the returned list.

use crate::errors::DripstoneError;
#[cfg(feature = "embedding-ort")]
use ort::{
    ep::{CPU, CUDA},
    session::{Session, builder::GraphOptimizationLevel},
};
use std::path::Path;

/// Device preference for ONNX Runtime execution.
#[cfg(feature = "embedding-ort")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrtDevice {
    /// CPU execution only
    #[default]
    Cpu,
    /// CUDA GPU execution (with CPU fallback)
    Cuda {
        /// GPU device ID (0 for first GPU, 1 for second, etc.)
        device_id: usize,
    },
    /// Auto-detect (CUDA if available, fallback to CPU)
    Auto,
}

#[cfg(feature = "embedding-ort")]
/// Configuration for building ONNX Runtime sessions.
#[derive(Debug, Clone)]
pub struct OrtSessionConfig {
    /// Graph optimization level
    pub optimization_level: GraphOptimizationLevel,
    /// Number of threads (None = use `num_cpus::get()`)
    pub num_threads: Option<usize>,
    /// Device preference for execution
    pub device: OrtDevice,
}

#[cfg(feature = "embedding-ort")]
impl Default for OrtSessionConfig {
    fn default() -> Self {
        Self {
            optimization_level: GraphOptimizationLevel::Level3,
            num_threads: None,
            device: OrtDevice::default(),
        }
    }
}

#[cfg(feature = "embedding-ort")]
impl OrtSessionConfig {
    /// Create a new configuration with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the optimization level.
    #[must_use]
    pub const fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Set the number of threads.
    #[must_use]
    pub const fn with_num_threads(mut self, threads: usize) -> Self {
        self.num_threads = Some(threads);
        self
    }

    /// Set device preference.
    #[must_use]
    pub const fn with_device(mut self, device: OrtDevice) -> Self {
        self.device = device;
        self
    }
}

/// Build the ordered list of ONNX Runtime execution providers for the given device.
///
/// Policy: CPU-only, CUDA-then-CPU, or Auto (CUDA then CPU). Extend with `CoreML`
/// (Apple) or `TensorRT` by adding the corresponding `ort` feature and pushing
/// the provider here (e.g. after CUDA for Auto, or in a new `OrtDevice` variant).
#[cfg(feature = "embedding-ort")]
fn execution_providers_for_device(
    device: OrtDevice,
) -> Result<Vec<ort::ep::ExecutionProviderDispatch>, DripstoneError> {
    Ok(match device {
        OrtDevice::Cpu => vec![CPU::default().build()],
        OrtDevice::Cuda { device_id } => {
            let device_id = i32::try_from(device_id)
                .map_err(|_| DripstoneError::configuration("CUDA device id exceeds i32 range"))?;
            vec![
                CUDA::default().with_device_id(device_id).build(),
                CPU::default().build(),
            ]
        }
        OrtDevice::Auto => vec![CUDA::default().build(), CPU::default().build()],
    })
}

/// Build an ONNX Runtime session from a model file path.
#[cfg(feature = "embedding-ort")]
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn build_ort_session(
    model_path: &Path,
    config: &OrtSessionConfig,
) -> Result<Session, DripstoneError> {
    let mut builder = Session::builder()
        .map_err(|e| DripstoneError::storage_with_source("failed to create session builder", e))?;

    let providers = execution_providers_for_device(config.device)?;
    tracing::info!(
        device = ?config.device,
        num_providers = providers.len(),
        "ORT execution providers (first wins; fallback to next on failure)"
    );
    builder = builder
        .with_execution_providers(providers)
        .map_err(|e| DripstoneError::storage_with_source("failed to set execution providers", e))?;

    builder = builder
        .with_optimization_level(config.optimization_level)
        .map_err(|e| DripstoneError::storage_with_source("failed to set optimization level", e))?;

    let num_threads = config.num_threads.unwrap_or_else(num_cpus::get);
    builder = builder
        .with_intra_threads(num_threads)
        .map_err(|e| DripstoneError::storage_with_source("failed to set thread count", e))?;

    builder.commit_from_file(model_path).map_err(|e| {
        DripstoneError::storage_with_source(
            format!("failed to load ONNX model from '{}'", model_path.display()),
            e,
        )
    })
}
