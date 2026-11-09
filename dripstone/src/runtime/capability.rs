//! Runtime-facing re-exports for host/model capability probes.
//!
//! The concrete estimation logic lives in [`crate::loader::fit`] so catalog and
//! download code can share one implementation. This module is the stable import
//! path for inference/runtime consumers.

#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "model-download"
))]
pub use crate::loader::fit::{
    CapabilityProbe, DefaultCapabilityProbe, FitOptions, FitStatus, HostCapabilities, ModelFit,
    ModelResourceHint, estimate_model_fit, estimate_model_resources, fit_catalog,
    host_capabilities,
};
