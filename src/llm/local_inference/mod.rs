//! Local model inference provider using `Dripstone` inference backends.
//!
//! Host/catalog memory guidance is exposed here as the ecosystem façade over
//! Dripstone `CapabilityProbe`: `LocalInferenceProvider::probe_host_capabilities`,
//! `fit_catalog_models`, and `safe_catalog_models`. Prefer these wrappers from
//! Caliche consumers; call `dripstone::loader::{fit_catalog, host_capabilities}`
//! directly only when you already depend on Dripstone without this crate.

mod bootstrap;
mod config;
mod provider;

pub use config::{
    CandleBootstrapConfig, GgufBootstrapConfig, LocalInferenceBootstrapConfig,
    LocalInferenceConfig, OrtBootstrapConfig,
};
pub use provider::{LazyLocalInferenceProvider, LocalInferenceProvider};
