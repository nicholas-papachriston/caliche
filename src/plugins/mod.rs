//! Schema-based plugin authoring: tools, providers, backends, skills, source plugins.
//!
//! Plugins are defined by JSON schemas and executed via a step DSL. They never
//! hold credentials; they receive `AgentCapabilities` at construction time.

pub mod adapters;
pub mod http_policy;
pub mod load_policy;
pub mod persistence;
pub mod registry;
pub mod runtime;
pub mod schema;
pub mod step_dsl;
pub mod tool;
pub mod validation;

pub use load_policy::PluginLoadPolicy;
pub use persistence::PluginStore;
