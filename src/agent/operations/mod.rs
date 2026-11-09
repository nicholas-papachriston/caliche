//! Operation + ToolTranslator layer (Unreal Agent-inspired).
//!
//! Model tool calls are validated by a sync, I/O-free [`ToolTranslator`] into
//! versioned serializable [`Operation`] specs. An [`OperationManager`] commits
//! and executes those specs asynchronously.
//!
//! Default agent behavior is unchanged: the existing `ToolExecutionEngine`
//! path runs unless [`crate::AgentBuilder::enable_operation_dispatch`] or
//! [`crate::AgentBuilder::with_operation_manager`] is used (or
//! `CALICHE_OPERATION_DISPATCH=1`).
//!
//! Local dispatch is audit-only: it commits specs, then
//! `ToolExecutionEngine` runs the original tool calls (sandbox, cancel,
//! workspace root, metrics, AgentFS, hook deny, parallel). Results are
//! written back with `record_engine_results` so ops do not stay `Ready`.
//! `LocalOperationManager::execute` refuses Ready records.
//! Remote dispatch uses `OperationStore::run_batch` (transport errors become
//! failed executions). A sandboxed agent with `RemoteOperationManager`
//! fails closed unless the transport reports `isolates_in_sandbox()` and
//! binds the agent's configured sandbox handle.
//!
//! | Unreal | Caliche |
//! | --- | --- |
//! | Tool translator | [`ToolTranslator`] |
//! | Operation | [`Operation`] |
//! | Operation manager | [`OperationManager`] |
//! | Local manager | [`LocalOperationManager`] |
//! | Proxy / remote manager | [`RemoteOperationManager`] |

mod codec;
mod dispatch;
mod local;
mod manager;
mod remote;
mod spec;
mod store;
mod translator;

pub use dispatch::{execute_tool_call_via_operations, shared_builtin_translators};
pub use local::LocalOperationManager;
pub use manager::{OperationExecutionMode, OperationManager};
#[cfg(feature = "sandbox")]
pub use remote::SandboxOperationTransport;
pub use remote::{RecordingTransport, RemoteOperationManager, RemoteOperationTransport};
pub use spec::{
    OPERATION_SPEC_VERSION, Operation, OperationExecution, OperationId, OperationKind,
    OperationStatus, ToolCallStatus, Translation,
};
pub use translator::{
    BashToolTranslator, FsToolTranslator, RegistryToolTranslator, ToolTranslator,
    TranslatorRegistry, format_executions,
};
