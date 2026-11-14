//! The gate's proof that one tool call may run.
//!
//! Every way to run a tool takes a [`ToolPermit`]: [`Tool::execute`],
//! [`ToolRegistry::execute`] and their variants. Only a gate builds a permit,
//! after the policy admits the call, so code outside the crate cannot run a
//! tool without some call having been admitted.
//!
//! The permit is not bound to the tool name, the arguments or a call id. A
//! host-Rust [`Tool`](crate::agent::tools::Tool) that holds one while it
//! runs can pass it to [`ToolRegistry::execute`] for another tool; that
//! tool's own checks (the workspace root, for one) still apply. Model calls
//! cannot do this: every model path admits each call, and hooks, scripted
//! tools (JavaScript, Python, WASM) and plugins never hold a permit.
//!
//! [`Tool::execute`]: crate::agent::tools::Tool::execute
//! [`ToolRegistry::execute`]: crate::agent::tools::ToolRegistry::execute

use std::marker::PhantomData;

/// Proof that a gate admitted one tool call.
///
/// A tool receives it and may pass it on to a tool it runs during the
/// same call. It cannot be cloned, and its lifetime ends with the call, so
/// it cannot be stored for a later one; it does not name which tool or
/// arguments were admitted.
///
/// Code outside this crate cannot build one:
///
/// ```compile_fail
/// let permit = caliche::agent::tools::ToolPermit::admitted();
/// ```
///
/// ```compile_fail
/// let permit = caliche::agent::tools::ToolPermit {
///     admitted: std::marker::PhantomData,
/// };
/// ```
#[derive(Debug)]
pub struct ToolPermit<'call> {
    admitted: PhantomData<&'call ()>,
}

impl ToolPermit<'_> {
    /// A permit for a call the gate just admitted.
    ///
    /// Only the gates call this:
    /// [`HookExecutionMiddleware`](crate::agent::tools::execution::HookExecutionMiddleware)
    /// and the MCP policy gate.
    pub(crate) const fn admitted() -> Self {
        Self {
            admitted: PhantomData,
        }
    }

    /// A permit for a test that runs a tool by itself.
    ///
    /// It exists only in this crate's tests and with the `testing` feature,
    /// which no release build turns on.
    #[cfg(any(test, feature = "testing"))]
    #[must_use]
    pub const fn for_tests() -> ToolPermit<'static> {
        ToolPermit {
            admitted: PhantomData,
        }
    }
}
