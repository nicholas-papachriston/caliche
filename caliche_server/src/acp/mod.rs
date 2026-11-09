mod catalog;
pub mod mapping;
pub mod mcp;
pub mod server;

mod driver;
mod outbound;
mod permission;
mod stdio;

pub use driver::AcpTurnDriver;
pub use outbound::{AcpFrameSink, StdFrameSink};
pub use permission::{ACP_PERMISSION_TIMEOUT, AcpPermissionGate};
pub use server::AcpServer;
pub use stdio::serve_acp;

#[cfg(test)]
mod tests;
