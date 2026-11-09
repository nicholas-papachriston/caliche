//! Sandbox backends.

#[cfg(feature = "agentfs")]
pub mod agentfs;
#[cfg(all(feature = "sandbox-linux", target_os = "linux"))]
mod cgroup;
mod child_output;
#[cfg(feature = "docker")]
pub mod docker;
#[cfg(all(feature = "sandbox-linux", target_os = "linux"))]
mod git_guard;
#[cfg(all(feature = "sandbox-linux", target_os = "linux"))]
mod landlock_restrict;
#[cfg(all(feature = "sandbox-linux", target_os = "linux"))]
mod landlock_roots;
#[cfg(all(feature = "sandbox-linux", target_os = "linux"))]
pub mod linux;
#[cfg(all(feature = "sandbox-macos", target_os = "macos"))]
pub mod macos;
pub mod noop;
#[cfg(feature = "sandbox-wasm")]
pub mod wasm;
#[cfg(all(feature = "sandbox-windows", target_os = "windows"))]
pub mod windows;
