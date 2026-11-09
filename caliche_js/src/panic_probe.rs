//! Test-only exports that panic. The tests use them to check that a Rust panic
//! rejects an async call or throws from a sync call, and does not end the Node
//! process.

#![expect(
    clippy::panic,
    clippy::unused_async,
    reason = "the probes panic on purpose, and the async one must stay async"
)]

use napi_derive::napi;

#[napi(js_name = "_panicProbe")]
pub async fn panic_probe() {
    panic!("caliche panic probe");
}

#[napi(js_name = "_panicProbeSync", catch_unwind)]
pub fn panic_probe_sync() {
    panic!("caliche sync panic probe");
}
