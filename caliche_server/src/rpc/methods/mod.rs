pub mod agent;
pub mod core;
pub mod features;
pub mod orchestration;
pub mod plugins;

use crate::rpc::dispatch::MethodRouter;

pub fn register_all(router: &mut MethodRouter) {
    core::register(router);
    agent::register(router);
    orchestration::register(router);
    plugins::register(router);
    features::register(router);
}
