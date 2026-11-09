pub mod connection;
pub mod dispatch;
pub mod helpers;
pub mod methods;
pub mod streaming;
pub mod subscriptions;
pub mod transport;
pub mod types;

use crate::adapter::ApiAdapter;
use crate::{AgentRouterState, AppState};
use dispatch::MethodRouter;
use std::sync::Arc;
use subscriptions::SubscriptionManager;

pub type MethodRegistrar = Arc<dyn Fn(&mut MethodRouter) + Send + Sync>;

#[derive(Clone)]
pub struct RpcState {
    pub app_state: AppState,
    pub agent_state: AgentRouterState,
    pub sessions_state: Arc<ApiAdapter>,
    pub method_router: Arc<MethodRouter>,
    pub subscription_manager: SubscriptionManager,
}

impl RpcState {
    pub fn new(
        app_state: AppState,
        agent_state: AgentRouterState,
        sessions_state: Arc<ApiAdapter>,
        extra_registrars: &[MethodRegistrar],
    ) -> Self {
        let mut router = MethodRouter::new();
        methods::register_all(&mut router);
        for register in extra_registrars {
            register(&mut router);
        }
        Self {
            app_state,
            agent_state,
            sessions_state,
            method_router: Arc::new(router),
            subscription_manager: SubscriptionManager::new(),
        }
    }
}
