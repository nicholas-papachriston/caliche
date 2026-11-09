use crate::gateway::routing_rules::RoutingRuleEngine;
use crate::gateway::session_binding::{
    BindingKey, SessionBinding, SessionBindingStore, SessionFactory,
};
use crate::gateway::types::{GatewayIngressEnvelope, RouteContext};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use chrono::Utc;
use dashmap::DashSet;
use std::collections::HashMap;
use std::sync::Arc;

/// Decision returned by gateway router after binding lookup/creation.
#[derive(Debug, Clone)]
pub struct GatewayRouteDecision {
    pub session_id: String,
    pub created_new_binding: bool,
    pub binding_key: BindingKey,
    pub reason: String,
}

#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("duplicate ingress event: {0}")]
    Duplicate(String),
    #[error("routing failed: {0}")]
    Routing(String),
}

#[async_trait]
pub trait MessageDedupeStore: Send + Sync {
    async fn mark_seen(&self, dedupe_key: &str) -> Result<bool>;
}

/// Fast in-memory dedupe for idempotency keys.
#[derive(Debug, Default)]
pub struct InMemoryDedupeStore {
    seen: DashSet<String>,
}

impl InMemoryDedupeStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl MessageDedupeStore for InMemoryDedupeStore {
    async fn mark_seen(&self, dedupe_key: &str) -> Result<bool> {
        Ok(self.seen.insert(dedupe_key.to_string()))
    }
}

#[async_trait]
pub trait GatewayRouter: Send + Sync {
    async fn route(
        &self,
        ingress: &GatewayIngressEnvelope,
        ctx: &RouteContext,
    ) -> Result<GatewayRouteDecision>;
}

/// Default router that resolves existing binding or creates one via SessionFactory.
pub struct DefaultGatewayRouter {
    binding_store: Arc<dyn SessionBindingStore>,
    session_factory: Arc<dyn SessionFactory>,
    dedupe: Option<Arc<dyn MessageDedupeStore>>,
    rule_engine: Option<Arc<dyn RoutingRuleEngine>>,
}

impl DefaultGatewayRouter {
    pub fn new(
        binding_store: Arc<dyn SessionBindingStore>,
        session_factory: Arc<dyn SessionFactory>,
    ) -> Self {
        Self {
            binding_store,
            session_factory,
            dedupe: None,
            rule_engine: None,
        }
    }

    pub fn with_dedupe(mut self, dedupe: Arc<dyn MessageDedupeStore>) -> Self {
        self.dedupe = Some(dedupe);
        self
    }

    pub fn with_rule_engine(mut self, engine: Option<Arc<dyn RoutingRuleEngine>>) -> Self {
        self.rule_engine = engine;
        self
    }

    /// Build a default router and optionally attach a rule engine from
    /// `CALICHE_GATEWAY_ROUTING_RULES_PATH` (JSON or YAML file). If the env is unset or the
    /// path is missing, no engine is attached.
    pub fn with_rules_from_env(self) -> anyhow::Result<Self> {
        let path = match std::env::var("CALICHE_GATEWAY_ROUTING_RULES_PATH").ok() {
            Some(p) => std::path::PathBuf::from(p),
            None => return Ok(self),
        };
        if !path.exists() {
            return Ok(self);
        }
        let rules = crate::gateway::routing_rules::load_rules_from_path(&path)?;
        let engine = if rules.is_empty() {
            None
        } else {
            Some(crate::gateway::routing_rules::engine_from_rules(rules))
        };
        Ok(self.with_rule_engine(engine))
    }

    fn to_binding_key(ingress: &GatewayIngressEnvelope) -> BindingKey {
        BindingKey {
            tenant_id: ingress.tenant_id.clone(),
            channel_kind: ingress.channel_kind,
            channel_id: ingress.channel_id.clone(),
            user_id: ingress.user_id.clone(),
            thread_id: ingress.session_ref.thread_id.clone(),
        }
    }
}

#[async_trait]
impl GatewayRouter for DefaultGatewayRouter {
    async fn route(
        &self,
        ingress: &GatewayIngressEnvelope,
        ctx: &RouteContext,
    ) -> Result<GatewayRouteDecision> {
        let dedupe_key = ingress
            .idempotency_key
            .as_ref()
            .cloned()
            .unwrap_or_else(|| ingress.event_id.clone());

        if let Some(dedupe) = &self.dedupe {
            let inserted = dedupe.mark_seen(&dedupe_key).await?;
            if !inserted {
                return Err(RouterError::Duplicate(dedupe_key).into());
            }
        }

        let overrides = if let Some(engine) = &self.rule_engine {
            engine.evaluate(ingress, ctx).await?
        } else {
            crate::gateway::routing_rules::RoutingOverrides::default()
        };

        let key = Self::to_binding_key(ingress);
        if let Some(existing) = self.binding_store.get(&key).await? {
            return Ok(GatewayRouteDecision {
                session_id: existing.session_id,
                created_new_binding: false,
                binding_key: key,
                reason: "existing binding".to_owned(),
            });
        }

        let hint = overrides.system_prompt_hint.as_deref();
        let mut session_id = if let Some(sid) = &ingress.session_ref.session_id {
            sid.clone()
        } else {
            self.session_factory
                .create_session(ingress.tenant_id.as_deref(), hint)
                .await
                .map_err(|e| anyhow!("session creation failed: {e}"))?
        };

        if session_id.is_empty() {
            return Err(RouterError::Routing("empty session id returned".to_owned()).into());
        }

        if let Some(prefix) = &overrides.session_id_prefix {
            session_id = format!("{prefix}{session_id}");
        }

        let now = Utc::now();
        let (binding, created) = self
            .binding_store
            .get_or_insert(SessionBinding {
                key: key.clone(),
                session_id: session_id.clone(),
                created_at: now,
                updated_at: now,
                metadata: HashMap::new(),
            })
            .await?;

        Ok(GatewayRouteDecision {
            session_id: binding.session_id,
            created_new_binding: created,
            binding_key: key,
            reason: if created {
                "created binding".to_owned()
            } else {
                "existing binding".to_owned()
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::routing_rules::{
        FirstMatchEngine, MatchCondition, RoutingOverrides, RoutingRule, RoutingRuleEngine,
    };
    use crate::gateway::session_binding::{MemorySessionBindingStore, SessionBinding};
    use crate::gateway::types::{
        ChannelKind, GatewayIngressEnvelope, GatewaySessionRef, IngressPayload, RouteContext,
    };
    use chrono::Utc;

    #[derive(Clone)]
    struct StubFactory(String);

    #[async_trait]
    impl SessionFactory for StubFactory {
        async fn create_session(
            &self,
            _tenant_id: Option<&str>,
            _system_prompt_hint: Option<&str>,
        ) -> Result<String> {
            Ok(self.0.clone())
        }
    }

    fn ingress(
        tenant_id: Option<&str>,
        channel_kind: ChannelKind,
        channel_id: &str,
        user_id: &str,
        thread_id: Option<&str>,
    ) -> GatewayIngressEnvelope {
        GatewayIngressEnvelope {
            event_id: "e1".to_string(),
            idempotency_key: None,
            channel_kind,
            channel_id: channel_id.to_string(),
            adapter_id: "adp".to_string(),
            tenant_id: tenant_id.map(String::from),
            user_id: user_id.to_string(),
            user_display_name: None,
            received_at: Utc::now(),
            session_ref: GatewaySessionRef {
                session_id: None,
                thread_id: thread_id.map(String::from),
                conversation_id: None,
            },
            payload: IngressPayload {
                text: String::new(),
                attachments: Vec::new(),
                is_command: false,
                reply_to_message_id: None,
            },
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn to_binding_key_matches_ingress_fields() {
        let g = ingress(Some("t1"), ChannelKind::Slack, "C123", "U456", Some("th1"));
        let k = DefaultGatewayRouter::to_binding_key(&g);
        assert_eq!(k.tenant_id.as_deref(), Some("t1"));
        assert_eq!(k.channel_kind, ChannelKind::Slack);
        assert_eq!(k.channel_id, "C123");
        assert_eq!(k.user_id, "U456");
        assert_eq!(k.thread_id.as_deref(), Some("th1"));
    }

    #[tokio::test]
    async fn dedupe_store_first_insert_true_second_false() {
        let d = InMemoryDedupeStore::new();
        assert!(d.mark_seen("k1").await.expect("mark"));
        assert!(!d.mark_seen("k1").await.expect("mark2"));
        assert!(d.mark_seen("k2").await.expect("mark3"));
    }

    #[tokio::test]
    async fn route_creates_binding_via_factory() {
        let store = Arc::new(MemorySessionBindingStore::new());
        let router =
            DefaultGatewayRouter::new(store.clone(), Arc::new(StubFactory("sess-new".to_string())));
        let ing = ingress(
            Some("tenant-a"),
            ChannelKind::Slack,
            "C1",
            "U1",
            Some("thread-1"),
        );
        let key = DefaultGatewayRouter::to_binding_key(&ing);
        let ctx = RouteContext::default();

        let decision = router.route(&ing, &ctx).await.expect("first route");

        assert!(decision.created_new_binding);
        assert_eq!(decision.session_id, "sess-new");
        assert_eq!(decision.binding_key, key);

        let stored = store.get(&key).await.expect("get").expect("binding");
        assert_eq!(stored.session_id, "sess-new");
    }

    #[tokio::test]
    async fn route_reuses_existing_binding() {
        let store = Arc::new(MemorySessionBindingStore::new());
        let ing = ingress(None, ChannelKind::Webhook, "hook", "user", None);
        let key = DefaultGatewayRouter::to_binding_key(&ing);
        let now = Utc::now();
        store
            .put(SessionBinding {
                key: key.clone(),
                session_id: "existing-42".to_string(),
                created_at: now,
                updated_at: now,
                metadata: HashMap::new(),
            })
            .await
            .expect("seed");

        let router =
            DefaultGatewayRouter::new(store, Arc::new(StubFactory("should-not-run".to_string())));
        let decision = router
            .route(&ing, &RouteContext::default())
            .await
            .expect("route");

        assert!(!decision.created_new_binding);
        assert_eq!(decision.session_id, "existing-42");
        assert_eq!(decision.reason, "existing binding");
    }

    #[tokio::test]
    async fn route_uses_explicit_session_id_without_factory() {
        let store = Arc::new(MemorySessionBindingStore::new());
        let mut ing = ingress(Some("t"), ChannelKind::Api, "api", "u", None);
        ing.session_ref.session_id = Some("prebound".to_string());

        let router = DefaultGatewayRouter::new(
            store.clone(),
            Arc::new(StubFactory("factory-id".to_string())),
        );
        let decision = router
            .route(&ing, &RouteContext::default())
            .await
            .expect("route");

        assert!(decision.created_new_binding);
        assert_eq!(decision.session_id, "prebound");

        let key = DefaultGatewayRouter::to_binding_key(&ing);
        let stored = store.get(&key).await.expect("get").expect("binding");
        assert_eq!(stored.session_id, "prebound");
    }

    #[tokio::test]
    async fn route_rejects_empty_session_id() {
        let store = Arc::new(MemorySessionBindingStore::new());
        let mut ing = ingress(None, ChannelKind::Cli, "cli", "u", None);
        ing.session_ref.session_id = Some(String::new());

        let router = DefaultGatewayRouter::new(store, Arc::new(StubFactory("x".to_string())));
        let err = router
            .route(&ing, &RouteContext::default())
            .await
            .err()
            .expect("expected error");

        let router_err = err.downcast_ref::<RouterError>().expect("router error");
        match router_err {
            RouterError::Routing(msg) => assert!(msg.contains("empty session id")),
            RouterError::Duplicate(_) => panic!("unexpected duplicate"),
        }
    }

    #[tokio::test]
    async fn route_duplicate_when_dedupe_enabled() {
        let store = Arc::new(MemorySessionBindingStore::new());
        let router = DefaultGatewayRouter::new(store, Arc::new(StubFactory("s1".to_string())))
            .with_dedupe(Arc::new(InMemoryDedupeStore::new()));

        let mut ing = ingress(None, ChannelKind::Slack, "C", "U", None);
        ing.idempotency_key = Some("same-key".to_string());
        let ctx = RouteContext::default();

        router.route(&ing, &ctx).await.expect("first ok");

        let err = router
            .route(&ing, &ctx)
            .await
            .err()
            .expect("second duplicate");
        match err.downcast_ref::<RouterError>() {
            Some(RouterError::Duplicate(k)) => assert_eq!(k, "same-key"),
            _ => panic!("expected RouterError::Duplicate, got {err:?}"),
        }
    }

    #[tokio::test]
    async fn route_applies_session_id_prefix_from_rule_engine() {
        let store = Arc::new(MemorySessionBindingStore::new());
        let rules = vec![RoutingRule {
            r#match: MatchCondition::ChannelId {
                value: "C-pfx".to_string(),
            },
            overrides: RoutingOverrides {
                system_prompt_hint: None,
                session_id_prefix: Some("tenant1:".to_string()),
            },
        }];
        let engine: Arc<dyn RoutingRuleEngine> = Arc::new(FirstMatchEngine::new(rules));
        let router =
            DefaultGatewayRouter::new(store.clone(), Arc::new(StubFactory("abc".to_string())))
                .with_rule_engine(Some(engine));

        let ing = ingress(None, ChannelKind::Slack, "C-pfx", "U", None);
        let decision = router
            .route(&ing, &RouteContext::default())
            .await
            .expect("route");

        assert!(decision.created_new_binding);
        assert_eq!(decision.session_id, "tenant1:abc");

        let key = DefaultGatewayRouter::to_binding_key(&ing);
        let stored = store.get(&key).await.expect("get").expect("binding");
        assert_eq!(stored.session_id, "tenant1:abc");
    }

    #[tokio::test]
    async fn route_dedupe_falls_back_to_event_id() {
        let store = Arc::new(MemorySessionBindingStore::new());
        let router = DefaultGatewayRouter::new(store, Arc::new(StubFactory("sid".to_string())))
            .with_dedupe(Arc::new(InMemoryDedupeStore::new()));

        let mut ing = ingress(None, ChannelKind::Slack, "C", "U", None);
        ing.idempotency_key = None;
        ing.event_id = "ev-dedupe".to_string();
        let ctx = RouteContext::default();

        router.route(&ing, &ctx).await.expect("first");
        let err = router.route(&ing, &ctx).await.err().expect("dup");
        match err.downcast_ref::<RouterError>() {
            Some(RouterError::Duplicate(k)) => assert_eq!(k, "ev-dedupe"),
            _ => panic!("expected duplicate on event_id"),
        }
    }
}
