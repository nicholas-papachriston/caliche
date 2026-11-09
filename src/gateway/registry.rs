use crate::gateway::adapters::{
    DiscordAdapter, GatewayAdapter, GenericWebhookAdapter, SlackAdapter, TeamsAdapter,
    TelegramAdapter, TwilioAdapter,
};
use crate::gateway::types::ChannelKind;
use dashmap::DashMap;
use std::sync::Arc;

/// Registration metadata for adapter orchestration.
#[derive(Debug, Clone)]
pub struct AdapterRegistration {
    pub adapter_id: String,
    pub channel_kind: ChannelKind,
    pub enabled: bool,
    pub priority: u16,
}

/// Metadata-driven adapter registry.
#[derive(Default)]
pub struct GatewayAdapterRegistry {
    adapters: DashMap<String, Arc<dyn GatewayAdapter>>,
    metadata: DashMap<String, AdapterRegistration>,
}

impl GatewayAdapterRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        adapter: Arc<dyn GatewayAdapter>,
        enabled: bool,
        priority: u16,
    ) -> String {
        let id = adapter.id().to_string();
        let registration = AdapterRegistration {
            adapter_id: id.clone(),
            channel_kind: adapter.channel_kind(),
            enabled,
            priority,
        };
        self.adapters.insert(id.clone(), adapter);
        self.metadata.insert(id.clone(), registration);
        id
    }

    pub fn unregister(&self, adapter_id: &str) {
        self.adapters.remove(adapter_id);
        self.metadata.remove(adapter_id);
    }

    pub fn set_enabled(&self, adapter_id: &str, enabled: bool) -> bool {
        if let Some(mut entry) = self.metadata.get_mut(adapter_id) {
            entry.enabled = enabled;
            true
        } else {
            false
        }
    }

    pub fn get(&self, adapter_id: &str) -> Option<Arc<dyn GatewayAdapter>> {
        self.adapters.get(adapter_id).map(|v| Arc::clone(v.value()))
    }

    pub fn list(&self) -> Vec<AdapterRegistration> {
        let mut rows: Vec<AdapterRegistration> = self
            .metadata
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        rows.sort_by(|left, right| {
            left.priority
                .cmp(&right.priority)
                .then_with(|| left.adapter_id.cmp(&right.adapter_id))
        });
        rows
    }

    pub fn resolve_for_channel(&self, channel: ChannelKind) -> Vec<Arc<dyn GatewayAdapter>> {
        let mut rows: Vec<(u16, String, Arc<dyn GatewayAdapter>)> = Vec::new();
        for entry in &self.metadata {
            let meta = entry.value();
            if !meta.enabled || meta.channel_kind != channel {
                continue;
            }
            if let Some(adapter) = self.get(&meta.adapter_id) {
                rows.push((meta.priority, meta.adapter_id.clone(), adapter));
            }
        }
        rows.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        rows.into_iter().map(|(_, _, adapter)| adapter).collect()
    }

    /// Bootstrap baseline adapters from the architecture plan.
    pub fn with_default_adapters() -> Self {
        let registry = Self::new();
        registry.register(
            Arc::new(GenericWebhookAdapter::new("webhook-default")),
            true,
            10,
        );
        registry.register(Arc::new(TelegramAdapter::new("telegram-default")), true, 20);
        registry.register(Arc::new(DiscordAdapter::new("discord-default")), true, 30);
        registry.register(Arc::new(SlackAdapter::new("slack-default")), true, 40);
        registry.register(Arc::new(TeamsAdapter::new("teams-default")), true, 50);
        registry.register(Arc::new(TwilioAdapter::new("twilio-default")), true, 60);
        registry
    }
}

impl std::fmt::Debug for GatewayAdapterRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatewayAdapterRegistry")
            .field("adapter_count", &self.adapters.len())
            .field("metadata_count", &self.metadata.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::adapters::GenericWebhookAdapter;

    #[test]
    fn list_sorted_by_priority() {
        let reg = GatewayAdapterRegistry::new();
        reg.register(Arc::new(GenericWebhookAdapter::new("high_pri")), true, 100);
        reg.register(Arc::new(GenericWebhookAdapter::new("low_pri")), true, 5);
        let rows = reg.list();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].adapter_id, "low_pri");
        assert_eq!(rows[0].priority, 5);
        assert_eq!(rows[1].adapter_id, "high_pri");
        assert_eq!(rows[1].priority, 100);
    }

    #[test]
    fn resolve_for_channel_orders_by_priority_and_skips_disabled() {
        let reg = GatewayAdapterRegistry::new();
        reg.register(Arc::new(GenericWebhookAdapter::new("second")), true, 20);
        reg.register(Arc::new(GenericWebhookAdapter::new("first")), true, 10);
        reg.register(Arc::new(GenericWebhookAdapter::new("disabled")), false, 5);
        let resolved = reg.resolve_for_channel(ChannelKind::Webhook);
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].id(), "first");
        assert_eq!(resolved[1].id(), "second");
    }

    #[test]
    fn set_enabled_false_excludes_from_resolve() {
        let reg = GatewayAdapterRegistry::new();
        reg.register(Arc::new(GenericWebhookAdapter::new("a")), true, 1);
        assert_eq!(reg.resolve_for_channel(ChannelKind::Webhook).len(), 1);
        assert!(reg.set_enabled("a", false));
        assert!(reg.resolve_for_channel(ChannelKind::Webhook).is_empty());
        assert!(!reg.set_enabled("missing", true));
    }

    #[test]
    fn unregister_removes_adapter_and_metadata() {
        let reg = GatewayAdapterRegistry::new();
        reg.register(Arc::new(GenericWebhookAdapter::new("x")), true, 1);
        assert!(reg.get("x").is_some());
        reg.unregister("x");
        assert!(reg.get("x").is_none());
        assert!(reg.list().is_empty());
    }

    #[test]
    fn resolve_for_channel_empty_when_kind_mismatches() {
        let reg = GatewayAdapterRegistry::new();
        reg.register(Arc::new(GenericWebhookAdapter::new("wh")), true, 1);
        assert!(reg.resolve_for_channel(ChannelKind::Api).is_empty());
    }

    #[test]
    fn register_overwrites_same_adapter_id() {
        let reg = GatewayAdapterRegistry::new();
        reg.register(Arc::new(GenericWebhookAdapter::new("same")), true, 50);
        reg.register(Arc::new(GenericWebhookAdapter::new("same")), false, 5);
        let rows = reg.list();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].priority, 5);
        assert!(!rows[0].enabled);
        assert!(reg.resolve_for_channel(ChannelKind::Webhook).is_empty());
    }

    #[test]
    fn equal_priority_resolve_is_deterministic_by_adapter_id() {
        let reg = GatewayAdapterRegistry::new();
        reg.register(Arc::new(GenericWebhookAdapter::new("zeta")), true, 10);
        reg.register(Arc::new(GenericWebhookAdapter::new("alpha")), true, 10);
        reg.register(Arc::new(GenericWebhookAdapter::new("mu")), true, 10);
        let resolved: Vec<_> = reg
            .resolve_for_channel(ChannelKind::Webhook)
            .iter()
            .map(|adapter| adapter.id().to_string())
            .collect();
        assert_eq!(resolved, vec!["alpha", "mu", "zeta"]);
    }
}
