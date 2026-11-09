//! Configurable routing rules (openclaw-inspired).
//!
//! Evaluate ingress against rules to obtain overrides (e.g. system prompt hint,
//! session id prefix) before binding lookup and session creation.

use crate::gateway::types::{ChannelKind, GatewayIngressEnvelope, RouteContext};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Overrides applied when a routing rule matches (e.g. system prompt hint, session prefix).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingOverrides {
    /// Hint passed to session factory for agent system prompt selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt_hint: Option<String>,
    /// Prefix applied to new session ids for partitioning (router appends factory id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id_prefix: Option<String>,
}

/// Condition to match against ingress. First matching rule wins.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MatchCondition {
    #[serde(rename = "channel_id")]
    ChannelId { value: String },
    #[serde(rename = "adapter_id")]
    AdapterId { value: String },
    #[serde(rename = "user_id")]
    UserId { value: String },
    #[serde(rename = "tenant_id")]
    TenantId { value: String },
    #[serde(rename = "channel_kind")]
    ChannelKind { value: String },
    /// All nested conditions must match.
    AllOf { conditions: Vec<MatchCondition> },
    /// Any nested condition must match.
    AnyOf { conditions: Vec<MatchCondition> },
}

impl MatchCondition {
    fn matches(&self, ingress: &GatewayIngressEnvelope) -> bool {
        match self {
            MatchCondition::ChannelId { value } => ingress.channel_id == *value,
            MatchCondition::AdapterId { value } => ingress.adapter_id == *value,
            MatchCondition::UserId { value } => ingress.user_id == *value,
            MatchCondition::TenantId { value } => match (&ingress.tenant_id, value.as_str()) {
                (Some(t), v) => t == v,
                (None, "") => true,
                (None, _) => false,
            },
            MatchCondition::ChannelKind { value } => {
                let kind_str = match ingress.channel_kind {
                    ChannelKind::Telegram => "telegram",
                    ChannelKind::Discord => "discord",
                    ChannelKind::Teams => "teams",
                    ChannelKind::Twilio => "twilio",
                    ChannelKind::Slack => "slack",
                    ChannelKind::WhatsApp => "whatsapp",
                    ChannelKind::Webhook => "webhook",
                    ChannelKind::Cli => "cli",
                    ChannelKind::Api => "api",
                    ChannelKind::Custom => "custom",
                };
                kind_str.eq_ignore_ascii_case(value)
            }
            MatchCondition::AllOf { conditions } => conditions.iter().all(|c| c.matches(ingress)),
            MatchCondition::AnyOf { conditions } => conditions.iter().any(|c| c.matches(ingress)),
        }
    }
}

/// A single rule: when condition matches, apply overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    pub r#match: MatchCondition,
    pub overrides: RoutingOverrides,
}

/// Engine that evaluates ingress (and optional context) to produce routing overrides.
#[async_trait]
pub trait RoutingRuleEngine: Send + Sync {
    async fn evaluate(
        &self,
        ingress: &GatewayIngressEnvelope,
        _ctx: &RouteContext,
    ) -> Result<RoutingOverrides>;
}

/// Engine that returns the overrides of the first matching rule, or empty overrides.
#[derive(Debug, Clone, Default)]
pub struct FirstMatchEngine {
    rules: Vec<RoutingRule>,
}

impl FirstMatchEngine {
    pub fn new(rules: Vec<RoutingRule>) -> Self {
        Self { rules }
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

#[async_trait]
impl RoutingRuleEngine for FirstMatchEngine {
    async fn evaluate(
        &self,
        ingress: &GatewayIngressEnvelope,
        _ctx: &RouteContext,
    ) -> Result<RoutingOverrides> {
        for rule in &self.rules {
            if rule.r#match.matches(ingress) {
                return Ok(rule.overrides.clone());
            }
        }
        Ok(RoutingOverrides::default())
    }
}

/// No-op engine that always returns empty overrides.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoOpRoutingEngine;

#[async_trait]
impl RoutingRuleEngine for NoOpRoutingEngine {
    async fn evaluate(
        &self,
        _ingress: &GatewayIngressEnvelope,
        _ctx: &RouteContext,
    ) -> Result<RoutingOverrides> {
        Ok(RoutingOverrides::default())
    }
}

/// Convenience: build a router-ready engine from rules (Arc for trait object).
pub fn engine_from_rules(rules: Vec<RoutingRule>) -> Arc<dyn RoutingRuleEngine> {
    if rules.is_empty() {
        Arc::new(NoOpRoutingEngine)
    } else {
        Arc::new(FirstMatchEngine::new(rules))
    }
}

/// Load routing rules from a file path.
/// Supports `.json`; `.yaml`/`.yml` when the `serde_norway` feature is enabled.
pub fn load_rules_from_path(path: &std::path::Path) -> Result<Vec<RoutingRule>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read routing rules path {}: {}", path.display(), e))?;
    let ext = path.extension().and_then(|e| e.to_str());
    let rules: Vec<RoutingRule> = match ext {
        Some("json") => serde_json::from_str(&content).map_err(|e| anyhow::anyhow!("{e}"))?,
        #[cfg(feature = "serde_norway")]
        Some("yaml") | Some("yml") => {
            serde_norway::from_str(&content).map_err(|e| anyhow::anyhow!("{e}"))?
        }
        #[cfg(not(feature = "serde_norway"))]
        Some("yaml") | Some("yml") => {
            return Err(anyhow::anyhow!(
                "YAML routing rules require the serde_norway feature; use a .json file or enable the feature"
            ));
        }
        _ => serde_json::from_str(&content).map_err(|e| anyhow::anyhow!("{e}"))?,
    };
    Ok(rules)
}

/// Load routing rules and build an engine. Returns None if path is missing or empty.
pub fn load_engine_from_path(path: &std::path::Path) -> Result<Option<Arc<dyn RoutingRuleEngine>>> {
    if !path.exists() {
        return Ok(None);
    }
    let rules = load_rules_from_path(path)?;
    Ok(Some(engine_from_rules(rules)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::types::{
        ChannelKind, GatewayIngressEnvelope, GatewaySessionRef, IngressPayload,
    };
    use chrono::Utc;

    fn sample_ingress(
        channel_kind: ChannelKind,
        channel_id: &str,
        adapter_id: &str,
        user_id: &str,
        tenant_id: Option<&str>,
    ) -> GatewayIngressEnvelope {
        GatewayIngressEnvelope {
            event_id: "ev1".to_string(),
            idempotency_key: None,
            channel_kind,
            channel_id: channel_id.to_string(),
            adapter_id: adapter_id.to_string(),
            tenant_id: tenant_id.map(String::from),
            user_id: user_id.to_string(),
            user_display_name: None,
            received_at: Utc::now(),
            session_ref: GatewaySessionRef {
                session_id: None,
                thread_id: None,
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

    #[tokio::test]
    async fn engine_from_rules_empty_is_noop() {
        let eng = engine_from_rules(vec![]);
        let ingress = sample_ingress(ChannelKind::Slack, "c1", "a1", "u1", None);
        let out = eng
            .evaluate(&ingress, &RouteContext::default())
            .await
            .unwrap();
        assert!(out.system_prompt_hint.is_none() && out.session_id_prefix.is_none());
    }

    #[tokio::test]
    async fn first_match_wins_and_channel_kind_case_insensitive() {
        let rules = vec![
            RoutingRule {
                r#match: MatchCondition::ChannelKind {
                    value: "SLACK".to_string(),
                },
                overrides: RoutingOverrides {
                    system_prompt_hint: Some("slack_hint".to_string()),
                    session_id_prefix: None,
                },
            },
            RoutingRule {
                r#match: MatchCondition::ChannelKind {
                    value: "slack".to_string(),
                },
                overrides: RoutingOverrides {
                    system_prompt_hint: Some("second".to_string()),
                    session_id_prefix: None,
                },
            },
        ];
        let eng = FirstMatchEngine::new(rules);
        let ingress = sample_ingress(ChannelKind::Slack, "x", "y", "z", None);
        let out = eng
            .evaluate(&ingress, &RouteContext::default())
            .await
            .unwrap();
        assert_eq!(out.system_prompt_hint.as_deref(), Some("slack_hint"));
    }

    #[tokio::test]
    async fn tenant_id_none_matches_empty_rule_only() {
        let rule_empty = RoutingRule {
            r#match: MatchCondition::TenantId {
                value: String::new(),
            },
            overrides: RoutingOverrides {
                system_prompt_hint: Some("t0".to_string()),
                session_id_prefix: None,
            },
        };
        let rule_t1 = RoutingRule {
            r#match: MatchCondition::TenantId {
                value: "t1".to_string(),
            },
            overrides: RoutingOverrides {
                system_prompt_hint: Some("t1_hint".to_string()),
                session_id_prefix: None,
            },
        };
        let eng = FirstMatchEngine::new(vec![rule_t1.clone(), rule_empty]);
        let ingress_no_tenant = sample_ingress(ChannelKind::Api, "c", "a", "u", None);
        let out = eng
            .evaluate(&ingress_no_tenant, &RouteContext::default())
            .await
            .unwrap();
        assert_eq!(out.system_prompt_hint.as_deref(), Some("t0"));

        let eng2 = FirstMatchEngine::new(vec![rule_t1]);
        let ingress_with = sample_ingress(ChannelKind::Api, "c", "a", "u", Some("t1"));
        let out2 = eng2
            .evaluate(&ingress_with, &RouteContext::default())
            .await
            .unwrap();
        assert_eq!(out2.system_prompt_hint.as_deref(), Some("t1_hint"));
    }

    #[tokio::test]
    async fn all_of_any_of() {
        let ingress = sample_ingress(ChannelKind::Discord, "ch99", "adp", "usr", Some("ten"));
        let rule_ok = RoutingRule {
            r#match: MatchCondition::AllOf {
                conditions: vec![
                    MatchCondition::ChannelId {
                        value: "ch99".to_string(),
                    },
                    MatchCondition::AnyOf {
                        conditions: vec![
                            MatchCondition::UserId {
                                value: "other".to_string(),
                            },
                            MatchCondition::AdapterId {
                                value: "adp".to_string(),
                            },
                        ],
                    },
                ],
            },
            overrides: RoutingOverrides {
                system_prompt_hint: Some("nested".to_string()),
                session_id_prefix: Some("pre_".to_string()),
            },
        };
        let eng = FirstMatchEngine::new(vec![rule_ok]);
        let out = eng
            .evaluate(&ingress, &RouteContext::default())
            .await
            .unwrap();
        assert_eq!(out.system_prompt_hint.as_deref(), Some("nested"));
        assert_eq!(out.session_id_prefix.as_deref(), Some("pre_"));
    }

    #[test]
    fn load_rules_from_json_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rules.json");
        let json = r#"[
            {
                "match": { "type": "channel_id", "value": "abc" },
                "overrides": { "system_prompt_hint": "h1" }
            }
        ]"#;
        std::fs::write(&path, json).expect("write");
        let rules = load_rules_from_path(&path).expect("load");
        assert_eq!(rules.len(), 1);
        let eng = FirstMatchEngine::new(rules);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        let ingress = sample_ingress(ChannelKind::Cli, "abc", "x", "u", None);
        let out = rt
            .block_on(eng.evaluate(&ingress, &RouteContext::default()))
            .expect("eval");
        assert_eq!(out.system_prompt_hint.as_deref(), Some("h1"));
    }

    #[test]
    fn load_engine_from_missing_path_returns_none() {
        let path = std::path::Path::new("/nonexistent/caliche_routing_rules.json");
        let got = load_engine_from_path(path).expect("ok");
        assert!(got.is_none());
    }
}
