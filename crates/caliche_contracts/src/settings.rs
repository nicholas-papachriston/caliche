use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InteractionModeDto {
    #[default]
    Agent,
    Ask,
    Plan,
}

impl core::fmt::Display for InteractionModeDto {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Agent => write!(f, "agent"),
            Self::Ask => write!(f, "ask"),
            Self::Plan => write!(f, "plan"),
        }
    }
}

impl InteractionModeDto {
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Agent => "Full capabilities - can read and write files",
            Self::Ask => "Read-only - can only answer questions",
            Self::Plan => "Planning - interview and document requirements",
        }
    }

    #[must_use]
    pub const fn allows_writes(self) -> bool {
        matches!(self, Self::Agent)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentSettingsDto {
    pub provider: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<usize>,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<InteractionModeDto>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_sources: Option<bool>,
    /// Dripstone backend kind from `VECTOR_BACKEND` for this process.
    ///
    /// Read-only status. Empty or omitted means unset. This is retrieval
    /// configuration, not a tenant or authorization boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_backend: Option<String>,
    /// `unconfigured`, `configured`, or `unavailable`.
    ///
    /// `configured` means this binary can construct that `VECTOR_BACKEND` name
    /// because the feature is compiled in. It does not mean a remote store is
    /// reachable or healthy. `unavailable` is an unknown name or a feature that
    /// is off. `unconfigured` is unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_backend_state: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interaction_mode_dto_allows_writes_only_for_agent() {
        assert!(InteractionModeDto::Agent.allows_writes());
        assert!(!InteractionModeDto::Plan.allows_writes());
        assert!(!InteractionModeDto::Ask.allows_writes());
        assert_eq!(InteractionModeDto::Agent.to_string(), "agent");
        assert_eq!(InteractionModeDto::Plan.to_string(), "plan");
        assert_eq!(InteractionModeDto::Ask.to_string(), "ask");
        assert!(InteractionModeDto::Agent.description().contains("write"));
        assert!(InteractionModeDto::Ask.description().contains("Read-only"));
        assert!(InteractionModeDto::Plan.description().contains("Planning"));
    }

    #[test]
    fn interaction_mode_dto_serde_roundtrips_lowercase() {
        let json = serde_json::to_string(&InteractionModeDto::Ask).expect("serialize");
        assert_eq!(json, "\"ask\"");
        let parsed: InteractionModeDto = serde_json::from_str("\"plan\"").expect("deserialize");
        assert_eq!(parsed, InteractionModeDto::Plan);
    }

    #[test]
    fn agent_settings_dto_roundtrips_mode_and_tools() {
        let settings = AgentSettingsDto {
            provider: "openai".into(),
            model: "gpt-4o".into(),
            system_prompt: Some("be brief".into()),
            temperature: Some(0.1),
            max_tokens: Some(256),
            context_window: Some(8_000),
            tools: vec!["fs".into(), "bash".into()],
            mode: Some(InteractionModeDto::Ask),
            embedding_provider: Some("openai".into()),
            embedding_model: Some("text-embedding-3-small".into()),
            show_sources: Some(false),
            vector_backend: Some("memory".into()),
            vector_backend_state: Some("configured".into()),
        };
        let value = serde_json::to_value(&settings).expect("serialize");
        assert_eq!(value["mode"], "ask");
        assert_eq!(value["tools"].as_array().expect("tools").len(), 2);
        assert_eq!(value["embedding_provider"], "openai");
        assert_eq!(value["show_sources"], false);
        assert_eq!(value["vector_backend"], "memory");
        assert_eq!(value["vector_backend_state"], "configured");
        let again: AgentSettingsDto = serde_json::from_value(value).expect("deserialize");
        assert_eq!(again, settings);
    }

    #[test]
    fn agent_settings_dto_deserializes_legacy_payload_without_embedding_fields() {
        let parsed: AgentSettingsDto =
            serde_json::from_str(r#"{"provider":"openai","model":"gpt-4o","tools":[]}"#)
                .expect("deserialize");
        assert_eq!(parsed.provider, "openai");
        assert_eq!(parsed.model, "gpt-4o");
        assert_eq!(parsed.embedding_provider, None);
        assert_eq!(parsed.embedding_model, None);
        assert_eq!(parsed.show_sources, None);
        assert_eq!(parsed.vector_backend, None);
        assert_eq!(parsed.vector_backend_state, None);
    }
}
