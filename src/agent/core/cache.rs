use super::{ToolCallingConfig, message_builder};
#[cfg(feature = "tools-rag")]
use std::hash::{Hash, Hasher};

impl super::Agent {
    /// Get cached tools description for system message, computing and caching if needed.
    pub(crate) fn get_cached_tools_description(&self, include: bool) -> Option<String> {
        if !include || self.tools.list().is_empty() {
            return None;
        }
        {
            let guard = self.cached_tools_description.as_ref().load();
            if let Some(ref cached) = **guard {
                return Some(cached.clone());
            }
        }
        let description = message_builder::describe_tools(&self.tools);
        self.cached_tools_description
            .as_ref()
            .store(std::sync::Arc::new(Some(description.clone())));
        Some(description)
    }

    /// Get or build system messages, using cache when `workspace_context` is None or when
    /// workspace context hash matches.
    pub(crate) fn get_or_build_system_messages(
        &self,
        tool_cfg: &ToolCallingConfig,
        workspace_context: Option<&str>,
    ) -> Vec<crate::messages::Message> {
        if let Some(ctx) = workspace_context {
            #[cfg(feature = "tools-rag")]
            {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                ctx.hash(&mut hasher);
                let workspace_hash = hasher.finish();
                let now = chrono::Utc::now();
                use chrono::Datelike;
                use chrono::Timelike;
                let hour_key = now.date_naive().ordinal() * 24 + now.hour();
                {
                    let cache = self.cached_system_messages_with_workspace.read();
                    if let Some((cached_hour, cached_inc, cached_tools, cached_hash, ref msgs)) =
                        *cache
                        && cached_hour == hour_key
                        && cached_inc == self.include_context_in_system
                        && cached_tools == tool_cfg.include_tool_descriptions
                        && cached_hash == workspace_hash
                    {
                        return msgs.clone();
                    }
                }
                let cached_tools =
                    self.get_cached_tools_description(tool_cfg.include_tool_descriptions);
                let msgs = message_builder::build_system_message(
                    &self.system_prompt,
                    &self.current_dir,
                    &self.tools,
                    tool_cfg.include_tool_descriptions,
                    Some(ctx),
                    cached_tools.as_deref(),
                    self.include_context_in_system,
                    &self.skill_index,
                );
                *self.cached_system_messages_with_workspace.write() = Some((
                    hour_key,
                    self.include_context_in_system,
                    tool_cfg.include_tool_descriptions,
                    workspace_hash,
                    msgs.clone(),
                ));
                return msgs;
            }
            #[cfg(not(feature = "tools-rag"))]
            {
                let cached_tools =
                    self.get_cached_tools_description(tool_cfg.include_tool_descriptions);
                return message_builder::build_system_message(
                    &self.system_prompt,
                    &self.current_dir,
                    &self.tools,
                    tool_cfg.include_tool_descriptions,
                    Some(ctx),
                    cached_tools.as_deref(),
                    self.include_context_in_system,
                    &self.skill_index,
                );
            }
        }

        let now = chrono::Utc::now();
        use chrono::Datelike;
        use chrono::Timelike;
        let hour_key = now.date_naive().ordinal() * 24 + now.hour();

        {
            let cache = self.cached_system_messages.read();
            if let Some((cached_hour, cached_include_context, cached_include_tools, ref msgs)) =
                *cache
                && cached_hour == hour_key
                && cached_include_context == self.include_context_in_system
                && cached_include_tools == tool_cfg.include_tool_descriptions
            {
                return msgs.clone();
            }
        }

        let cached_tools = self.get_cached_tools_description(tool_cfg.include_tool_descriptions);
        let msgs = message_builder::build_system_message(
            &self.system_prompt,
            &self.current_dir,
            &self.tools,
            tool_cfg.include_tool_descriptions,
            None,
            cached_tools.as_deref(),
            self.include_context_in_system,
            &self.skill_index,
        );
        *self.cached_system_messages.write() = Some((
            hour_key,
            self.include_context_in_system,
            tool_cfg.include_tool_descriptions,
            msgs.clone(),
        ));
        msgs
    }
}
