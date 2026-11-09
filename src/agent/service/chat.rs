use super::{AgentService, Arc, Context, MemoryStrategy, Message, PathBuf, SessionStore};

impl AgentService {
    /// Chat with structured output and automatic deserialization
    ///
    /// This method enforces a JSON schema and automatically retries on validation failure.
    /// Returns the deserialized type and token usage.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentService, AgentServiceConfig, ProviderType};
    /// use serde::Deserialize;
    /// use serde_json::json;
    ///
    /// #[derive(Deserialize)]
    /// struct Person {
    ///     name: String,
    ///     age: u32,
    /// }
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let mut agent = AgentService::from_config(
    ///     AgentServiceConfig::new()
    ///         .with_provider(Some(ProviderType::OpenAI), Some("gpt-4o".to_string()))
    ///         .with_all_default_tools(),
    /// ).await?;
    ///
    /// let schema = json!({
    ///     "type": "object",
    ///     "properties": {
    ///         "name": {"type": "string"},
    ///         "age": {"type": "integer"}
    ///     },
    ///     "required": ["name", "age"]
    /// });
    ///
    /// let (person, usage) = agent.chat_structured::<Person>(
    ///     "Extract person info: John is 30 years old",
    ///     schema,
    /// ).await?;
    ///
    /// println!("Name: {}, Age: {}", person.name, person.age);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn chat_structured<T: serde::de::DeserializeOwned>(
        &mut self,
        input: &str,
        schema: serde_json::Value,
    ) -> anyhow::Result<(T, Option<crate::messages::TokenUsage>)> {
        use crate::messages::{CompletionRequest, Message};
        use crate::output::{
            JsonSchemaDefinition, ResponseFormat, StructuredOutputConfig, StructuredOutputEnforcer,
        };
        use std::sync::Arc;

        // Sanitize type name to match OpenAI's pattern: ^[a-zA-Z0-9_-]+$
        let type_name = std::any::type_name::<T>();
        let sanitized_name = type_name
            .split("::")
            .last()
            .unwrap_or(type_name)
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .collect::<String>();

        // Ensure it's not empty and starts with a letter or underscore
        let schema_name = if sanitized_name.is_empty()
            || !sanitized_name
                .chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
        {
            "Response".to_string()
        } else {
            sanitized_name
        };

        // Normalize schema to ensure OpenAI compatibility (adds additionalProperties: false if needed)
        let normalized_schema = JsonSchemaDefinition::normalize_schema(schema);

        // Create response format with schema
        let format = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaDefinition {
                name: schema_name,
                description: None,
                schema: normalized_schema,
                strict: true,
            },
        };

        // Create enforcer
        let enforcer = StructuredOutputEnforcer::new(StructuredOutputConfig::new(format))?;

        // Build messages
        let mut messages = vec![Message::user(input.to_string())];
        if let Some(memory) = &self.session.memory {
            let history = memory
                .read()
                .await
                .get_context_arc()
                .await
                .context("Failed to gather memory context")?;
            if !history.is_empty() {
                let history_messages =
                    crate::agent::core::message_history::sanitize_history_for_request(
                        history.iter().cloned(),
                    );
                messages.splice(0..0, history_messages);
            }
        }

        // Add system prompt if present
        if !self.agent.system_prompt.is_empty() {
            messages.insert(0, Message::system(self.agent.system_prompt.clone()));
        }

        // Create completion request
        let request = CompletionRequest {
            messages: Arc::new(messages),
            max_tokens: Some(4096),
            temperature: Some(0.7),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None, // Will be set by enforcer
        };

        // Execute with schema enforcement
        let provider_ref = self.agent.llm_provider();
        let response = enforcer
            .complete_with_schema(&**provider_ref, request)
            .await?;

        // Deserialize
        let parsed: T = serde_json::from_str(response.content.as_text())
            .map_err(|e| anyhow::anyhow!("Failed to deserialize structured output: {e}"))?;

        // Update memory if present
        if let Some(memory) = &self.session.memory {
            memory
                .write()
                .await
                .add_messages(vec![
                    Message::user(input.to_string()),
                    Message::assistant(response.content.as_text().to_string()),
                ])
                .await
                .context("Failed to update memory after structured chat")?;
        }

        Ok((parsed, response.usage))
    }

    pub async fn chat(&mut self, message: &str) -> anyhow::Result<crate::messages::ChatResponse> {
        tracing::info!(
            session_id = %self.session.id,
            model = %self.model,
            "agent_service_chat"
        );
        #[cfg(feature = "session-tools")]
        let _guard =
            crate::agent::tools::session_tools::CallerSessionIdGuard::new(self.session.id.clone());
        // If response_format is set, call provider directly to use it
        if let Some(ref response_format) = self.response_format {
            use crate::messages::{CompletionRequest, Message as MessageType};
            use std::sync::Arc;

            let history_arc = if let Some(memory) = &self.session.memory {
                Some(
                    memory
                        .read()
                        .await
                        .get_context_arc()
                        .await
                        .context("Failed to gather memory context")?,
                )
            } else {
                None
            };

            // Build messages
            let mut messages = vec![];
            if !self.agent.system_prompt.is_empty() {
                messages.push(MessageType::system(self.agent.system_prompt.clone()));
            }
            if let Some(ref history) = history_arc {
                messages.extend(
                    crate::agent::core::message_history::sanitize_history_for_request(
                        history.iter().cloned(),
                    ),
                );
            }
            messages.push(MessageType::user(message.to_string()));

            // Call provider directly with response_format
            let provider_ref = self.agent.llm_provider();
            let request = CompletionRequest {
                messages: Arc::new(messages),
                max_tokens: self.agent.max_tokens.or(Some(4096)),
                temperature: self.agent.temperature.or(Some(0.7)),
                reasoning_effort: self
                    .agent
                    .reasoning_effort_slot
                    .as_ref()
                    .and_then(|slot| slot.current()),
                tools: None,
                tool_choice: None,
                response_format: Some(response_format.clone()),
            };

            let completion_response = provider_ref
                .complete(request)
                .await
                .map_err(|e| anyhow::anyhow!("LLM completion failed: {e}"))?;

            let response_text = completion_response.content.as_text().to_string();
            let usage = completion_response.usage;
            // Review copies live on `ChatResponse`. Service memory stays
            // text-only so the next turn does not resend the trace.
            let reasoning = completion_response
                .reasoning_content
                .clone()
                .unwrap_or_default();

            // Update memory if present
            if let Some(memory) = &self.session.memory {
                memory
                    .write()
                    .await
                    .add_messages(vec![
                        MessageType::user(message.to_string()),
                        MessageType::assistant(response_text.clone()),
                    ])
                    .await
                    .context("Failed to update memory after chat")?;
            }

            return Ok(crate::messages::ChatResponse::new(
                response_text,
                usage.unwrap_or_default(),
                false, // No tools used when calling provider directly
            )
            .with_reasoning_content(reasoning));
        }

        // Normal path: use agent.chat() which handles tools, hooks, etc.
        let history_arc = if let Some(memory) = &self.session.memory {
            // Use get_context_arc() for zero-copy access when cached
            Some(
                memory
                    .read()
                    .await
                    .get_context_arc()
                    .await
                    .context("Failed to gather memory context")?,
            )
        } else {
            None
        };

        let mut last_error: Option<anyhow::Error> = None;
        let mut response_opt: Option<crate::messages::ChatResponse> = None;
        for _attempt in 0..self.retry_attempts {
            let attempt_result = if let Some(ref history) = history_arc {
                if history.is_empty() {
                    self.agent
                        .chat(message)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                } else {
                    self.agent
                        .chat_with_history_arc(message, std::sync::Arc::clone(history))
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))
                }
            } else {
                self.agent
                    .chat(message)
                    .await
                    .map_err(|e| anyhow::anyhow!("{e}"))
            };

            match attempt_result {
                Ok(resp) => {
                    response_opt = Some(resp);
                    break;
                }
                Err(err) => {
                    last_error = Some(err);
                }
            }
        }
        let response = if let Some(resp) = response_opt {
            resp
        } else {
            return Err(last_error.unwrap_or_else(|| anyhow::anyhow!("chat failed")));
        };

        if let Some(memory) = &self.session.memory {
            memory
                .write()
                .await
                .add_messages(vec![
                    Message::user(message),
                    Message::assistant(response.text()),
                ])
                .await
                .context("Failed to update memory after chat")?;
        }

        self.capture_plan_turn(message, response.text()).await;

        #[cfg(feature = "session-tools")]
        {
            let reg = crate::agent::session::live_registry::LiveSessionRegistry::global();
            reg.record_history(&self.session.id, "user", message);
            reg.record_history(&self.session.id, "assistant", response.text());
        }

        if let Some(store) = &self.session_store {
            self.session
                .save(store.as_ref())
                .await
                .context("Failed to autosave session after chat")?;
        }

        Ok(response)
    }

    pub fn set_retry_attempts(&mut self, attempts: usize) {
        self.retry_attempts = attempts.max(1);
    }

    /// Persist the current session snapshot when a session store is configured.
    pub async fn save_session_snapshot(&self) -> anyhow::Result<()> {
        if let Some(store) = &self.session_store {
            self.session
                .save(store.as_ref())
                .await
                .context("Failed to save session snapshot")?;
        }
        Ok(())
    }

    /// Persist session snapshots as rkyv `.cls1` files under `path`.
    pub fn enable_session_autosave(&mut self, path: PathBuf) -> anyhow::Result<()> {
        let store = crate::agent::session::BinaryFileSessionStore::new(path);
        self.session_store = Some(Arc::new(store) as Arc<dyn SessionStore>);
        Ok(())
    }

    /// Enable AgentFS for session storage, tool audit trail, and optional memory persistence.
    #[cfg(feature = "agentfs")]
    pub fn enable_agentfs(&mut self, agentfs: Arc<agentfs_sdk::AgentFS>) -> anyhow::Result<()> {
        let store = crate::agent::session::AgentFsSessionStore::new(Arc::clone(&agentfs));
        self.session_store = Some(Arc::new(store) as Arc<dyn SessionStore>);
        let logger = crate::agent::tools::AgentFsToolLogger::new(agentfs);
        self.agent.set_agentfs_tool_logger(Some(Arc::new(logger)));
        Ok(())
    }

    /// Process a chat message with conversation history.
    pub async fn chat_with_history(
        &mut self,
        message: &str,
        history: Vec<crate::messages::Message>,
    ) -> anyhow::Result<crate::messages::ChatResponse> {
        #[cfg(feature = "session-tools")]
        let _guard =
            crate::agent::tools::session_tools::CallerSessionIdGuard::new(self.session.id.clone());
        let response = self
            .agent
            .chat_with_history(message, history)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        if let Some(memory) = &self.session.memory {
            memory
                .write()
                .await
                .add_messages(vec![
                    Message::user(message),
                    Message::assistant(response.text()),
                ])
                .await
                .context("Failed to update memory after chat_with_history")?;
        }

        Ok(response)
    }

    /// Chat with multimodal attachments (images, audio, video, documents)
    pub async fn chat_with_attachments(
        &mut self,
        message: &str,
        attachments: Vec<crate::messages::Attachment>,
    ) -> anyhow::Result<crate::messages::ChatResponse> {
        #[cfg(feature = "session-tools")]
        let _guard =
            crate::agent::tools::session_tools::CallerSessionIdGuard::new(self.session.id.clone());
        use crate::messages::Message as MessageType;

        // Check if provider supports vision
        let provider_ref = self.agent.llm_provider();
        if !provider_ref.supports_vision() {
            return Err(anyhow::anyhow!(
                "Provider '{}' does not support vision/multimodal inputs",
                provider_ref.name()
            ));
        }

        let history_arc = if let Some(memory) = &self.session.memory {
            Some(
                memory
                    .read()
                    .await
                    .get_context_arc()
                    .await
                    .context("Failed to gather memory context")?,
            )
        } else {
            None
        };

        // Create multimodal message with attachments
        let user_message = MessageType::user_with_attachments(message, attachments);

        // For multimodal messages, we need to call the provider directly
        // since chat_with_history expects a string message
        // Build messages using the agent's internal structure

        use std::sync::Arc;

        // Get system prompt and build messages
        let mut messages = if self.agent.system_prompt.is_empty() {
            Vec::new()
        } else {
            vec![MessageType::system(self.agent.system_prompt.clone())]
        };

        if let Some(ref history) = history_arc {
            messages.extend(
                crate::agent::core::message_history::sanitize_history_for_request(
                    history.iter().cloned(),
                ),
            );
        }
        messages.push(user_message.clone());

        // Use step() method which handles tools properly
        let messages_arc = Arc::new(messages);
        let tool_cfg = self.agent.tool_calling_config();
        let query = crate::agent::core::Agent::extract_last_user_query(messages_arc.as_ref());
        let step_result = self
            .agent
            .step(
                messages_arc.clone(),
                &tool_cfg,
                None,
                query.as_ref(),
                1,
                &self.session.id,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Agent step failed: {e}"))?;

        // Convert Arc<String> to String for response
        let response_text =
            crate::agent::core::message_history::arc_unwrap_or_clone(step_result.response_text);
        // Collect tool names from tool_results
        let tools_executed: Vec<String> = step_result
            .tool_results
            .iter()
            .map(|(tool_name, _)| tool_name.clone())
            .collect();
        let response = crate::messages::ChatResponse::new(
            response_text,
            step_result.usage,
            !step_result.tool_calls.is_empty(),
        )
        .with_tools_executed(tools_executed);

        // Update memory if present
        if let Some(memory) = &self.session.memory {
            memory
                .write()
                .await
                .add_messages(vec![
                    user_message,
                    MessageType::assistant(response.text().to_string()),
                ])
                .await
                .context("Failed to update memory after chat_with_attachments")?;
        }

        Ok(response)
    }

    /// Chat with an image from a file path
    pub async fn chat_with_image_file(
        &mut self,
        message: &str,
        image_path: &std::path::Path,
    ) -> anyhow::Result<crate::messages::ChatResponse> {
        let attachment = crate::messages::Attachment::from_file(image_path).await?;
        self.chat_with_attachments(message, vec![attachment]).await
    }

    /// Execute a single step: one LLM call and its immediate tool executions.
    /// This allows reasoning patterns to control the execution loop themselves.
    ///
    /// The messages parameter should include the full conversation history up to this point.
    /// Returns a `StepResult` containing the response, tool calls, and updated history.
    ///
    /// Note: This method does NOT update the `AgentService`'s memory. Callers are responsible
    /// for managing their own conversation history, which is important for reasoning patterns
    /// that need isolated state (e.g., `TreeOfThoughts` with parallel branches).
    pub async fn step(
        &self,
        messages: std::sync::Arc<Vec<crate::messages::Message>>,
    ) -> anyhow::Result<crate::agent::core::chat::StepResult> {
        #[cfg(feature = "session-tools")]
        let _guard =
            crate::agent::tools::session_tools::CallerSessionIdGuard::new(self.session.id.clone());
        let tool_cfg = self.agent.tool_calling_config();
        let query =
            crate::agent::core::Agent::extract_last_user_query(messages.as_ref()).into_owned();
        self.agent
            .step(messages, &tool_cfg, None, &query, 1, &self.session.id)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::AgentService;
    use crate::agent::Agent;
    use crate::agent::memory::{
        ApproximateTokenCounter, BufferMemory, BufferMemoryConfig, MemoryStrategy,
    };
    use crate::agent::tools::ToolRegistry;
    use crate::llm::{Provider, ProviderType};
    use crate::messages::{CompletionContent, CompletionResponse, TokenUsage};
    use crate::testing::MockProvider;
    use anyhow::Context;
    use std::sync::Arc;

    fn reasoned(text: &str, reasoning: String) -> CompletionResponse {
        CompletionResponse {
            content: CompletionContent::Text {
                content: text.to_string(),
            },
            finish_reason: Some(beef::lean::Cow::borrowed("end_turn")),
            usage: Some(TokenUsage::default()),
            reasoning_content: Some(reasoning),
        }
    }

    /// Turn 2 must not carry turn 1's trace. Storing it on service memory
    /// makes `chars` for the second request equal the trace length.
    #[tokio::test]
    async fn live_memory_second_turn_carries_no_reasoning() -> anyhow::Result<()> {
        let trace = "r".repeat(25_500);
        let provider = Arc::new(MockProvider::with_responses(vec![
            reasoned("visible answer", trace.clone()),
            reasoned("second answer", "later trace".to_string()),
        ]));
        let owned = Arc::clone(&provider);
        let llm: Arc<dyn Provider> = owned;
        let agent = Agent::new(llm, Arc::new(ToolRegistry::new()));
        let mut service = AgentService::new_with_agent(agent, ProviderType::OpenAI)?;
        service.set_memory_strategy(Some(Box::new(BufferMemory::new(
            BufferMemoryConfig::default().with_max_tokens(16_000),
            Arc::new(ApproximateTokenCounter::default()),
        ))));

        let first = service.chat("turn one").await?;
        assert_eq!(first.reasoning_content(), trace);

        let after_first = provider.recorded_prompts().len();
        let second = service.chat("turn two").await?;
        assert_eq!(second.text(), "second answer");
        assert_eq!(second.reasoning_content(), "later trace");

        let prompts = provider.recorded_prompts();
        let chars = provider.recorded_reasoning_chars();
        assert!(
            prompts.len() > after_first && chars.len() == prompts.len(),
            "turn 2 made no provider call"
        );
        assert!(
            prompts[after_first..]
                .iter()
                .any(|prompt| prompt.contains("visible answer")),
            "turn 2 dropped turn 1 text, so a resend could hide: {prompts:?}"
        );
        assert!(
            chars[after_first..].iter().all(|count| *count == 0),
            "turn 2 carried reasoning chars {:?}",
            &chars[after_first..]
        );

        let memory = service.memory().context("memory installed")?;
        let stored = memory.read().await.get_context().await?;
        let stored_reasoning: usize = stored
            .iter()
            .map(|message| message.reasoning_content().map(str::len).unwrap_or(0))
            .sum();
        assert_eq!(stored_reasoning, 0, "service memory stored reasoning");
        Ok(())
    }
}
