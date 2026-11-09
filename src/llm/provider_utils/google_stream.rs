//! Shared Google Gemini streaming accumulator for function-call chunks.

use crate::llm::provider_utils::StreamingToolCallAccumulator;
use crate::messages::StreamEvent;
use std::collections::HashMap;

/// Parses Google `generateContent` stream chunks into [`StreamEvent`]s.
pub struct GoogleStreamAccumulator {
    tool_call_accumulator: StreamingToolCallAccumulator,
    next_tool_index: u32,
    thought_signatures: HashMap<u32, Option<String>>,
    open_call: Option<(u32, String)>,
}

impl GoogleStreamAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tool_call_accumulator: StreamingToolCallAccumulator::new(),
            next_tool_index: 0,
            thought_signatures: HashMap::new(),
            open_call: None,
        }
    }

    fn resolve_index(&mut self, name: &str) -> u32 {
        if let Some((index, open_name)) = &self.open_call
            && open_name == name
        {
            return *index;
        }
        let index = self.next_tool_index;
        self.next_tool_index = self.next_tool_index.saturating_add(1);
        self.open_call = Some((index, name.to_owned()));
        index
    }

    pub fn feed_function_call(
        &mut self,
        name: String,
        args_json: String,
        thought_signature: Option<String>,
    ) {
        let index = self.resolve_index(name.as_str());
        if thought_signature.is_some() {
            self.thought_signatures.insert(index, thought_signature);
        }
        self.tool_call_accumulator.accumulate_cumulative(
            index,
            Some(format!("call_{index}")),
            Some(name),
            Some(args_json),
        );
    }

    pub fn finalize_all(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        for evt in self.tool_call_accumulator.finalize_all(false) {
            let evt = match evt {
                StreamEvent::ToolCallStart {
                    id,
                    name,
                    arguments,
                    thought_signature,
                    action_id,
                } if thought_signature.is_none() => {
                    let signature = id
                        .strip_prefix("call_")
                        .and_then(|raw| raw.parse::<u32>().ok())
                        .and_then(|index| self.thought_signatures.remove(&index))
                        .flatten();
                    StreamEvent::ToolCallStart {
                        id,
                        name,
                        arguments,
                        thought_signature: signature,
                        action_id,
                    }
                }
                other => other,
            };
            events.push(evt);
        }
        self.open_call = None;
        events
    }
}

impl Default for GoogleStreamAccumulator {
    fn default() -> Self {
        Self::new()
    }
}
