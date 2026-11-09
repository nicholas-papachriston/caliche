//! Shared Cohere streaming tool-call parser.

use crate::llm::provider_utils::StreamingToolCallAccumulator;
use crate::messages::StreamEvent;

/// Parses Cohere tool-call stream chunks into [`StreamEvent`]s.
pub struct CohereStreamParser {
    tool_call_accumulator: StreamingToolCallAccumulator,
}

impl CohereStreamParser {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tool_call_accumulator: StreamingToolCallAccumulator::new(),
        }
    }

    pub const fn accumulator_mut(&mut self) -> &mut StreamingToolCallAccumulator {
        &mut self.tool_call_accumulator
    }

    /// Finalize pending tool calls, requiring arguments first, then flushing stragglers.
    pub fn flush_remaining(&mut self) -> Vec<StreamEvent> {
        let mut events = self.tool_call_accumulator.finalize_all(true);
        events.extend(self.tool_call_accumulator.finalize_all(false));
        events
    }

    pub fn finalize_index(&mut self, index: u32, require_arguments: bool) -> Option<StreamEvent> {
        self.tool_call_accumulator
            .finalize(index, require_arguments)
    }
}

impl Default for CohereStreamParser {
    fn default() -> Self {
        Self::new()
    }
}
