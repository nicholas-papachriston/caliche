//! Reusable loop detection primitives for streamed agent output.

use std::collections::{HashMap, VecDeque};

/// Default minimum substantial lines before text loop detection is evaluated.
pub const DEFAULT_TEXT_LOOP_MIN_LINES: usize = 6;
/// Default minimum non-whitespace line length considered for repetition.
pub const DEFAULT_TEXT_LOOP_LINE_MIN_LEN: usize = 15;
/// Default number of repeated lines required before we treat output as a loop.
pub const DEFAULT_TEXT_LOOP_REPEAT_THRESHOLD: usize = 5;
/// Default maximum bytes retained in the text buffer.
pub const DEFAULT_TEXT_LOOP_BUFFER_MAX: usize = 3000;
/// Default repeated tool signatures required to detect a tool loop.
pub const DEFAULT_TOOL_LOOP_REPEAT_THRESHOLD: usize = 4;
/// Default bounded history size for tool signature tracking.
pub const DEFAULT_TOOL_LOOP_HISTORY_MAX: usize = 8;

/// Detect repeated textual output loops in a streamed response.
#[derive(Debug, Clone)]
pub struct TextLoopDetector {
    buffer: String,
    min_lines: usize,
    line_min_len: usize,
    repeat_threshold: usize,
    buffer_max: usize,
}

impl TextLoopDetector {
    #[must_use]
    pub fn new_default() -> Self {
        Self::new(
            DEFAULT_TEXT_LOOP_MIN_LINES,
            DEFAULT_TEXT_LOOP_LINE_MIN_LEN,
            DEFAULT_TEXT_LOOP_REPEAT_THRESHOLD,
            DEFAULT_TEXT_LOOP_BUFFER_MAX,
        )
    }

    #[must_use]
    pub const fn new(
        min_lines: usize,
        line_min_len: usize,
        repeat_threshold: usize,
        buffer_max: usize,
    ) -> Self {
        Self {
            buffer: String::new(),
            min_lines,
            line_min_len,
            repeat_threshold,
            buffer_max,
        }
    }

    #[must_use]
    pub const fn repeat_threshold(&self) -> usize {
        self.repeat_threshold
    }

    pub fn on_text(&mut self, text: &str) {
        self.buffer.push_str(text);
        if self.buffer.len() > self.buffer_max {
            let excess = self.buffer.len() - self.buffer_max / 2;
            let mut drop_to = excess;
            while drop_to < self.buffer.len() && !self.buffer.is_char_boundary(drop_to) {
                drop_to += 1;
            }
            self.buffer.drain(..drop_to);
        }
    }

    pub fn on_tool_call(&mut self) {
        self.buffer.clear();
    }

    pub fn detect(&self) -> Option<String> {
        let lines: Vec<&str> = self
            .buffer
            .lines()
            .map(str::trim)
            .filter(|l| l.len() >= self.line_min_len)
            .collect();

        if lines.len() < self.min_lines {
            return None;
        }

        let mut counts: HashMap<&str, usize> = HashMap::new();
        for &line in &lines {
            let count = counts.entry(line).and_modify(|c| *c += 1).or_insert(1);
            if *count >= self.repeat_threshold {
                return Some(line.to_string());
            }
        }
        None
    }
}

/// Detect repeated tool-call signature loops in a single turn.
#[derive(Debug, Clone)]
pub struct ToolLoopDetector {
    recent: VecDeque<String>,
    repeat_threshold: usize,
    history_max: usize,
}

impl ToolLoopDetector {
    #[must_use]
    pub fn new_default() -> Self {
        Self::new(
            DEFAULT_TOOL_LOOP_REPEAT_THRESHOLD,
            DEFAULT_TOOL_LOOP_HISTORY_MAX,
        )
    }

    #[must_use]
    pub const fn new(repeat_threshold: usize, history_max: usize) -> Self {
        Self {
            recent: VecDeque::new(),
            repeat_threshold,
            history_max,
        }
    }

    #[must_use]
    pub const fn repeat_threshold(&self) -> usize {
        self.repeat_threshold
    }

    pub fn on_tool_call(&mut self, name: &str, args: &serde_json::Value) -> Option<String> {
        let sig = format!("{name}:{args}");
        self.recent.push_back(sig.clone());
        while self.recent.len() > self.history_max {
            self.recent.pop_front();
        }

        let repeats = self.recent.iter().filter(|s| *s == &sig).count();
        if repeats >= self.repeat_threshold {
            return Some(sig);
        }
        None
    }

    pub fn reset(&mut self) {
        self.recent.clear();
    }
}
