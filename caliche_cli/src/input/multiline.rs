//! Multi-line input capture.

/// Returns the complete multi-line text when user presses Ctrl+D or enters an empty line twice
pub fn read_multiline(prompt: &str) -> std::io::Result<String> {
    use std::io::{self, BufRead, Write};

    println!(
        "\x1b[33mEntering multi-line mode. Press Ctrl+D or enter empty line twice to submit.\x1b[0m"
    );
    println!("\x1b[2m{}\x1b[0m...", prompt);

    let stdin = io::stdin();
    let mut lines = Vec::new();
    let mut empty_count = 0;

    loop {
        print!("\x1b[2m{:>3} |\x1b[0m ", lines.len() + 1);
        io::stdout().flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                // EOF (Ctrl+D)
                break;
            }
            Ok(_) => {
                let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
                if trimmed.is_empty() {
                    empty_count += 1;
                    if empty_count >= 2 {
                        // Two consecutive empty lines - submit
                        break;
                    }
                    lines.push(String::new());
                } else {
                    empty_count = 0;
                    lines.push(trimmed.to_string());
                }
            }
            Err(e) => return Err(e),
        }
    }

    // Remove trailing empty lines
    while lines.last().map(|s| s.is_empty()).unwrap_or(false) {
        lines.pop();
    }

    println!("\x1b[2m--- End of input ({} lines) ---\x1b[0m", lines.len());

    Ok(lines.join("\n"))
}

/// Multi-line input state
#[derive(Debug, Clone, Default)]
pub struct MultiLineState {
    /// Whether multi-line mode is enabled
    pub enabled: bool,
    /// Buffer for accumulating lines
    pub buffer: Vec<String>,
}

impl MultiLineState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Toggle multi-line mode
    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
        if !self.enabled {
            self.buffer.clear();
        }
    }

    /// Add a line to the buffer
    pub fn add_line(&mut self, line: String) {
        self.buffer.push(line);
    }

    /// Get the complete text and clear the buffer
    pub fn take_text(&mut self) -> String {
        let text = self.buffer.join("\n");
        self.buffer.clear();
        text
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty() || self.buffer.iter().all(|s| s.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::MultiLineState;

    #[test]
    fn take_text_joins_lines_and_clears_the_buffer() {
        let mut state = MultiLineState::new();
        state.add_line("one".to_owned());
        state.add_line("two".to_owned());
        assert_eq!(state.take_text(), "one\ntwo");
        assert!(state.is_empty());
    }

    #[test]
    fn toggle_off_clears_the_buffer() {
        let mut state = MultiLineState::new();
        state.toggle();
        state.add_line("kept while enabled".to_owned());
        state.toggle();
        assert!(!state.enabled);
        assert!(state.is_empty());
    }

    #[test]
    fn blank_lines_count_as_empty() {
        let mut state = MultiLineState::new();
        state.add_line(String::new());
        assert!(state.is_empty());
    }
}
