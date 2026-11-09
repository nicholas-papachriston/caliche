//! Shared helpers for lightweight task decomposition.

/// Build the planning prompt used to break a complex task into 1-5 sub-tasks.
pub fn planning_prompt(task: &str) -> String {
    format!(
        "Break down the following task into 1-5 concrete sub-tasks. \
         Reply only with a numbered list, one sub-task per line (e.g. '1. First do X'). \
         Task: {}",
        task
    )
}

/// Parse numbered sub-tasks from a planning response (e.g. `1. Do X`).
pub fn parse_subtasks(response: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in response.lines() {
        let trimmed = line.trim();
        if let Some((num, rest)) = trimmed.split_once('.') {
            if num.trim().chars().all(|c: char| c.is_ascii_digit()) && !rest.trim().is_empty() {
                out.push(rest.trim().to_string());
            }
        }
    }
    out
}

/// Heuristic: true if the message looks like a multi-step task (long or action verbs).
pub fn is_complex_message(message: &str) -> bool {
    let words = message.split_whitespace().count();
    if words > 50 {
        return true;
    }
    let lower = message.to_lowercase();
    let action_verbs = [
        "refactor",
        "implement",
        "add",
        "fix",
        "create",
        "write",
        "update",
        "replace",
        "move",
        "extract",
        "introduce",
        "remove",
        "rewrite",
    ];
    action_verbs.iter().any(|v| lower.contains(v))
}
