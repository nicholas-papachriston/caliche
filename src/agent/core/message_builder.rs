use crate::messages::{Message, MessageRole};
use chrono::Datelike;
use chrono::Timelike;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::sync::Arc;

/// Cached system prompt components that only change hourly.
struct CachedDateStamp {
    hour_key: u32,
    formatted: Arc<str>, // Use Arc<str> instead of String for efficient sharing
}

static DATE_CACHE: parking_lot::Mutex<Option<CachedDateStamp>> = parking_lot::Mutex::new(None);

/// Get the formatted date string, caching for up to 1 hour.
fn get_cached_date_stamp() -> Arc<str> {
    let now = chrono::Utc::now();
    let hour_key = now.date_naive().ordinal() * 24 + now.hour();

    let mut cache = DATE_CACHE.lock();
    if let Some(ref cached) = *cache
        && cached.hour_key == hour_key
    {
        return cached.formatted.clone();
    }

    let formatted: Arc<str> = Arc::from(
        now.format("%Y-%m-%d %H:00 UTC")
            .to_string()
            .into_boxed_str(),
    );
    *cache = Some(CachedDateStamp {
        hour_key,
        formatted: formatted.clone(),
    });
    formatted
}

/// Minimal default system prompt for API/benchmark parity with direct SDK calls.
/// Keeps token count low when no custom prompt is configured.
#[must_use]
pub const fn default_system_prompt(_current_dir: &str) -> &'static str {
    "You are a helpful AI assistant."
}

/// Tool call format instructions to append when tools are registered (text-based tool providers).
const TOOL_CALL_FORMAT: &str = r#" When you need to use tools, prefer parallel tool calls for speed:
- Single: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
- Parallel (preferred): <function_calls>[{"name": "fs", "arguments": {"op":"read","path": "a.rs"}}, {"name": "fs", "arguments": {"op":"read","path": "b.rs"}}]</function_calls>
- Batch independent tools in one round; speculative prefetch when you anticipate needing multiple files.
- For real build/test/lint/install commands, use bash (e.g. {"name":"bash","arguments":{"command":"bun test","working_directory":"."}})."#;

/// Active-tool guidelines for the coding default set (`fs` + `bash`).
const CODING_TOOL_GUIDELINES: &str = r"
Guidelines:
- Explore the codebase with fs (op=list/glob/grep/read). Prefer grep/glob before broad reads.
- Use bash only for project commands (build/test/lint/git/install/scripts). Do not use bash for file search or reads.
- Be concise; keep going until the task is done.";

/// Appended only when `web_fetch` is registered. Not a browser.
const WEB_FETCH_GUIDELINE: &str = "
- Use web_fetch to read a public http(s) URL as text. Order is TinyFish when TINYFISH_API_KEY is set, then Firecrawl when FIRECRAWL_API_KEY is set, then plain HTTP. With no fetch keys, only plain HTTP runs. web_fetch is not a browser and does not run page JavaScript. Do not fetch private or metadata hosts.";

pub fn build_system_message(
    system_prompt: &str,
    current_dir: &str,
    tools: &crate::agent::tools::ToolRegistry,
    include_tool_descriptions: bool,
    workspace_context: Option<&str>,
    cached_tools_description: Option<&str>,
    include_context_in_system: bool,
    skill_index: &[crate::agent::skills::SkillIndexEntry],
) -> Vec<Message> {
    // Only append date/time and cwd when tools are enabled (useful for file ops, bash, etc.).
    // For pure chat agents, keep system prompt minimal to reduce tokens and response length.
    let base_prompt = if system_prompt.trim().is_empty() {
        "You are a helpful AI assistant."
    } else {
        system_prompt
    };

    let has_tools = !tools.is_empty();

    // Fast path: no tools and no workspace context — skip date cache and formatting.
    // Skill index is only injected when `fs` is registered, so ignore it here.
    if !has_tools && !include_tool_descriptions && workspace_context.is_none() {
        return vec![Message {
            role: MessageRole::System,
            content: crate::messages::MessageContent::text(base_prompt.to_string()),
            metadata: None,
        }];
    }

    // Append tool call format / coding guidelines only when tools are registered.
    let prompt_with_tools = if has_tools {
        let mut prompt = base_prompt.to_string();
        if include_tool_descriptions {
            prompt.push_str(TOOL_CALL_FORMAT);
        }
        if has_coding_default_tools(tools) {
            prompt.push_str(CODING_TOOL_GUIDELINES);
        }
        if tools
            .get(crate::agent::tools::names::TOOL_WEB_FETCH)
            .is_some()
        {
            prompt.push_str(WEB_FETCH_GUIDELINE);
        }
        // Progressive disclosure requires `fs` so the model can load SKILL.md.
        // Do not inject the skill catalog when tools are absent or `fs` is missing.
        if !skill_index.is_empty()
            && tools
                .get(crate::agent::tools::names::TOOL_FILESYSTEM)
                .is_some()
        {
            prompt.push_str(&crate::agent::skills::format_skills_for_prompt(skill_index));
        }
        prompt
    } else {
        base_prompt.to_string()
    };
    let append_date_cwd = include_context_in_system && (include_tool_descriptions || has_tools);
    let system_content = if append_date_cwd {
        let current_date = get_cached_date_stamp();
        format!(
            "{prompt_with_tools}\n\nCurrent date and time: {current_date}\nCurrent working directory: {current_dir}"
        )
    } else {
        prompt_with_tools
    };

    let mut messages = vec![Message {
        role: MessageRole::System,
        content: crate::messages::MessageContent::text(system_content),
        metadata: None,
    }];

    if let Some(ctx) = workspace_context
        && !ctx.is_empty()
    {
        messages.push(Message {
            role: MessageRole::System,
            content: crate::messages::MessageContent::text(format!(
                "Relevant workspace context (use this to guide your answers):\n{ctx}"
            )),
            metadata: None,
        });
    }

    // Only include tool descriptions in system message if not using native tools API.
    // Native tools API sends schemas separately; prose lists use short snippets only.
    if include_tool_descriptions && has_tools {
        let tools_description =
            cached_tools_description.map_or_else(|| describe_tools(tools), String::from);
        messages.push(Message {
            role: MessageRole::System,
            content: crate::messages::MessageContent::text(format!(
                "Available tools:\n{tools_description}"
            )),
            metadata: None,
        });
    }

    messages
}

/// Build chat messages from system config and a single user text message.
///
/// **No preprocessing** is applied for media attachments or URLs: the user input is used
/// as plain text only. The types in [`crate::messages::MessageContent`] (e.g. `Multimodal`,
/// `ImageUrl`, `VideoUrl`) and related runners are in place so that callers can add
/// media/URL handling or preprocessing in a later layer without changing this function.
#[allow(clippy::too_many_arguments)]
pub fn build_messages(
    system_prompt: &str,
    current_dir: &str,
    tools: &crate::agent::tools::ToolRegistry,
    user_message: &str,
    include_tool_descriptions: bool,
    workspace_context: Option<&str>,
    cached_tools_description: Option<&str>,
    include_context_in_system: bool,
    skill_index: &[crate::agent::skills::SkillIndexEntry],
) -> Vec<Message> {
    let mut messages = build_system_message(
        system_prompt,
        current_dir,
        tools,
        include_tool_descriptions,
        workspace_context,
        cached_tools_description,
        include_context_in_system,
        skill_index,
    );

    messages.push(Message {
        role: MessageRole::User,
        content: crate::messages::MessageContent::text(user_message.to_string()),
        metadata: None,
    });

    messages
}

fn has_coding_default_tools(tools: &crate::agent::tools::ToolRegistry) -> bool {
    tools
        .get(crate::agent::tools::names::TOOL_FILESYSTEM)
        .is_some()
        && tools.get(crate::agent::tools::names::TOOL_BASH).is_some()
}

/// Short one-line tool listing for text-tool providers (pi-style snippets; no param dump).
pub fn describe_tools(tools: &crate::agent::tools::ToolRegistry) -> String {
    describe_tools_where(tools, |_name| true)
}

pub fn describe_tools_filtered(
    tools: &crate::agent::tools::ToolRegistry,
    allowed_tool_names: &HashSet<String>,
) -> String {
    describe_tools_where(tools, |name| allowed_tool_names.contains(name))
}

fn describe_tools_where(
    tools: &crate::agent::tools::ToolRegistry,
    keep: impl Fn(&str) -> bool,
) -> String {
    let mut out = String::new();
    for (name, description, _params) in tools.list().iter().filter(|(name, ..)| keep(name)) {
        if !out.is_empty() {
            out.push('\n');
        }
        let _ = write!(out, "- {name}: {}", tool_prompt_snippet(name, description));
    }
    out
}

fn tool_prompt_snippet<'a>(name: &str, description: &'a str) -> beef::lean::Cow<'a, str> {
    match name {
        "fs" => beef::lean::Cow::borrowed(
            "Filesystem: read/list/glob/grep/write/edit/delete. Prefer grep/glob before broad reads.",
        ),
        "bash" => beef::lean::Cow::borrowed(
            "Shell for build/test/lint/git/install. Never for file search/read (use fs).",
        ),
        "git" => beef::lean::Cow::borrowed("Git status/diff/log/branch/commit via structured ops."),
        "http" => {
            beef::lean::Cow::borrowed("HTTP request/fetch for remote URLs (not local files).")
        }
        "web_fetch" => beef::lean::Cow::borrowed(
            "Fetch a public URL as text. Auto: TinyFish (key), Firecrawl (key), then plain HTTP. Not a browser.",
        ),
        _ => {
            let first_line = description.lines().next().unwrap_or(description).trim();
            if first_line.chars().count() <= 140 {
                beef::lean::Cow::borrowed(first_line)
            } else {
                let truncated: String = first_line.chars().take(137).collect();
                beef::lean::Cow::owned(format!("{truncated}..."))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::skills::SkillIndexEntry;
    use crate::agent::tools::ToolRegistry;

    #[test]
    fn skill_index_not_injected_without_tools() {
        let tools = ToolRegistry::new();
        let skills = vec![SkillIndexEntry {
            name: "review".to_string(),
            description: "Code review".to_string(),
            location: "/tmp/SKILL.md".to_string(),
        }];
        let messages = build_system_message(
            "You are a helpful AI assistant.",
            "/tmp",
            &tools,
            false,
            None,
            None,
            false,
            &skills,
        );
        assert_eq!(messages.len(), 1);
        let text = messages[0].content.as_text().unwrap_or_default();
        assert!(
            !text.contains("<available_skills>"),
            "skill catalog must not inflate tool-less prompts: {text}"
        );
    }

    #[test]
    fn web_fetch_guideline_is_added_only_when_registered() {
        let without = ToolRegistry::new();
        let bare = build_system_message("base", "/tmp", &without, false, None, None, false, &[]);
        let bare_text = bare[0].content.as_text().unwrap_or_default();
        assert!(!bare_text.contains("web_fetch"));

        let with_fetch = ToolRegistry::new();
        with_fetch.register(Arc::new(crate::agent::tools::WebFetchTool::from_env()));
        let messages =
            build_system_message("base", "/tmp", &with_fetch, false, None, None, false, &[]);
        let text = messages[0].content.as_text().unwrap_or_default();
        assert!(text.contains("web_fetch"));
        assert!(text.contains("not a browser"));
    }
}
