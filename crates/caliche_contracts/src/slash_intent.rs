//! Shared slash-command intent parsing for CLI/UI hosts (ROADMAP 2.10).

use crate::slash::{SlashSurface, find_slash_command};
use serde::{Deserialize, Serialize};

/// High-traffic slash intents shared across CLI and UI hosts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "intent")]
pub enum SlashIntent {
    Plan,
    Ask,
    Agent {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        arg: Option<String>,
    },
    Mode,
    ShowPlan,
    Resume {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
    },
    Attach {
        #[serde(default)]
        paths: Vec<String>,
    },
}

/// Parse shared host intents from a slash-command line.
///
/// Returns `None` for non-slash input or commands outside this shared set so
/// hosts can fall through to surface-specific matching.
#[must_use]
pub fn parse_slash_intent(input: &str) -> Option<SlashIntent> {
    let input = input.trim();
    if !input.starts_with('/') {
        return None;
    }

    let parts: Vec<&str> = input.split_whitespace().collect();
    let raw = parts.first().copied().unwrap_or("");
    let command = find_slash_command(raw, SlashSurface::Cli)
        .or_else(|| find_slash_command(raw, SlashSurface::Ui))
        .map_or(raw, |spec| spec.primary_name);
    let args = parts.get(1..).unwrap_or(&[]);

    match command {
        "/plan" => Some(SlashIntent::Plan),
        "/ask" => Some(SlashIntent::Ask),
        "/agent" => Some(SlashIntent::Agent {
            arg: args.first().map(|s| (*s).to_string()),
        }),
        "/mode" => Some(SlashIntent::Mode),
        "/show-plan" | "/showplan" => Some(SlashIntent::ShowPlan),
        "/resume" => Some(SlashIntent::Resume {
            id: args.first().map(|s| (*s).to_string()),
        }),
        "/attach" => Some(SlashIntent::Attach {
            paths: args.iter().map(|s| (*s).to_string()).collect(),
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mode_intents() {
        assert_eq!(parse_slash_intent("/plan"), Some(SlashIntent::Plan));
        assert_eq!(parse_slash_intent("/ask"), Some(SlashIntent::Ask));
        assert_eq!(parse_slash_intent("/mode"), Some(SlashIntent::Mode));
        assert_eq!(
            parse_slash_intent("/show-plan"),
            Some(SlashIntent::ShowPlan)
        );
        assert_eq!(parse_slash_intent("/showplan"), Some(SlashIntent::ShowPlan));
        assert_eq!(
            parse_slash_intent("/agent"),
            Some(SlashIntent::Agent { arg: None })
        );
        assert_eq!(
            parse_slash_intent("/agent research"),
            Some(SlashIntent::Agent {
                arg: Some("research".into()),
            })
        );
    }

    #[test]
    fn parse_resume_and_attach() {
        assert_eq!(
            parse_slash_intent("/resume"),
            Some(SlashIntent::Resume { id: None })
        );
        assert_eq!(
            parse_slash_intent("/resume abc123"),
            Some(SlashIntent::Resume {
                id: Some("abc123".into()),
            })
        );
        assert_eq!(
            parse_slash_intent("/attach ./a.png ./b.pdf"),
            Some(SlashIntent::Attach {
                paths: vec!["./a.png".into(), "./b.pdf".into()],
            })
        );
    }

    #[test]
    fn returns_none_for_unshared_commands() {
        assert_eq!(parse_slash_intent("hello"), None);
        assert_eq!(parse_slash_intent("/help"), None);
        assert_eq!(parse_slash_intent("/provider openai"), None);
    }

    #[test]
    fn attach_with_zero_paths_and_agent_keeps_first_arg_only() {
        assert_eq!(
            parse_slash_intent("/attach"),
            Some(SlashIntent::Attach { paths: vec![] })
        );
        assert_eq!(
            parse_slash_intent("/agent research deep dive"),
            Some(SlashIntent::Agent {
                arg: Some("research".into()),
            })
        );
    }
}
