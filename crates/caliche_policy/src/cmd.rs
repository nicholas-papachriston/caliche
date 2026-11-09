//! cmd.exe denylist.
//!
//! `cmd /c` is parsed again. `%VAR%` stays in the argument text. A `/s` delete
//! or remove is denied. This is not a full cmd grammar.

use crate::action::ShellPlan;
use crate::analyzer::ShellAnalyzer;
use crate::error::PolicyError;
use crate::posture::ShellDialect;

/// cmd analyzer. Facts only. The policy crate decides the effect.
#[derive(Clone, Copy, Debug, Default)]
pub struct CmdShellAnalyzer;

impl ShellAnalyzer for CmdShellAnalyzer {
    fn dialect(&self) -> ShellDialect {
        ShellDialect::Cmd
    }

    fn analyze(&self, command: &str) -> Result<ShellPlan, PolicyError> {
        Ok(analyze_cmd(command, 0))
    }
}

fn analyze_cmd(command: &str, depth: u8) -> ShellPlan {
    if depth > 3 {
        return opaque_plan();
    }
    let pieces = match split_pieces(command) {
        Ok(pieces) => pieces,
        Err(()) => return opaque_plan(),
    };
    let mut denied = false;
    let mut segments = 0_u32;
    for piece in pieces {
        let words = match split_words(&piece) {
            Ok(words) => words,
            Err(()) => return opaque_plan(),
        };
        if words.is_empty() {
            continue;
        }
        if let Some(script) = nested_script(&words) {
            let inner = analyze_cmd(&script, depth + 1);
            denied = denied || inner.denied;
            segments = segments.saturating_add(inner.segment_count);
            if inner.opaque {
                return ShellPlan {
                    opaque: true,
                    segment_count: segments,
                    denied,
                };
            }
            continue;
        }
        let exe = token_name(&words[0]);
        segments = segments.saturating_add(1);
        if is_recursive_delete(&exe, &words[1..]) {
            denied = true;
        }
    }
    ShellPlan {
        opaque: false,
        segment_count: segments,
        denied,
    }
}

fn opaque_plan() -> ShellPlan {
    ShellPlan {
        opaque: true,
        segment_count: 0,
        denied: false,
    }
}

fn token_name(token: &str) -> String {
    basename(token).to_ascii_lowercase()
}

fn basename(path: &str) -> &str {
    path.rsplit(['/', '\\'])
        .find(|part| !part.is_empty())
        .unwrap_or(path)
}

fn is_cmd(name: &str) -> bool {
    name.trim_end_matches(".exe") == "cmd"
}

fn is_recursive_delete(exe: &str, args: &[String]) -> bool {
    let name = exe.trim_end_matches(".exe");
    if !matches!(name, "del" | "erase" | "rd" | "rmdir") {
        return false;
    }
    args.iter().any(|arg| {
        let lower = arg.to_ascii_lowercase();
        lower == "/s" || lower.starts_with("/s:")
    })
}

fn nested_script(words: &[String]) -> Option<String> {
    if !is_cmd(&token_name(&words[0])) {
        return None;
    }
    let mut index = 1;
    while index < words.len() {
        let flag = words[index].to_ascii_lowercase();
        if flag == "/c" || flag == "/k" {
            let rest = words[index + 1..].join(" ");
            if rest.is_empty() {
                return None;
            }
            return Some(rest);
        }
        index += 1;
    }
    None
}

fn split_pieces(input: &str) -> Result<Vec<String>, ()> {
    let chars: Vec<char> = input.chars().collect();
    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut index = 0;
    let mut quote = false;
    while index < chars.len() {
        let ch = chars[index];
        if ch == '^' && !quote {
            if let Some(next) = chars.get(index + 1).copied() {
                current.push(next);
                index += 2;
                continue;
            }
        }
        if ch == '"' {
            quote = !quote;
            current.push(ch);
            index += 1;
            continue;
        }
        if !quote && is_separator(&chars, index) {
            push_piece(&mut pieces, &mut current);
            index += separator_width(&chars, index);
            continue;
        }
        current.push(ch);
        index += 1;
    }
    if quote {
        return Err(());
    }
    push_piece(&mut pieces, &mut current);
    Ok(pieces)
}

fn is_separator(chars: &[char], index: usize) -> bool {
    matches!(
        (chars[index], chars.get(index + 1).copied()),
        ('&', Some('&')) | ('|', Some('|')) | ('&', _) | ('|', _) | ('\n', _)
    )
}

fn separator_width(chars: &[char], index: usize) -> usize {
    match (chars[index], chars.get(index + 1).copied()) {
        ('&', Some('&')) | ('|', Some('|')) => 2,
        _ => 1,
    }
}

fn push_piece(pieces: &mut Vec<String>, current: &mut String) {
    let text = std::mem::take(current);
    if !text.trim().is_empty() {
        pieces.push(text);
    }
}

fn split_words(input: &str) -> Result<Vec<String>, ()> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote = false;
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch == '^' && !quote {
            if let Some(next) = chars.next() {
                current.push(next);
            }
            continue;
        }
        if ch == '"' {
            quote = !quote;
            current.push(ch);
            continue;
        }
        if ch.is_whitespace() && !quote {
            push_word(&mut words, &mut current);
            continue;
        }
        current.push(ch);
    }
    if quote {
        return Err(());
    }
    push_word(&mut words, &mut current);
    Ok(words)
}

fn push_word(words: &mut Vec<String>, current: &mut String) {
    let word = std::mem::take(current);
    let trimmed = word.trim();
    if trimmed.is_empty() {
        return;
    }
    words.push(unquote(trimmed));
}

fn unquote(word: &str) -> String {
    let bytes = word.as_bytes();
    if bytes.len() >= 2 && bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"' {
        return word[1..word.len() - 1].to_owned();
    }
    word.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::Effect;
    use crate::policy::PolicyBuilder;
    use crate::posture::Preset;
    use crate::principal::{ApprovalWaiter, HumanChannel, Principal};
    use crate::reason::ReasonCode;
    use crate::request::{Request, SessionScope};

    fn decide(command: &str) -> crate::Decision {
        let policy = PolicyBuilder::preset(Preset::Standard)
            .build()
            .expect("standard");
        let plan = policy
            .analyze_shell(&CmdShellAnalyzer, command)
            .expect("facet")
            .expect("plan");
        let action = crate::action::Action::tool(
            crate::action::ToolClass::Shell,
            "cmd",
            crate::action::ToolFacts::Shell {
                dialect: ShellDialect::Cmd,
                plan: Some(plan),
            },
        );
        let principal = Principal::human("operator", HumanChannel::CliTty);
        let session = SessionScope::new("cmd");
        let args = serde_json::json!({"command": command});
        policy.decide(&Request::new(
            &principal,
            ApprovalWaiter::Present(HumanChannel::CliTty),
            &session,
            action,
            &args,
        ))
    }

    #[test]
    fn trusted_preset_does_not_call_the_cmd_analyzer() {
        let trusted = PolicyBuilder::preset(Preset::Trusted)
            .build()
            .expect("trusted");
        assert!(
            trusted
                .analyze_shell(&CmdShellAnalyzer, "del /s /q C:\\temp")
                .is_none()
        );
    }

    #[test]
    fn fixtures_deny_under_standard() {
        let fixtures = [
            "del /s /q %USERPROFILE%",
            r"rd /s /q C:\",
            r"cmd /c del /s /q C:\temp",
        ];
        for command in fixtures {
            let decision = decide(command);
            assert_eq!(decision.effect, Effect::Deny, "{command}");
            assert_eq!(decision.reason.code(), ReasonCode::CmdDenylist, "{command}");
        }
    }

    #[test]
    fn echo_is_not_on_the_denylist() {
        let decision = decide("echo hi");
        assert_eq!(decision.effect, Effect::Ask);
        assert_ne!(decision.reason.code(), ReasonCode::CmdDenylist);
    }
}
