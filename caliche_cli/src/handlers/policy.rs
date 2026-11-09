//! `policy show` and `policy set preset`.

use std::path::PathBuf;

use caliche::policy::{
    default_policy_path, host_for_process, open_policy_file, parse_host, parse_preset,
    render_effective, write_policy_file,
};

use crate::commands::{PolicyAction, PolicySetAction};

/// Run a policy subcommand. The caller prints nothing else.
///
/// # Errors
///
/// Returns an error when the host label, preset, file, or write fails.
pub fn run(action: PolicyAction) -> Result<(), String> {
    match action {
        PolicyAction::Show { json, host, path } => show(json, host, path),
        PolicyAction::Set { action } => match action {
            PolicySetAction::Preset { name, path } => set_preset(&name, path),
        },
    }
}

fn show(json: bool, host: Option<String>, path: Option<String>) -> Result<(), String> {
    let view =
        open_policy_file(&policy_path(path), host_of(host)?).map_err(|err| err.to_string())?;
    if json {
        let text = serde_json::to_string_pretty(&view).map_err(|err| err.to_string())?;
        println!("{text}");
    } else {
        print!("{}", render_effective(&view));
    }
    Ok(())
}

fn set_preset(name: &str, path: Option<String>) -> Result<(), String> {
    parse_preset(name).map_err(|err| err.to_string())?;
    let path = policy_path(path);
    let host = default_host();
    let mut view = open_policy_file(&path, host).map_err(|err| err.to_string())?;
    view.document.preset = name.to_owned();
    let written = write_policy_file(&path, &view.document, host).map_err(|err| err.to_string())?;
    println!(
        "preset {} written to {}",
        written.preset, written.source_path
    );
    Ok(())
}

fn policy_path(path: Option<String>) -> PathBuf {
    path.map_or_else(default_policy_path, PathBuf::from)
}

fn default_host() -> caliche::policy::HostCapability {
    host_for_process(caliche_sandbox::landlock_tcp_enforceable())
}

fn host_of(host: Option<String>) -> Result<caliche::policy::HostCapability, String> {
    match host {
        None => Ok(default_host()),
        Some(token) => parse_host(&token).map_err(|err| err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::Args;
    use clap::Parser;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn policy_show_parses() {
        let args = Args::parse_from(["caliche", "policy", "show", "--json"]);
        assert!(matches!(
            args.command,
            Some(crate::commands::Commands::Policy {
                action: PolicyAction::Show { json: true, .. }
            })
        ));
    }

    #[test]
    fn set_preset_writes_the_named_file() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("caliche-policy-cli-{stamp}"));
        let path = dir.join("policy.toml");
        run(PolicyAction::Set {
            action: PolicySetAction::Preset {
                name: "standard".to_owned(),
                path: Some(path.display().to_string()),
            },
        })
        .expect("write");
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.contains("standard"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
