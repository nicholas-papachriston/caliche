//! Shell-related handlers: completions and man pages.

use crate::commands::{Args, CompletionShell, ManAction};
use anyhow::Result;
use std::io::Write;

/// Handle shell completion generation.
pub fn handle_completions_command(shell: CompletionShell, output: Option<String>) -> Result<()> {
    use clap::CommandFactory;
    use clap_complete::{Shell, generate};

    let shell_type: Shell = match shell {
        CompletionShell::Bash => Shell::Bash,
        CompletionShell::Zsh => Shell::Zsh,
        CompletionShell::Fish => Shell::Fish,
        CompletionShell::PowerShell => Shell::PowerShell,
        CompletionShell::Elvish => Shell::Elvish,
    };

    let mut cmd = Args::command();
    let mut buf = Vec::new();
    generate(shell_type, &mut cmd, "caliche", &mut buf);

    if let Some(output_path) = output {
        std::fs::write(&output_path, &buf)?;
        println!("Completions written to: {}", output_path);
    } else {
        std::io::stdout().write_all(&buf)?;
    }

    Ok(())
}

/// Handle man page generation and install.
pub fn handle_man_command(action: ManAction) -> Result<()> {
    use clap::CommandFactory;
    use clap_mangen::Man;

    let cmd = Args::command();

    match action {
        ManAction::Generate { output, all } => {
            if all {
                let output_dir = output
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| std::path::PathBuf::from("."));
                std::fs::create_dir_all(&output_dir)?;
                let written = write_all_man_pages(&cmd, &output_dir)?;
                println!("Wrote {} man page(s) to {}", written, output_dir.display());
            } else {
                let mut buf = Vec::new();
                Man::new(cmd).render(&mut buf)?;
                if let Some(output_path) = output {
                    std::fs::write(&output_path, &buf)?;
                    println!("Man page written to: {}", output_path);
                } else {
                    std::io::stdout().write_all(&buf)?;
                }
            }
        }
        ManAction::Install { dir } => {
            let install_dir = dir.unwrap_or_else(|| "/usr/local/share/man/man1".to_string());
            let install_dir_path = std::path::PathBuf::from(&install_dir);
            std::fs::create_dir_all(&install_dir_path)?;
            let written = write_all_man_pages(&cmd, &install_dir_path)?;
            println!(
                "Installed {} man page(s) to {}",
                written,
                install_dir_path.display()
            );
        }
    }

    Ok(())
}

/// Recursively render man pages for `cmd` and all reachable subcommands into
/// `output_dir`. Returns the number of files written.
fn write_all_man_pages(cmd: &clap::Command, output_dir: &std::path::Path) -> Result<usize> {
    use clap_mangen::Man;

    fn walk(
        cmd: &clap::Command,
        prefix: &str,
        output_dir: &std::path::Path,
        count: &mut usize,
    ) -> Result<()> {
        let name = if prefix.is_empty() {
            cmd.get_name().to_string()
        } else {
            format!("{}-{}", prefix, cmd.get_name())
        };
        let file_path = output_dir.join(format!("{}.1", name));
        let mut buf = Vec::new();
        let renamed: clap::Command = cmd.clone().bin_name(name.clone());
        Man::new(renamed)
            .title(name.to_uppercase())
            .render(&mut buf)?;
        std::fs::write(&file_path, &buf)?;
        *count += 1;
        for sub in cmd.get_subcommands() {
            if sub.get_name() == "help" {
                continue;
            }
            walk(sub, &name, output_dir, count)?;
        }
        Ok(())
    }

    let mut count = 0usize;
    walk(cmd, "", output_dir, &mut count)?;
    Ok(count)
}
