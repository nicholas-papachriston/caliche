//! `caliche skills` installs and updates `SKILL.md` packages.

use std::path::PathBuf;

use anyhow::{Context, Result};
use caliche::agent::skills::{
    SkillUpdateAction, SkillUpdatePreview, apply_updates, install_source, parse_remote_source,
    preview_updates, skill_freshness,
};

use crate::commands::SkillsAction;

/// Run `skills install`, `skills update`, or `skills list`.
pub fn handle_skills(action: SkillsAction) -> Result<()> {
    match action {
        SkillsAction::Install { source, global } => install(&source, global),
        SkillsAction::Update {
            names,
            apply,
            global,
        } => update(&names, apply, global),
        SkillsAction::List { global } => list(global),
    }
}

fn install(source: &str, global: bool) -> Result<()> {
    let parsed = parse_remote_source(source)?;
    let root = skill_root(global)?;
    let reports = install_source(&parsed, &root)?;
    for report in reports {
        println!(
            "installed {} {} from {}",
            report.name,
            short_hash(&report.computed_hash),
            parsed.owner_repo
        );
    }
    Ok(())
}

fn update(names: &[String], apply: bool, global: bool) -> Result<()> {
    let root = skill_root(global)?;
    if apply {
        return apply_named(&root, names);
    }
    review_named(&root, names)
}

fn review_named(root: &std::path::Path, names: &[String]) -> Result<()> {
    let previews = preview_updates(root, names)?;
    if previews.is_empty() {
        println!("no installed skills");
        return Ok(());
    }
    let mut pending = 0;
    for preview in &previews {
        print_preview(preview);
        if preview.changed {
            pending += 1;
        }
    }
    if pending == 0 {
        println!("no skill updates");
        return Ok(());
    }
    println!("Review the diff above.");
    println!("Run the same command with --apply to accept it.");
    Ok(())
}

fn print_preview(preview: &SkillUpdatePreview) {
    if !preview.changed {
        println!("{} is current", preview.name);
        return;
    }
    println!(
        "{} {} -> {}",
        preview.name,
        short_hash(&preview.locked_hash),
        short_hash(&preview.remote_hash)
    );
    if let (Some(locked), Some(remote)) = (&preview.locked_git_sha, &preview.remote_git_sha)
        && locked != remote
    {
        println!("git {} -> {}", short_hash(locked), short_hash(remote));
    }
    if preview.diff.is_empty() {
        println!("files match");
        return;
    }
    print!("{}", preview.diff);
}

fn apply_named(root: &std::path::Path, names: &[String]) -> Result<()> {
    let actions = apply_updates(root, names)?;
    if actions.is_empty() {
        println!("no installed skills");
        return Ok(());
    }
    for action in actions {
        match action {
            SkillUpdateAction::Current { name } => println!("{name} is current"),
            SkillUpdateAction::Updated(report) => {
                println!(
                    "updated {} {}",
                    report.name,
                    short_hash(&report.computed_hash)
                );
            }
        }
    }
    Ok(())
}

fn list(global: bool) -> Result<()> {
    let root = skill_root(global)?;
    let lock_path = root.join("skills-lock.json");
    if !lock_path.is_file() {
        println!("no installed skills");
        return Ok(());
    }
    let text = std::fs::read_to_string(&lock_path)
        .with_context(|| format!("read {}", lock_path.display()))?;
    let lock: caliche::agent::skills::SkillLock =
        serde_json::from_str(&text).context("parse skill lock")?;
    if lock.skills.is_empty() {
        println!("no installed skills");
        return Ok(());
    }
    for (name, entry) in &lock.skills {
        let freshness = skill_freshness(entry, &root.join(name), None)?;
        let state = match freshness {
            caliche::agent::skills::SkillFreshness::Current => "current",
            caliche::agent::skills::SkillFreshness::Missing => "missing",
            caliche::agent::skills::SkillFreshness::ContentChanged { .. } => "changed",
            caliche::agent::skills::SkillFreshness::GitChanged { .. } => "git-moved",
        };
        println!(
            "{name}\t{state}\t{}\t{}",
            short_hash(&entry.computed_hash),
            entry.source
        );
    }
    Ok(())
}

fn skill_root(global: bool) -> Result<PathBuf> {
    if global {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .context("home directory is not set")?;
        return Ok(PathBuf::from(home).join(".caliche").join("skills"));
    }
    Ok(PathBuf::from(".caliche").join("skills"))
}

fn short_hash(hash: &str) -> &str {
    let end = hash.len().min(12);
    &hash[..end]
}
