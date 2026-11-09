//! Vault command handler -- directory-scoped credential management.

use crate::commands::VaultAction;
use crate::commands::credentials_service::CredentialService;
use crate::storage::env_scanner;
use anyhow::Result;

pub async fn handle_vault_command(action: VaultAction) -> Result<()> {
    match action {
        VaultAction::Status => handle_status().await,
        VaultAction::List => handle_list().await,
        VaultAction::Show { scope } => handle_show(scope).await,
        VaultAction::Link {
            source_dir,
            provider,
        } => handle_link(source_dir, provider).await,
        VaultAction::Unlink {
            source_dir,
            provider,
        } => handle_unlink(source_dir, provider).await,
        VaultAction::Import { dir, parents } => handle_import(dir, parents).await,
        VaultAction::Promote { provider } => handle_promote(provider).await,
    }
}

async fn handle_status() -> Result<()> {
    let service = CredentialService::new().await?;
    let status = service.vault_status().await;

    println!("\nCredential Vault Status");
    println!("=======================");
    println!("CWD: {}\n", status.cwd);

    if status.sources.is_empty() && status.env_providers.is_empty() {
        println!("  No credentials available for this directory.");
        println!("  Run 'caliche config vault import' to load credentials from a .env file,");
        println!("  or 'caliche config setup' to configure a provider globally.");
        return Ok(());
    }

    let mut all_reachable: Vec<String> = Vec::new();

    for source in &status.sources {
        let marker = if source.is_linked { " [linked]" } else { "" };
        println!("  {}{}", source.label, marker);
        for provider in &source.providers {
            let desc = CredentialService::provider_description(provider);
            println!("    {} -- {}", provider, desc);
            if !all_reachable.contains(provider) {
                all_reachable.push(provider.clone());
            }
        }
        println!();
    }

    if !status.env_providers.is_empty() {
        println!("  Environment variables (shell-level):");
        for provider in &status.env_providers {
            println!("    {}", provider);
            if !all_reachable.contains(provider) {
                all_reachable.push(provider.clone());
            }
        }
        println!();
    }

    println!(
        "  {} provider(s) reachable: {}\n",
        all_reachable.len(),
        all_reachable.join(", ")
    );

    Ok(())
}

async fn handle_list() -> Result<()> {
    let service = CredentialService::new().await?;
    let scopes = service.vault_list_scopes().await;

    if scopes.is_empty() {
        println!("No credential scopes found.");
        println!(
            "Use 'caliche config vault import' or 'caliche config set <provider> --key ...' to add credentials."
        );
        return Ok(());
    }

    println!("\nCredential Scopes");
    println!("=================\n");

    for scope in &scopes {
        let label = scope.label.as_deref().unwrap_or(scope.path.as_str());
        println!("  {}", label);
        if scope.hash != "global" && label != scope.path {
            println!("    Path:     {}", scope.path);
        }
        println!("    Hash:     {}", &scope.hash[..12.min(scope.hash.len())]);
        if !scope.providers.is_empty() {
            println!("    Providers: {}", scope.providers.join(", "));
        } else {
            println!("    Providers: (none)");
        }
        if !scope.updated_at.is_empty() {
            println!("    Updated:   {}", scope.updated_at);
        }
        println!();
    }

    Ok(())
}

async fn handle_show(scope_arg: Option<String>) -> Result<()> {
    let service = CredentialService::new().await?;

    let scope_hash = match scope_arg {
        Some(ref path) => CredentialService::scope_hash_for_path(path),
        None => service.current_scope_hash().to_string(),
    };

    let creds = service.vault_scope_credentials(&scope_hash);

    if creds.is_empty() {
        let label = scope_arg.as_deref().unwrap_or("current directory");
        println!("No credentials stored in scope: {}", label);
        return Ok(());
    }

    let label = scope_arg.as_deref().unwrap_or_else(|| {
        service
            .manager()
            .current_scope()
            .map(|scope| scope.path.as_str())
            .unwrap_or("global")
    });

    println!("\nCredentials in scope: {}", label);
    println!("{}", "-".repeat(40));

    for (provider, key) in &creds {
        println!(
            "  {:12} {}",
            provider,
            CredentialService::mask_for_display(key)
        );
    }

    println!();
    Ok(())
}

async fn handle_link(source_dir: String, provider: Option<String>) -> Result<()> {
    let source_path = CredentialService::resolve_dir(&source_dir)?;
    let source_str = source_path.to_string_lossy().to_string();

    let service = CredentialService::new().await?;
    let prov = provider.as_deref().unwrap_or("*");
    service.vault_link(&source_str, prov).await?;

    let target_label = service.current_scope_path();
    let prov_display = if prov == "*" {
        "all providers".to_string()
    } else {
        format!("provider '{}'", prov)
    };

    println!(
        "Linked {} from \"{}\" into \"{}\".",
        prov_display, source_str, target_label
    );
    println!("Run 'caliche config vault status' to verify.");

    Ok(())
}

async fn handle_unlink(source_dir: String, provider: Option<String>) -> Result<()> {
    let source_path = CredentialService::resolve_dir(&source_dir)?;
    let source_str = source_path.to_string_lossy().to_string();

    let service = CredentialService::new().await?;
    let removed = service
        .vault_unlink(&source_str, provider.as_deref())
        .await?;

    if removed {
        println!("Unlinked \"{}\" from current directory.", source_str);
    } else {
        println!("No link found for \"{}\" in current directory.", source_str);
    }

    Ok(())
}

async fn handle_import(dir_arg: Option<String>, check_parents: bool) -> Result<()> {
    let dir = match dir_arg {
        Some(ref path) => CredentialService::resolve_dir(path)?,
        None => std::env::current_dir()?,
    };

    let detected = env_scanner::scan_for_credentials(&dir, check_parents);

    if detected.is_empty() {
        println!(
            "No provider credentials found in .env files under \"{}\".",
            dir.display()
        );
        println!(
            "Checked for: {}",
            CredentialService::env_file_names_display()
        );
        return Ok(());
    }

    println!(
        "\nFound {} credential(s) in \"{}\":\n",
        detected.len(),
        dir.display()
    );
    for item in &detected {
        println!(
            "  {} ({}) from {}",
            item.provider_name,
            item.env_var,
            item.source_path.display()
        );
    }

    let service = CredentialService::new_with_scope(Some(&dir)).await?;
    let scope_label = service.current_scope_path();

    println!("\nWhere should these be stored in the keychain?");
    println!(
        "  \x1b[1m[G]\x1b[0m Globally — available from any directory \x1b[2m(recommended)\x1b[0m"
    );
    println!("  \x1b[1m[D]\x1b[0m This directory only — {}", scope_label);
    println!("  \x1b[1m[N]\x1b[0m Cancel");
    print!("Choice [G/d/n]: ");

    use std::io::Write;
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    match input.trim().to_lowercase().as_str() {
        "" | "g" | "global" | "y" | "yes" => {
            let global = CredentialService::new().await?;
            let stored = global.vault_import_global(&detected).await?;
            println!(
                "\x1b[32m{} credential(s) stored globally in keychain.\x1b[0m",
                stored
            );
        }
        "d" | "dir" | "directory" => {
            let stored = service.vault_import_directory(&detected).await?;
            println!(
                "\x1b[32m{} credential(s) stored in keychain for \"{}\".\x1b[0m",
                stored, scope_label
            );
        }
        _ => {
            println!("\x1b[2mImport cancelled.\x1b[0m");
        }
    }

    Ok(())
}

async fn handle_promote(provider: String) -> Result<()> {
    let service = CredentialService::new().await?;
    let current_scope_label = service.current_scope_path();

    if service.current_scope_hash() == "global" {
        println!("Already in the global scope -- nothing to promote.");
        return Ok(());
    }

    match service.vault_promote(&provider).await? {
        true => {
            println!(
                "\x1b[32mMoved '{}' credentials from \"{}\" to the global scope.\x1b[0m",
                provider, current_scope_label
            );
            println!("The credential is now available from any directory.");
        }
        false => {
            println!(
                "No '{}' credential found in scope \"{}\".",
                provider, current_scope_label
            );
            println!("Use 'caliche config vault show' to see what's stored here.");
        }
    }

    Ok(())
}
