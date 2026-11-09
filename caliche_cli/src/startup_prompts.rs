use crate::storage::{ScopeRecord, SecureCredentialManager, env_scanner};

fn interactive_startup_prompts_disabled() -> bool {
    std::env::var("CALICHE_DISABLE_STARTUP_PROMPTS")
        .ok()
        .as_deref()
        == Some("1")
}

/// Scan `.env` files for provider credentials that are not yet in the keychain.
///
/// Returns a banner notice instead of blocking stdin. Interactive ingest lives
/// on `/creds import` so the REPL prompt can appear immediately.
pub(super) fn env_credential_ingest_notice(cm: &SecureCredentialManager) -> Option<String> {
    if interactive_startup_prompts_disabled() {
        return None;
    }

    let cwd = std::env::current_dir().ok()?;
    let detected = env_scanner::scan_for_credentials(&cwd, false);
    if detected.is_empty() {
        return None;
    }

    let global_existing = cm.get_scope_credentials("global");
    let new_detected: Vec<_> = detected
        .into_iter()
        .filter(|d| !global_existing.contains_key(&d.provider_name))
        .collect();

    if new_detected.is_empty() {
        return None;
    }

    let names: Vec<&str> = new_detected
        .iter()
        .map(|d| d.provider_name.as_str())
        .collect();
    Some(format!(
        "Detected credentials in .env ({names}). Run /creds import to save them to the keychain.",
        names = names.join(", ")
    ))
}

/// When `caliche` starts from a directory with no credentials, check the keychain for
/// credentials stored under other scopes and offer the user a way to use them.
///
/// Returns `true` if recovery succeeded (credentials are now available), `false`
/// if no usable credentials were found anywhere.
pub(super) async fn try_recover_credentials(cm: &SecureCredentialManager) -> bool {
    if interactive_startup_prompts_disabled() {
        return false;
    }

    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| "this directory".to_string());

    let scopes = cm.list_scopes().await;
    let scopes_with_creds: Vec<_> = scopes
        .iter()
        .filter(|s| !s.providers.is_empty() && s.hash != cm.current_scope_hash())
        .collect();

    if scopes_with_creds.is_empty() {
        return false;
    }

    print_recovery_menu(&cwd, &scopes_with_creds);
    let Some(choice) = read_recovery_choice(scopes_with_creds.len()) else {
        return false;
    };
    apply_recovery_choice(cm, &scopes_with_creds, &choice).await
}

fn print_recovery_menu(cwd: &str, scopes_with_creds: &[&ScopeRecord]) {
    println!(
        "\n\x1b[33mNo credentials configured for \"{}\".\x1b[0m",
        cwd
    );
    println!("Found credentials in other locations:\n");

    for (i, scope) in scopes_with_creds.iter().enumerate() {
        let label = scope.label.as_deref().unwrap_or(scope.path.as_str());
        println!(
            "  \x1b[1m[{}]\x1b[0m {} — {}",
            i + 1,
            label,
            scope.providers.join(", ")
        );
    }

    println!("\n  \x1b[1m[a]\x1b[0m All — use all and promote to global");
    println!("  \x1b[1m[n]\x1b[0m Skip — run setup instead");
    print!("\nChoice [1");
    if scopes_with_creds.len() > 1 {
        print!("-{}", scopes_with_creds.len());
    }
    print!(" a n]: ");
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

fn recovery_choice_is_valid(choice: &str, scope_count: usize) -> bool {
    let valid_numeric = (1..=scope_count).any(|i| choice == i.to_string());
    valid_numeric
        || choice.is_empty()
        || choice == "a"
        || choice == "all"
        || (choice == "1" && scope_count == 1)
        || choice == "n"
        || choice == "no"
        || choice == "skip"
}

fn read_recovery_choice(scope_count: usize) -> Option<String> {
    loop {
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            return None;
        }
        let choice = input.trim().to_lowercase();
        if recovery_choice_is_valid(&choice, scope_count) {
            return Some(choice);
        }
        let opts = if scope_count > 1 {
            format!("1-{}, a, or n", scope_count)
        } else {
            "1, a, or n".to_string()
        };
        println!("\x1b[33mInvalid choice. Please enter {}.\x1b[0m", opts);
        print!("Choice: ");
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
}

fn should_promote_all(choice: &str, scope_count: usize) -> bool {
    choice.is_empty() || choice == "a" || choice == "all" || (choice == "1" && scope_count == 1)
}

async fn apply_recovery_choice(
    cm: &SecureCredentialManager,
    scopes_with_creds: &[&ScopeRecord],
    choice: &str,
) -> bool {
    if should_promote_all(choice, scopes_with_creds.len()) {
        return promote_scopes(
            cm,
            scopes_with_creds,
            "Credentials promoted to global keychain. Now available from any directory.",
        )
        .await;
    }

    let Ok(idx) = choice.parse::<usize>() else {
        return false;
    };
    if idx < 1 || idx > scopes_with_creds.len() {
        return false;
    }
    let chosen = scopes_with_creds[idx - 1];
    let label = chosen.label.as_deref().unwrap_or(chosen.path.as_str());
    let message = format!(
        "Credentials from \"{}\" promoted to global keychain.",
        label
    );
    promote_scopes(cm, &[chosen], &message).await
}

async fn promote_scopes(
    cm: &SecureCredentialManager,
    scopes: &[&ScopeRecord],
    success_message: &str,
) -> bool {
    let mut promoted = false;
    for scope in scopes {
        let creds = cm.get_scope_credentials(&scope.hash);
        for (provider, key) in creds {
            if cm.set_api_key(&provider, &key).await.is_ok() {
                promoted = true;
            }
        }
    }
    if !promoted {
        return false;
    }
    cm.apply_to_credential_store().await;
    println!("\x1b[32m{success_message}\x1b[0m\n");
    true
}
