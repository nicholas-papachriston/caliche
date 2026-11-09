//! Status bar component - workspace context, git state, and session info.

use dioxus::prelude::*;

use crate::api::{ApiClient, WorkspaceContextInfo};

/// Props for the StatusBar component
#[derive(Props, Clone, PartialEq)]
pub struct StatusBarProps {
    /// Session ID (None if no session yet)
    pub session_id: Option<String>,
    /// API client for fetching context
    pub api_client: ApiClient,
}

/// A compact status bar showing workspace context below the header.
#[component]
pub fn StatusBar(props: StatusBarProps) -> Element {
    let mut context: Signal<Option<WorkspaceContextInfo>> = use_signal(|| None);
    let mut loading = use_signal(|| false);
    let mut last_loaded_session_id = use_signal(|| None::<String>);

    // Fetch context once per session id change.
    let session_id_for_effect = props.session_id.clone();
    let api = props.api_client.clone();

    use_effect(move || {
        let sid = session_id_for_effect.clone();
        if sid == *last_loaded_session_id.read() {
            return;
        }

        last_loaded_session_id.set(sid.clone());

        if let Some(sid) = sid {
            loading.set(true);
            let api = api.clone();
            spawn(async move {
                match api.get_context(&sid).await {
                    Ok(ctx) => context.set(Some(ctx)),
                    Err(_) => context.set(None),
                }
                loading.set(false);
            });
        } else {
            context.set(None);
            loading.set(false);
        }
    });

    let ctx = context.read();

    // Only show when we have context info
    let Some(ref ctx) = *ctx else {
        return rsx! {};
    };

    let branch = ctx
        .git_context
        .as_ref()
        .map(|g| g.branch.clone())
        .unwrap_or_default();

    let git_changes = ctx.git_context.as_ref().map(|g| {
        let staged = g.staged_count;
        let modified = g.modified_count;
        (staged, modified, g.has_changes)
    });

    let project_type = ctx.project_type.as_deref().unwrap_or("");
    let tracked_count = ctx.tracked_files.len();

    // Workspace root - just basename
    let workspace = std::path::Path::new(&ctx.workspace_root)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&ctx.workspace_root);

    rsx! {
        div {
            class: "status-bar",
            style: "display: flex; align-items: center; gap: 2ch; padding: 0.2em 1ch; background: var(--secondary-bg); border-bottom: 1px solid var(--border-color); font-size: var(--font-size-xs); color: var(--muted-color); flex-shrink: 0; overflow: hidden; white-space: nowrap;",

            // Workspace root
            span { style: "color: var(--accent); overflow: hidden; text-overflow: ellipsis; max-width: 20ch;",
                title: "{ctx.workspace_root}",
                "{workspace}"
            }

            // Project type
            if !project_type.is_empty() {
                span { style: "color: var(--border-color);", "·" }
                span { style: "color: var(--info);", "{project_type}" }
            }

            // Git info
            if !branch.is_empty() {
                span { style: "color: var(--border-color);", "·" }
                span { style: "display: flex; align-items: center; gap: 0.5ch;",
                    span { style: "color: var(--muted-color);", "⎇" }
                    span { style: "color: var(--foreground);", "{branch}" }
                    if let Some((staged, modified, has_changes)) = git_changes {
                        if has_changes {
                            span { style: "color: var(--warning);",
                                if staged > 0 && modified > 0 {
                                    "+{staged} ~{modified}"
                                } else if staged > 0 {
                                    "+{staged}"
                                } else if modified > 0 {
                                    "~{modified}"
                                }
                            }
                        }
                    }
                }
            }

            // Tracked files
            if tracked_count > 0 {
                span { style: "color: var(--border-color);", "·" }
                span { title: "files tracked in this session",
                    "{tracked_count} tracked"
                }
            }
        }
    }
}
