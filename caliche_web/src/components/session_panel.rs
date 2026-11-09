//! Session sidebar panel — memory inspector, checkpoint manager, diff viewer.

use dioxus::prelude::*;

use crate::api::{
    ApiClient, ContextItemsResponse, DiffChangeKind, DiffMessageSummary, DiffResponse,
    MemorySearchResponse, MemorySnapshot, MemoryStatsResponse, PlanResponse,
};

#[derive(Debug, Clone, PartialEq)]
enum PanelTab {
    Memory,
    Checkpoints,
    Diff,
    Plan,
    Context,
}

/// Checkpoint entry stored locally after creation.
#[derive(Debug, Clone)]
struct CheckpointEntry {
    id: String,
    label: String,
    created_at: String,
}

fn diff_change_label(change: DiffChangeKind) -> &'static str {
    match change {
        DiffChangeKind::Created => "created",
        DiffChangeKind::Modified => "modified",
        DiffChangeKind::Deleted => "deleted",
    }
}

fn diff_size_summary(before_len: Option<usize>, after_len: Option<usize>) -> String {
    match (before_len, after_len) {
        (Some(before), Some(after)) => format!("{before}B -> {after}B"),
        (None, Some(after)) => format!("new {after}B"),
        (Some(before), None) => format!("was {before}B"),
        (None, None) => "size unknown".to_string(),
    }
}

fn render_diff_entry(entry: &DiffMessageSummary, is_reverted: bool) -> Element {
    let message_id = entry.message_id.clone();
    let changed_files = entry.changed_files;
    let border_style = if is_reverted {
        "border: 1px dashed var(--border-color); padding: 0.5em 0.75ch; display: flex; flex-direction: column; gap: 0.35em;"
    } else {
        "border: 1px solid var(--border-color); padding: 0.5em 0.75ch; display: flex; flex-direction: column; gap: 0.35em;"
    };

    rsx! {
        div { style: border_style,
            div { style: "display: flex; justify-content: space-between; gap: 1ch; align-items: baseline; font-size: var(--font-size-xs);",
                span { style: "color: var(--foreground);", "{message_id}" }
                span { style: "color: var(--muted-color);", "{changed_files} file(s)" }
            }
            div { style: "display: flex; flex-direction: column; gap: 0.25em;",
                for snapshot in &entry.snapshots {
                    {
                        let label = diff_change_label(snapshot.change);
                        let size_summary = diff_size_summary(snapshot.before_len, snapshot.after_len);
                        rsx! {
                            div { style: "display: flex; justify-content: space-between; gap: 1ch; font-size: var(--font-size-xs);",
                                span { style: "color: var(--foreground); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;", title: "{snapshot.path}", "{snapshot.path}" }
                                span { style: "color: var(--muted-color); flex-shrink: 0;", "{label} • {size_summary}" }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Props, Clone, PartialEq)]
pub struct SessionPanelProps {
    /// Active session ID (None → panel shows empty state)
    pub session_id: Option<String>,
    pub api_client: ApiClient,
    /// Called when panel should be closed
    pub on_close: EventHandler<()>,
    /// Called after the user confirms "End session"; parent clears the chat
    /// and forgets the current session ID so a fresh one is created on the
    /// next turn.
    #[props(default)]
    pub on_end_session: EventHandler<String>,
}

/// Panel header text. A stored title wins. An untitled session keeps the
/// default label.
fn session_header_label(title: Option<&str>) -> String {
    match title {
        Some(title) if !title.is_empty() => title.to_owned(),
        _ => "session panel".to_owned(),
    }
}

/// Slide-in panel for session-scoped inspector tools.
#[component]
pub fn SessionPanel(props: SessionPanelProps) -> Element {
    let mut active_tab: Signal<PanelTab> = use_signal(|| PanelTab::Memory);

    let mut mem_stats: Signal<Option<MemoryStatsResponse>> = use_signal(|| None);
    let mut mem_search_query = use_signal(|| String::new());
    let mut mem_search_results: Signal<Option<MemorySearchResponse>> = use_signal(|| None);
    let mut mem_loading = use_signal(|| false);
    let mut mem_error: Signal<Option<String>> = use_signal(|| None);
    let mut mem_export_data: Signal<Option<String>> = use_signal(|| None);
    let mut mem_export_loading = use_signal(|| false);
    let mut mem_import_data = use_signal(|| String::new());
    let mut mem_import_loading = use_signal(|| false);
    let mut mem_import_status: Signal<Option<String>> = use_signal(|| None);

    let mut checkpoints: Signal<Vec<CheckpointEntry>> = use_signal(Vec::new);
    let mut cp_label = use_signal(|| String::new());
    let mut cp_busy = use_signal(|| false);
    let mut cp_error: Signal<Option<String>> = use_signal(|| None);
    let mut rollback_status: Signal<Option<String>> = use_signal(|| None);

    let mut diff: Signal<Option<DiffResponse>> = use_signal(|| None);
    let mut diff_loading = use_signal(|| false);
    let mut diff_error: Signal<Option<String>> = use_signal(|| None);
    let mut revert_status: Signal<Option<String>> = use_signal(|| None);

    // Plan tab
    let mut plan: Signal<Option<PlanResponse>> = use_signal(|| None);
    let mut plan_loading = use_signal(|| false);
    let mut plan_error: Signal<Option<String>> = use_signal(|| None);
    let mut plan_status: Signal<Option<String>> = use_signal(|| None);

    // Context tab
    let mut ctx_query = use_signal(String::new);
    let mut ctx_results: Signal<Option<ContextItemsResponse>> = use_signal(|| None);
    let mut ctx_loading = use_signal(|| false);
    let mut ctx_error: Signal<Option<String>> = use_signal(|| None);
    let mut project_info: Signal<Option<crate::api::WorkspaceContextInfo>> = use_signal(|| None);
    let mut project_loading = use_signal(|| false);
    let mut session_title: Signal<Option<String>> = use_signal(|| None);

    let sid_title = props.session_id.clone();
    let api_title = props.api_client.clone();
    use_effect(move || {
        let Some(sid) = sid_title.clone() else {
            session_title.set(None);
            return;
        };
        let api = api_title.clone();
        spawn(async move {
            match api.get_live_session(&sid).await {
                Ok(info) => session_title.set(info.title),
                Err(err) => {
                    tracing::debug!(
                        error = %err,
                        session_id = %sid,
                        "session panel title unavailable"
                    );
                    session_title.set(None);
                }
            }
        });
    });

    let sid_mem = props.session_id.clone();
    let api_mem = props.api_client.clone();
    use_effect(move || {
        if *active_tab.read() != PanelTab::Memory {
            return;
        }
        let Some(sid) = sid_mem.clone() else {
            return;
        };
        mem_loading.set(true);
        mem_error.set(None);
        let api = api_mem.clone();
        spawn(async move {
            match api.get_memory_stats(&sid).await {
                Ok(stats) => mem_stats.set(Some(stats)),
                Err(e) => mem_error.set(Some(e.to_string())),
            }
            mem_loading.set(false);
        });
    });

    let sid_diff = props.session_id.clone();
    let api_diff = props.api_client.clone();
    use_effect(move || {
        if *active_tab.read() != PanelTab::Diff {
            return;
        }
        let Some(sid) = sid_diff.clone() else {
            return;
        };
        diff_loading.set(true);
        diff_error.set(None);
        let api = api_diff.clone();
        spawn(async move {
            match api.get_diff(&sid).await {
                Ok(d) => diff.set(Some(d)),
                Err(e) => diff_error.set(Some(e.to_string())),
            }
            diff_loading.set(false);
        });
    });

    let Some(ref session_id) = props.session_id else {
        return rsx! {
            div { style: "width: 320px; border-left: 1px solid var(--border-color); background: var(--card-bg); display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--muted-color); font-size: var(--font-size-xs); gap: 1em; padding: 2em 1ch;",
                span { "no active session" }
                button {
                    style: "background: transparent; border: 1px solid var(--border-color); color: var(--muted-color); padding: 0.3em 1ch; cursor: pointer;",
                    onclick: move |_| props.on_close.call(()),
                    "[x]"
                }
            }
        };
    };
    let session_id = session_id.clone();

    let tab = active_tab.read().clone();
    let header_label = session_header_label(session_title.read().as_deref());

    rsx! {
        div {
            style: "width: 320px; min-width: 240px; border-left: 1px solid var(--border-color); background: var(--card-bg); display: flex; flex-direction: column; flex-shrink: 0; overflow: hidden;",

            // Panel header
            div {
                style: "display: flex; align-items: center; gap: 1ch; padding: 0.4em 1ch; border-bottom: 1px solid var(--border-color); flex-shrink: 0;",
                span {
                    style: "color: var(--accent); font-size: var(--font-size-xs); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;",
                    title: "{header_label}",
                    "{header_label}"
                }
                {
                    let sid_for_end = session_id.clone();
                    let api_for_end = props.api_client.clone();
                    let on_end = props.on_end_session.clone();
                    rsx! {
                        button {
                            style: "background: transparent; border: 1px solid var(--border-color); color: var(--error); cursor: pointer; padding: 0.2em 0.8ch; font-size: var(--font-size-xs); margin-right: 0.5ch;",
                            title: "End this session (deletes it on the server and clears the chat)",
                            onclick: move |_| {
                                let sid = sid_for_end.clone();
                                let api = api_for_end.clone();
                                let on_end = on_end.clone();
                                spawn(async move {
                                    if let Err(err) = api.delete_live_session(&sid).await {
                                        tracing::warn!(
                                            error = %err,
                                            session_id = %sid,
                                            "failed to delete session on backend"
                                        );
                                    }
                                    on_end.call(sid);
                                });
                            },
                            "[end session]"
                        }
                    }
                }
                button {
                    style: "background: transparent; border: none; color: var(--muted-color); cursor: pointer; padding: 0.2em 0.5ch; font-size: var(--font-size-xs);",
                    title: "Close panel",
                    onclick: move |_| props.on_close.call(()),
                    "[x]"
                }
            }

            // Tab strip
            div {
                style: "display: flex; border-bottom: 1px solid var(--border-color); flex-shrink: 0;",
                {
                    let t = tab.clone();
                    rsx! {
                        button {
                            style: if t == PanelTab::Memory { "flex: 1; padding: 0.4em; font-size: var(--font-size-xs); background: var(--secondary-bg); color: var(--accent); border: none; border-bottom: 2px solid var(--accent); cursor: pointer;" } else { "flex: 1; padding: 0.4em; font-size: var(--font-size-xs); background: transparent; color: var(--muted-color); border: none; cursor: pointer;" },
                            onclick: move |_| active_tab.set(PanelTab::Memory),
                            "memory"
                        }
                        button {
                            style: if t == PanelTab::Checkpoints { "flex: 1; padding: 0.4em; font-size: var(--font-size-xs); background: var(--secondary-bg); color: var(--accent); border: none; border-bottom: 2px solid var(--accent); cursor: pointer;" } else { "flex: 1; padding: 0.4em; font-size: var(--font-size-xs); background: transparent; color: var(--muted-color); border: none; cursor: pointer;" },
                            onclick: move |_| active_tab.set(PanelTab::Checkpoints),
                            "checkpoints"
                        }
                        button {
                            style: if t == PanelTab::Diff { "flex: 1; padding: 0.4em; font-size: var(--font-size-xs); background: var(--secondary-bg); color: var(--accent); border: none; border-bottom: 2px solid var(--accent); cursor: pointer;" } else { "flex: 1; padding: 0.4em; font-size: var(--font-size-xs); background: transparent; color: var(--muted-color); border: none; cursor: pointer;" },
                            onclick: move |_| active_tab.set(PanelTab::Diff),
                            "diff"
                        }
                        button {
                            style: if t == PanelTab::Plan { "flex: 1; padding: 0.4em; font-size: var(--font-size-xs); background: var(--secondary-bg); color: var(--accent); border: none; border-bottom: 2px solid var(--accent); cursor: pointer;" } else { "flex: 1; padding: 0.4em; font-size: var(--font-size-xs); background: transparent; color: var(--muted-color); border: none; cursor: pointer;" },
                            onclick: move |_| active_tab.set(PanelTab::Plan),
                            "plan"
                        }
                        button {
                            style: if t == PanelTab::Context { "flex: 1; padding: 0.4em; font-size: var(--font-size-xs); background: var(--secondary-bg); color: var(--accent); border: none; border-bottom: 2px solid var(--accent); cursor: pointer;" } else { "flex: 1; padding: 0.4em; font-size: var(--font-size-xs); background: transparent; color: var(--muted-color); border: none; cursor: pointer;" },
                            onclick: move |_| active_tab.set(PanelTab::Context),
                            "context"
                        }
                    }
                }
            }

            // Tab body
            div { style: "flex: 1; overflow-y: auto; padding: 1ch; display: flex; flex-direction: column; gap: 0.75em;",

                match tab {
                    PanelTab::Memory => {
                        rsx! {
                            // Stats
                            if *mem_loading.read() {
                                span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "loading..." }
                            } else if let Some(ref e) = *mem_error.read() {
                                span { style: "color: var(--error-color); font-size: var(--font-size-xs);", "{e}" }
                            } else if let Some(ref stats) = *mem_stats.read() {
                                {
                                    let entry_count = stats.entry_count;
                                    let token_str = stats.total_tokens
                                        .map(|t| format!("{} tokens", t))
                                        .unwrap_or_else(|| "—".to_string());
                                    let summary = stats.summary.clone().unwrap_or_default();
                                    rsx! {
                                        div { style: "display: flex; flex-direction: column; gap: 0.3em; font-size: var(--font-size-xs);",
                                            div { style: "display: flex; justify-content: space-between;",
                                                span { style: "color: var(--muted-color);", "entries" }
                                                span { style: "color: var(--foreground);", "{entry_count}" }
                                            }
                                            div { style: "display: flex; justify-content: space-between;",
                                                span { style: "color: var(--muted-color);", "tokens" }
                                                span { style: "color: var(--foreground);", "{token_str}" }
                                            }
                                            if !summary.is_empty() {
                                                div { style: "border-top: 1px solid var(--border-color); padding-top: 0.5em; color: var(--muted-color); font-style: italic;",
                                                    "{summary}"
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Search
                            div { style: "display: flex; flex-direction: column; gap: 0.5em;",
                                span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "search memory:" }
                                div { style: "display: flex; gap: 0.5ch;",
                                    input {
                                        style: "flex: 1; background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3em 0.75ch; font-size: var(--font-size-xs);",
                                        placeholder: "query...",
                                        value: "{mem_search_query.read()}",
                                        oninput: move |e| mem_search_query.set(e.value()),
                                        onkeydown: {
                                            let sid = session_id.clone();
                                            let api = props.api_client.clone();
                                            move |e: KeyboardEvent| {
                                                if e.key() == Key::Enter {
                                                    let q = mem_search_query.read().clone();
                                                    if q.is_empty() { return; }
                                                    let sid = sid.clone();
                                                    let api = api.clone();
                                                    spawn(async move {
                                                        match api.search_memory(&sid, &q).await {
                                                            Ok(r) => mem_search_results.set(Some(r)),
                                                            Err(e) => mem_error.set(Some(e.to_string())),
                                                        }
                                                    });
                                                }
                                            }
                                        },
                                    }
                                    button {
                                        style: "background: var(--secondary-bg); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.3em 0.75ch; font-size: var(--font-size-xs); cursor: pointer;",
                                        onclick: {
                                            let sid = session_id.clone();
                                            let api = props.api_client.clone();
                                            move |_| {
                                                let q = mem_search_query.read().clone();
                                                if q.is_empty() { return; }
                                                let sid = sid.clone();
                                                let api = api.clone();
                                                spawn(async move {
                                                    match api.search_memory(&sid, &q).await {
                                                        Ok(r) => mem_search_results.set(Some(r)),
                                                        Err(e) => mem_error.set(Some(e.to_string())),
                                                    }
                                                });
                                            }
                                        },
                                        "go"
                                    }
                                }
                                if let Some(ref results) = *mem_search_results.read() {
                                    if results.results.is_empty() {
                                        span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "no results" }
                                    } else {
                                        div { style: "display: flex; flex-direction: column; gap: 0.5em;",
                                            for entry in &results.results {
                                                {
                                                    let content = entry.content.clone();
                                                    let rel_str = entry.relevance
                                                        .map(|r| format!(" ({:.0}%)", r * 100.0))
                                                        .unwrap_or_default();
                                                    rsx! {
                                                        div { style: "border: 1px solid var(--border-color); padding: 0.4em 0.75ch; font-size: var(--font-size-xs);",
                                                            if !rel_str.is_empty() {
                                                                span { style: "color: var(--accent); margin-right: 0.5ch;", "{rel_str}" }
                                                            }
                                                            span { "{content}" }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Clear memory
                            div { style: "border-top: 1px solid var(--border-color); padding-top: 0.5em;",
                                button {
                                    style: "background: transparent; border: 1px solid var(--error-color); color: var(--error-color); padding: 0.3em 1ch; font-size: var(--font-size-xs); cursor: pointer; width: 100%;",
                                    onclick: {
                                        let sid = session_id.clone();
                                        let api = props.api_client.clone();
                                        move |_| {
                                            let sid = sid.clone();
                                            let api = api.clone();
                                            spawn(async move {
                                                let _ = api.clear_memory(&sid).await;
                                                mem_stats.set(None);
                                                mem_search_results.set(None);
                                            });
                                        }
                                    },
                                    "clear memory"
                                }
                            }

                            // Export memory
                            div {
                                style: "display: flex; gap: 0.6ch; align-items: center; flex-wrap: wrap; margin-top: 0.4ch;",
                                button {
                                    style: "background: transparent; color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3ch 0.8ch; font-family: var(--font-mono); font-size: var(--font-size-xs); cursor: pointer;",
                                    disabled: *mem_export_loading.read(),
                                    onclick: {
                                        let api = props.api_client.clone();
                                        let sid = props.session_id.clone();
                                        move |_| {
                                            if let Some(sid) = sid.clone() {
                                                mem_export_loading.set(true);
                                                mem_export_data.set(None);
                                                let api = api.clone();
                                                spawn(async move {
                                                    match api.export_memory(&sid).await {
                                                        Ok(snap) => {
                                                            let json = serde_json::to_string_pretty(&snap).unwrap_or_default();
                                                            mem_export_data.set(Some(json));
                                                        }
                                                        Err(e) => mem_error.set(Some(e.to_string())),
                                                    }
                                                    mem_export_loading.set(false);
                                                });
                                            }
                                        }
                                    },
                                    if *mem_export_loading.read() { "exporting..." } else { "export snapshot" }
                                }
                            }

                            // Export data display
                            if let Some(ref data) = *mem_export_data.read() {
                                textarea {
                                    style: "width: 100%; height: 8ch; background: var(--secondary-bg); color: var(--muted-color); border: 1px solid var(--border-color); padding: 0.4ch; font-family: var(--font-mono); font-size: var(--font-size-xs); resize: vertical; box-sizing: border-box; margin-top: 0.4ch;",
                                    readonly: true,
                                    value: "{data}",
                                }
                            }

                            // Import memory
                            div {
                                style: "display: flex; flex-direction: column; gap: 0.4ch; margin-top: 0.6ch;",
                                label {
                                    style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs);",
                                    "import snapshot (paste JSON)"
                                }
                                textarea {
                                    style: "height: 6ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.4ch; font-family: var(--font-mono); font-size: var(--font-size-xs); resize: vertical; width: 100%; box-sizing: border-box;",
                                    placeholder: "Paste exported snapshot JSON here...",
                                    value: "{mem_import_data}",
                                    oninput: move |e| mem_import_data.set(e.value()),
                                }
                                div {
                                    style: "display: flex; gap: 0.6ch; align-items: center;",
                                    button {
                                        style: "background: transparent; color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3ch 0.8ch; font-family: var(--font-mono); font-size: var(--font-size-xs); cursor: pointer;",
                                        disabled: *mem_import_loading.read(),
                                        onclick: {
                                            let api = props.api_client.clone();
                                            let sid = props.session_id.clone();
                                            move |_| {
                                                if let Some(sid) = sid.clone() {
                                                    let raw = mem_import_data.read().clone();
                                                    match serde_json::from_str::<MemorySnapshot>(&raw) {
                                                        Ok(snap) => {
                                                            mem_import_loading.set(true);
                                                            mem_import_status.set(None);
                                                            let api = api.clone();
                                                            spawn(async move {
                                                                match api.import_memory(&sid, &snap).await {
                                                                    Ok(_) => mem_import_status.set(Some("imported successfully".to_string())),
                                                                    Err(e) => mem_import_status.set(Some(format!("error: {}", e))),
                                                                }
                                                                mem_import_loading.set(false);
                                                            });
                                                        }
                                                        Err(e) => mem_import_status.set(Some(format!("invalid JSON: {}", e))),
                                                    }
                                                }
                                            }
                                        },
                                        if *mem_import_loading.read() { "importing..." } else { "import" }
                                    }
                                    if let Some(ref status) = *mem_import_status.read() {
                                        span {
                                            style: if status.starts_with("error") || status.starts_with("invalid") {
                                                "color: #f44; font-family: var(--font-mono); font-size: var(--font-size-xs);"
                                            } else {
                                                "color: #4f4; font-family: var(--font-mono); font-size: var(--font-size-xs);"
                                            },
                                            "{status}"
                                        }
                                    }
                                }
                            }
                        }
                    }

                    PanelTab::Checkpoints => {
                        rsx! {
                            // Create checkpoint
                            div { style: "display: flex; flex-direction: column; gap: 0.5em;",
                                span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "create checkpoint:" }
                                div { style: "display: flex; gap: 0.5ch;",
                                    input {
                                        style: "flex: 1; background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3em 0.75ch; font-size: var(--font-size-xs);",
                                        placeholder: "label...",
                                        value: "{cp_label.read()}",
                                        oninput: move |e| cp_label.set(e.value()),
                                    }
                                    button {
                                        disabled: *cp_busy.read(),
                                        style: "background: var(--secondary-bg); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.3em 0.75ch; font-size: var(--font-size-xs); cursor: pointer;",
                                        onclick: {
                                            let sid = session_id.clone();
                                            let api = props.api_client.clone();
                                            move |_| {
                                                let label = cp_label.read().clone();
                                                if label.is_empty() { return; }
                                                let sid = sid.clone();
                                                let api = api.clone();
                                                cp_busy.set(true);
                                                cp_error.set(None);
                                                spawn(async move {
                                                    match api.create_checkpoint(&sid, &label).await {
                                                        Ok(resp) => {
                                                            let entry = CheckpointEntry {
                                                                id: resp.checkpoint_id,
                                                                label: label.clone(),
                                                                created_at: chrono::Utc::now().format("%H:%M:%S").to_string(),
                                                            };
                                                            checkpoints.write().insert(0, entry);
                                                            cp_label.set(String::new());
                                                        }
                                                        Err(e) => cp_error.set(Some(e.to_string())),
                                                    }
                                                    cp_busy.set(false);
                                                });
                                            }
                                        },
                                        if *cp_busy.read() { "..." } else { "save" }
                                    }
                                }
                                if let Some(ref e) = *cp_error.read() {
                                    span { style: "color: var(--error-color); font-size: var(--font-size-xs);", "{e}" }
                                }
                                if let Some(ref s) = *rollback_status.read() {
                                    span { style: "color: var(--success-color, #0f0); font-size: var(--font-size-xs);", "{s}" }
                                }
                            }

                            // Checkpoint list
                            if checkpoints.read().is_empty() {
                                span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "no checkpoints yet" }
                            } else {
                                div { style: "display: flex; flex-direction: column; gap: 0.4em;",
                                    for cp in checkpoints.read().iter() {
                                        {
                                            let cp_id = cp.id.clone();
                                            let cp_lbl = cp.label.clone();
                                            let cp_ts = cp.created_at.clone();
                                            rsx! {
                                                div { style: "border: 1px solid var(--border-color); padding: 0.4em 0.75ch; font-size: var(--font-size-xs); display: flex; flex-direction: column; gap: 0.2em;",
                                                    div { style: "display: flex; justify-content: space-between; align-items: center;",
                                                        span { style: "color: var(--foreground); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1;", "{cp_lbl}" }
                                                        span { style: "color: var(--muted-color); flex-shrink: 0; margin-left: 0.5ch;", "{cp_ts}" }
                                                    }
                                                    button {
                                                        style: "background: transparent; border: 1px solid var(--warning-color, #fa0); color: var(--warning-color, #fa0); padding: 0.2em 0.5ch; font-size: var(--font-size-xs); cursor: pointer; align-self: flex-start;",
                                                        onclick: {
                                                            let sid = session_id.clone();
                                                            let api = props.api_client.clone();
                                                            let id = cp_id.clone();
                                                            move |_| {
                                                                let sid = sid.clone();
                                                                let api = api.clone();
                                                                let id = id.clone();
                                                                rollback_status.set(None);
                                                                spawn(async move {
                                                                    match api.rollback(&sid, &id).await {
                                                                        Ok(r) => {
                                                                            let n = r.files_restored.len();
                                                                            let msg = format!("restored {} file(s)", n);
                                                                            rollback_status.set(Some(msg));
                                                                        }
                                                                        Err(e) => cp_error.set(Some(e.to_string())),
                                                                    }
                                                                });
                                                            }
                                                        },
                                                        "rollback"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    PanelTab::Plan => {
                        rsx! {
                            div { style: "display: flex; align-items: center; gap: 1ch;",
                                button {
                                    disabled: *plan_loading.read(),
                                    style: "background: var(--secondary-bg); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.3em 1ch; font-size: var(--font-size-xs); cursor: pointer;",
                                    onclick: {
                                        let sid = session_id.clone();
                                        let api = props.api_client.clone();
                                        move |_| {
                                            let sid = sid.clone();
                                            let api = api.clone();
                                            plan_loading.set(true);
                                            plan_error.set(None);
                                            spawn(async move {
                                                match api.get_plan(&sid).await {
                                                    Ok(p) => plan.set(Some(p)),
                                                    Err(e) => plan_error.set(Some(e.to_string())),
                                                }
                                                plan_loading.set(false);
                                            });
                                        }
                                    },
                                    if *plan_loading.read() { "loading..." } else { "show plan" }
                                }
                                button {
                                    style: "background: transparent; border: 1px solid var(--error-color); color: var(--error-color); padding: 0.3em 1ch; font-size: var(--font-size-xs); cursor: pointer;",
                                    onclick: {
                                        let sid = session_id.clone();
                                        let api = props.api_client.clone();
                                        move |_| {
                                            let sid = sid.clone();
                                            let api = api.clone();
                                            plan_status.set(None);
                                            spawn(async move {
                                                match api.delete_plan(&sid).await {
                                                    Ok(_) => {
                                                        plan.set(None);
                                                        plan_status.set(Some("plan cleared".to_string()));
                                                    }
                                                    Err(e) => plan_error.set(Some(e.to_string())),
                                                }
                                            });
                                        }
                                    },
                                    "clear plan"
                                }
                                if let Some(ref s) = *plan_status.read() {
                                    span { style: "color: var(--success-color, #0f0); font-size: var(--font-size-xs);", "{s}" }
                                }
                            }
                            if let Some(ref e) = *plan_error.read() {
                                span { style: "color: var(--error-color); font-size: var(--font-size-xs);", "{e}" }
                            }
                            if let Some(ref p) = *plan.read() {
                                if let Some(ref content) = p.content {
                                    pre { style: "background: var(--secondary-bg); border: 1px solid var(--border-color); padding: 0.5em 1ch; font-size: 11px; overflow-x: auto; white-space: pre-wrap; color: var(--text-color); max-height: 300px; overflow-y: auto;",
                                        "{content}"
                                    }
                                } else {
                                    span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "no plan" }
                                }
                            }
                        }
                    }

                    PanelTab::Context => {
                        rsx! {
                            // detect project button
                            div { style: "display: flex; align-items: center; gap: 1ch;",
                                button {
                                    disabled: *project_loading.read(),
                                    style: "background: var(--secondary-bg); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.3em 1ch; font-size: var(--font-size-xs); cursor: pointer;",
                                    onclick: {
                                        let sid = session_id.clone();
                                        let api = props.api_client.clone();
                                        move |_| {
                                            let sid = sid.clone();
                                            let api = api.clone();
                                            project_loading.set(true);
                                            project_info.set(None);
                                            spawn(async move {
                                                match api.detect_project(&sid).await {
                                                    Ok(v) => project_info.set(Some(v)),
                                                    Err(_) => {}
                                                }
                                                project_loading.set(false);
                                            });
                                        }
                                    },
                                    if *project_loading.read() { "detecting..." } else { "detect project" }
                                }
                            }
                            if let Some(ref info) = *project_info.read() {
                                pre { style: "background: var(--secondary-bg); border: 1px solid var(--border-color); padding: 0.5em 1ch; font-size: 11px; white-space: pre-wrap; color: var(--text-color); max-height: 100px; overflow-y: auto;",
                                    { serde_json::to_string_pretty(info).unwrap_or_default() }
                                }
                            }

                            // relevant context search
                            div { style: "display: flex; flex-direction: column; gap: 0.5em; border-top: 1px solid var(--border-color); padding-top: 0.5em;",
                                span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "relevant context:" }
                                div { style: "display: flex; gap: 0.5ch;",
                                    input {
                                        style: "flex: 1; background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3em 0.75ch; font-size: var(--font-size-xs);",
                                        placeholder: "query...",
                                        value: "{ctx_query.read()}",
                                        oninput: move |e| ctx_query.set(e.value()),
                                    }
                                    button {
                                        disabled: *ctx_loading.read(),
                                        style: "background: var(--secondary-bg); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.3em 0.75ch; font-size: var(--font-size-xs); cursor: pointer;",
                                        onclick: {
                                            let sid = session_id.clone();
                                            let api = props.api_client.clone();
                                            move |_| {
                                                let q = ctx_query.read().trim().to_string();
                                                if q.is_empty() { return; }
                                                let sid = sid.clone();
                                                let api = api.clone();
                                                ctx_loading.set(true);
                                                ctx_error.set(None);
                                                spawn(async move {
                                                    match api.get_relevant_context(&sid, &q).await {
                                                        Ok(r) => ctx_results.set(Some(r)),
                                                        Err(e) => ctx_error.set(Some(e.to_string())),
                                                    }
                                                    ctx_loading.set(false);
                                                });
                                            }
                                        },
                                        if *ctx_loading.read() { "..." } else { "search" }
                                    }
                                }
                                if let Some(ref e) = *ctx_error.read() {
                                    span { style: "color: var(--error-color); font-size: var(--font-size-xs);", "{e}" }
                                }
                                if let Some(ref results) = *ctx_results.read() {
                                    if results.items.is_empty() {
                                        span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "no results" }
                                    } else {
                                        div { style: "display: flex; flex-direction: column; gap: 0.5em;",
                                            for item in &results.items {
                                                {
                                                    let itype = item.item_type.clone();
                                                    let icontent = item.content.clone();
                                                    let irel = format!("{:.0}%", item.relevance * 100.0);
                                                    rsx! {
                                                        div { style: "border: 1px solid var(--border-color); padding: 0.4em 0.75ch; font-size: var(--font-size-xs);",
                                                            div { style: "display: flex; gap: 0.5ch; color: var(--muted-color);",
                                                                span { "[{itype}]" }
                                                                span { "{irel}" }
                                                            }
                                                            span { "{icontent}" }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    PanelTab::Diff => {
                        let diff_state = diff.read().clone();
                        let can_undo = diff_state.as_ref().map(|d| d.can_undo).unwrap_or(false);
                        let can_redo = diff_state.as_ref().map(|d| d.can_redo).unwrap_or(false);
                        let has_changes = diff_state
                            .as_ref()
                            .map(DiffResponse::has_any_changes)
                            .unwrap_or(false);

                        rsx! {
                            // Refresh button
                            div { style: "display: flex; align-items: center; gap: 1ch;",
                                button {
                                    disabled: *diff_loading.read(),
                                    style: "background: var(--secondary-bg); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.3em 1ch; font-size: var(--font-size-xs); cursor: pointer;",
                                    onclick: {
                                        let sid = session_id.clone();
                                        let api = props.api_client.clone();
                                        move |_| {
                                            let sid = sid.clone();
                                            let api = api.clone();
                                            diff_loading.set(true);
                                            diff_error.set(None);
                                            spawn(async move {
                                                match api.get_diff(&sid).await {
                                                    Ok(d) => diff.set(Some(d)),
                                                    Err(e) => diff_error.set(Some(e.to_string())),
                                                }
                                                diff_loading.set(false);
                                            });
                                        }
                                    },
                                    if *diff_loading.read() { "loading..." } else { "refresh" }
                                }
                                button {
                                    disabled: !can_undo,
                                    style: "background: transparent; border: 1px solid var(--warning-color, #fa0); color: var(--warning-color, #fa0); padding: 0.3em 1ch; font-size: var(--font-size-xs); cursor: pointer;",
                                    onclick: {
                                        let sid = session_id.clone();
                                        let api = props.api_client.clone();
                                        move |_| {
                                            let sid = sid.clone();
                                            let api = api.clone();
                                            revert_status.set(None);
                                            spawn(async move {
                                                match api.undo(&sid).await {
                                                    Ok(response) => {
                                                        revert_status.set(Some(response.status_message()));
                                                        match api.get_diff(&sid).await {
                                                            Ok(updated) => diff.set(Some(updated)),
                                                            Err(e) => diff_error.set(Some(e.to_string())),
                                                        }
                                                    }
                                                    Err(e) => diff_error.set(Some(e.to_string())),
                                                }
                                            });
                                        }
                                    },
                                    "undo"
                                }
                                button {
                                    disabled: !can_redo,
                                    style: "background: transparent; border: 1px solid var(--accent-dim); color: var(--accent); padding: 0.3em 1ch; font-size: var(--font-size-xs); cursor: pointer;",
                                    onclick: {
                                        let sid = session_id.clone();
                                        let api = props.api_client.clone();
                                        move |_| {
                                            let sid = sid.clone();
                                            let api = api.clone();
                                            revert_status.set(None);
                                            spawn(async move {
                                                match api.redo(&sid).await {
                                                    Ok(response) => {
                                                        revert_status.set(Some(response.status_message()));
                                                        match api.get_diff(&sid).await {
                                                            Ok(updated) => diff.set(Some(updated)),
                                                            Err(e) => diff_error.set(Some(e.to_string())),
                                                        }
                                                    }
                                                    Err(e) => diff_error.set(Some(e.to_string())),
                                                }
                                            });
                                        }
                                    },
                                    "redo"
                                }
                                if let Some(ref s) = *revert_status.read() {
                                    span { style: "color: var(--success-color, #0f0); font-size: var(--font-size-xs);", "{s}" }
                                }
                            }

                            if let Some(ref e) = *diff_error.read() {
                                span { style: "color: var(--error-color); font-size: var(--font-size-xs);", "{e}" }
                            }

                            if let Some(ref d) = *diff.read() {
                                div { style: "display: flex; flex-direction: column; gap: 0.35em; border: 1px solid var(--border-color); padding: 0.5em 0.75ch; font-size: var(--font-size-xs);",
                                    div { style: "display: flex; justify-content: space-between; gap: 1ch;",
                                        span { style: "color: var(--muted-color);", "active" }
                                        span { style: "color: var(--foreground);", "{d.active_change_sets}" }
                                    }
                                    div { style: "display: flex; justify-content: space-between; gap: 1ch;",
                                        span { style: "color: var(--muted-color);", "redo queue" }
                                        span { style: "color: var(--foreground);", "{d.reverted_change_sets}" }
                                    }
                                    div { style: "display: flex; justify-content: space-between; gap: 1ch;",
                                        span { style: "color: var(--muted-color);", "files touched" }
                                        span { style: "color: var(--foreground);", "{d.files_changed.len()}" }
                                    }
                                    div { style: "display: flex; justify-content: space-between; gap: 1ch;",
                                        span { style: "color: var(--muted-color);", "undo" }
                                        span { style: if d.can_undo { "color: var(--warning-color, #fa0);" } else { "color: var(--muted-color);" }, if d.can_undo { "ready" } else { "idle" } }
                                    }
                                    div { style: "display: flex; justify-content: space-between; gap: 1ch;",
                                        span { style: "color: var(--muted-color);", "redo" }
                                        span { style: if d.can_redo { "color: var(--accent);" } else { "color: var(--muted-color);" }, if d.can_redo { "ready" } else { "idle" } }
                                    }
                                }

                                if !has_changes {
                                    span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "no changes" }
                                } else {
                                    div { style: "display: flex; flex-direction: column; gap: 0.6em;",
                                        if !d.messages.is_empty() {
                                            span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "active change sets:" }
                                            for entry in &d.messages {
                                                { render_diff_entry(entry, false) }
                                            }
                                        }
                                        if !d.reverted_messages.is_empty() {
                                            span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "redo queue:" }
                                            for entry in &d.reverted_messages {
                                                { render_diff_entry(entry, true) }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::session_header_label;

    #[test]
    fn header_uses_title_when_present() {
        assert_eq!(session_header_label(Some("desk")), "desk");
        assert_eq!(session_header_label(None), "session panel");
        assert_eq!(session_header_label(Some("")), "session panel");
    }
}
