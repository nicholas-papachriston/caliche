//! File browser component for exploring and selecting local files.

use dioxus::prelude::*;

use crate::api::{ApiClient, FileEntry};

/// Props for the FileBrowser component
#[derive(Props, Clone, PartialEq)]
pub struct FileBrowserProps {
    /// API client for file operations
    pub api_client: ApiClient,
    /// Callback when a file is selected for attachment
    pub on_file_select: EventHandler<FileEntry>,
    /// Callback to close the browser
    pub on_close: EventHandler<()>,
}

/// File browser component
#[component]
pub fn FileBrowser(props: FileBrowserProps) -> Element {
    // Current directory state
    let mut current_path = use_signal(|| None::<String>);
    let mut entries = use_signal(|| Vec::<FileEntry>::new());
    let mut parent_path = use_signal(|| None::<String>);
    let mut loading = use_signal(|| true);
    let mut error = use_signal(|| None::<String>);

    // Selected files for attachment
    let mut selected_files = use_signal(|| Vec::<FileEntry>::new());

    // Load directory on mount and when path changes
    let api = props.api_client.clone();
    let path_value = current_path.read().clone();

    use_effect(move || {
        let api = api.clone();
        let path = path_value.clone();

        spawn(async move {
            loading.set(true);
            error.set(None);

            match api.list_directory(path.as_deref()).await {
                Ok(response) => {
                    current_path.set(Some(response.path));
                    parent_path.set(response.parent);
                    entries.set(response.entries);
                }
                Err(e) => {
                    error.set(Some(e.to_string()));
                }
            }

            loading.set(false);
        });
    });

    // Navigate to a directory
    let mut navigate = move |path: String| {
        current_path.set(Some(path));
    };

    // Navigate to parent
    let go_up = move |_| {
        if let Some(parent) = parent_path.read().clone() {
            current_path.set(Some(parent));
        }
    };

    // Toggle file selection
    let mut toggle_select = move |file: FileEntry| {
        let mut selected = selected_files.write();
        if let Some(idx) = selected.iter().position(|f| f.path == file.path) {
            selected.remove(idx);
        } else {
            selected.push(file);
        }
    };

    // Check if a file is selected
    let is_selected = |path: &str| -> bool { selected_files.read().iter().any(|f| f.path == path) };

    // Attach selected files
    let attach_files = {
        let on_file_select = props.on_file_select.clone();
        move |_| {
            for file in selected_files.read().iter() {
                on_file_select.call(file.clone());
            }
            selected_files.write().clear();
        }
    };

    // Get current path display
    let path_display = current_path
        .read()
        .clone()
        .unwrap_or_else(|| ".".to_string());

    // Format file size
    let format_size = |size: Option<u64>| -> String {
        match size {
            Some(s) if s < 1024 => format!("{}B", s),
            Some(s) if s < 1024 * 1024 => format!("{:.1}K", s as f64 / 1024.0),
            Some(s) => format!("{:.1}M", s as f64 / (1024.0 * 1024.0)),
            None => String::new(),
        }
    };

    let selected_count = selected_files.read().len();
    let has_parent = parent_path.read().is_some();

    rsx! {
        div { class: "file-browser",
            style: "display: flex; flex-direction: column; height: 100%; border: 1px solid var(--border-color); background: var(--card-bg);",

            // Header
            div { class: "file-browser-header",
                style: "display: flex; align-items: center; justify-content: space-between; padding: 0.5em 1ch; border-bottom: 1px solid var(--border-color); background: var(--secondary-bg);",

                span { style: "color: var(--accent); font-weight: bold;",
                    "files"
                }

                button {
                    style: "background: transparent; border: none; color: var(--muted-color); padding: 0.25em 0.5ch;",
                    onclick: move |_| props.on_close.call(()),
                    "[x]"
                }
            }

            // Path bar
            div { class: "file-browser-path",
                style: "display: flex; align-items: center; gap: 1ch; padding: 0.5em 1ch; border-bottom: 1px dashed var(--border-color); font-size: var(--font-size-xs);",

                if has_parent {
                    button {
                        style: "background: var(--secondary-bg); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.25em 0.5ch;",
                        onclick: go_up,
                        "[..]"
                    }
                }

                span { style: "color: var(--muted-color); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;",
                    "{path_display}"
                }
            }

            // File list
            div { class: "file-browser-list",
                style: "flex: 1; overflow-y: auto; min-height: 0;",

                if *loading.read() {
                    div { style: "padding: 2ch; color: var(--muted-color); text-align: center;",
                        "[loading...]"
                    }
                } else if let Some(err) = error.read().as_ref() {
                    div { style: "padding: 2ch; color: var(--error-color);",
                        "[error] {err}"
                    }
                } else if entries.read().is_empty() {
                    div { style: "padding: 2ch; color: var(--muted-color); text-align: center;",
                        "(empty)"
                    }
                } else {
                    for entry in entries.read().iter() {
                        {
                            let entry_clone = entry.clone();
                            let entry_for_nav = entry.clone();
                            let entry_for_select = entry.clone();
                            let is_dir = entry.is_dir;
                            let selected = is_selected(&entry.path);
                            let size_str = format_size(entry.size);

                            rsx! {
                                div {
                                    key: "{entry_clone.path}",
                                    class: "file-entry",
                                    style: "display: flex; align-items: center; gap: 1ch; padding: 0.5em 1ch; border-bottom: 1px dotted var(--border-color); cursor: pointer;",
                                    style: if selected { "background: var(--accent-dim); color: #000;" } else { "" },

                                    // Icon
                                    span { style: "color: var(--muted-color); width: 2ch;",
                                        if is_dir { "[d]" } else { "[f]" }
                                    }

                                    // Name (clickable for navigation or selection)
                                    span {
                                        style: "flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;",
                                        onclick: move |_| {
                                            if is_dir {
                                                navigate(entry_for_nav.path.clone());
                                            } else {
                                                toggle_select(entry_for_select.clone());
                                            }
                                        },
                                        "{entry_clone.name}"
                                    }

                                    // Size (for files)
                                    if !is_dir {
                                        span { style: "color: var(--muted-color); font-size: var(--font-size-xs);",
                                            "{size_str}"
                                        }
                                    }

                                    // Select checkbox for files
                                    if !is_dir {
                                        {
                                            let entry_for_check = entry.clone();
                                            rsx! {
                                                button {
                                                    style: "background: transparent; border: 1px solid var(--border-color); color: var(--text-color); padding: 0 0.5ch; min-width: 3ch;",
                                                    onclick: move |_| toggle_select(entry_for_check.clone()),
                                                    if selected { "[*]" } else { "[ ]" }
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

            // Footer with actions
            div { class: "file-browser-footer",
                style: "display: flex; align-items: center; justify-content: space-between; padding: 0.5em 1ch; border-top: 1px solid var(--border-color); background: var(--secondary-bg);",

                span { style: "color: var(--muted-color); font-size: var(--font-size-xs);",
                    "{selected_count} selected"
                }

                button {
                    style: "background: var(--accent-dim); border: 1px solid var(--accent); color: #000; padding: 0.25em 1ch;",
                    disabled: selected_count == 0,
                    onclick: attach_files,
                    "[attach]"
                }
            }
        }
    }
}

/// Compact file attachment display
#[derive(Props, Clone, PartialEq)]
pub struct AttachedFileProps {
    /// The attached file
    pub file: FileEntry,
    /// Callback to remove the attachment
    pub on_remove: EventHandler<String>,
}

/// Display an attached file with remove button
#[component]
pub fn AttachedFile(props: AttachedFileProps) -> Element {
    let name = &props.file.name;
    let path = props.file.path.clone();

    rsx! {
        span { class: "attached-file",
            style: "display: inline-flex; align-items: center; gap: 0.5ch; background: var(--secondary-bg); border: 1px solid var(--border-color); padding: 0.25em 0.5ch; font-size: var(--font-size-xs);",

            span { style: "color: var(--accent);", "[f]" }
            span { style: "max-width: 15ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;",
                "{name}"
            }
            button {
                style: "background: transparent; border: none; color: var(--muted-color); padding: 0; cursor: pointer;",
                onclick: move |_| props.on_remove.call(path.clone()),
                "x"
            }
        }
    }
}
