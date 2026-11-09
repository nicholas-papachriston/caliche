//! Chat input component - monospace terminal style.

use dioxus::prelude::*;
use serde_json::Value;
use uuid::Uuid;

use super::chat_view::AttachedFileContext;
use crate::api::types::MediaAttachment;
use crate::commands::SLASH_COMMANDS;
use crate::context::{CONTEXT_REFERENCES, ContextReference};

// Platform-specific media pickers. On WASM (browser) we drive a hidden DOM
// `<input type=file>` via `dioxus::document::eval`; on native desktop/mobile
// we use `rfd` so users get the OS-native dialog and bytes come straight off
// disk without going through a webview FileReader.
#[cfg(target_arch = "wasm32")]
mod media_web;
#[cfg(target_arch = "wasm32")]
use media_web::{open_media_picker, parse_media_attachments};

#[cfg(not(target_arch = "wasm32"))]
mod media_native;
#[cfg(not(target_arch = "wasm32"))]
mod media_web; // Still needed for the drag-drop DOM event hook below.
#[cfg(not(target_arch = "wasm32"))]
use media_native::open_media_picker;
#[cfg(not(target_arch = "wasm32"))]
use media_web::parse_media_attachments;

/// Props for the ChatInput component
#[derive(Props, Clone, PartialEq)]
pub struct ChatInputProps {
    /// Callback when a message is sent
    pub on_send: EventHandler<String>,
    /// Callback when stop is clicked
    pub on_stop: EventHandler<()>,
    /// Callback when clear is clicked
    pub on_clear: EventHandler<()>,
    /// Whether the chat is currently streaming
    pub is_streaming: bool,
    /// Total tokens used in conversation
    #[props(default = 0)]
    pub total_tokens: u32,
    /// Attached files
    #[props(default = Vec::new())]
    pub attached_files: Vec<AttachedFileContext>,
    /// Context references attached via @
    #[props(default = Vec::new())]
    pub context_refs: Vec<ContextReference>,
    /// Callback to remove an attachment
    pub on_remove_attachment: EventHandler<String>,
    /// Callback to remove a context reference
    pub on_remove_context: EventHandler<usize>,
    /// Callback to toggle file browser
    pub on_toggle_files: EventHandler<()>,
    /// Whether file browsing is enabled
    #[props(default = false)]
    pub file_browsing_enabled: bool,
    /// Media attachments (images, audio, video, docs)
    #[props(default = Vec::new())]
    pub media_attachments: Vec<MediaAttachment>,
    /// Callback to add media attachments
    pub on_add_media: EventHandler<Vec<MediaAttachment>>,
    /// Callback to remove a media attachment by index
    pub on_remove_media: EventHandler<usize>,
}

/// Completion type - either a command or a context reference
#[derive(Clone, PartialEq)]
enum CompletionType {
    Command,
    Context,
}

/// A completion item
#[derive(Clone)]
struct CompletionItem {
    display: String,
    description: String,
    replacement: String,
    completion_type: CompletionType,
}

/// Get command completions for a partial input
fn get_command_completions(input: &str) -> Vec<CompletionItem> {
    if !input.starts_with('/') {
        return Vec::new();
    }

    let input_lower = input.to_lowercase();
    SLASH_COMMANDS
        .iter()
        .filter(|(cmd, _)| cmd.starts_with(&input_lower))
        .map(|(cmd, desc)| CompletionItem {
            display: cmd.to_string(),
            description: desc.to_string(),
            replacement: format!("{} ", cmd),
            completion_type: CompletionType::Command,
        })
        .collect()
}

/// Get context completions for a partial @ input
fn get_context_completions(partial: &str) -> Vec<CompletionItem> {
    let partial_lower = partial.to_lowercase();
    CONTEXT_REFERENCES
        .iter()
        .filter(|(ref_name, _)| ref_name.starts_with(&partial_lower))
        .map(|(ref_name, desc)| CompletionItem {
            display: ref_name.to_string(),
            description: desc.to_string(),
            replacement: format!("{} ", ref_name),
            completion_type: CompletionType::Context,
        })
        .collect()
}

/// Find the current @ reference being typed (if any)
fn find_current_at_reference(input: &str) -> Option<(usize, String)> {
    // Find the last @ that starts a word
    let chars: Vec<char> = input.chars().collect();
    let mut last_at_pos = None;

    for (i, &c) in chars.iter().enumerate() {
        if c == '@' && (i == 0 || chars[i - 1].is_whitespace()) {
            last_at_pos = Some(i);
        }
    }

    if let Some(at_pos) = last_at_pos {
        // Check if there's a space after the @reference (meaning it's complete)
        let after_at: String = chars[at_pos..].iter().collect();
        if !after_at.contains(' ') {
            return Some((at_pos, after_at));
        }
    }

    None
}

/// Chat input component with textarea and inline action buttons
#[component]
pub fn ChatInput(props: ChatInputProps) -> Element {
    let mut input_value = use_signal(String::new);
    let mut show_completions = use_signal(|| false);
    let mut selected_completion = use_signal(|| 0usize);
    let mut completion_start_pos = use_signal(|| 0usize);
    let drop_active = use_signal(|| false);
    let drop_zone_id = use_signal(|| format!("chat-input-dropzone-{}", Uuid::new_v4()));
    let mut drop_listener_registered = use_signal(|| false);
    let register_drop_listener = !*drop_listener_registered.read();
    let drop_zone_id_value = drop_zone_id.read().clone();
    let drop_zone_id_for_effect = drop_zone_id_value.clone();
    let on_add_media = props.on_add_media.clone();

    use_effect(move || {
        if !register_drop_listener {
            return;
        }

        drop_listener_registered.set(true);

        let mut drop_active_signal = drop_active;
        let on_add_media = on_add_media.clone();
        let js = r#"
            (function() {
                var id = __DROP_ZONE_ID__;
                var target = document.getElementById(id);
                if (!target || target.dataset.dropBound === 'true') {
                    return;
                }
                target.dataset.dropBound = 'true';

                var counter = 0;
                var setActive = function(active) {
                    dioxus.send({ type: active ? 'drag_enter' : 'drag_leave' });
                };

                var readFiles = function(files) {
                    var accepted = Array.from(files || []).filter(function(file) {
                        return file && !file.type.startsWith('text/directory');
                    });
                    if (accepted.length === 0) {
                        return;
                    }

                    Promise.all(accepted.map(function(file) {
                        return new Promise(function(resolve) {
                            var reader = new FileReader();
                            reader.onload = function(e) {
                                var result = typeof e.target.result === 'string' ? e.target.result : '';
                                resolve({
                                    filename: file.name,
                                    media_type: file.type || 'application/octet-stream',
                                    data_b64: result.split(',')[1] || '',
                                    preview_url: file.type.startsWith('image/') ? result : null
                                });
                            };
                            reader.onerror = function() { resolve(null); };
                            reader.readAsDataURL(file);
                        });
                    })).then(function(results) {
                        dioxus.send({ type: 'files', files: results.filter(Boolean) });
                    });
                };

                target.addEventListener('dragenter', function(event) {
                    event.preventDefault();
                    counter += 1;
                    setActive(true);
                });

                target.addEventListener('dragover', function(event) {
                    event.preventDefault();
                    if (event.dataTransfer) {
                        event.dataTransfer.dropEffect = 'copy';
                    }
                });

                target.addEventListener('dragleave', function(event) {
                    event.preventDefault();
                    counter = Math.max(0, counter - 1);
                    if (counter === 0) {
                        setActive(false);
                    }
                });

                target.addEventListener('drop', function(event) {
                    event.preventDefault();
                    counter = 0;
                    setActive(false);
                    readFiles(event.dataTransfer && event.dataTransfer.files);
                });
            })();
            "#
        .replace(
            "__DROP_ZONE_ID__",
            &serde_json::to_string(&drop_zone_id_for_effect)
                .unwrap_or_else(|_| "\"chat-input\"".to_string()),
        );

        spawn(async move {
            let mut eval = dioxus::document::eval(&js);
            while let Ok(value) = eval.recv::<Value>().await {
                match value.get("type").and_then(Value::as_str) {
                    Some("drag_enter") => drop_active_signal.set(true),
                    Some("drag_leave") => drop_active_signal.set(false),
                    Some("files") => {
                        drop_active_signal.set(false);
                        let attachments = parse_media_attachments(
                            value.get("files").cloned().unwrap_or(Value::Null),
                        );
                        if !attachments.is_empty() {
                            on_add_media.call(attachments);
                        }
                    }
                    _ => {}
                }
            }
        });
    });

    let is_streaming = props.is_streaming;
    let has_content = !input_value.read().trim().is_empty()
        || !props.attached_files.is_empty()
        || !props.context_refs.is_empty()
        || !props.media_attachments.is_empty();
    let can_send = has_content && !is_streaming;
    let placeholder_text = if is_streaming {
        "waiting..."
    } else {
        "message, /help, /chat, /settings, @file, or drop files..."
    };
    let attachment_count = props.attached_files.len();
    let context_count = props.context_refs.len();
    let media_count = props.media_attachments.len();
    let container_style = if *drop_active.read() {
        "border: 1px solid var(--accent); box-shadow: inset 0 0 0 1px var(--accent); flex-shrink: 0; display: flex; flex-direction: column; position: relative;"
    } else if is_streaming {
        "border: 1px solid var(--warning); flex-shrink: 0; display: flex; flex-direction: column; position: relative;"
    } else {
        "border: 1px solid var(--border-color); flex-shrink: 0; display: flex; flex-direction: column; position: relative;"
    };
    let add_media_handler = props.on_add_media.clone();

    // Get completions based on current input
    let completions: Vec<CompletionItem> = {
        let input = input_value.read();

        // Check for slash command at start
        if input.starts_with('/') && !input.contains(' ') {
            get_command_completions(&input)
        }
        // Check for @ reference anywhere in input
        else if let Some((_, partial)) = find_current_at_reference(&input) {
            get_context_completions(&partial)
        } else {
            Vec::new()
        }
    };
    let has_completions = !completions.is_empty() && *show_completions.read();

    let handle_send = {
        let on_send = props.on_send.clone();
        move |_| {
            let value = input_value.read().trim().to_string();
            if has_content && !is_streaming {
                on_send.call(value);
                input_value.set(String::new());
                show_completions.set(false);
                // Reset textarea height after clearing
                spawn(async move {
                    let _ = dioxus::document::eval(
                        "var t=document.getElementById('chat-main-textarea');if(t){t.style.height='auto';}",
                    );
                });
            }
        }
    };

    let handle_input = move |evt: FormEvent| {
        let value = evt.value();
        input_value.set(value.clone());

        // Auto-resize textarea to fit content
        spawn(async move {
            let _ = dioxus::document::eval(
                "var t=document.getElementById('chat-main-textarea');if(t){t.style.height='auto';t.style.height=Math.min(t.scrollHeight,200)+'px';}",
            );
        });

        // Show completions when typing a command
        if value.starts_with('/') && !value.contains(' ') {
            show_completions.set(true);
            selected_completion.set(0);
            completion_start_pos.set(0);
        }
        // Show completions when typing an @ reference
        else if let Some((pos, _)) = find_current_at_reference(&value) {
            show_completions.set(true);
            selected_completion.set(0);
            completion_start_pos.set(pos);
        } else {
            show_completions.set(false);
        }
    };

    let handle_keydown = {
        let on_send = props.on_send.clone();
        let completions_for_keydown = completions.clone();
        move |evt: KeyboardEvent| {
            let completions = &completions_for_keydown;

            // Handle completion navigation
            if has_completions {
                match evt.key() {
                    Key::ArrowDown => {
                        evt.prevent_default();
                        let current = *selected_completion.read();
                        if current < completions.len().saturating_sub(1) {
                            selected_completion.set(current + 1);
                        }
                        return;
                    }
                    Key::ArrowUp => {
                        evt.prevent_default();
                        let current = *selected_completion.read();
                        if current > 0 {
                            selected_completion.set(current - 1);
                        }
                        return;
                    }
                    Key::Tab | Key::Enter => {
                        if !completions.is_empty() {
                            evt.prevent_default();
                            let idx = *selected_completion.read();
                            if let Some(item) = completions.get(idx) {
                                let current_input = input_value.read().clone();
                                let start = *completion_start_pos.read();

                                // Replace from start position to end with the completion
                                let new_value = if start == 0 {
                                    item.replacement.clone()
                                } else {
                                    let prefix: String =
                                        current_input.chars().take(start).collect();
                                    format!("{}{}", prefix, item.replacement)
                                };

                                input_value.set(new_value);
                                show_completions.set(false);
                            }
                            return;
                        }
                    }
                    Key::Escape => {
                        show_completions.set(false);
                        return;
                    }
                    _ => {}
                }
            }

            // Send on Enter (without Shift), allow Shift+Enter for new line
            if evt.key() == Key::Enter && !evt.modifiers().shift() {
                evt.prevent_default();
                let value = input_value.read().trim().to_string();
                if has_content && !is_streaming {
                    on_send.call(value);
                    input_value.set(String::new());
                    show_completions.set(false);
                    spawn(async move {
                        let _ = dioxus::document::eval(
                            "var t=document.getElementById('chat-main-textarea');if(t){t.style.height='auto';}",
                        );
                    });
                }
            }
        }
    };

    let handle_stop = move |_| {
        props.on_stop.call(());
    };

    let handle_clear = move |_| {
        props.on_clear.call(());
    };

    rsx! {
        div {
            id: "{drop_zone_id_value}",
            class: "chat-input-container",
            style: "{container_style}",

            if *drop_active.read() {
                div {
                    class: "chat-drop-overlay",
                    style: "position: absolute; inset: 0; z-index: 5; display: flex; align-items: center; justify-content: center; background: var(--secondary-bg); border: 1px dashed var(--accent); color: var(--text-color); pointer-events: none;",
                    span { style: "background: var(--card-bg); border: 1px solid var(--accent); padding: 0.5em 1ch; font-size: var(--font-size-sm);",
                        "drop files to attach"
                    }
                }
            }

            // Media attachment strip
            if !props.media_attachments.is_empty() {
                div { class: "media-strip",
                    for (idx, att) in props.media_attachments.iter().enumerate() {
                        {
                            let filename = att.filename.clone();
                            let media_type = att.media_type.clone();
                            let preview = att.preview_url.clone();
                            let on_remove = props.on_remove_media.clone();
                            rsx! {
                                div {
                                    key: "{idx}",
                                    class: "media-thumb-wrap",
                                    if let Some(url) = preview {
                                        img {
                                            class: "media-thumb",
                                            src: "{url}",
                                            alt: "{filename}",
                                            title: "{filename}",
                                        }
                                    } else {
                                        div {
                                            class: "media-icon",
                                            title: "{filename}",
                                            {
                                                if media_type.starts_with("audio/") { "[aud]" }
                                                else if media_type.starts_with("video/") { "[vid]" }
                                                else { "[doc]" }
                                            }
                                        }
                                    }
                                    button {
                                        class: "media-remove",
                                        title: "remove",
                                        onclick: move |_| on_remove.call(idx),
                                        "x"
                                    }
                                    span { class: "media-label", "{filename}" }
                                }
                            }
                        }
                    }
                }
            }

            // Attached files and context references display
            if !props.attached_files.is_empty() || !props.context_refs.is_empty() {
                div { style: "display: flex; flex-wrap: wrap; gap: 0.5ch; padding: 0.5em 1ch; border-bottom: 1px dashed var(--border-color); background: var(--secondary-bg);",
                    // File attachments
                    for file_ctx in &props.attached_files {
                        {
                            let path = file_ctx.file.path.clone();
                            let name = file_ctx.file.name.clone();
                            let on_remove = props.on_remove_attachment.clone();
                            rsx! {
                                span {
                                    key: "{path}",
                                    class: "attached-file",
                                    style: "display: inline-flex; align-items: center; gap: 0.5ch; background: var(--card-bg); border: 1px solid var(--border-color); padding: 0.25em 0.5ch; font-size: var(--font-size-xs);",

                                    span { style: "color: var(--accent);", "[f]" }
                                    span { style: "max-width: 15ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;",
                                        "{name}"
                                    }
                                    button {
                                        style: "background: transparent; border: none; color: var(--muted-color); padding: 0; cursor: pointer;",
                                        onclick: move |_| on_remove.call(path.clone()),
                                        "x"
                                    }
                                }
                            }
                        }
                    }

                    // Context references
                    for (idx, ctx_ref) in props.context_refs.iter().enumerate() {
                        {
                            let display = ctx_ref.display_name();
                            let label = ctx_ref.short_label();
                            let on_remove = props.on_remove_context.clone();
                            rsx! {
                                span {
                                    key: "{idx}",
                                    class: "context-ref",
                                    style: "display: inline-flex; align-items: center; gap: 0.5ch; background: var(--card-bg); border: 1px solid var(--info); padding: 0.25em 0.5ch; font-size: var(--font-size-xs);",

                                    span { style: "color: var(--info);", "{label}" }
                                    span { style: "max-width: 20ch; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;",
                                        "{display}"
                                    }
                                    button {
                                        style: "background: transparent; border: none; color: var(--muted-color); padding: 0; cursor: pointer;",
                                        onclick: move |_| on_remove.call(idx),
                                        "x"
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Input prompt line with completions
            div { style: "position: relative; display: flex; align-items: flex-start; padding: 0.5em 1ch; gap: 1ch;",
                // Prompt character
                span { style: "color: var(--accent); flex-shrink: 0; line-height: var(--lh);", ">" }

                // Input wrapper for positioning completions
                div { style: "flex: 1; position: relative;",
                    // Textarea
                    textarea {
                        id: "chat-main-textarea",
                        class: "chat-textarea",
                        style: "width: 100%; min-height: calc(var(--lh) * 1em); max-height: 200px; padding: 0; border: none; background: transparent; color: var(--text-color); resize: none; outline: none; line-height: var(--lh); overflow-y: auto;",
                        placeholder: "{placeholder_text}",
                        value: "{input_value}",
                        disabled: is_streaming,
                        oninput: handle_input,
                        onkeydown: handle_keydown,
                        onfocusout: move |_| {
                            // Delay hiding to allow click on completion
                        },
                    }

                    // Completions dropdown (commands and context references)
                    if has_completions {
                        div {
                            class: "completions-dropdown",
                            style: "position: absolute; bottom: 100%; left: 0; right: 0; max-height: 200px; overflow-y: auto; background: var(--card-bg); border: 1px solid var(--border-color); border-bottom: none; z-index: 100;",

                            for (idx, item) in completions.iter().enumerate() {
                                {
                                    let is_selected = idx == *selected_completion.read();
                                    let display = item.display.clone();
                                    let description = item.description.clone();
                                    let replacement = item.replacement.clone();
                                    let start_pos = *completion_start_pos.read();
                                    let is_context = item.completion_type == CompletionType::Context;
                                    let bg = if is_selected { "var(--secondary-bg)" } else { "transparent" };
                                    let color = if is_context { "var(--info)" } else { "var(--accent)" };

                                    rsx! {
                                        div {
                                            key: "{display}",
                                            style: "padding: 0.5em 1ch; cursor: pointer; background: {bg}; display: flex; gap: 2ch;",
                                            onclick: move |_| {
                                                let current_input = input_value.read().clone();
                                                let new_value = if start_pos == 0 {
                                                    replacement.clone()
                                                } else {
                                                    let prefix: String = current_input.chars().take(start_pos).collect();
                                                    format!("{}{}", prefix, replacement)
                                                };
                                                input_value.set(new_value);
                                                show_completions.set(false);
                                            },
                                            onmouseenter: move |_| {
                                                selected_completion.set(idx);
                                            },

                                            span { style: "color: {color}; font-weight: bold;", "{display}" }
                                            span { style: "color: var(--muted-color); flex: 1;", "{description}" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Bottom bar - separator and actions
            div { style: "display: flex; justify-content: space-between; align-items: center; padding: 0.25em 1ch; border-top: 1px dashed var(--border-color); background: var(--card-bg);",

                // Left side - status info
                div { style: "display: flex; align-items: center; gap: 2ch; color: var(--muted-color); font-size: var(--font-size-xs);",
                    // Token count
                    if props.total_tokens > 0 {
                        span { title: "total tokens used", "{props.total_tokens}t" }
                    }

                    // Attachment count
                    if attachment_count > 0 {
                        span { style: "color: var(--accent);", "{attachment_count} files" }
                    }

                    // Context reference count
                    if context_count > 0 {
                        span { style: "color: var(--info);", "{context_count} refs" }
                    }

                    if media_count > 0 {
                        span { style: "color: var(--accent);", "{media_count} media" }
                    }

                    // Keyboard hint
                    span { class: "keyboard-hint", "↵ send · ⇧↵ newline · @ ctx · / cmd · drop files" }
                }

                // Right side - action buttons
                div { style: "display: flex; gap: 1ch;",
                    // Media attachment button
                    if !is_streaming {
                        button {
                            title: "attach image, audio, video, or document",
                            style: "background: transparent; border: 1px solid var(--border-color); color: var(--muted-color); padding: 0.25em 1ch;",
                            onclick: move |_| open_media_picker(add_media_handler.clone()),
                            "[+media]"
                        }
                    }

                    // File browser button (if enabled)
                    if props.file_browsing_enabled {
                        button {
                            title: "browse files",
                            style: "background: transparent; border: 1px solid var(--border-color); color: var(--muted-color); padding: 0.25em 1ch;",
                            onclick: move |_| props.on_toggle_files.call(()),
                            "[files]"
                        }
                    }

                    // Clear button
                    button {
                        title: "clear conversation",
                        style: "background: transparent; border: 1px solid var(--border-color); color: var(--muted-color); padding: 0.25em 1ch;",
                        onclick: handle_clear,
                        "[clear]"
                    }

                    // Stop button (only when streaming)
                    if is_streaming {
                        button {
                            title: "stop generation",
                            style: "background: var(--error); border: 1px solid var(--error); color: #000; padding: 0.25em 1ch;",
                            onclick: handle_stop,
                            "[stop]"
                        }
                    }

                    // Send button
                    if can_send {
                        button {
                            class: "primary send-btn",
                            title: "send message (Enter)",
                            onclick: handle_send,
                            "send"
                        }
                    } else {
                        button {
                            class: "send-btn",
                            disabled: true,
                            title: "type a message to send",
                            "send"
                        }
                    }
                }
            }
        }
    }
}
