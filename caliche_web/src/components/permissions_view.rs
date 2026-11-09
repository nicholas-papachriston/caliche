//! Settings → Permissions. Renders GET `/api/policy/effective`.

use dioxus::prelude::*;

use crate::api::{ApiClient, EffectivePolicy};

const PRESETS: [&str; 5] = ["locked", "review", "standard", "trusted", "danger"];

/// Props for the permissions section.
#[derive(Props, Clone, PartialEq)]
pub struct PermissionsViewProps {
    /// API client that loads and saves the deployment file.
    pub api_client: ApiClient,
}

/// Preset picker, rule table, and host capability panel.
#[component]
pub fn PermissionsView(props: PermissionsViewProps) -> Element {
    let mut view = use_signal(|| None::<EffectivePolicy>);
    let mut status = use_signal(|| None::<(bool, String)>);
    let mut saving = use_signal(|| false);
    let mut loaded = use_signal(|| false);

    {
        let client = props.api_client.clone();
        use_effect(move || {
            if *loaded.read() {
                return;
            }
            loaded.set(true);
            let client = client.clone();
            spawn(async move {
                match client.get_policy().await {
                    Ok(policy) => view.set(Some(policy)),
                    Err(err) => status.set(Some((false, err.to_string()))),
                }
            });
        });
    }

    let row = "display: flex; align-items: flex-start; gap: 2ch; margin-bottom: 1.25em;";
    let label_style = "color: var(--muted-color); width: 16ch; flex-shrink: 0; padding-top: 0.2em;";
    let input_style = "background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.25em 1ch; flex: 1; min-width: 0;";

    let current = view.read().clone();
    let preset = current
        .as_ref()
        .map(|policy| policy.document.preset.clone())
        .unwrap_or_default();

    let client = props.api_client;
    let save = move |_| {
        let Some(policy) = view.read().clone() else {
            return;
        };
        saving.set(true);
        status.set(None);
        let client = client.clone();
        spawn(async move {
            match client.put_policy(&policy.document).await {
                Ok(saved) => {
                    view.set(Some(saved));
                    status.set(Some((true, "policy saved".to_owned())));
                }
                Err(err) => status.set(Some((false, err.to_string()))),
            }
            saving.set(false);
        });
    };

    rsx! {
        div { style: "margin-bottom: 2em;",
            div { style: "color: var(--muted-color); margin-bottom: 1em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5em;",
                "permissions"
            }
            if let Some(policy) = current {
                div { style: "{row}",
                    span { style: "{label_style}", "preset" }
                    select {
                        style: "{input_style}",
                        aria_label: "policy preset",
                        value: "{preset}",
                        onchange: move |evt| {
                            let next = evt.value();
                            if let Some(policy) = view.write().as_mut() {
                                policy.document.preset = next;
                            }
                        },
                        for name in PRESETS {
                            option { value: "{name}", selected: name == preset, "{name}" }
                        }
                    }
                }
                div { style: "{row}",
                    span { style: "{label_style}", "default" }
                    span { "{policy.default_effect}" }
                }
                div { style: "{row}",
                    span { style: "{label_style}", "sandbox" }
                    span { "{policy.sandbox}" }
                }
                div { style: "{row}",
                    span { style: "{label_style}", "host" }
                    span { "{policy.host}" }
                }
                div { style: "{row}",
                    span { style: "{label_style}", "file" }
                    span { "{policy.source_path}" }
                }
                p { style: "color: var(--muted-color); font-size: var(--font-size-xs); margin: 0 0 1em 0;",
                    if policy.file_present {
                        "Save writes this file. The live tool gate stays on {policy.live_gate}."
                    } else {
                        "No policy file yet. Save writes the selected preset. The live tool gate stays on {policy.live_gate}."
                    }
                }
                pre { style: "white-space: pre-wrap; background: var(--input-bg); border: 1px solid var(--border-color); padding: 0.75em 1ch; margin: 0 0 1em 0;",
                    "{policy.capability_panel}"
                }
                div { style: "margin-bottom: 1em;",
                    div { style: "color: var(--muted-color); margin-bottom: 0.5em;", "rules" }
                    if policy.rules.is_empty() {
                        span { style: "color: var(--muted-color);", "none" }
                    }
                    for rule in policy.rules {
                        div { key: "{rule.id}", style: "font-family: var(--font-mono); font-size: var(--font-size-xs); margin-bottom: 0.25em;",
                            "{rule.layer} {rule.effect} {rule.source} {rule.matcher}"
                        }
                    }
                }
                div { style: "margin-bottom: 1em;",
                    div { style: "color: var(--muted-color); margin-bottom: 0.5em;", "grants" }
                    if policy.grants.is_empty() {
                        span { style: "color: var(--muted-color);", "none" }
                    }
                    for grant in policy.grants {
                        div { key: "{grant.id}", style: "font-family: var(--font-mono); font-size: var(--font-size-xs);",
                            "{grant.layer} {grant.effect} {grant.source}"
                        }
                    }
                }
                div { style: "margin-bottom: 1em;",
                    span { style: "color: var(--muted-color);",
                        if policy.pending.is_empty() {
                            "pending: none"
                        } else {
                            "pending: {policy.pending.len()}"
                        }
                    }
                }
                button {
                    class: "primary",
                    style: "padding: 0.4em 2ch;",
                    disabled: *saving.read(),
                    onclick: save,
                    if *saving.read() { "saving..." } else { "save policy" }
                }
            } else {
                span { style: "color: var(--muted-color);", "loading permissions" }
            }
            if let Some((ok, ref msg)) = *status.read() {
                p {
                    style: if ok {
                        "color: var(--accent); font-size: var(--font-size-sm);"
                    } else {
                        "color: var(--error-color); font-size: var(--font-size-sm);"
                    },
                    "{msg}"
                }
            }
        }
    }
}
