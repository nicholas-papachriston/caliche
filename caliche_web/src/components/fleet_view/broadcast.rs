use std::collections::HashSet;

use dioxus::prelude::*;

use super::helpers::{BTN_ACCENT, BTN_MUTED, INPUT_STYLE, snippet};
use super::types::BroadcastResult;
use crate::api::{EndpointInfo, HarnessInfo};
use crate::components::chrome::SectionHeader;
use crate::state::agents::SubAgent;

#[derive(Props, Clone, PartialEq)]
pub struct BroadcastPanelProps {
    pub bcast_sel_label: String,
    pub bcast_btn_label: String,
    pub sel_count: usize,
    pub show_broadcast: Signal<bool>,
    pub selected_ids: Signal<HashSet<String>>,
    pub bcast_msg: Signal<String>,
    pub bcast_busy: Signal<bool>,
    pub bcast_results: Signal<Vec<BroadcastResult>>,
    pub endpoints: Signal<Vec<EndpointInfo>>,
    pub harnesses: Signal<Vec<HarnessInfo>>,
    pub agents: Signal<Vec<SubAgent>>,
    pub on_broadcast: EventHandler<()>,
}

#[component]
pub fn BroadcastPanel(props: BroadcastPanelProps) -> Element {
    let mut show_broadcast = props.show_broadcast;
    let mut selected_ids = props.selected_ids;
    let mut bcast_msg = props.bcast_msg;
    let bcast_busy = props.bcast_busy;
    let bcast_results = props.bcast_results;
    let endpoints = props.endpoints;
    let harnesses = props.harnesses;
    let agents = props.agents;
    let bcast_sel_label = props.bcast_sel_label;
    let bcast_btn_label = props.bcast_btn_label;
    let sel_count = props.sel_count;
    let on_broadcast = props.on_broadcast;

    rsx! {
        div { class: "broadcast-panel",
            SectionHeader {
                title: "BROADCAST".to_string(),
                summary: bcast_sel_label,
                button {
                    style: "{BTN_MUTED}",
                    onclick: move |_| {
                        let mut sel = selected_ids.write();
                        for ep in endpoints.read().iter().filter(|e| e.status == "running") {
                            sel.insert(ep.id.clone());
                        }
                    },
                    "all running"
                }
                button {
                    style: "{BTN_MUTED}",
                    onclick: move |_| {
                        let mut sel = selected_ids.write();
                        for harness in harnesses.read().iter() {
                            sel.insert(harness.id.clone());
                        }
                    },
                    "all harnesses"
                }
                button {
                    style: "{BTN_MUTED}",
                    onclick: move |_| {
                        let mut sel = selected_ids.write();
                        for a in agents.read().iter().filter(|a| a.status.is_active()) {
                            sel.insert(a.id.clone());
                        }
                    },
                    "all active tasks"
                }
                button {
                    style: "{BTN_MUTED}",
                    onclick: move |_| selected_ids.write().clear(),
                    "clear"
                }
                button {
                    style: if *show_broadcast.read() { "padding: 0.2em 1.25ch; font-size: var(--font-size-xs); background: var(--accent); border: 1px solid var(--accent); color: var(--background); cursor: pointer;" } else { "{BTN_ACCENT}" },
                    onclick: move |_| {
                        let next = !*show_broadcast.read();
                        show_broadcast.set(next);
                    },
                    if *show_broadcast.read() { "hide" } else { "compose" }
                }
            }

            if *show_broadcast.read() {
                div { class: "harness-form",
                    textarea {
                        style: "{INPUT_STYLE} min-height: 3.5em; resize: vertical;",
                        placeholder: "Message or command to broadcast…",
                        value: "{bcast_msg}",
                        oninput: move |e| bcast_msg.set(e.value()),
                    }

                    div { class: "form-actions",
                        button {
                            class: "primary",
                            style: "padding: 0.3em 1.5ch;",
                            disabled: *bcast_busy.read() || sel_count == 0 || bcast_msg.read().trim().is_empty(),
                            onclick: move |_| on_broadcast.call(()),
                            "{bcast_btn_label}"
                        }
                        if *bcast_busy.read() {
                            span { class: "worker-meta", "in progress…" }
                        }
                    }

                    if !bcast_results.read().is_empty() {
                        div { class: "worker-list",
                            for r in bcast_results.read().iter() {
                                {
                                    let r = r.clone();
                                    let status_class = if r.ok { "worker-ready" } else { "worker-error" };
                                    let name = r.name.clone();
                                    let kind = r.kind.clone();
                                    let snip = snippet(&r.text);
                                    rsx! {
                                        div { class: "worker-row-main",
                                            span { class: "worker-kind", "{kind}" }
                                            span { class: "worker-name", "{name}" }
                                            span { class: "{status_class}", if r.ok { "ok" } else { "err" } }
                                            span { class: "worker-meta", "{snip}" }
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
