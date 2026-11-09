use std::collections::{HashMap, HashSet};

use dioxus::prelude::*;

use super::helpers::{
    BTN_ACCENT, BTN_DANGER, BTN_MUTED, INPUT_STYLE, status_color, status_icon, truncate,
};
use super::types::ChatMsg;
use crate::api::{ApiClient, EndpointInfo};

#[derive(Props, Clone, PartialEq)]
pub struct EndpointRowProps {
    pub ep: EndpointInfo,
    pub ep_expanded: Signal<Option<String>>,
    pub selected_ids: Signal<HashSet<String>>,
    pub ep_op_busy: Signal<Option<String>>,
    pub ep_chat: Signal<HashMap<String, Vec<ChatMsg>>>,
    pub ep_chat_inp: Signal<HashMap<String, String>>,
    pub ep_chat_busy: Signal<HashSet<String>>,
    pub ep_ping_ms: Signal<HashMap<String, Option<u64>>>,
    pub ep_ping_busy: Signal<HashSet<String>>,
    pub ep_metrics: Signal<HashMap<String, Vec<(String, u64)>>>,
    pub ep_metrics_busy: Signal<HashSet<String>>,
    pub ep_show_metrics: Signal<HashSet<String>>,
    pub on_connect: EventHandler<String>,
    pub on_start: EventHandler<String>,
    pub on_stop: EventHandler<String>,
    pub on_delete: EventHandler<String>,
    pub on_chat_send: EventHandler<(String, String)>,
}

#[component]
pub fn EndpointRow(props: EndpointRowProps) -> Element {
    let ep = props.ep.clone();
    let ep_id = ep.id.clone();
    let mut ep_expanded = props.ep_expanded;
    let mut selected_ids = props.selected_ids;
    let ep_op_busy = props.ep_op_busy;
    let ep_chat = props.ep_chat;
    let mut ep_chat_inp = props.ep_chat_inp;
    let ep_chat_busy = props.ep_chat_busy;
    let mut ep_ping_ms = props.ep_ping_ms;
    let mut ep_ping_busy = props.ep_ping_busy;
    let mut ep_metrics = props.ep_metrics;
    let mut ep_metrics_busy = props.ep_metrics_busy;
    let mut ep_show_metrics = props.ep_show_metrics;
    let on_connect = props.on_connect;
    let on_start = props.on_start;
    let on_stop = props.on_stop;
    let on_delete = props.on_delete;
    let on_chat_send = props.on_chat_send;

    let is_expanded = ep_expanded.read().as_deref() == Some(&ep_id);
    let is_running = ep.status == "running";
    let is_active = ep.status == "running" || ep.status == "starting";
    let is_busy = ep_op_busy.read().as_deref() == Some(&ep_id);
    let is_selected = selected_ids.read().contains(&ep_id);
    let is_chat_busy = ep_chat_busy.read().contains(&ep_id);
    let sc = status_color(&ep.status);
    let si = status_icon(&ep.status);
    let sl = ep.status.clone();
    let ws_short = truncate(&ep.workspace, 38);
    let toggle_id = ep_id.clone();
    let start_id = ep_id.clone();
    let stop_id = ep_id.clone();
    let delete_id = ep_id.clone();
    let connect_url = ep.url.clone();
    let chat_ep_id = ep_id.clone();
    let chat_ep_url = ep.url.clone();
    let chat_history = ep_chat.read().get(&ep_id).cloned().unwrap_or_default();
    let chat_input_val = ep_chat_inp.read().get(&ep_id).cloned().unwrap_or_default();

    rsx! {
        div {
            key: "{ep_id}",
            style: if is_selected { "border-bottom: 1px solid var(--border-color); background: color-mix(in srgb, var(--accent) 6%, transparent);" } else { "border-bottom: 1px solid var(--border-color);" },

            div {
                style: "display: flex; align-items: center; padding: 0.4em 1ch; gap: 0.5ch; cursor: pointer;",
                onclick: move |_| {
                    let mut exp = ep_expanded.write();
                    if exp.as_deref() == Some(&toggle_id) {
                        *exp = None;
                    } else {
                        *exp = Some(toggle_id.clone());
                    }
                },

                input {
                    r#type: "checkbox",
                    style: "accent-color: var(--accent); flex-shrink: 0; cursor: pointer;",
                    checked: is_selected,
                    onclick: move |e| e.stop_propagation(),
                    onchange: move |e| {
                        if e.checked() {
                            selected_ids.write().insert(ep_id.clone());
                        } else {
                            selected_ids.write().remove(&ep_id);
                        }
                    },
                }
                span {
                    style: "color: var(--muted-color); font-size: var(--font-size-xs); width: 0.9ch; flex-shrink: 0;",
                    if is_expanded { "▼" } else { "▶" }
                }
                span {
                    style: "font-size: var(--font-size-xs); flex-shrink: 0; color: {sc};",
                    if is_running {
                        span { class: "blink", "{si}" }
                    } else {
                        "{si}"
                    }
                }
                span { style: "font-size: var(--font-size-sm); font-weight: 500; flex-shrink: 0;", "{ep.name}" }
                span { style: "color: var(--muted-color); font-size: var(--font-size-xs); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;",
                    "{ws_short}" }
                span { style: "color: var(--muted-color); font-size: var(--font-size-xs); flex-shrink: 0;", ":{ep.port}" }
                span { style: "font-size: var(--font-size-xs); color: {sc}; flex-shrink: 0;", "{sl}" }
            }

            if is_expanded {
                div {
                    style: "padding: 0.4em 1ch 0.6em 2.5ch; border-top: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 0.4em;",

                    div { style: "font-size: var(--font-size-xs); color: var(--muted-color);",
                        "workspace: " span { style: "color: var(--text-color);", "{ep.workspace}" }
                    }
                    if is_running {
                        div { style: "font-size: var(--font-size-xs); color: var(--muted-color);",
                            "url: " span { style: "color: var(--accent);", "{ep.url}" }
                        }
                    }
                    if let Some(ref prov) = ep.provider {
                        div { style: "font-size: var(--font-size-xs); color: var(--muted-color);",
                            "provider: " span { style: "color: var(--text-color);", "{prov}" }
                            if let Some(ref m) = ep.model {
                                span { " / {m}" }
                            }
                        }
                    }
                    if let Some(ref err) = ep.error {
                        div { style: "font-size: var(--font-size-xs); color: var(--error); padding: 0.25em 0.5ch; border: 1px solid var(--error);",
                            "{err}"
                        }
                    }
                    if let Some(pid) = ep.pid {
                        div { style: "font-size: var(--font-size-xs); color: var(--muted-color);", "pid: {pid}" }
                    }

                    div { style: "display: flex; gap: 0.75ch; flex-wrap: wrap;",
                        if is_running {
                            button {
                                class: "primary",
                                style: "padding: 0.2em 1.25ch; font-size: var(--font-size-xs);",
                                onclick: move |_| on_connect.call(connect_url.clone()),
                                "connect"
                            }
                        }
                        if is_active {
                            button {
                                style: "{BTN_MUTED}",
                                disabled: is_busy,
                                onclick: move |_| on_stop.call(stop_id.clone()),
                                if is_busy { "…" } else { "stop" }
                            }
                        } else {
                            button {
                                style: "{BTN_MUTED}",
                                disabled: is_busy,
                                onclick: move |_| on_start.call(start_id.clone()),
                                if is_busy { "…" } else { "start" }
                            }
                        }
                        button {
                            style: "{BTN_DANGER}",
                            disabled: is_busy,
                            onclick: move |_| on_delete.call(delete_id.clone()),
                            "delete"
                        }
                        if is_running {
                            {
                                let ping_ep_id = ep.id.clone();
                                let ping_url = ep.url.clone();
                                let is_ping_busy = ep_ping_busy.read().contains(&ping_ep_id);
                                let ping_result = ep_ping_ms.read().get(&ping_ep_id).cloned().flatten();
                                let ping_label = if is_ping_busy {
                                    "pinging…".to_string()
                                } else if let Some(ms) = ping_result {
                                    format!("ping {}ms", ms)
                                } else {
                                    "ping".to_string()
                                };
                                rsx! {
                                    button {
                                        style: "{BTN_MUTED}",
                                        disabled: is_ping_busy,
                                        onclick: move |_| {
                                            let eid = ping_ep_id.clone();
                                            let url = ping_url.clone();
                                            ep_ping_busy.write().insert(eid.clone());
                                            spawn(async move {
                                                let ms = ApiClient::ping_url(&url).await;
                                                ep_ping_ms.write().insert(eid.clone(), ms);
                                                ep_ping_busy.write().remove(&eid);
                                            });
                                        },
                                        "{ping_label}"
                                    }
                                }
                            }
                        }
                        if is_running {
                            {
                                let met_ep_id = ep.id.clone();
                                let met_url = ep.url.clone();
                                let is_met_busy = ep_metrics_busy.read().contains(&met_ep_id);
                                let is_met_shown = ep_show_metrics.read().contains(&met_ep_id);
                                rsx! {
                                    button {
                                        style: if is_met_shown { "padding: 0.2em 1.25ch; font-size: var(--font-size-xs); background: var(--accent); border: 1px solid var(--accent); color: var(--background); cursor: pointer;" } else { "{BTN_MUTED}" },
                                        disabled: is_met_busy,
                                        onclick: move |_| {
                                            let eid = met_ep_id.clone();
                                            let url = met_url.clone();
                                            let currently_shown = ep_show_metrics.read().contains(&eid);
                                            if currently_shown {
                                                ep_show_metrics.write().remove(&eid);
                                            } else {
                                                ep_show_metrics.write().insert(eid.clone());
                                                if !ep_metrics.read().contains_key(&eid) {
                                                    ep_metrics_busy.write().insert(eid.clone());
                                                    spawn(async move {
                                                        if let Ok(resp) = ApiClient::endpoint_metrics(&url).await {
                                                            let mut top: Vec<(String, u64)> = resp
                                                                .tools
                                                                .into_iter()
                                                                .map(|(k, v)| (k, v.total_calls))
                                                                .collect();
                                                            top.sort_by(|a, b| b.1.cmp(&a.1));
                                                            top.truncate(8);
                                                            ep_metrics.write().insert(eid.clone(), top);
                                                        }
                                                        ep_metrics_busy.write().remove(&eid);
                                                    });
                                                }
                                            }
                                        },
                                        if is_met_busy { "loading…" } else if is_met_shown { "hide metrics" } else { "metrics" }
                                    }
                                }
                            }
                        }
                    }

                    if is_running && ep_show_metrics.read().contains(&ep.id) {
                        {
                            let met_data = ep_metrics.read().get(&ep.id).cloned().unwrap_or_default();
                            rsx! {
                                div {
                                    style: "border: 1px solid var(--border-color); background: var(--background); margin-top: 0.25em;",
                                    div { style: "padding: 0.3em 0.75ch; font-size: var(--font-size-xs); color: var(--muted-color); border-bottom: 1px solid var(--border-color); background: var(--card-bg); display: flex; align-items: center; gap: 1ch;",
                                        span { "endpoint tool metrics" }
                                        button {
                                            style: "margin-left: auto; background: transparent; border: 1px solid var(--border-color); color: var(--muted-color); padding: 0 0.5ch; font-size: var(--font-size-xs); cursor: pointer;",
                                            onclick: {
                                                let eid = ep.id.clone();
                                                let url = ep.url.clone();
                                                move |_| {
                                                    ep_metrics.write().remove(&eid);
                                                    ep_metrics_busy.write().insert(eid.clone());
                                                    let eid2 = eid.clone();
                                                    let url2 = url.clone();
                                                    spawn(async move {
                                                        if let Ok(resp) = ApiClient::endpoint_metrics(&url2).await {
                                                            let mut top: Vec<(String, u64)> = resp
                                                                .tools
                                                                .into_iter()
                                                                .map(|(k, v)| (k, v.total_calls))
                                                                .collect();
                                                            top.sort_by(|a, b| b.1.cmp(&a.1));
                                                            top.truncate(8);
                                                            ep_metrics.write().insert(eid2.clone(), top);
                                                        }
                                                        ep_metrics_busy.write().remove(&eid2);
                                                    });
                                                }
                                            },
                                            "↻"
                                        }
                                    }
                                    if met_data.is_empty() {
                                        div { style: "padding: 0.5em 0.75ch; font-size: var(--font-size-xs); color: var(--muted-color);", "no tool calls yet" }
                                    }
                                    for (tool_name, call_count) in met_data.iter() {
                                        {
                                            let tool_name = tool_name.clone();
                                            let call_count = *call_count;
                                            rsx! {
                                                div { style: "display: flex; align-items: center; gap: 0.5ch; padding: 0.2em 0.75ch; border-bottom: 1px solid var(--border-color); font-size: var(--font-size-xs);",
                                                    span { style: "flex: 1; color: var(--text-color);", "{tool_name}" }
                                                    span { style: "color: var(--accent); font-weight: 500;", "{call_count}" }
                                                    span { style: "color: var(--muted-color);", "calls" }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    div {
                        style: "margin-top: 0.5em; border: 1px solid var(--border-color); background: var(--background);",

                        div {
                            style: "padding: 0.3em 0.75ch; font-size: var(--font-size-xs); color: var(--muted-color); border-bottom: 1px solid var(--border-color); background: var(--card-bg);",
                            "inline chat"
                        }

                        if !chat_history.is_empty() {
                            div {
                                style: "max-height: 8em; overflow-y: auto; padding: 0.4em 0.75ch; display: flex; flex-direction: column; gap: 0.3em;",
                                for msg in chat_history.iter() {
                                    {
                                        let msg = msg.clone();
                                        let (role_color, role_lbl) = match msg.role.as_str() {
                                            "user" => ("var(--accent)", "you"),
                                            "assistant" => ("var(--text-color)", "agent"),
                                            _ => ("var(--error)", "error"),
                                        };
                                        let text = msg.text.clone();
                                        rsx! {
                                            div { style: "font-size: var(--font-size-xs); display: flex; gap: 0.5ch; align-items: flex-start;",
                                                span { style: "color: {role_color}; flex-shrink: 0; font-weight: 500;", "{role_lbl}" }
                                                span { style: "color: var(--text-color); white-space: pre-wrap; word-break: break-word;", "{text}" }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if is_running {
                            div {
                                style: "display: flex; gap: 0.5ch; padding: 0.4em 0.75ch; border-top: 1px solid var(--border-color);",
                                input {
                                    r#type: "text",
                                    style: "flex: 1; {INPUT_STYLE} border-color: transparent;",
                                    placeholder: "send a message…",
                                    value: "{chat_input_val}",
                                    disabled: is_chat_busy,
                                    oninput: {
                                        let ceid = chat_ep_id.clone();
                                        move |e| {
                                            ep_chat_inp.write().insert(ceid.clone(), e.value());
                                        }
                                    },
                                    onkeydown: {
                                        let ceid2 = chat_ep_id.clone();
                                        let ceurl = chat_ep_url.clone();
                                        move |e: KeyboardEvent| {
                                            if e.key() == Key::Enter && !e.modifiers().shift() {
                                                on_chat_send.call((ceid2.clone(), ceurl.clone()));
                                            }
                                        }
                                    },
                                }
                                button {
                                    style: "{BTN_ACCENT}",
                                    disabled: is_chat_busy || chat_input_val.trim().is_empty(),
                                    onclick: {
                                        let cid = ep.id.clone();
                                        let curl = ep.url.clone();
                                        move |_| on_chat_send.call((cid.clone(), curl.clone()))
                                    },
                                    if is_chat_busy { "…" } else { "send" }
                                }
                            }
                        } else {
                            div { style: "padding: 0.4em 0.75ch; font-size: var(--font-size-xs); color: var(--muted-color);",
                                "start endpoint to chat"
                            }
                        }
                    }
                }
            }
        }
    }
}
