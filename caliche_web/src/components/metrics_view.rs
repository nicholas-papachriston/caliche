//! Metrics view — tool call statistics dashboard.

use dioxus::prelude::*;
use std::collections::HashMap;

use crate::api::{
    ApiClient, LiveSessionInfo,
    types::{CacheMetricsResponse, ToolMetricsDto},
};
use crate::components::chrome::{
    EMPTY_SENTENCE, EmptyHint, RETRY_ACTION, START_CHAT_ACTION, load_hint_body,
    load_kind_from_error, server_message_from_error, session_display_title,
};

#[derive(Props, Clone, PartialEq)]
pub struct MetricsViewProps {
    pub api_client: ApiClient,
    /// Open chat session. Inspect is about this session.
    pub session_id: Option<String>,
    /// API address included when a request fails.
    pub api_base_url: String,
    /// Leave inspect and open chat when there is no session.
    pub on_start_chat: EventHandler<()>,
}

/// Tool metrics dashboard showing call counts, success rates, and durations.
#[component]
pub fn MetricsView(props: MetricsViewProps) -> Element {
    let mut tools: Signal<HashMap<String, ToolMetricsDto>> = use_signal(HashMap::new);
    let mut loading = use_signal(|| false);
    let mut error: Signal<Option<String>> = use_signal(|| None);
    let mut sort_col: Signal<&'static str> = use_signal(|| "calls");
    let mut load_count = use_signal(|| 0_u32);

    let mut cache: Signal<Option<CacheMetricsResponse>> = use_signal(|| None);
    let mut cache_loading = use_signal(|| false);
    let mut cache_error: Signal<Option<String>> = use_signal(|| None);
    let mut cache_load_count = use_signal(|| 0_u32);
    let mut session_info: Signal<Option<LiveSessionInfo>> = use_signal(|| None);
    let mut session_error: Signal<Option<String>> = use_signal(|| None);
    let mut session_reload = use_signal(|| 0_u32);

    let session_for_title = props.session_id.clone();
    let api_for_title = props.api_client.clone();
    use_effect(move || {
        let _reload = *session_reload.read();
        let Some(sid) = session_for_title.clone() else {
            session_info.set(None);
            session_error.set(None);
            return;
        };
        let api = api_for_title.clone();
        session_error.set(None);
        spawn(async move {
            match api.get_live_session(&sid).await {
                Ok(info) => {
                    session_info.set(Some(info));
                    session_error.set(None);
                }
                Err(err) => {
                    session_info.set(None);
                    session_error.set(Some(err.to_string()));
                }
            }
        });
    });

    // Initial load triggered by incrementing load_count
    let api_for_effect = props.api_client.clone();
    use_effect(move || {
        let _trigger = *load_count.read();
        let api = api_for_effect.clone();
        loading.set(true);
        error.set(None);
        spawn(async move {
            match api.get_tool_metrics().await {
                Ok(resp) => tools.set(resp.tools),
                Err(e) => error.set(Some(e.to_string())),
            }
            loading.set(false);
        });
    });

    let api_for_cache = props.api_client.clone();
    use_effect(move || {
        let _trigger = *cache_load_count.read();
        let api = api_for_cache.clone();
        cache_loading.set(true);
        cache_error.set(None);
        spawn(async move {
            match api.get_cache_metrics().await {
                Ok(resp) => cache.set(Some(resp)),
                Err(e) => cache_error.set(Some(e.to_string())),
            }
            cache_loading.set(false);
        });
    });

    let col = *sort_col.read();

    // Build sorted rows — compute before rsx!
    let mut rows: Vec<(String, ToolMetricsDto)> = tools
        .read()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    rows.sort_by(|(an, av), (bn, bv)| match col {
        "calls" => bv.total_calls.cmp(&av.total_calls),
        "success" => bv
            .success_rate
            .partial_cmp(&av.success_rate)
            .unwrap_or(std::cmp::Ordering::Equal),
        "avg_ms" => bv
            .avg_duration_ms
            .partial_cmp(&av.avg_duration_ms)
            .unwrap_or(std::cmp::Ordering::Equal),
        _ => an.cmp(bn),
    });

    let status_text = if *loading.read() {
        "loading...".to_string()
    } else {
        format!("{} tools", rows.len())
    };

    let address = props.api_base_url.clone();
    let session_missing = props.session_id.is_none();
    let session_heading = match session_info.read().as_ref() {
        Some(info) => session_display_title(info.title.as_deref()),
        None => props.session_id.clone().unwrap_or_default(),
    };
    let session_meta = session_info
        .read()
        .as_ref()
        .map(|info| format!("{} turns · {}", info.turn_count, info.session_id));
    let tool_failure = error.read().as_ref().map(|err| {
        load_hint_body(
            load_kind_from_error(err),
            &address,
            server_message_from_error(err).as_deref(),
        )
    });
    let cache_failure = cache_error.read().as_ref().map(|err| {
        load_hint_body(
            load_kind_from_error(err),
            &address,
            server_message_from_error(err).as_deref(),
        )
    });
    let session_failure = session_error.read().as_ref().map(|err| {
        load_hint_body(
            load_kind_from_error(err),
            &address,
            server_message_from_error(err).as_deref(),
        )
    });

    if session_missing {
        return rsx! {
            EmptyHint {
                body: START_CHAT_ACTION.to_string(),
                action: START_CHAT_ACTION.to_string(),
                on_action: move |_| props.on_start_chat.call(()),
            }
        };
    }

    rsx! {
        div {
            style: "display: flex; flex-direction: column; height: 100%; overflow: hidden; padding: 1ch; gap: 1em;",

            div { style: "display: flex; flex-direction: column; gap: 0.25em; flex-shrink: 0;",
                span { style: "color: var(--accent); font-weight: bold;", "{session_heading}" }
                if let Some(meta) = session_meta {
                    span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "{meta}" }
                }
            }

            if let Some(body) = session_failure {
                EmptyHint {
                    body: body,
                    action: RETRY_ACTION.to_string(),
                    on_action: move |_| {
                        let next = *session_reload.read() + 1;
                        session_reload.set(next);
                    },
                }
            }

            // Toolbar
            div { style: "display: flex; align-items: center; gap: 2ch; flex-shrink: 0;",
                span { style: "color: var(--accent); font-weight: bold;", "tool metrics" }
                span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "{status_text}" }
                button {
                    disabled: *loading.read(),
                    style: "background: var(--secondary-bg); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.3em 1ch; font-size: var(--font-size-xs); cursor: pointer; margin-left: auto;",
                    onclick: move |_| {
                        let next = *load_count.read() + 1;
                        load_count.set(next);
                    },
                    "refresh"
                }
            }

            if let Some(body) = tool_failure.clone() {
                EmptyHint {
                    body: body,
                    action: RETRY_ACTION.to_string(),
                    on_action: move |_| {
                        let next = *load_count.read() + 1;
                        load_count.set(next);
                    },
                }
            }

            // Cache metrics panel
            div { style: "flex-shrink: 0; border-top: 1px solid var(--border-color); padding-top: 1em; display: flex; flex-direction: column; gap: 0.8em;",
                div { style: "display: flex; align-items: center; gap: 2ch;",
                    span { style: "color: var(--accent); font-weight: bold;", "cache metrics" }
                    if *cache_loading.read() {
                        span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "loading..." }
                    }
                    button {
                        disabled: *cache_loading.read(),
                        style: "background: var(--secondary-bg); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.3em 1ch; font-size: var(--font-size-xs); cursor: pointer; margin-left: auto;",
                        onclick: move |_| {
                            let next = *cache_load_count.read() + 1;
                            cache_load_count.set(next);
                        },
                        "refresh"
                    }
                }
                if let Some(body) = cache_failure {
                    EmptyHint {
                        body: body,
                        action: RETRY_ACTION.to_string(),
                        on_action: move |_| {
                            let next = *cache_load_count.read() + 1;
                            cache_load_count.set(next);
                        },
                    }
                }
                if let Some(ref cm) = *cache.read() {
                    div { style: "display: flex; gap: 2ch; flex-wrap: wrap;",
                        div { style: "background: var(--secondary-bg); border: 1px solid var(--border-color); padding: 0.6em 1.2ch; display: flex; flex-direction: column; align-items: center; min-width: 10ch;",
                            span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "provider hits" }
                            span { style: "color: #4f4; font-size: 1.1rem; font-weight: bold;", "{cm.provider_hits}" }
                        }
                        div { style: "background: var(--secondary-bg); border: 1px solid var(--border-color); padding: 0.6em 1.2ch; display: flex; flex-direction: column; align-items: center; min-width: 10ch;",
                            span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "provider misses" }
                            span { style: "color: #f44; font-size: 1.1rem; font-weight: bold;", "{cm.provider_misses}" }
                        }
                        div { style: "background: var(--secondary-bg); border: 1px solid var(--border-color); padding: 0.6em 1.2ch; display: flex; flex-direction: column; align-items: center; min-width: 12ch;",
                            span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "provider hit rate" }
                            span { style: "color: var(--accent-color, #4af); font-size: 1.1rem; font-weight: bold;",
                                { format!("{:.1}%", cm.provider_hit_rate) }
                            }
                        }
                        div { style: "background: var(--secondary-bg); border: 1px solid var(--border-color); padding: 0.6em 1.2ch; display: flex; flex-direction: column; align-items: center; min-width: 10ch;",
                            span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "registry hits" }
                            span { style: "color: #4f4; font-size: 1.1rem; font-weight: bold;", "{cm.registry_hits}" }
                        }
                        div { style: "background: var(--secondary-bg); border: 1px solid var(--border-color); padding: 0.6em 1.2ch; display: flex; flex-direction: column; align-items: center; min-width: 10ch;",
                            span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "registry misses" }
                            span { style: "color: #f44; font-size: 1.1rem; font-weight: bold;", "{cm.registry_misses}" }
                        }
                        div { style: "background: var(--secondary-bg); border: 1px solid var(--border-color); padding: 0.6em 1.2ch; display: flex; flex-direction: column; align-items: center; min-width: 12ch;",
                            span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "registry hit rate" }
                            span { style: "color: var(--accent-color, #4af); font-size: 1.1rem; font-weight: bold;",
                                { format!("{:.1}%", cm.registry_hit_rate) }
                            }
                        }
                    }
                }
            }

            // Table
            div { style: "flex: 1; overflow-y: auto;",
                table { style: "width: 100%; border-collapse: collapse; font-size: var(--font-size-xs);",

                    thead {
                        tr { style: "border-bottom: 2px solid var(--border-color); color: var(--muted-color);",
                            th { style: "text-align: left; padding: 0.4em 1ch; cursor: pointer; white-space: nowrap;",
                                onclick: move |_| sort_col.set("name"),
                                "tool"
                                if col == "name" { " ▲" }
                            }
                            th { style: "text-align: right; padding: 0.4em 1ch; cursor: pointer; white-space: nowrap;",
                                onclick: move |_| sort_col.set("calls"),
                                "calls"
                                if col == "calls" { " ▼" }
                            }
                            th { style: "text-align: right; padding: 0.4em 1ch; cursor: pointer; white-space: nowrap;",
                                onclick: move |_| sort_col.set("success"),
                                "success%"
                                if col == "success" { " ▼" }
                            }
                            th { style: "text-align: right; padding: 0.4em 1ch; cursor: pointer; white-space: nowrap;",
                                onclick: move |_| sort_col.set("avg_ms"),
                                "avg ms"
                                if col == "avg_ms" { " ▼" }
                            }
                            th { style: "text-align: left; padding: 0.4em 1ch; white-space: nowrap;",
                                "last run"
                            }
                        }
                    }

                    tbody {
                        if rows.is_empty() && !*loading.read() && tool_failure.is_none() {
                            tr {
                                td {
                                    colspan: "5",
                                    style: "color: var(--muted-color); padding: 1em 1ch; text-align: center;",
                                    "{EMPTY_SENTENCE}"
                                }
                            }
                        }

                        for (name, dto) in &rows {
                            {
                                let success_pct = format!("{:.1}%", dto.success_rate * 100.0);
                                let avg_ms = format!("{:.1}", dto.avg_duration_ms);
                                let calls = dto.total_calls;
                                let last = dto.last_executed.as_deref().unwrap_or("—");
                                // Truncate ISO timestamp to HH:MM:SS date portion
                                let last_short = if last.len() > 10 {
                                    &last[..10]
                                } else {
                                    last
                                };
                                let success_color = if dto.success_rate >= 0.9 {
                                    "var(--success-color, #0f0)"
                                } else if dto.success_rate >= 0.5 {
                                    "var(--warning-color, #fa0)"
                                } else {
                                    "var(--error-color)"
                                };
                                rsx! {
                                    tr { style: "border-bottom: 1px solid var(--border-color);",
                                        td { style: "padding: 0.4em 1ch; color: var(--foreground); font-family: monospace;",
                                            "{name}"
                                        }
                                        td { style: "padding: 0.4em 1ch; text-align: right; color: var(--foreground);",
                                            "{calls}"
                                        }
                                        td { style: "padding: 0.4em 1ch; text-align: right;",
                                            span { style: "color: {success_color};", "{success_pct}" }
                                        }
                                        td { style: "padding: 0.4em 1ch; text-align: right; color: var(--muted-color);",
                                            "{avg_ms}"
                                        }
                                        td { style: "padding: 0.4em 1ch; color: var(--muted-color);",
                                            "{last_short}"
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
