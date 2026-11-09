use std::collections::{HashMap, HashSet};

use dioxus::prelude::*;

use super::endpoint_launch::EndpointLaunchForm;
use super::endpoint_row::EndpointRow;
use super::helpers::{BTN_ACCENT, default_provider_and_model};
use super::types::ChatMsg;
use crate::api::{ApiClient, EndpointInfo, ModelInfo, ProviderCatalogEntry};
use crate::components::chrome::{
    EMPTY_SENTENCE, EmptyHint, RETRY_ACTION, SectionHeader, load_hint_body, load_kind_from_error,
    server_message_from_error,
};

#[derive(Props, Clone, PartialEq)]
pub struct EndpointsSectionProps {
    pub api_client: ApiClient,
    pub ep_label: String,
    pub ep_list: Vec<EndpointInfo>,
    pub ep_error: Signal<Option<String>>,
    pub endpoints: Signal<Vec<EndpointInfo>>,
    pub provider_options: Signal<Vec<ProviderCatalogEntry>>,
    pub model_options: Signal<Vec<ModelInfo>>,
    pub current_provider_setting: Signal<Option<String>>,
    pub current_model_setting: Signal<Option<String>>,
    pub show_launch: Signal<bool>,
    pub form_name: Signal<String>,
    pub form_workspace: Signal<String>,
    pub form_port: Signal<String>,
    pub form_provider: Signal<String>,
    pub form_model: Signal<String>,
    pub form_auto_start: Signal<bool>,
    pub form_error: Signal<Option<String>>,
    pub form_busy: Signal<bool>,
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
    pub on_refresh: EventHandler<()>,
    pub on_connect: EventHandler<String>,
    pub on_start: EventHandler<String>,
    pub on_stop: EventHandler<String>,
    pub on_delete: EventHandler<String>,
    pub on_chat_send: EventHandler<(String, String)>,
    /// API address included when the endpoint list request fails.
    pub address: String,
}

#[component]
pub fn EndpointsSection(props: EndpointsSectionProps) -> Element {
    let api_client = props.api_client;
    let ep_label = props.ep_label;
    let ep_list = props.ep_list;
    let ep_error = props.ep_error;
    let mut show_launch = props.show_launch;
    let mut form_provider = props.form_provider;
    let mut form_model = props.form_model;
    let provider_options = props.provider_options;
    let model_options = props.model_options;
    let current_provider_setting = props.current_provider_setting;
    let current_model_setting = props.current_model_setting;
    let on_refresh = props.on_refresh;
    let on_refresh_retry = on_refresh.clone();
    let address = props.address.clone();
    let failure_body = ep_error.read().as_ref().map(|err| {
        load_hint_body(
            load_kind_from_error(err),
            &address,
            server_message_from_error(err).as_deref(),
        )
    });

    rsx! {
        div { class: "workers-section",
            SectionHeader {
                title: "ENDPOINTS".to_string(),
                summary: ep_label,
                button {
                    style: "{BTN_ACCENT}",
                    onclick: move |_| {
                        let next = !*show_launch.read();
                        if next {
                            let providers = provider_options.read();
                            let models = model_options.read();
                            let current_provider = current_provider_setting.read();
                            let current_model = current_model_setting.read();
                            let (provider, model) = default_provider_and_model(
                                &providers,
                                &models,
                                current_provider.as_deref(),
                                current_model.as_deref(),
                            );
                            form_provider.set(provider);
                            form_model.set(model);
                        }
                        show_launch.set(next);
                    },
                    if *show_launch.read() { "✕ cancel" } else { "+ new endpoint" }
                }
            }

            if let Some(body) = failure_body {
                EmptyHint {
                    body: body,
                    action: RETRY_ACTION.to_string(),
                    on_action: move |_| on_refresh_retry.call(()),
                }
            }

            if *show_launch.read() {
                EndpointLaunchForm {
                    api_client: api_client.clone(),
                    provider_options,
                    model_options,
                    current_provider_setting,
                    current_model_setting,
                    form_name: props.form_name,
                    form_workspace: props.form_workspace,
                    form_port: props.form_port,
                    form_provider: props.form_provider,
                    form_model: props.form_model,
                    form_auto_start: props.form_auto_start,
                    form_error: props.form_error,
                    form_busy: props.form_busy,
                    show_launch,
                    on_launched: on_refresh,
                }
            }

            div { class: "worker-list",
                if ep_list.is_empty() && ep_error.read().is_none() && !*show_launch.read() {
                    EmptyHint {
                        body: EMPTY_SENTENCE.to_string(),
                        action: "add an endpoint".to_string(),
                        on_action: move |_| show_launch.set(true),
                    }
                }

                for ep in ep_list.iter() {
                    EndpointRow {
                        key: "{ep.id}",
                        ep: ep.clone(),
                        ep_expanded: props.ep_expanded,
                        selected_ids: props.selected_ids,
                        ep_op_busy: props.ep_op_busy,
                        ep_chat: props.ep_chat,
                        ep_chat_inp: props.ep_chat_inp,
                        ep_chat_busy: props.ep_chat_busy,
                        ep_ping_ms: props.ep_ping_ms,
                        ep_ping_busy: props.ep_ping_busy,
                        ep_metrics: props.ep_metrics,
                        ep_metrics_busy: props.ep_metrics_busy,
                        ep_show_metrics: props.ep_show_metrics,
                        on_connect: props.on_connect,
                        on_start: props.on_start,
                        on_stop: props.on_stop,
                        on_delete: props.on_delete,
                        on_chat_send: props.on_chat_send,
                    }
                }
            }
        }
    }
}
