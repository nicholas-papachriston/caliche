use std::collections::{HashMap, HashSet};

use dioxus::prelude::*;

use super::harness_form::HarnessForm;
use super::harness_row::HarnessRow;
use super::helpers::BTN_ACCENT;
use crate::api::{ApiClient, CreateHarnessRequest, HarnessInfo};
use crate::components::chrome::{
    EMPTY_SENTENCE, EmptyHint, RETRY_ACTION, SectionHeader, load_hint_body, load_kind_from_error,
    server_message_from_error,
};
use crate::harness::HarnessDraft;

#[derive(Props, Clone, PartialEq)]
pub struct HarnessesSectionProps {
    pub api_client: ApiClient,
    pub selected_ids: Signal<HashSet<String>>,
    pub harnesses: Signal<Vec<HarnessInfo>>,
    /// API address included when the harness list request fails.
    pub address: String,
}

#[component]
pub fn HarnessesSection(props: HarnessesSectionProps) -> Element {
    let api_client = props.api_client;
    let selected_ids = props.selected_ids;
    let harnesses = props.harnesses;
    let mut list_error = use_signal(|| Option::<String>::None);
    let mut poll_started = use_signal(|| false);
    let mut show_form = use_signal(|| false);
    let mut draft = use_signal(HarnessDraft::default);
    let mut form_error = use_signal(|| Option::<String>::None);
    let mut form_busy = use_signal(|| false);
    let mut op_busy = use_signal(|| Option::<String>::None);
    let mut dispatch_inp = use_signal(HashMap::<String, String>::new);

    {
        let client = api_client.clone();
        use_effect(move || {
            if *poll_started.read() {
                return;
            }
            poll_started.set(true);
            let client = client.clone();
            spawn(async move {
                loop {
                    refresh_harnesses(&client, harnesses, list_error).await;
                    #[cfg(target_arch = "wasm32")]
                    gloo_timers::future::TimeoutFuture::new(10_000).await;
                    #[cfg(not(target_arch = "wasm32"))]
                    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                }
            });
        });
    }

    let on_submit = {
        let client = api_client.clone();
        move |_| {
            let input = match draft.read().to_create_input() {
                Ok(input) => input,
                Err(err) => {
                    form_error.set(Some(err));
                    return;
                }
            };
            form_busy.set(true);
            form_error.set(None);
            let client = client.clone();
            spawn(async move {
                let req = CreateHarnessRequest {
                    name: input.name,
                    preset_id: input.preset_id,
                    command: input.command,
                    args: input.args,
                    cwd: input.cwd,
                    prompt_placement: input.prompt_placement,
                    capabilities: input.capabilities,
                };
                match client.create_harness(req).await {
                    Ok(_) => {
                        draft.set(HarnessDraft::default());
                        show_form.set(false);
                        refresh_harnesses(&client, harnesses, list_error).await;
                    }
                    Err(err) => form_error.set(Some(err.to_string())),
                }
                form_busy.set(false);
            });
        }
    };

    let on_delete = {
        let client = api_client.clone();
        move |id: String| {
            op_busy.set(Some(id.clone()));
            let client = client.clone();
            spawn(async move {
                let _ = client.delete_harness(&id).await;
                op_busy.set(None);
                refresh_harnesses(&client, harnesses, list_error).await;
            });
        }
    };

    let on_dispatch = {
        let client = api_client.clone();
        move |id: String| {
            let msg = dispatch_inp.read().get(&id).cloned().unwrap_or_default();
            let msg = msg.trim().to_string();
            if msg.is_empty() {
                return;
            }
            dispatch_inp.write().insert(id.clone(), String::new());
            op_busy.set(Some(id.clone()));
            let client = client.clone();
            spawn(async move {
                let dispatch_err = client.dispatch_harness(&id, &msg).await.err();
                op_busy.set(None);
                refresh_harnesses(&client, harnesses, list_error).await;
                if let Some(err) = dispatch_err {
                    list_error.set(Some(err.to_string()));
                }
            });
        }
    };

    let list = harnesses.read().clone();
    let failure_body = list_error.read().as_ref().map(|err| {
        load_hint_body(
            load_kind_from_error(err),
            &props.address,
            server_message_from_error(err).as_deref(),
        )
    });
    let summary = if list.is_empty() {
        "no CLI harnesses".to_string()
    } else {
        format!("{} registered", list.len())
    };

    rsx! {
        div { class: "harnesses-section",
            SectionHeader {
                title: "HARNESSES".to_string(),
                summary,
                button {
                    style: "{BTN_ACCENT}",
                    onclick: move |_| {
                        let next = !*show_form.read();
                        if next {
                            draft.set(HarnessDraft::default());
                            form_error.set(None);
                        }
                        show_form.set(next);
                    },
                    if *show_form.read() { "✕ cancel" } else { "+ add harness" }
                }
            }

            if let Some(body) = failure_body {
                EmptyHint {
                    body: body,
                    action: RETRY_ACTION.to_string(),
                    on_action: move |_| {
                        let client = api_client.clone();
                        spawn(async move {
                            refresh_harnesses(&client, harnesses, list_error).await;
                        });
                    },
                }
            }

            if *show_form.read() {
                HarnessForm {
                    draft,
                    error: form_error,
                    busy: form_busy,
                    on_submit,
                    on_cancel: move |_| show_form.set(false),
                }
            }

            if list.is_empty() && list_error.read().is_none() && !*show_form.read() {
                EmptyHint {
                    body: EMPTY_SENTENCE.to_string(),
                    action: "add a harness".to_string(),
                    on_action: move |_| {
                        draft.set(HarnessDraft::default());
                        form_error.set(None);
                        show_form.set(true);
                    },
                }
            }

            div { class: "worker-list",
                for item in list.iter() {
                    HarnessRow {
                        key: "{item.id}",
                        harness: item.clone(),
                        selected_ids,
                        dispatch_inp,
                        op_busy,
                        on_dispatch: on_dispatch.clone(),
                        on_delete: on_delete.clone(),
                    }
                }
            }
        }
    }
}

async fn refresh_harnesses(
    client: &ApiClient,
    mut harnesses: Signal<Vec<HarnessInfo>>,
    mut list_error: Signal<Option<String>>,
) {
    match client.list_harnesses().await {
        Ok(list) => {
            harnesses.set(list);
            list_error.set(None);
        }
        Err(err) => list_error.set(Some(err.to_string())),
    }
}
