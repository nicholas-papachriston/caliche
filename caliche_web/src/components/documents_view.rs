//! Documents / RAG view — manage collections, index text, and search semantically.

use crate::api::{
    ApiClient, CollectionInfo, DocumentDto, DocumentIndexRequest, DocumentSearchRequest,
    DocumentSearchResponse, DocumentSplitRequest,
};
use crate::state::ConfigState;
use dioxus::prelude::*;

#[derive(Debug, Clone, PartialEq)]
enum DocTab {
    Index,
    Search,
    Collections,
}

#[component]
pub fn DocumentsView(api_client: ApiClient, config: Signal<ConfigState>) -> Element {
    let mut active_tab = use_signal(|| DocTab::Index);

    // Index state
    let mut index_text = use_signal(String::new);
    let mut collection_name = use_signal(|| "default".to_string());
    let mut chunk_size = use_signal(|| 512usize);
    let mut chunk_overlap = use_signal(|| 50usize);
    let mut is_indexing = use_signal(|| false);
    let mut index_result: Signal<Option<String>> = use_signal(|| None);
    let mut index_error: Signal<Option<String>> = use_signal(|| None);

    // Search state
    let mut search_query = use_signal(String::new);
    let mut search_collection = use_signal(|| "default".to_string());
    let mut search_top_k = use_signal(|| 5usize);
    let mut is_searching = use_signal(|| false);
    let mut search_results: Signal<Option<DocumentSearchResponse>> = use_signal(|| None);
    let mut search_error: Signal<Option<String>> = use_signal(|| None);

    // Collections state
    let mut collections: Signal<Vec<CollectionInfo>> = use_signal(Vec::new);
    let mut collections_loading = use_signal(|| false);
    let mut collections_error: Signal<Option<String>> = use_signal(|| None);
    let mut delete_confirm: Signal<Option<String>> = use_signal(|| None);

    let do_load_collections = {
        let api = api_client.clone();
        move || {
            collections_loading.set(true);
            collections_error.set(None);
            let api = api.clone();
            spawn(async move {
                match api.list_collections().await {
                    Ok(resp) => collections.set(resp.collections),
                    Err(e) => collections_error.set(Some(e.to_string())),
                }
                collections_loading.set(false);
            });
        }
    };

    let do_index = {
        let api = api_client.clone();
        move |_| {
            if *is_indexing.read() {
                return;
            }
            let text = index_text.read().trim().to_string();
            if text.is_empty() {
                return;
            }
            let cname = collection_name.read().trim().to_string();
            if cname.is_empty() {
                return;
            }
            let cs = *chunk_size.read();
            let co = *chunk_overlap.read();
            let embedding = {
                let cfg = config.read();
                (cfg.embedding_provider.clone(), cfg.embedding_model.clone())
            };

            is_indexing.set(true);
            index_result.set(None);
            index_error.set(None);

            let api = api.clone();
            spawn(async move {
                let doc = DocumentDto {
                    content: text.clone(),
                    metadata: std::collections::HashMap::new(),
                };
                let split_req = DocumentSplitRequest {
                    documents: vec![doc],
                    chunk_size: cs,
                    chunk_overlap: Some(co),
                };
                let chunks = match api.split_documents(&split_req).await {
                    Ok(r) => r.chunks,
                    Err(e) => {
                        index_error.set(Some(format!("Split failed: {}", e)));
                        is_indexing.set(false);
                        return;
                    }
                };

                let docs: Vec<DocumentDto> = chunks
                    .into_iter()
                    .map(|c| DocumentDto {
                        content: c.content,
                        metadata: c.metadata,
                    })
                    .collect();

                let index_req = DocumentIndexRequest {
                    documents: docs,
                    collection_name: cname,
                    provider: embedding.0,
                    embedding_model: embedding.1,
                };

                match api.index_documents(&index_req).await {
                    Ok(r) => {
                        index_result.set(Some(format!(
                            "indexed {} chunks into collection \"{}\"",
                            r.indexed_count, r.collection_name
                        )));
                    }
                    Err(e) => index_error.set(Some(e.to_string())),
                }
                is_indexing.set(false);
            });
        }
    };

    // do_search as a shared closure — we run it via a trigger signal to avoid
    // the borrow issue of two move closures each trying to own it.
    let mut search_trigger = use_signal(|| false);

    let api_for_search = api_client.clone();
    use_effect(move || {
        if !*search_trigger.read() {
            return;
        }
        search_trigger.set(false);

        if *is_searching.read() {
            return;
        }
        let q = search_query.read().trim().to_string();
        if q.is_empty() {
            return;
        }
        let cname = search_collection.read().trim().to_string();
        let top_k = *search_top_k.read();
        let api = api_for_search.clone();

        is_searching.set(true);
        search_results.set(None);
        search_error.set(None);

        spawn(async move {
            let req = DocumentSearchRequest {
                query: q,
                collection_name: cname,
                top_k: Some(top_k),
            };
            match api.search_documents(&req).await {
                Ok(r) => search_results.set(Some(r)),
                Err(e) => search_error.set(Some(e.to_string())),
            }
            is_searching.set(false);
        });
    });

    rsx! {
        div {
            style: "display: flex; flex-direction: column; height: 100%; overflow: hidden; padding: 1.5ch; gap: 1.2ch; box-sizing: border-box;",

            // Header
            div {
                style: "color: var(--accent-color, #4af); font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); font-weight: bold; border-bottom: 1px solid var(--border-color); padding-bottom: 0.6ch;",
                "documents / rag"
            }

            // Sub-tab bar
            div {
                style: "display: flex; gap: 0.8ch; flex-shrink: 0;",
                for (tab_val, tab_label) in [(DocTab::Index, "index"), (DocTab::Search, "search"), (DocTab::Collections, "collections")] {
                    {
                        let tab_clone = tab_val.clone();
                        let mut load_clone = do_load_collections.clone();
                        let is_active = *active_tab.read() == tab_val;
                        rsx! {
                            button {
                                style: if is_active {
                                    "background: var(--accent-color, #4af); color: var(--background); border: 1px solid var(--accent-color, #4af); padding: 0.3ch 1ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); cursor: pointer; border-radius: 2px;"
                                } else {
                                    "background: transparent; color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3ch 1ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); cursor: pointer; border-radius: 2px;"
                                },
                                onclick: move |_| {
                                    active_tab.set(tab_clone.clone());
                                    if tab_clone == DocTab::Collections {
                                        load_clone();
                                    }
                                },
                                "{tab_label}"
                            }
                        }
                    }
                }
            }

            // Tab content
            div {
                style: "flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 1ch; min-height: 0;",

                match *active_tab.read() {
                    DocTab::Index => rsx! {
                        // Collection name + settings row
                        div {
                            style: "display: flex; gap: 1ch; align-items: center; flex-wrap: wrap; flex-shrink: 0;",
                            span { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "collection:" }
                            input {
                                value: "{collection_name}",
                                style: "flex: 1; min-width: 12ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3ch 0.6ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                oninput: move |e| collection_name.set(e.value()),
                            }
                            span { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "chunk:" }
                            input {
                                r#type: "number",
                                value: "{chunk_size}",
                                min: "64",
                                max: "4096",
                                style: "width: 6ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.2ch 0.4ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                oninput: move |e| { if let Ok(n) = e.value().parse::<usize>() { chunk_size.set(n); } }
                            }
                            span { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "overlap:" }
                            input {
                                r#type: "number",
                                value: "{chunk_overlap}",
                                min: "0",
                                max: "512",
                                style: "width: 5ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.2ch 0.4ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                oninput: move |e| { if let Ok(n) = e.value().parse::<usize>() { chunk_overlap.set(n); } }
                            }
                        }

                        // Text area
                        textarea {
                            style: "flex: 1; min-height: 12ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.6ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); resize: vertical; box-sizing: border-box; width: 100%;",
                            placeholder: "Paste document text to index...",
                            value: "{index_text}",
                            oninput: move |e| index_text.set(e.value()),
                        }

                        // Index button
                        button {
                            style: if *is_indexing.read() {
                                "background: var(--secondary-bg); color: var(--muted-color); border: 1px solid var(--border-color); padding: 0.4ch 2ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); cursor: not-allowed; flex-shrink: 0;"
                            } else {
                                "background: var(--accent-color, #4af); color: var(--background); border: none; padding: 0.4ch 2ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); cursor: pointer; flex-shrink: 0;"
                            },
                            disabled: *is_indexing.read(),
                            onclick: do_index,
                            if *is_indexing.read() { "indexing..." } else { "split & index" }
                        }

                        // Result / error
                        if let Some(ref msg) = *index_result.read() {
                            div { style: "color: #4f4; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); background: var(--secondary-bg); padding: 0.6ch 1ch; border-left: 2px solid #4f4;", "{msg}" }
                        }
                        if let Some(ref err) = *index_error.read() {
                            div { style: "color: #f44; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); background: var(--secondary-bg); padding: 0.6ch 1ch; border-left: 2px solid #f44;", "error: {err}" }
                        }
                    },

                    DocTab::Search => rsx! {
                        // Search controls
                        div {
                            style: "display: flex; gap: 1ch; align-items: center; flex-shrink: 0; flex-wrap: wrap;",
                            span { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "collection:" }
                            input {
                                value: "{search_collection}",
                                style: "flex: 1; min-width: 12ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3ch 0.6ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                oninput: move |e| search_collection.set(e.value()),
                            }
                            span { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "top k:" }
                            input {
                                r#type: "number",
                                value: "{search_top_k}",
                                min: "1",
                                max: "20",
                                style: "width: 5ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.2ch 0.4ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                oninput: move |e| { if let Ok(n) = e.value().parse::<usize>() { search_top_k.set(n); } }
                            }
                        }

                        // Query input
                        div {
                            style: "display: flex; gap: 0.8ch; flex-shrink: 0;",
                            input {
                                style: "flex: 1; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.4ch 0.8ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem);",
                                placeholder: "Enter semantic search query...",
                                value: "{search_query}",
                                oninput: move |e| search_query.set(e.value()),
                                onkeydown: move |e| {
                                    if e.key() == Key::Enter {
                                        search_trigger.set(true);
                                    }
                                }
                            }
                            button {
                                style: if *is_searching.read() {
                                    "background: var(--secondary-bg); color: var(--muted-color); border: 1px solid var(--border-color); padding: 0.4ch 1.5ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); cursor: not-allowed;"
                                } else {
                                    "background: var(--accent-color, #4af); color: var(--background); border: none; padding: 0.4ch 1.5ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); cursor: pointer;"
                                },
                                disabled: *is_searching.read(),
                                onclick: move |_| search_trigger.set(true),
                                if *is_searching.read() { "..." } else { "search" }
                            }
                        }

                        // Search error
                        if let Some(ref err) = *search_error.read() {
                            div { style: "color: #f44; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); background: var(--secondary-bg); padding: 0.6ch 1ch; border-left: 2px solid #f44;", "error: {err}" }
                        }

                        // Search results
                        if let Some(ref results) = *search_results.read() {
                            div {
                                style: "display: flex; flex-direction: column; gap: 0.6ch;",
                                div {
                                    style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                    "{results.results.len()} results"
                                }
                                for result in &results.results {
                                    div {
                                        style: "border: 1px solid var(--border-color); background: var(--secondary-bg); padding: 0.8ch 1ch;",
                                        div {
                                            style: "display: flex; justify-content: space-between; margin-bottom: 0.4ch;",
                                            span { style: "color: var(--accent-color, #4af); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "score: {result.score:.3}" }
                                            if !result.metadata.is_empty() {
                                                span {
                                                    style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                                    {result.metadata.iter().map(|(k, v)| format!("{}: {}", k, v)).collect::<Vec<_>>().join(", ")}
                                                }
                                            }
                                        }
                                        pre {
                                            style: "margin: 0; color: var(--text-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); white-space: pre-wrap; word-break: break-word;",
                                            "{result.content}"
                                        }
                                    }
                                }
                            }
                        }
                    },

                    DocTab::Collections => rsx! {
                        // Refresh button
                        div {
                            style: "display: flex; align-items: center; gap: 1ch; flex-shrink: 0;",
                            {
                                let mut load = do_load_collections.clone();
                                rsx! {
                                    button {
                                        style: "background: transparent; color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3ch 1ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); cursor: pointer;",
                                        onclick: move |_| load(),
                                        if *collections_loading.read() { "loading..." } else { "refresh" }
                                    }
                                }
                            }
                        }

                        if let Some(ref err) = *collections_error.read() {
                            div { style: "color: #f44; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); background: var(--secondary-bg); padding: 0.6ch 1ch; border-left: 2px solid #f44;", "error: {err}" }
                        }

                        if collections.read().is_empty() && !*collections_loading.read() {
                            div {
                                style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); text-align: center; padding: 2ch;",
                                "no collections yet — index some documents first"
                            }
                        }

                        for col in collections.read().clone() {
                            {
                                let col_name = col.name.clone();
                                let col_name2 = col_name.clone();
                                let is_confirming = delete_confirm.read().as_deref() == Some(&col_name);
                                let api = api_client.clone();
                                rsx! {
                                    div {
                                        style: "border: 1px solid var(--border-color); background: var(--secondary-bg); padding: 0.8ch 1ch; display: flex; align-items: center; justify-content: space-between;",
                                        div {
                                            style: "display: flex; flex-direction: column; gap: 0.2ch;",
                                            span { style: "color: var(--text-color); font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem);", "{col_name}" }
                                            span { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "{col.document_count} documents" }
                                        }
                                        div {
                                            style: "display: flex; gap: 0.6ch; align-items: center;",
                                            if is_confirming {
                                                span { style: "color: #f44; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "confirm?" }
                                                button {
                                                    style: "background: #f44; color: #fff; border: none; padding: 0.2ch 0.8ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); cursor: pointer;",
                                                    onclick: move |_| {
                                                        let name = col_name2.clone();
                                                        let api = api.clone();
                                                        delete_confirm.set(None);
                                                        spawn(async move {
                                                            let _ = api.delete_collection(&name).await;
                                                            match api.list_collections().await {
                                                                Ok(resp) => collections.set(resp.collections),
                                                                Err(_) => {}
                                                            }
                                                        });
                                                    },
                                                    "delete"
                                                }
                                                button {
                                                    style: "background: transparent; color: var(--text-color); border: 1px solid var(--border-color); padding: 0.2ch 0.8ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); cursor: pointer;",
                                                    onclick: move |_| delete_confirm.set(None),
                                                    "cancel"
                                                }
                                            } else {
                                                button {
                                                    style: "background: transparent; color: #f44; border: 1px solid #f44; padding: 0.2ch 0.8ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); cursor: pointer;",
                                                    onclick: move |_| delete_confirm.set(Some(col_name.clone())),
                                                    "delete"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                }
            }
        }
    }
}
