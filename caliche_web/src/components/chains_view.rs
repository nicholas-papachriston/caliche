//! Chains view — LLM, Map-Reduce, Retrieval, and Document Map-Reduce chains.

use crate::api::{
    ApiClient, ChainResponse, DocumentMapReduceRequest, LLMChainRequest, MapReduceRequest,
    RetrievalChainRequest,
};
use dioxus::prelude::*;

#[derive(Debug, Clone, PartialEq)]
enum ChainType {
    Llm,
    MapReduce,
    Retrieval,
    DocMapReduce,
}

impl ChainType {
    fn label(&self) -> &'static str {
        match self {
            Self::Llm => "LLM",
            Self::MapReduce => "Map-Reduce",
            Self::Retrieval => "Retrieval (RAG)",
            Self::DocMapReduce => "Doc Map-Reduce",
        }
    }
    fn description(&self) -> &'static str {
        match self {
            Self::Llm => "Fill a prompt template with variables and get a single LLM response.",
            Self::MapReduce => {
                "Apply a map prompt to each item in parallel, then reduce all results into one answer."
            }
            Self::Retrieval => {
                "Retrieve relevant documents from a collection, then answer a query using that context."
            }
            Self::DocMapReduce => {
                "Chunk a long document, map over chunks, then reduce to a final summary."
            }
        }
    }
}

fn render_chain_output(result: &ChainResponse) -> Element {
    rsx! {
        div {
            style: "display: flex; flex-direction: column; gap: 0.8ch;",

            // Intermediate steps
            if !result.intermediate_steps.is_empty() {
                div {
                    style: "display: flex; flex-direction: column; gap: 0.4ch;",
                    div {
                        style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                        "intermediate steps ({result.intermediate_steps.len()})"
                    }
                    for (step_name, step_val) in &result.intermediate_steps {
                        div {
                            style: "border: 1px solid var(--border-color); background: var(--secondary-bg); padding: 0.6ch 1ch;",
                            div {
                                style: "color: var(--accent-color, #4af); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); margin-bottom: 0.3ch;",
                                "{step_name}"
                            }
                            pre {
                                style: "margin: 0; color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); white-space: pre-wrap; word-break: break-word; max-height: 10ch; overflow-y: auto;",
                                {
                                    if let Some(s) = step_val.as_str() {
                                        s.to_string()
                                    } else {
                                        serde_json::to_string_pretty(step_val).unwrap_or_default()
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Final output
            div {
                style: "border: 1px solid var(--accent-color, #4af); background: var(--secondary-bg); padding: 0.8ch 1ch;",
                div {
                    style: "color: var(--accent-color, #4af); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); margin-bottom: 0.4ch;",
                    "output"
                }
                pre {
                    style: "margin: 0; color: var(--text-color); font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); white-space: pre-wrap; word-break: break-word;",
                    {
                        if let Some(s) = result.output.as_str() {
                            s.to_string()
                        } else {
                            serde_json::to_string_pretty(&result.output).unwrap_or_default()
                        }
                    }
                }
            }
        }
    }
}

#[component]
pub fn ChainsView(api_client: ApiClient) -> Element {
    let mut selected = use_signal(|| ChainType::Llm);
    let mut is_running = use_signal(|| false);
    let mut result: Signal<Option<ChainResponse>> = use_signal(|| None);
    let mut error_msg: Signal<Option<String>> = use_signal(|| None);

    // LLM chain state
    let mut llm_template = use_signal(|| "Answer this question concisely: {question}".to_string());
    let mut llm_vars =
        use_signal(|| "{\n  \"question\": \"What is the capital of France?\"\n}".to_string());

    // Map-Reduce state
    let mut mr_map_template =
        use_signal(|| "Summarize this item in one sentence: {item}".to_string());
    let mut mr_reduce_template =
        use_signal(|| "Combine these summaries into a final summary:\n{summaries}".to_string());
    let mut mr_items_text = use_signal(|| {
        "\"First item to process\"\n\"Second item to process\"\n\"Third item to process\""
            .to_string()
    });

    // Retrieval chain state
    let mut ret_query = use_signal(|| String::new());
    let mut ret_collection = use_signal(|| "default".to_string());
    let mut ret_top_k = use_signal(|| 5usize);

    // Doc Map-Reduce state
    let mut doc_text = use_signal(|| String::new());
    let mut doc_map_template =
        use_signal(|| "Extract the key points from this section:\n{document}".to_string());
    let mut doc_reduce_template = use_signal(|| {
        "Synthesize these key points into a coherent summary:\n{summaries}".to_string()
    });
    let mut doc_chunk_size = use_signal(|| 512usize);

    let run_chain = {
        let api = api_client.clone();
        move |_| {
            if *is_running.read() {
                return;
            }
            let chain_type = selected.read().clone();

            is_running.set(true);
            result.set(None);
            error_msg.set(None);

            let api = api.clone();
            spawn(async move {
                let res: Result<ChainResponse, anyhow::Error> = match chain_type {
                    ChainType::Llm => {
                        let vars: serde_json::Value = serde_json::from_str(&llm_vars.read())
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        let req = LLMChainRequest {
                            prompt_template: llm_template.read().clone(),
                            variables: vars,
                            provider: None,
                            model: None,
                        };
                        api.chains_llm(&req).await
                    }
                    ChainType::MapReduce => {
                        let items_raw = mr_items_text.read().clone();
                        let items: Vec<serde_json::Value> = items_raw
                            .lines()
                            .filter(|l| !l.trim().is_empty())
                            .filter_map(|l| serde_json::from_str(l).ok())
                            .collect();
                        let req = MapReduceRequest {
                            map_template: mr_map_template.read().clone(),
                            reduce_template: mr_reduce_template.read().clone(),
                            items,
                            provider: None,
                            model: None,
                        };
                        api.chains_map_reduce(&req).await
                    }
                    ChainType::Retrieval => {
                        let req = RetrievalChainRequest {
                            query: ret_query.read().clone(),
                            collection: Some(ret_collection.read().clone()),
                            top_k: Some(*ret_top_k.read()),
                            provider: None,
                            model: None,
                        };
                        api.chains_retrieval(&req).await
                    }
                    ChainType::DocMapReduce => {
                        let req = DocumentMapReduceRequest {
                            document: doc_text.read().clone(),
                            map_template: doc_map_template.read().clone(),
                            reduce_template: doc_reduce_template.read().clone(),
                            chunk_size: Some(*doc_chunk_size.read()),
                            provider: None,
                            model: None,
                        };
                        api.chains_document_map_reduce(&req).await
                    }
                };
                match res {
                    Ok(r) => result.set(Some(r)),
                    Err(e) => error_msg.set(Some(e.to_string())),
                }
                is_running.set(false);
            });
        }
    };

    let btn_style = if *is_running.read() {
        "background: var(--secondary-bg); color: var(--muted-color); border: 1px solid var(--border-color); padding: 0.4ch 2ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); cursor: not-allowed; flex-shrink: 0;"
    } else {
        "background: var(--accent-color, #4af); color: var(--background); border: none; padding: 0.4ch 2ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); cursor: pointer; flex-shrink: 0;"
    };

    rsx! {
        div {
            style: "display: flex; flex-direction: column; height: 100%; overflow: hidden; padding: 1.5ch; gap: 1.2ch; box-sizing: border-box;",

            // Header
            div {
                style: "color: var(--accent-color, #4af); font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); font-weight: bold; border-bottom: 1px solid var(--border-color); padding-bottom: 0.6ch;",
                "chains"
            }

            // Chain type selector
            div {
                style: "display: flex; gap: 0.8ch; flex-wrap: wrap; flex-shrink: 0;",
                for chain in [ChainType::Llm, ChainType::MapReduce, ChainType::Retrieval, ChainType::DocMapReduce] {
                    {
                        let c = chain.clone();
                        let is_sel = *selected.read() == chain;
                        let label = chain.label();
                        rsx! {
                            button {
                                style: if is_sel {
                                    "background: var(--accent-color, #4af); color: var(--background); border: 1px solid var(--accent-color, #4af); padding: 0.3ch 1ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); cursor: pointer; border-radius: 2px;"
                                } else {
                                    "background: transparent; color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3ch 1ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); cursor: pointer; border-radius: 2px;"
                                },
                                onclick: move |_| {
                                    selected.set(c.clone());
                                    result.set(None);
                                    error_msg.set(None);
                                },
                                "{label}"
                            }
                        }
                    }
                }
            }

            // Description
            div {
                style: "color: var(--muted-color); font-size: var(--font-size-xs, 0.75rem); font-family: var(--font-mono); background: var(--secondary-bg); padding: 0.6ch 1ch; border-left: 2px solid var(--accent-color, #4af); flex-shrink: 0;",
                "{selected.read().description()}"
            }

            // Chain-specific inputs
            div {
                style: "flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 1ch; min-height: 0;",

                match *selected.read() {
                    ChainType::Llm => rsx! {
                        div {
                            style: "display: flex; flex-direction: column; gap: 0.6ch;",
                            label {
                                style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                {"prompt template (use {variable_name} for substitution)"}
                            }
                            textarea {
                                style: "height: 6ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); resize: vertical; width: 100%; box-sizing: border-box;",
                                value: "{llm_template}",
                                oninput: move |e| llm_template.set(e.value()),
                            }
                        }
                        div {
                            style: "display: flex; flex-direction: column; gap: 0.6ch;",
                            label {
                                style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                "variables (JSON object)"
                            }
                            textarea {
                                style: "height: 6ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); resize: vertical; width: 100%; box-sizing: border-box;",
                                value: "{llm_vars}",
                                oninput: move |e| llm_vars.set(e.value()),
                            }
                        }
                        button {
                            style: btn_style,
                            disabled: *is_running.read(),
                            onclick: run_chain,
                            if *is_running.read() { "running..." } else { "run llm chain" }
                        }
                    },

                    ChainType::MapReduce => rsx! {
                        div {
                            style: "display: flex; flex-direction: column; gap: 0.6ch;",
                            label { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", {"map template (use {item})"} }
                            textarea {
                                style: "height: 5ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); resize: vertical; width: 100%; box-sizing: border-box;",
                                value: "{mr_map_template}",
                                oninput: move |e| mr_map_template.set(e.value()),
                            }
                        }
                        div {
                            style: "display: flex; flex-direction: column; gap: 0.6ch;",
                            label { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", {"reduce template (use {summaries})"} }
                            textarea {
                                style: "height: 5ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); resize: vertical; width: 100%; box-sizing: border-box;",
                                value: "{mr_reduce_template}",
                                oninput: move |e| mr_reduce_template.set(e.value()),
                            }
                        }
                        div {
                            style: "display: flex; flex-direction: column; gap: 0.6ch;",
                            label { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "items (one JSON value per line)" }
                            textarea {
                                style: "height: 8ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); resize: vertical; width: 100%; box-sizing: border-box;",
                                value: "{mr_items_text}",
                                oninput: move |e| mr_items_text.set(e.value()),
                            }
                        }
                        button {
                            style: btn_style,
                            disabled: *is_running.read(),
                            onclick: run_chain,
                            if *is_running.read() { "running..." } else { "run map-reduce" }
                        }
                    },

                    ChainType::Retrieval => rsx! {
                        div {
                            style: "display: flex; gap: 1ch; align-items: center; flex-wrap: wrap; flex-shrink: 0;",
                            span { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "collection:" }
                            input {
                                value: "{ret_collection}",
                                style: "flex: 1; min-width: 12ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.3ch 0.6ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                oninput: move |e| ret_collection.set(e.value()),
                            }
                            span { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "top k:" }
                            input {
                                r#type: "number",
                                value: "{ret_top_k}",
                                min: "1", max: "20",
                                style: "width: 5ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.2ch 0.4ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                oninput: move |e| { if let Ok(n) = e.value().parse::<usize>() { ret_top_k.set(n); } }
                            }
                        }
                        div {
                            style: "display: flex; gap: 0.8ch;",
                            input {
                                style: "flex: 1; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.4ch 0.8ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem);",
                                placeholder: "Enter your question to answer with retrieved context...",
                                value: "{ret_query}",
                                oninput: move |e| ret_query.set(e.value()),
                                onkeydown: move |e: KeyboardEvent| {
                                    if e.key() == Key::Enter {
                                        if !*is_running.read() {
                                            let chain_type = selected.read().clone();
                                            is_running.set(true);
                                            result.set(None);
                                            error_msg.set(None);
                                            let api = api_client.clone();
                                            spawn(async move {
                                                let req = RetrievalChainRequest {
                                                    query: ret_query.read().clone(),
                                                    collection: Some(ret_collection.read().clone()),
                                                    top_k: Some(*ret_top_k.read()),
                                                    provider: None,
                                                    model: None,
                                                };
                                                let _ = chain_type; // suppress unused warning
                                                match api.chains_retrieval(&req).await {
                                                    Ok(r) => result.set(Some(r)),
                                                    Err(e) => error_msg.set(Some(e.to_string())),
                                                }
                                                is_running.set(false);
                                            });
                                        }
                                    }
                                }
                            }
                            button {
                                style: btn_style,
                                disabled: *is_running.read(),
                                onclick: run_chain,
                                if *is_running.read() { "..." } else { "run" }
                            }
                        }
                    },

                    ChainType::DocMapReduce => rsx! {
                        div {
                            style: "display: flex; gap: 1ch; align-items: center; flex-shrink: 0;",
                            span { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "chunk size:" }
                            input {
                                r#type: "number",
                                value: "{doc_chunk_size}",
                                min: "128", max: "4096",
                                style: "width: 6ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.2ch 0.4ch; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);",
                                oninput: move |e| { if let Ok(n) = e.value().parse::<usize>() { doc_chunk_size.set(n); } }
                            }
                        }
                        div {
                            style: "display: flex; flex-direction: column; gap: 0.6ch;",
                            label { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", "document text" }
                            textarea {
                                style: "height: 10ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); resize: vertical; width: 100%; box-sizing: border-box;",
                                placeholder: "Paste a long document to process with map-reduce...",
                                value: "{doc_text}",
                                oninput: move |e| doc_text.set(e.value()),
                            }
                        }
                        div {
                            style: "display: flex; flex-direction: column; gap: 0.6ch;",
                            label { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", {"map template (use {document})"} }
                            textarea {
                                style: "height: 5ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); resize: vertical; width: 100%; box-sizing: border-box;",
                                value: "{doc_map_template}",
                                oninput: move |e| doc_map_template.set(e.value()),
                            }
                        }
                        div {
                            style: "display: flex; flex-direction: column; gap: 0.6ch;",
                            label { style: "color: var(--muted-color); font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem);", {"reduce template (use {summaries})"} }
                            textarea {
                                style: "height: 5ch; background: var(--secondary-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5ch; font-family: var(--font-mono); font-size: var(--font-size-sm, 0.85rem); resize: vertical; width: 100%; box-sizing: border-box;",
                                value: "{doc_reduce_template}",
                                oninput: move |e| doc_reduce_template.set(e.value()),
                            }
                        }
                        button {
                            style: btn_style,
                            disabled: *is_running.read(),
                            onclick: run_chain,
                            if *is_running.read() { "running..." } else { "run doc map-reduce" }
                        }
                    },
                }

                // Error
                if let Some(ref err) = *error_msg.read() {
                    div {
                        style: "color: #f44; font-family: var(--font-mono); font-size: var(--font-size-xs, 0.75rem); background: var(--secondary-bg); padding: 0.6ch 1ch; border-left: 2px solid #f44;",
                        "error: {err}"
                    }
                }

                // Result
                if let Some(ref r) = result.read().clone() {
                    { render_chain_output(r) }
                }
            }
        }
    }
}
