//! CLI handlers for chain operations.

use caliche::agent::service::AgentService;
use caliche::chains::*;
use caliche::prompts::PromptTemplate;
use serde::Deserialize;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Deserialize)]
struct SequentialSpec {
    stages: Vec<SequentialStageSpec>,
}

#[derive(Debug, Deserialize)]
struct SequentialStageSpec {
    template: String,
}

#[derive(Debug, Deserialize)]
struct MapReduceSpec {
    map_template: String,
    reduce_template: String,
    #[serde(default = "default_input_key")]
    input_key: String,
}

fn default_input_key() -> String {
    "input".to_string()
}

#[derive(Debug, Deserialize)]
struct ChainConfigFile {
    sequential: Option<SequentialSpec>,
    map_reduce: Option<MapReduceSpec>,
}

fn chain_config_preview(s: &str) -> String {
    let t = s.trim();
    let prefix: String = t.chars().take(80).collect();
    if t.chars().count() > 80 {
        format!("{prefix}...")
    } else {
        prefix
    }
}

/// Load JSON chain config from inline string or from a file path.
fn load_chain_config_json(template: &str) -> anyhow::Result<serde_json::Value> {
    let trimmed = template.trim();
    let text = if trimmed.starts_with('{') {
        trimmed.to_string()
    } else if Path::new(trimmed).is_file() {
        std::fs::read_to_string(trimmed)
            .map_err(|e| anyhow::anyhow!("Failed to read chain config file {}: {}", trimmed, e))?
    } else {
        anyhow::bail!(
            "Chain config must be JSON starting with '{{' or a path to a JSON file. Got: {}",
            chain_config_preview(trimmed)
        );
    };
    serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("Invalid chain config JSON: {}", e))
}

/// Execute a chain from CLI
pub async fn execute_chain(
    chain_type: &str,
    template: Option<String>,
    template_vars: Option<String>,
    message: &str,
    service: AgentService,
) -> anyhow::Result<serde_json::Value> {
    let service_arc = Arc::new(Mutex::new(service));
    match chain_type.to_lowercase().as_str() {
        "llm" => execute_llm_chain(service_arc, template, template_vars, message).await,
        "sequential" => {
            execute_sequential_chain(service_arc, template, template_vars, message).await
        }
        "map-reduce" | "map_reduce" => {
            execute_map_reduce_chain(service_arc, template, template_vars, message).await
        }
        _ => anyhow::bail!("Unknown chain type: {}", chain_type),
    }
}

fn parse_vars_or_message(
    template_vars: Option<String>,
    message: &str,
) -> anyhow::Result<serde_json::Value> {
    if let Some(vars_str) = template_vars {
        Ok(serde_json::from_str(&vars_str)?)
    } else {
        Ok(serde_json::json!({ "input": message }))
    }
}

async fn execute_llm_chain(
    service_arc: Arc<Mutex<AgentService>>,
    template: Option<String>,
    template_vars: Option<String>,
    message: &str,
) -> anyhow::Result<serde_json::Value> {
    let template_str = template.unwrap_or_else(|| "{input}".to_string());
    let prompt = PromptTemplate::new(&template_str);
    let chain = LLMChain::new(service_arc, prompt);
    let vars = parse_vars_or_message(template_vars, message)?;
    let result = chain.run(&vars).await?;
    Ok(result.output)
}

async fn execute_sequential_chain(
    service_arc: Arc<Mutex<AgentService>>,
    template: Option<String>,
    template_vars: Option<String>,
    message: &str,
) -> anyhow::Result<serde_json::Value> {
    let raw = template.ok_or_else(|| {
        anyhow::anyhow!(
            "Sequential chain requires --template with JSON: \
             {{\"sequential\":{{\"stages\":[{{\"template\":\"...\"}},...]}}}} \
             or a path to that JSON file"
        )
    })?;
    let v = load_chain_config_json(&raw)?;
    let cfg: ChainConfigFile = serde_json::from_value(v)
        .map_err(|e| anyhow::anyhow!("Chain config shape invalid for sequential: {}", e))?;
    if cfg.map_reduce.is_some() {
        anyhow::bail!(
            "Chain config cannot include both \"sequential\" and \"map_reduce\"; remove \"map_reduce\" for sequential mode"
        );
    }
    let spec = cfg.sequential.ok_or_else(|| {
        anyhow::anyhow!("Sequential chain config must include a top-level \"sequential\" object")
    })?;
    anyhow::ensure!(
        !spec.stages.is_empty(),
        "sequential.stages must contain at least one stage"
    );

    let vars = parse_vars_or_message(template_vars, message)?;
    let first = &spec.stages[0];
    let mut composed: Arc<dyn Chain> = Arc::new(LLMChain::new(
        Arc::clone(&service_arc),
        PromptTemplate::new(&first.template),
    ));
    for stage in spec.stages.iter().skip(1) {
        let next = Arc::new(LLMChain::new(
            Arc::clone(&service_arc),
            PromptTemplate::new(&stage.template),
        ));
        composed = Arc::new(CompositionSequentialChain::new(composed, next));
    }
    let result = composed.run(&vars).await?;
    Ok(result.output)
}

async fn execute_map_reduce_chain(
    service_arc: Arc<Mutex<AgentService>>,
    template: Option<String>,
    template_vars: Option<String>,
    message: &str,
) -> anyhow::Result<serde_json::Value> {
    let raw = template.ok_or_else(|| {
        anyhow::anyhow!(
            "Map-reduce chain requires --template with JSON: \
             {{\"map_reduce\":{{\"map_template\":\"...\",\"reduce_template\":\"...\",\"input_key\":\"input\"}}}} \
             or a path to that JSON file"
        )
    })?;
    let v = load_chain_config_json(&raw)?;
    let cfg: ChainConfigFile = serde_json::from_value(v)
        .map_err(|e| anyhow::anyhow!("Chain config shape invalid for map-reduce: {}", e))?;
    if cfg.sequential.is_some() {
        anyhow::bail!(
            "Chain config cannot include both \"sequential\" and \"map_reduce\"; remove \"sequential\" for map-reduce mode"
        );
    }
    let spec = cfg.map_reduce.ok_or_else(|| {
        anyhow::anyhow!("Map-reduce chain config must include a top-level \"map_reduce\" object")
    })?;

    let items: serde_json::Value = if let Some(vars_str) = template_vars {
        serde_json::from_str(&vars_str)?
    } else {
        serde_json::json!([{ spec.input_key.clone(): message }])
    };
    let serde_json::Value::Array(arr) = items else {
        anyhow::bail!(
            "Map-reduce requires --template-vars to be a JSON array of objects (one per item), \
             or omit it to map a single message using input_key"
        );
    };
    anyhow::ensure!(
        !arr.is_empty(),
        "Map-reduce requires at least one item in the input array"
    );

    let map_chain: Arc<dyn Chain> = Arc::new(LLMChain::new(
        Arc::clone(&service_arc),
        PromptTemplate::new(&spec.map_template),
    ));
    let reduce_chain: Arc<dyn Chain> = Arc::new(LLMChain::new(
        Arc::clone(&service_arc),
        PromptTemplate::new(&spec.reduce_template),
    ));
    let chain = MapReduceChain::new(map_chain, reduce_chain);
    let result = chain.run(&serde_json::Value::Array(arr)).await?;
    Ok(result.output)
}

/// Display chain result
pub fn display_chain_result(result: &serde_json::Value) {
    println!("\n=== Chain Result ===");
    println!(
        "{}",
        serde_json::to_string_pretty(result).unwrap_or_else(|_| format!("{}", result))
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_config_preview_truncates() {
        let short = "hello";
        assert_eq!(chain_config_preview(short), "hello");
        let long = "x".repeat(100);
        let p = chain_config_preview(&long);
        assert_eq!(p.len(), 83);
        assert!(p.ends_with("..."));
    }

    #[test]
    fn load_chain_config_json_inline_object() {
        let Ok(v) = load_chain_config_json(r#"  {"sequential":{"stages":[]}}  "#) else {
            panic!("inline json should parse");
        };
        assert!(v.get("sequential").is_some());
    }

    #[test]
    fn load_chain_config_json_from_file() {
        let Ok(dir) = tempfile::tempdir() else {
            panic!("tempdir for chain config test");
        };
        let path = dir.path().join("chain.json");
        let body = r#"{"map_reduce":{"map_template":"m","reduce_template":"r"}}"#;
        if let Err(e) = std::fs::write(&path, body) {
            panic!("write chain fixture: {e}");
        }
        let Some(p) = path.to_str() else {
            panic!("fixture path should be utf-8");
        };
        let Ok(v) = load_chain_config_json(p) else {
            panic!("chain json file should load");
        };
        assert!(v.get("map_reduce").is_some());
    }

    #[test]
    fn load_chain_config_json_rejects_non_json_non_file() {
        let Err(err) = load_chain_config_json("not-json-and-not-a-file-path") else {
            panic!("non-json non-path should fail");
        };
        let s = format!("{err:#}");
        assert!(
            s.contains("Chain config must be JSON") || s.contains("Got:"),
            "{s}"
        );
    }
}
