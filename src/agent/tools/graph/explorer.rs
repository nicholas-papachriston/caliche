//! Graph explorer tool (Level 3): stateful traversal with frontier and annotations.

use super::state::{ExplorationAction, ExplorationState, ExplorationStep, FrontierPrioritizer};
use crate::agent::tools::common::params;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use async_trait::async_trait;
use dripstone::graph::QueryableGraphBackend;
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Stateful graph exploration tool with frontier and annotations.
pub struct GraphExplorerTool {
    backend: Arc<dyn QueryableGraphBackend>,
    state: Arc<RwLock<ExplorationState>>,
    prioritizer: FrontierPrioritizer,
}

impl std::fmt::Debug for GraphExplorerTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphExplorerTool")
            .field("prioritizer", &self.prioritizer)
            .finish()
    }
}

impl GraphExplorerTool {
    pub fn new(backend: Arc<dyn QueryableGraphBackend>) -> Self {
        Self {
            backend,
            state: Arc::new(RwLock::new(ExplorationState::new())),
            prioritizer: FrontierPrioritizer::default(),
        }
    }

    pub fn with_prioritizer(
        backend: Arc<dyn QueryableGraphBackend>,
        prioritizer: FrontierPrioritizer,
    ) -> Self {
        Self {
            backend,
            state: Arc::new(RwLock::new(ExplorationState::new())),
            prioritizer,
        }
    }

    fn timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    async fn op_start_exploration(&self, arguments: &serde_json::Value) -> ToolResult {
        let node_id = match arguments["node_id"].as_str() {
            Some(id) => id,
            None => {
                return ToolResult::error("Missing 'node_id' for start_exploration".to_string());
            }
        };
        let node = match self.backend.get_node(node_id).await {
            Ok(Some(n)) => n,
            Ok(None) => return ToolResult::error(format!("Node not found: {node_id}")),
            Err(e) => return ToolResult::error(e.to_string()),
        };
        let neighbors = match self.backend.get_neighbors(node_id, None).await {
            Ok(n) => n,
            Err(e) => return ToolResult::error(e.to_string()),
        };
        let mut state = self.state.write().await;
        *state = ExplorationState::new();
        state.start_node_id = Some(node_id.to_string());
        state.current_focus_node = Some(node_id.to_string());
        state.visit_node(node, 0);
        let num_neighbors = neighbors.len();
        for n in neighbors {
            state.visit_edge(n.edge, 0, &self.prioritizer);
        }
        let step_num = state.exploration_log.len() + 1;
        state.exploration_log.push(ExplorationStep {
            step_number: step_num,
            action: ExplorationAction::StartedFrom {
                node_id: node_id.to_string(),
            },
            nodes_discovered: 1,
            edges_discovered: num_neighbors,
            timestamp_ms: Self::timestamp_ms(),
        });
        let summary = state.to_summary();
        ToolResult::success(format!(
            "Started exploration from node {}. Frontier: {} entries.\n\n{}",
            node_id,
            state.frontier.len(),
            summary
        ))
    }

    async fn op_expand(&self, arguments: &serde_json::Value) -> ToolResult {
        let count = arguments["count"].as_u64().unwrap_or(5) as usize;
        let count = count.clamp(1, 20);
        let edge_type_filter: Option<Vec<String>> =
            arguments["edge_type_filter"].as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            });
        let priority_threshold: Option<f32> =
            arguments["priority_threshold"].as_f64().map(|f| f as f32);
        let mut state = self.state.write().await;
        let to_expand = select_frontier_entries(
            &mut state,
            count,
            edge_type_filter.as_deref(),
            priority_threshold,
        );
        let mut discovered = 0usize;
        for entry in to_expand {
            let Some(target_id) = entry.target_node_id.as_deref() else {
                continue;
            };
            let Ok(Some(node)) = self.backend.get_node(target_id).await else {
                continue;
            };
            state.visit_node(node, entry.depth);
            discovered += 1;
            let Ok(neighbors) = self.backend.get_neighbors(target_id, None).await else {
                continue;
            };
            for n in neighbors {
                state.visit_edge(n.edge, entry.depth, &self.prioritizer);
            }
        }
        let step_num = state.exploration_log.len() + 1;
        let from_node = state.current_focus_node.clone().unwrap_or_default();
        state.exploration_log.push(ExplorationStep {
            step_number: step_num,
            action: ExplorationAction::Expanded {
                count: discovered,
                from_node,
            },
            nodes_discovered: discovered,
            edges_discovered: 0,
            timestamp_ms: Self::timestamp_ms(),
        });
        let summary = state.to_summary();
        ToolResult::success(format!(
            "Expanded {} nodes. Frontier: {} entries.\n\n{}",
            discovered,
            state.frontier.len(),
            summary
        ))
    }

    async fn op_get_state(&self) -> ToolResult {
        let state = self.state.read().await;
        let summary = state.to_summary();
        let frontier_preview: Vec<serde_json::Value> = state
            .frontier
            .iter()
            .take(20)
            .map(|f| {
                json!({
                    "source": f.source_node_id,
                    "edge_type": f.edge_type,
                    "target": f.target_node_id,
                    "depth": f.depth,
                    "priority": f.priority
                })
            })
            .collect();
        ToolResult::success(format!(
            "{}\n\nFrontier (first 20): {:?}\nAnnotations: {:?}",
            summary, frontier_preview, state.annotations
        ))
    }

    async fn op_search_explored(&self, arguments: &serde_json::Value) -> ToolResult {
        let state = self.state.read().await;
        let matches = state.search_explored(
            arguments["search_type"].as_str(),
            arguments["search_property"].as_str(),
            arguments["search_value"].as_str(),
        );
        let list: Vec<serde_json::Value> = matches
            .iter()
            .map(|n| {
                json!({
                    "id": n.id,
                    "node_type": n.node_type,
                    "properties": n.properties
                })
            })
            .collect();
        let summary = state.to_summary();
        ToolResult::success(format!(
            "Found {} nodes.\n{}\n\n{}",
            list.len(),
            serde_json::to_string_pretty(&list).unwrap_or_else(|_| "[]".to_string()),
            summary
        ))
    }

    async fn op_annotate(&self, arguments: &serde_json::Value) -> ToolResult {
        let node_id = match arguments["node_id"].as_str() {
            Some(id) => id,
            None => return ToolResult::error("Missing 'node_id' for annotate"),
        };
        let annotation = match arguments["annotation"].as_str() {
            Some(a) => a,
            None => return ToolResult::error("Missing 'annotation' for annotate"),
        };
        let mut state = self.state.write().await;
        if !state.visited_node_ids.contains(node_id) {
            return ToolResult::error(format!("Node {node_id} not in explored set"));
        }
        state
            .annotations
            .entry(node_id.to_string())
            .or_default()
            .push(annotation.to_string());
        for f in &mut state.frontier {
            if f.source_node_id == node_id {
                f.priority += 1.0;
            }
        }
        let step_num = state.exploration_log.len() + 1;
        state.exploration_log.push(ExplorationStep {
            step_number: step_num,
            action: ExplorationAction::Annotated {
                node_id: node_id.to_string(),
                annotation: annotation.to_string(),
            },
            nodes_discovered: 0,
            edges_discovered: 0,
            timestamp_ms: Self::timestamp_ms(),
        });
        let summary = state.to_summary();
        ToolResult::success(format!("Annotated node {node_id}.\n\n{summary}"))
    }

    async fn op_backtrack(&self, arguments: &serde_json::Value) -> ToolResult {
        let node_id = match arguments["node_id"].as_str() {
            Some(id) => id,
            None => return ToolResult::error("Missing 'node_id' for backtrack"),
        };
        let mut state = self.state.write().await;
        if !state.visited_node_ids.contains(node_id) {
            return ToolResult::error(format!("Node {node_id} not in explored set"));
        }
        state.current_focus_node = Some(node_id.to_string());
        let neighbors = match self.backend.get_neighbors(node_id, None).await {
            Ok(n) => n,
            Err(e) => return ToolResult::error(e.to_string()),
        };
        let num_neighbors = neighbors.len();
        for n in neighbors {
            if !state.visited_node_ids.contains(&n.node.id) {
                state.visit_edge(n.edge, 0, &self.prioritizer);
            }
        }
        let step_num = state.exploration_log.len() + 1;
        state.exploration_log.push(ExplorationStep {
            step_number: step_num,
            action: ExplorationAction::Backtracked {
                to_node: node_id.to_string(),
            },
            nodes_discovered: 0,
            edges_discovered: num_neighbors,
            timestamp_ms: Self::timestamp_ms(),
        });
        let summary = state.to_summary();
        ToolResult::success(format!("Backtracked to {node_id}.\n\n{summary}"))
    }

    async fn op_get_subgraph(&self) -> ToolResult {
        let state = self.state.read().await;
        let subgraph = json!({
            "nodes": state.visited_nodes.values().collect::<Vec<_>>(),
            "edges": state.visited_edges,
            "annotations": state.annotations
        });
        let summary = state.to_summary();
        ToolResult::success(format!(
            "{}\n\n{}",
            serde_json::to_string_pretty(&subgraph).unwrap_or_else(|_| "{}".to_string()),
            summary
        ))
    }

    async fn op_reset(&self) -> ToolResult {
        let mut state = self.state.write().await;
        *state = ExplorationState::new();
        ToolResult::success("Exploration state reset.")
    }

    async fn op_save_state(&self) -> ToolResult {
        let state = self.state.read().await;
        let json_str = match serde_json::to_string(&*state) {
            Ok(s) => s,
            Err(e) => return ToolResult::error(e.to_string()),
        };
        ToolResult::success(format!(
            "State saved ({} bytes). Persist this JSON for load_state.\n\n{}",
            json_str.len(),
            json_str
        ))
    }

    async fn op_load_state(&self, arguments: &serde_json::Value) -> ToolResult {
        let state_json = match arguments["state_json"].as_str() {
            Some(s) => s,
            None => return ToolResult::error("Missing 'state_json' for load_state".to_string()),
        };
        let loaded: ExplorationState = match serde_json::from_str(state_json) {
            Ok(s) => s,
            Err(e) => return ToolResult::error(format!("Invalid state JSON: {e}")),
        };
        if loaded.stats.nodes_visited != loaded.visited_nodes.len()
            || loaded.stats.edges_traversed != loaded.visited_edges.len()
        {
            return ToolResult::error("State validation failed: stats mismatch".to_string());
        }
        {
            let mut state = self.state.write().await;
            *state = loaded;
        }
        let state = self.state.read().await;
        let summary = state.to_summary();
        ToolResult::success(format!("State restored.\n\n{summary}"))
    }
}

fn select_frontier_entries(
    state: &mut ExplorationState,
    count: usize,
    edge_type_filter: Option<&[String]>,
    priority_threshold: Option<f32>,
) -> Vec<super::state::FrontierEntry> {
    let mut to_expand = Vec::new();
    let mut rest = Vec::new();
    while to_expand.len() < count {
        let Some(entry) = state.frontier.pop_front() else {
            break;
        };
        if let Some(filter) = edge_type_filter
            && !filter.contains(&entry.edge_type)
        {
            rest.push(entry);
            continue;
        }
        if let Some(thresh) = priority_threshold
            && entry.priority < thresh
        {
            rest.push(entry);
            continue;
        }
        if let Some(ref tid) = entry.target_node_id
            && !state.visited_node_ids.contains(tid)
        {
            to_expand.push(entry);
        }
    }
    for e in rest {
        state.frontier.push_back(e);
    }
    state.stats.frontier_size = state.frontier.len();
    to_expand
}

const NAME: &str = "graph_explorer";
const DESCRIPTION: &str = "Stateful graph exploration tool. Maintains a working model of the subgraph you've explored so far. Start with 'start_exploration' from a node, then 'expand' to traverse frontier edges. Every response includes a state summary showing what's been explored, frontier size, and annotations. Use 'annotate' to mark nodes (e.g., 'suspicious', 'dead-end'). Use 'search_explored' to find nodes in the explored subgraph without new DB queries. Use 'backtrack' to return to a previously visited node and explore its other edges.";

impl ToolMeta for GraphExplorerTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["start_exploration", "expand", "get_state", "search_explored",
                             "annotate", "backtrack", "get_subgraph", "reset",
                             "save_state", "load_state"],
                    "description": "Exploration operation. Use save_state/load_state to persist exploration across sessions."
                },
                "node_id": { "type": "string",
                    "description": "Node ID for start_exploration, backtrack, or annotate" },
                "count": { "type": "integer", "default": 5, "minimum": 1, "maximum": 20,
                    "description": "Number of frontier entries to expand" },
                "edge_type_filter": { "type": "array", "items": {"type": "string"},
                    "description": "Only expand frontier entries with these edge types" },
                "priority_threshold": { "type": "number",
                    "description": "Only expand entries with priority >= this value" },
                "annotation": { "type": "string",
                    "description": "Text to annotate a node with (for 'annotate')" },
                "search_type": { "type": "string",
                    "description": "Node type filter for search_explored" },
                "search_property": { "type": "string",
                    "description": "Property key for search_explored" },
                "search_value": { "type": "string",
                    "description": "Property value for search_explored" },
                "state_json": { "type": "string",
                    "description": "JSON string of a previously saved ExplorationState (for 'load_state')" }
            },
            "required": ["operation"]
        })
    }
}

#[async_trait]
impl Tool for GraphExplorerTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let operation = match params::extract_operation(&arguments) {
            Ok(op) => op,
            Err(e) => return ToolResult::error(e),
        };
        match operation.as_str() {
            "start_exploration" => self.op_start_exploration(&arguments).await,
            "expand" => self.op_expand(&arguments).await,
            "get_state" => self.op_get_state().await,
            "search_explored" => self.op_search_explored(&arguments).await,
            "annotate" => self.op_annotate(&arguments).await,
            "backtrack" => self.op_backtrack(&arguments).await,
            "get_subgraph" => self.op_get_subgraph().await,
            "reset" => self.op_reset().await,
            "save_state" => self.op_save_state().await,
            "load_state" => self.op_load_state(&arguments).await,
            _ => ToolResult::error(format!("Unknown operation: '{operation}'")),
        }
    }
}
