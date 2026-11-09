//! Exploration state types for the graph explorer tool (Level 3).

use dripstone::graph::{Edge, Node};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Configurable frontier prioritization scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierPrioritizer {
    pub edge_type_weights: HashMap<String, f32>,
    pub default_weight: f32,
    pub degree_boost: bool,
    pub recency_boost: bool,
}

impl Default for FrontierPrioritizer {
    fn default() -> Self {
        Self {
            edge_type_weights: HashMap::new(),
            default_weight: 1.0,
            degree_boost: false,
            recency_boost: false,
        }
    }
}

impl FrontierPrioritizer {
    /// Security-oriented default: weights identity and credential edges highest
    #[must_use]
    pub fn security_default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("AUTHENTICATES".to_string(), 3.0);
        weights.insert("HAS_CREDENTIAL".to_string(), 3.0);
        weights.insert("OWNS".to_string(), 2.5);
        weights.insert("MEMBER_OF".to_string(), 2.0);
        weights.insert("HAS_PERMISSION".to_string(), 2.0);
        weights.insert("CONNECTS_TO".to_string(), 1.5);
        weights.insert("RUNS_ON".to_string(), 1.5);
        weights.insert("MONITORS".to_string(), 1.0);
        weights.insert("LOGS_TO".to_string(), 0.8);
        Self {
            edge_type_weights: weights,
            default_weight: 1.0,
            degree_boost: true,
            recency_boost: true,
        }
    }

    #[must_use]
    pub fn compute_priority(&self, edge_type: &str, _target_node: Option<&Node>) -> f32 {
        self.edge_type_weights
            .get(edge_type)
            .copied()
            .unwrap_or(self.default_weight)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierEntry {
    pub source_node_id: String,
    pub edge_type: String,
    pub target_node_id: Option<String>,
    pub depth: usize,
    pub priority: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationStep {
    pub step_number: usize,
    pub action: ExplorationAction,
    pub nodes_discovered: usize,
    pub edges_discovered: usize,
    pub timestamp_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplorationAction {
    StartedFrom { node_id: String },
    Expanded { count: usize, from_node: String },
    Backtracked { to_node: String },
    Annotated { node_id: String, annotation: String },
    SearchedExplored { query: String, matches: usize },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExplorationStats {
    pub nodes_visited: usize,
    pub edges_traversed: usize,
    pub frontier_size: usize,
    pub max_depth_reached: usize,
    pub types_encountered: HashMap<String, usize>,
    pub edge_types_encountered: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationState {
    pub visited_nodes: HashMap<String, Node>,
    pub visited_edges: Vec<Edge>,
    pub visited_node_ids: HashSet<String>,
    pub frontier: VecDeque<FrontierEntry>,
    pub annotations: HashMap<String, Vec<String>>,
    pub exploration_log: Vec<ExplorationStep>,
    pub stats: ExplorationStats,
    pub current_focus_node: Option<String>,
    pub start_node_id: Option<String>,
}

impl Default for ExplorationState {
    fn default() -> Self {
        Self::new()
    }
}

impl ExplorationState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            visited_nodes: HashMap::new(),
            visited_edges: Vec::new(),
            visited_node_ids: HashSet::new(),
            frontier: VecDeque::new(),
            annotations: HashMap::new(),
            exploration_log: Vec::new(),
            stats: ExplorationStats::default(),
            current_focus_node: None,
            start_node_id: None,
        }
    }

    pub fn visit_node(&mut self, node: Node, depth: usize) {
        if self.visited_node_ids.insert(node.id.clone()) {
            *self
                .stats
                .types_encountered
                .entry(node.node_type.clone())
                .or_insert(0) += 1;
            self.stats.nodes_visited += 1;
            self.stats.max_depth_reached = self.stats.max_depth_reached.max(depth);
            self.visited_nodes.insert(node.id.clone(), node);
        }
    }

    pub fn visit_edge(&mut self, edge: Edge, depth: usize, prioritizer: &FrontierPrioritizer) {
        *self
            .stats
            .edge_types_encountered
            .entry(edge.edge_type.clone())
            .or_insert(0) += 1;
        self.stats.edges_traversed += 1;
        if !self.visited_node_ids.contains(&edge.target_id) {
            let priority = prioritizer.compute_priority(&edge.edge_type, None);
            self.frontier.push_back(FrontierEntry {
                source_node_id: edge.source_id.clone(),
                edge_type: edge.edge_type.clone(),
                target_node_id: Some(edge.target_id.clone()),
                depth: depth + 1,
                priority,
            });
        }
        self.visited_edges.push(edge);
        self.stats.frontier_size = self.frontier.len();
    }

    #[must_use]
    pub fn to_summary(&self) -> String {
        let types: Vec<String> = self
            .stats
            .types_encountered
            .iter()
            .map(|(t, c)| format!("{t}({c})"))
            .collect();
        let frontier_types: HashMap<&str, usize> =
            self.frontier.iter().fold(HashMap::new(), |mut acc, f| {
                *acc.entry(f.edge_type.as_str()).or_insert(0) += 1;
                acc
            });
        let ft: Vec<String> = frontier_types
            .iter()
            .map(|(t, c)| format!("{t}({c})"))
            .collect();
        let ann_count: usize = self.annotations.values().map(std::vec::Vec::len).sum();
        format!(
            "[Explored {} nodes, {} edges | Frontier: {} [{}] | Types: {} | Depth: {} | Annotations: {} | Focus: {}]",
            self.stats.nodes_visited,
            self.stats.edges_traversed,
            self.stats.frontier_size,
            ft.join(", "),
            types.join(", "),
            self.stats.max_depth_reached,
            ann_count,
            self.current_focus_node.as_deref().unwrap_or("none")
        )
    }

    #[must_use]
    pub fn search_explored(
        &self,
        node_type: Option<&str>,
        prop_key: Option<&str>,
        prop_val: Option<&str>,
    ) -> Vec<&Node> {
        self.visited_nodes
            .values()
            .filter(|n| {
                node_type.is_none_or(|t| n.node_type == t)
                    && prop_key.is_none_or(|k| {
                        prop_val.map_or_else(
                            || n.properties.contains_key(k),
                            |v| n.properties.get(k).and_then(|pv| pv.as_str()) == Some(v),
                        )
                    })
            })
            .collect()
    }
}
