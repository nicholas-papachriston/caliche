use crate::agent::tools::metadata::ToolMetadata;
use anyhow::{Result, bail};
use std::collections::{HashMap, HashSet, VecDeque};

/// Represents a chained tool call in an execution plan
#[derive(Debug, Clone)]
pub struct ChainedToolCall {
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub phase: usize,
    pub can_run_in_parallel: bool,
}

impl ChainedToolCall {
    pub fn new(tool_name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments,
            phase: 0,
            can_run_in_parallel: true,
        }
    }
}

/// A phase of execution containing tools that can run in parallel
#[derive(Debug, Clone)]
pub struct ExecutionPhase {
    pub phase_number: usize,
    /// Tools that can run in parallel in this phase
    pub tools: Vec<String>,
    /// Tools that must complete before this phase
    pub dependencies: HashSet<String>,
}

impl ExecutionPhase {
    #[must_use]
    pub fn new(phase_number: usize) -> Self {
        Self {
            phase_number,
            tools: Vec::new(),
            dependencies: HashSet::new(),
        }
    }

    pub fn add_tool(&mut self, tool_name: impl Into<String>) {
        self.tools.push(tool_name.into());
    }

    pub fn add_dependency(&mut self, tool_name: impl Into<String>) {
        self.dependencies.insert(tool_name.into());
    }
}

/// Execution plan with ordered phases
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub phases: Vec<ExecutionPhase>,
    /// Mapping of tool names to their phase number
    pub tool_to_phase: HashMap<String, usize>,
    /// Estimated total cost
    pub estimated_cost: u32,
    /// Estimated execution time in milliseconds
    pub estimated_time_ms: u64,
    /// Explanation of the plan
    pub explanation: String,
}

impl ExecutionPlan {
    #[must_use]
    pub fn new() -> Self {
        Self {
            phases: Vec::new(),
            tool_to_phase: HashMap::new(),
            estimated_cost: 0,
            estimated_time_ms: 0,
            explanation: String::new(),
        }
    }

    pub fn add_phase(&mut self, phase: ExecutionPhase) {
        for tool in &phase.tools {
            self.tool_to_phase.insert(tool.clone(), phase.phase_number);
        }
        self.phases.push(phase);
    }

    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = explanation.into();
        self
    }

    /// Get tools that can run in parallel
    #[must_use]
    pub fn parallel_groups(&self) -> Vec<Vec<String>> {
        self.phases.iter().map(|p| p.tools.clone()).collect()
    }

    /// Get sequential steps (flattened phases)
    #[must_use]
    pub fn sequential_steps(&self) -> Vec<String> {
        self.phases.iter().flat_map(|p| p.tools.clone()).collect()
    }
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds and analyzes dependency graphs for tools
pub struct DependencyGraph {
    /// Adjacency list: tool -> dependencies
    dependencies: HashMap<String, HashSet<String>>,
    /// Reverse adjacency list: tool -> dependents
    dependents: HashMap<String, HashSet<String>>,
    /// All tools in the graph
    all_tools: HashSet<String>,
}

impl DependencyGraph {
    /// Create a new dependency graph from tool metadata
    #[must_use]
    pub fn build_from_tools(tools: &[ToolMetadata]) -> Self {
        let mut graph = Self {
            dependencies: HashMap::new(),
            dependents: HashMap::new(),
            all_tools: HashSet::new(),
        };

        // Add all tools
        for tool in tools {
            graph.all_tools.insert(tool.name.clone());
            graph.dependencies.entry(tool.name.clone()).or_default();
            graph.dependents.entry(tool.name.clone()).or_default();

            // Add explicit dependencies
            for dep in &tool.dependencies.requires_prior_tools {
                graph
                    .dependencies
                    .entry(tool.name.clone())
                    .or_default()
                    .insert(dep.clone());

                graph
                    .dependents
                    .entry(dep.clone())
                    .or_default()
                    .insert(tool.name.clone());
            }
        }

        graph
    }

    /// Check if the graph has cycles (returns true if acyclic)
    pub fn validate_acyclic(&self) -> Result<()> {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for tool in &self.all_tools {
            if !visited.contains(tool) && self.has_cycle_dfs(tool, &mut visited, &mut rec_stack) {
                bail!("Cyclic dependency detected involving tool: {tool}");
            }
        }

        Ok(())
    }

    /// DFS helper for cycle detection
    fn has_cycle_dfs(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        if let Some(deps) = self.dependencies.get(node) {
            for dep in deps {
                if !visited.contains(dep) {
                    if self.has_cycle_dfs(dep, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(dep) {
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }

    /// Find execution order using topological sort
    pub fn find_execution_order(&self) -> Result<Vec<Vec<String>>> {
        self.validate_acyclic()?;

        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        let mut phases: Vec<Vec<String>> = Vec::new();

        // Calculate in-degrees
        for tool in &self.all_tools {
            let degree = self
                .dependencies
                .get(tool)
                .map_or(0, std::collections::HashSet::len);
            in_degree.insert(tool.clone(), degree);

            if degree == 0 {
                queue.push_back(tool.clone());
            }
        }

        // Process by phases (BFS-style topological sort)
        while !queue.is_empty() {
            let mut current_phase = Vec::new();
            let phase_size = queue.len();

            for _ in 0..phase_size {
                if let Some(tool) = queue.pop_front() {
                    current_phase.push(tool.clone());

                    // Reduce in-degree of dependents
                    if let Some(deps) = self.dependents.get(&tool) {
                        for dependent in deps {
                            if let Some(degree) = in_degree.get_mut(dependent) {
                                *degree -= 1;
                                if *degree == 0 {
                                    queue.push_back(dependent.clone());
                                }
                            }
                        }
                    }
                }
            }

            if !current_phase.is_empty() {
                phases.push(current_phase);
            }
        }

        if phases.iter().flat_map(|p| p.iter()).count() != self.all_tools.len() {
            bail!("Failed to process all tools in topological sort");
        }

        Ok(phases)
    }

    /// Get tools with no dependencies
    #[must_use]
    pub fn get_root_tools(&self) -> Vec<String> {
        self.all_tools
            .iter()
            .filter(|tool| {
                self.dependencies
                    .get(*tool)
                    .map_or(true, std::collections::HashSet::is_empty)
            })
            .cloned()
            .collect()
    }

    /// Get direct dependencies of a tool
    #[must_use]
    pub fn get_dependencies(&self, tool_name: &str) -> Option<HashSet<String>> {
        self.dependencies.get(tool_name).cloned()
    }

    /// Get tools that depend on this tool
    #[must_use]
    pub fn get_dependents(&self, tool_name: &str) -> Option<HashSet<String>> {
        self.dependents.get(tool_name).cloned()
    }
}

/// Plans tool execution based on dependencies
pub struct ExecutionPlanner;

impl ExecutionPlanner {
    /// Create an execution plan from tools and their metadata
    pub fn plan(tools: &[ToolMetadata]) -> Result<ExecutionPlan> {
        let graph = DependencyGraph::build_from_tools(tools);
        let execution_order = graph.find_execution_order()?;

        let mut plan = ExecutionPlan::new();

        for (phase_num, phase_tools) in execution_order.iter().enumerate() {
            let mut phase = ExecutionPhase::new(phase_num);

            for tool_name in phase_tools {
                phase.add_tool(tool_name.clone());

                // Add dependencies as phase dependencies
                if let Some(deps) = graph.get_dependencies(tool_name) {
                    for dep in deps {
                        phase.add_dependency(dep);
                    }
                }
            }

            plan.add_phase(phase);
        }

        let phase_count = plan.phases.len();
        let tool_count = tools.len();
        Ok(plan.with_explanation(format!(
            "Execution plan with {phase_count} phases for {tool_count} tools"
        )))
    }

    /// Estimate cost of execution plan
    #[must_use]
    pub fn estimate_cost(plan: &ExecutionPlan, tools: &[ToolMetadata]) -> u32 {
        let tool_map: HashMap<&str, &ToolMetadata> =
            tools.iter().map(|t| (t.name.as_str(), t)).collect();

        let mut total_cost = 0;

        for phase in &plan.phases {
            // Costs in parallel phase are counted once (not multiplied)
            for tool_name in &phase.tools {
                if let Some(tool) = tool_map.get(tool_name.as_str()) {
                    total_cost += tool.cost.score();
                }
            }
        }

        total_cost
    }

    /// Estimate execution time (assumes parallel execution within phases)
    #[must_use]
    pub fn estimate_time(plan: &ExecutionPlan, tools: &[ToolMetadata]) -> u64 {
        let tool_map: HashMap<&str, &ToolMetadata> =
            tools.iter().map(|t| (t.name.as_str(), t)).collect();

        let mut total_time = 0u64;

        for phase in &plan.phases {
            // Time per phase is max time of tools in that phase
            let phase_time = phase
                .tools
                .iter()
                .filter_map(|name| tool_map.get(name.as_str()))
                .map(|tool| {
                    // Estimate time based on cost
                    match tool.cost {
                        crate::agent::tools::metadata::ToolCost::Free => 10,
                        crate::agent::tools::metadata::ToolCost::Low => 100,
                        crate::agent::tools::metadata::ToolCost::Medium => 500,
                        crate::agent::tools::metadata::ToolCost::High => 2000,
                    }
                })
                .max()
                .unwrap_or(0);

            total_time += phase_time;
        }

        total_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::metadata::ToolCategory;

    #[test]
    fn test_execution_phase() {
        let mut phase = ExecutionPhase::new(0);
        phase.add_tool("tool1");
        phase.add_tool("tool2");
        phase.add_dependency("tool0");

        assert_eq!(phase.tools.len(), 2);
        assert_eq!(phase.dependencies.len(), 1);
    }

    #[test]
    fn test_dependency_graph_acyclic() {
        let tools = vec![
            ToolMetadata::new("tool1", "First", ToolCategory::Utility),
            ToolMetadata::new("tool2", "Second", ToolCategory::Utility),
        ];

        let graph = DependencyGraph::build_from_tools(&tools);
        assert!(graph.validate_acyclic().is_ok());
    }

    #[test]
    fn test_dependency_graph_root_tools() {
        let tools = vec![
            ToolMetadata::new("tool1", "First", ToolCategory::Utility),
            ToolMetadata::new("tool2", "Second", ToolCategory::Utility),
        ];

        let graph = DependencyGraph::build_from_tools(&tools);
        let roots = graph.get_root_tools();

        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_execution_planner() {
        let tools = vec![
            ToolMetadata::new("tool1", "First", ToolCategory::Utility),
            ToolMetadata::new("tool2", "Second", ToolCategory::Utility),
        ];

        let plan = ExecutionPlanner::plan(&tools);
        assert!(plan.is_ok());
    }

    #[test]
    fn test_execution_plan_parallel_groups() {
        let mut plan = ExecutionPlan::new();
        let mut phase = ExecutionPhase::new(0);
        phase.add_tool("tool1");
        phase.add_tool("tool2");
        plan.add_phase(phase);

        let groups = plan.parallel_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }
}
