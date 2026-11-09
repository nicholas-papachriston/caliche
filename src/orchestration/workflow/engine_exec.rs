//! `WorkflowEngine` method implementations.

use super::*;

impl WorkflowEngine {
    /// Execute workflow as a state machine (supports cycles)
    pub async fn execute_state_machine(
        &self,
        nodes: &[WorkflowNode],
        edges: &[WorkflowEdge],
        entry_point: &str,
        context: &mut ExecutionContext,
    ) -> Result<Vec<NodeExecutionResult>> {
        let nodes_map = build_nodes_map(nodes);
        let outgoing_edges = build_outgoing_edges(edges);

        let mut current_node_id = entry_point.to_string();
        let mut results = Vec::new();
        let mut step_count = 0;

        loop {
            if context.is_cancelled() {
                context.state = WorkflowState::Cancelled;
                context.end_time = Some(Utc::now());
                break;
            }
            if step_count >= MAX_STEPS {
                anyhow::bail!(
                    "State machine exceeded maximum steps ({}). Possible infinite loop.",
                    MAX_STEPS
                );
            }
            step_count += 1;

            // Get current node
            let node = nodes_map
                .get(current_node_id.as_str())
                .context(format!("Node '{}' not found", current_node_id))?;

            context.current_node_id = Some(current_node_id.clone());

            // Execute node
            let result = self.execute_node(node, nodes, edges, context).await?;
            record_and_update_context(context, node, &result);
            let node_output_ref = result.output.clone();
            let failed = self.should_halt_on_node_error(&result);
            results.push(result);

            if failed {
                if let Some(last) = results.last() {
                    self.mark_failed_on_node_error(context, last);
                }
                break;
            }

            // Nodes may override control flow (e.g., Parallel/Loop)
            if let Some(next) = get_forced_next_node(&results) {
                current_node_id = next;
                continue;
            }

            // Check if workflow was paused (HumanInterrupt)
            if context.state == WorkflowState::Paused {
                // Workflow paused, stop execution
                tracing::info!(
                    execution_id = %context.execution_id,
                    node_id = %current_node_id,
                    "Workflow paused at HumanInterrupt node"
                );
                break;
            }

            // Find next node by evaluating outgoing edges - use string reference
            let edges_from_current = outgoing_edges.get(current_node_id.as_str());

            let Some(edges) = edges_from_current else {
                // No outgoing edges - end state reached
                break;
            };

            if edges.is_empty() {
                // No outgoing edges - end state reached
                break;
            }

            let next_node_id = select_next_edge(
                edges,
                context,
                node_output_ref.as_ref(),
                Self::evaluate_condition,
            );

            // If we found a next node, transition; otherwise end
            if let Some(next_id) = next_node_id {
                current_node_id = next_id;
            } else {
                // No matching edge - end state
                break;
            }
        }

        Ok(results)
    }

    /// Execute nodes in topological order
    pub async fn execute_dag(
        &self,
        nodes: &[WorkflowNode],
        edges: &[WorkflowEdge],
        context: &mut ExecutionContext,
    ) -> Result<Vec<NodeExecutionResult>> {
        // Build adjacency list and in-degree map
        // Note: We need to clone node IDs here because we need owned Strings for the queue
        // However, we can reduce clones by reusing the same String references where possible
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        let mut in_degree: HashMap<String, usize> = HashMap::new();

        for node in nodes {
            in_degree.insert(node.id.clone(), 0);
        }

        for edge in edges {
            adj.entry(edge.from_node.clone())
                .or_default()
                .push(edge.to_node.clone());
            if let Some(deg) = in_degree.get_mut(&edge.to_node) {
                *deg += 1;
            } else {
                anyhow::bail!("Edge references non-existent node: {}", edge.to_node);
            }
        }

        // Topological sort using Kahn's algorithm
        // Clone node IDs only once when initializing the queue
        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(id, _)| id.clone())
            .collect();

        let mut execution_order = Vec::new();
        let nodes_map = build_nodes_map(nodes);

        while let Some(node_id) = queue.pop_front() {
            // Clone node_id for execution_order (need owned value)
            execution_order.push(node_id.clone());

            if let Some(neighbors) = adj.get(&node_id) {
                for neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            // Clone only when adding to queue
                            queue.push_back(neighbor.clone());
                        }
                    } else {
                        anyhow::bail!("Neighbor node not found in in_degree map: {}", neighbor);
                    }
                }
            }
        }

        if execution_order.len() != nodes.len() {
            anyhow::bail!("Workflow contains cycles or disconnected nodes");
        }

        // Execute nodes in order
        let mut results = Vec::new();
        for node_id in execution_order {
            // Use string reference for lookup
            let node = nodes_map
                .get(node_id.as_str())
                .ok_or_else(|| anyhow::anyhow!("Node not found in nodes_map: {}", node_id))?;
            context.current_node_id = Some(node_id.clone());
            let result = self.execute_node(node, nodes, edges, context).await?;
            record_and_update_context(context, node, &result);
            let failed = self.should_halt_on_node_error(&result);
            results.push(result);

            if failed {
                if let Some(last) = results.last() {
                    self.mark_failed_on_node_error(context, last);
                }
                break;
            }

            // DAG mode cannot safely honor forced control flow in a topological schedule.
            // Fail fast to avoid silent partial execution.
            if get_forced_next_node(&results).is_some() {
                anyhow::bail!(
                    "Node requested forced control flow in DAG mode (unsupported). Use state machine mode with entry_point()."
                );
            }
        }

        Ok(results)
    }

    /// Execute a single node.
    ///
    /// This is implemented as a boxed future so that control-flow nodes (e.g. Parallel)
    /// can execute subgraphs without running into `async fn` recursion limits.
    pub(super) fn execute_node<'a>(
        &'a self,
        node: &'a WorkflowNode,
        nodes: &'a [WorkflowNode],
        edges: &'a [WorkflowEdge],
        context: &'a mut ExecutionContext,
    ) -> Pin<Box<dyn Future<Output = Result<NodeExecutionResult>> + Send + 'a>> {
        Box::pin(async move {
            let start_time = Utc::now();

            // Handle HumanInterrupt nodes specially
            if let NodeType::HumanInterrupt {
                message,
                required_fields,
            } = &node.node_type
            {
                // Create pending interrupt
                let interrupt = PendingInterrupt {
                    interrupt_id: Uuid::now_v7().to_string(),
                    execution_id: context.execution_id.clone(),
                    node_id: node.id.clone(),
                    node_name: node.name.clone(),
                    message: message.clone(),
                    required_fields: required_fields.clone(),
                    created_at: Utc::now(),
                    context_snapshot: context.variables.clone(),
                };

                // Pause workflow execution
                context.state = WorkflowState::Paused;
                context.pending_interrupt = Some(interrupt.clone());
                context.current_node_id = Some(node.id.clone());

                // Return result indicating pause
                return Ok(NodeExecutionResult::ok(
                    node.id.clone(),
                    serde_json::json!({
                        "interrupt_id": interrupt.interrupt_id,
                        "message": message,
                        "required_fields": required_fields,
                        "paused": true
                    }),
                    start_time,
                    HashMap::new(),
                ));
            }

            // Build lookup maps for control-flow nodes.
            let nodes_map = build_nodes_map(nodes);
            let outgoing_edges = build_outgoing_edges(edges);

            let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();

            let result = match &node.node_type {
                NodeType::Agent { .. } => {
                    self.run_agent_node(node, nodes, edges, context, &mut metadata, start_time)
                        .await?
                }
                NodeType::Condition { .. } => Self::run_condition_node(node, context, start_time),
                NodeType::Transform { .. } => {
                    self.run_transform_node(node, nodes, edges, context, &mut metadata, start_time)
                        .await?
                }
                NodeType::Parallel => {
                    self.run_parallel_node(
                        node,
                        nodes,
                        edges,
                        context,
                        &nodes_map,
                        &outgoing_edges,
                        &mut metadata,
                        start_time,
                    )
                    .await?
                }
                NodeType::Merge { .. } => {
                    self.run_merge_node(node, nodes, edges, context, &mut metadata, start_time)
                        .await?
                }
                NodeType::Loop { .. } => {
                    Self::run_loop_node(node, context, &outgoing_edges, &mut metadata, start_time)
                }
                NodeType::HumanInterrupt { .. } => NodeExecutionResult::err(
                    node.id.clone(),
                    "HumanInterrupt nodes must be handled in the interrupt branch",
                    start_time,
                    HashMap::new(),
                ),
            };

            Ok(result)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_until<F>(
        &self,
        nodes_map: &HashMap<&str, &WorkflowNode>,
        outgoing_edges: &HashMap<&str, Vec<&WorkflowEdge>>,
        start_node_id: &str,
        nodes: &[WorkflowNode],
        edges: &[WorkflowEdge],
        context: &mut ExecutionContext,
        stop_predicate: F,
    ) -> Result<(Vec<NodeExecutionResult>, Option<String>)>
    where
        F: Fn(&str, Option<&WorkflowNode>) -> Option<String>,
    {
        let mut current_node_id = start_node_id.to_string();
        let mut results = Vec::new();
        let mut step_count: usize = 0;

        loop {
            if context.is_cancelled() {
                context.state = WorkflowState::Cancelled;
                context.end_time = Some(Utc::now());
                return Ok((results, None));
            }
            if step_count >= MAX_STEPS {
                anyhow::bail!(
                    "Branch exceeded maximum steps ({}). Possible infinite loop.",
                    MAX_STEPS
                );
            }
            step_count += 1;

            if let Some(id) = stop_predicate(current_node_id.as_str(), None) {
                return Ok((results, Some(id)));
            }

            let node = nodes_map
                .get(current_node_id.as_str())
                .ok_or_else(|| anyhow::anyhow!("Node '{}' not found", current_node_id))?;

            if let Some(id) = stop_predicate(current_node_id.as_str(), Some(node)) {
                return Ok((results, Some(id)));
            }

            let result = self
                .execute_node(node, nodes, edges, context)
                .await
                .context("Branch node execution failed")?;
            let node_output = result.output.clone();
            let node_output_ref = node_output.as_ref();
            publish_template_outputs(
                context,
                node,
                node_output_ref.unwrap_or(&serde_json::Value::Null),
            );
            let failed = self.should_halt_on_node_error(&result);
            results.push(result);

            if failed {
                if let Some(last) = results.last() {
                    self.mark_failed_on_node_error(context, last);
                }
                return Ok((results, None));
            }

            if let Some(next) = get_forced_next_node(&results) {
                current_node_id = next;
                continue;
            }

            let Some(next) = Self::select_next_node_id(
                current_node_id.as_str(),
                outgoing_edges,
                context,
                node_output_ref,
            ) else {
                return Ok((results, None));
            };
            current_node_id = next;
        }
    }

    pub(super) async fn execute_until_first_merge(
        &self,
        nodes_map: &HashMap<&str, &WorkflowNode>,
        outgoing_edges: &HashMap<&str, Vec<&WorkflowEdge>>,
        start_node_id: &str,
        nodes: &[WorkflowNode],
        edges: &[WorkflowEdge],
        context: &mut ExecutionContext,
    ) -> Result<(Vec<NodeExecutionResult>, Option<String>)> {
        self.execute_until(
            nodes_map,
            outgoing_edges,
            start_node_id,
            nodes,
            edges,
            context,
            |_current_id, node| {
                node.and_then(|n| {
                    if matches!(n.node_type, NodeType::Merge { .. }) {
                        Some(n.id.clone())
                    } else {
                        None
                    }
                })
            },
        )
        .await
    }

    #[allow(clippy::too_many_arguments, dead_code)]
    pub(super) async fn execute_until_node(
        &self,
        nodes_map: &HashMap<&str, &WorkflowNode>,
        outgoing_edges: &HashMap<&str, Vec<&WorkflowEdge>>,
        start_node_id: &str,
        stop_node_id: &str,
        nodes: &[WorkflowNode],
        edges: &[WorkflowEdge],
        context: &mut ExecutionContext,
    ) -> Result<(Vec<NodeExecutionResult>, Option<String>)> {
        self.execute_until(
            nodes_map,
            outgoing_edges,
            start_node_id,
            nodes,
            edges,
            context,
            |current_id, node| {
                if current_id == stop_node_id {
                    return Some(stop_node_id.to_string());
                }
                node.and_then(|n| {
                    if matches!(n.node_type, NodeType::Merge { .. }) && n.id != stop_node_id {
                        Some(n.id.clone())
                    } else {
                        None
                    }
                })
            },
        )
        .await
    }

    pub(super) fn select_next_node_id(
        current_node_id: &str,
        outgoing_edges: &HashMap<&str, Vec<&WorkflowEdge>>,
        context: &ExecutionContext,
        node_output: Option<&serde_json::Value>,
    ) -> Option<String> {
        let edges = outgoing_edges.get(current_node_id)?;
        select_next_edge(edges, context, node_output, Self::evaluate_condition)
    }

    pub(super) fn run_condition_node(
        node: &WorkflowNode,
        context: &ExecutionContext,
        start_time: DateTime<Utc>,
    ) -> NodeExecutionResult {
        let NodeType::Condition { predicate } = &node.node_type else {
            return NodeExecutionResult::err(
                node.id.clone(),
                format!("expected Condition node, got {:?}", node.node_type),
                start_time,
                HashMap::new(),
            );
        };
        let context_value = serde_json::json!(context.variables);
        let value = Self::evaluate_condition(predicate, &context_value, None);
        NodeExecutionResult::ok(
            node.id.clone(),
            serde_json::json!({ "value": value, "predicate": predicate }),
            start_time,
            HashMap::new(),
        )
    }

    pub(super) async fn run_transform_node(
        &self,
        node: &WorkflowNode,
        _nodes: &[WorkflowNode],
        _edges: &[WorkflowEdge],
        context: &mut ExecutionContext,
        metadata: &mut HashMap<String, serde_json::Value>,
        start_time: DateTime<Utc>,
    ) -> Result<NodeExecutionResult> {
        let NodeType::Transform { function } = &node.node_type else {
            return Ok(NodeExecutionResult::err(
                node.id.clone(),
                format!("expected Transform node, got {:?}", node.node_type),
                start_time,
                std::mem::take(metadata),
            ));
        };
        let last_output = context.get_variable(VAR_LAST_OUTPUT);
        match self
            .executor
            .transform(function, context, last_output)
            .await
        {
            Ok(value) => {
                context.set_variable(format!("node_{}_output", node.id), value.clone());
                Ok(NodeExecutionResult::ok(
                    node.id.clone(),
                    value,
                    start_time,
                    std::mem::take(metadata),
                ))
            }
            Err(e) => Ok(NodeExecutionResult::err(
                node.id.clone(),
                e.to_string(),
                start_time,
                std::mem::take(metadata),
            )),
        }
    }

    pub(super) fn run_loop_node(
        node: &WorkflowNode,
        context: &mut ExecutionContext,
        outgoing_edges: &HashMap<&str, Vec<&WorkflowEdge>>,
        metadata: &mut HashMap<String, serde_json::Value>,
        start_time: DateTime<Utc>,
    ) -> NodeExecutionResult {
        let NodeType::Loop {
            max_iterations,
            condition,
        } = &node.node_type
        else {
            return NodeExecutionResult::err(
                node.id.clone(),
                format!("expected Loop node, got {:?}", node.node_type),
                start_time,
                std::mem::take(metadata),
            );
        };
        let iter_key = format!("loop_{}_iter", node.id);
        let current_iter = context
            .get_variable(iter_key.as_str())
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let should_continue = if current_iter >= *max_iterations as u64 {
            false
        } else {
            let context_value = serde_json::json!(context.variables);
            Self::evaluate_condition(condition, &context_value, None)
        };

        let edges_from = outgoing_edges
            .get(node.id.as_str())
            .cloned()
            .unwrap_or_default();
        let body_edge = edges_from
            .iter()
            .find(|e| e.label.as_deref() == Some("body"))
            .or_else(|| {
                edges_from
                    .iter()
                    .find(|e| e.label.as_deref() != Some("exit"))
            });
        let exit_edge = edges_from
            .iter()
            .find(|e| e.label.as_deref() == Some("exit"));

        if should_continue {
            if let Some(body) = body_edge {
                context.set_variable(iter_key, serde_json::json!(current_iter + 1));
                metadata.insert(
                    META_FORCED_NEXT_NODE_ID.to_owned(),
                    serde_json::Value::String(body.to_node.clone()),
                );
                NodeExecutionResult::ok(
                    node.id.clone(),
                    serde_json::json!({
                        "continue": true,
                        "iteration": current_iter + 1,
                        "max_iterations": max_iterations
                    }),
                    start_time,
                    std::mem::take(metadata),
                )
            } else {
                NodeExecutionResult::err(
                    node.id.clone(),
                    "Loop node has no body edge",
                    start_time,
                    std::mem::take(metadata),
                )
            }
        } else {
            if let Some(exit) = exit_edge {
                metadata.insert(
                    META_FORCED_NEXT_NODE_ID.to_owned(),
                    serde_json::Value::String(exit.to_node.clone()),
                );
            }
            NodeExecutionResult::ok(
                node.id.clone(),
                serde_json::json!({
                    "continue": false,
                    "iteration": current_iter,
                    "max_iterations": max_iterations
                }),
                start_time,
                std::mem::take(metadata),
            )
        }
    }
}
