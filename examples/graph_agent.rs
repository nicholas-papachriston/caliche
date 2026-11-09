//! Agent with graph tools (graph_query, graph_explorer) using Apache AGE or Neo4j.
//!
//! This example demonstrates:
//! - Building an agent with a queryable graph backend (AGE or Neo4j)
//! - Enabling graph_query and graph_explorer tools
//! - Seeding the graph with sample data
//! - Chat turns that use the graph (schema, find nodes, neighbors, path)
//! - Cleaning up graph data
//!
//! Prerequisites:
//!   docker-compose up -d age neo4j
//!
//! Run with (set AGE_URL for Apache AGE, or NEO4J_URI and NEO4J_PASSWORD for Neo4j):
//!   cargo run --example graph_agent --features "tools-graph,graph-age,graph-neo4j,openai"
//!
//! Configuration: AGE_URL, NEO4J_URI (set to use Neo4j instead, e.g. bolt://127.0.0.1:7687),
//! NEO4J_USER, and NEO4J_PASSWORD.
//!
//! Set OPENAI_API_KEY (or configure another provider explicitly) for the LLM.

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::{Context, Result};
use caliche::AgentBuilder;
use std::sync::Arc;

#[cfg(any(feature = "graph-age", feature = "graph-neo4j"))]
use dripstone::graph::{Edge, GraphBackend, Node, QueryableGraphBackend};

#[cfg(feature = "graph-age")]
use dripstone::graph::age::{AgeBackend, AgeConfig};

#[cfg(feature = "graph-neo4j")]
use dripstone::graph::neo4j::{Neo4jBackend, Neo4jConfig};

const NUM_PERSONS: usize = 15;
const NUM_COMPANIES: usize = 5;
const NUM_EDGES: usize = 25;

fn make_sample_data() -> (Vec<Node>, Vec<Edge>) {
    let mut nodes: Vec<Node> = Vec::with_capacity(NUM_PERSONS + NUM_COMPANIES);
    for i in 0..NUM_PERSONS {
        nodes.push(
            Node::new(format!("person-{}", i), "Person".to_string()).with_property(
                "name".to_string(),
                serde_json::Value::String(format!("Person {}", i)),
            ),
        );
    }
    for i in 0..NUM_COMPANIES {
        nodes.push(
            Node::new(format!("company-{}", i), "Company".to_string()).with_property(
                "name".to_string(),
                serde_json::Value::String(format!("Company {}", i)),
            ),
        );
    }
    let mut edges: Vec<Edge> = Vec::with_capacity(NUM_EDGES);
    for i in 0..NUM_EDGES {
        if i % 2 == 0 {
            let p = i % NUM_PERSONS;
            let c = (i / 2) % NUM_COMPANIES;
            edges.push(Edge::new(
                format!("person-{}", p),
                format!("company-{}", c),
                "WORKS_AT".to_string(),
            ));
        } else {
            let a = i % NUM_PERSONS;
            let b = (a + 1) % NUM_PERSONS;
            if a != b {
                edges.push(Edge::new(
                    format!("person-{}", a),
                    format!("person-{}", b),
                    "KNOWS".to_string(),
                ));
            }
        }
    }
    (nodes, edges)
}

#[cfg(any(feature = "graph-age", feature = "graph-neo4j"))]
async fn seed_graph<B: GraphBackend + ?Sized>(
    backend: &B,
) -> Result<(), dripstone::graph::GraphError> {
    let (nodes, edges) = make_sample_data();
    for node in &nodes {
        backend.add_node(node.clone()).await?;
    }
    for edge in &edges {
        backend.add_edge(edge.clone()).await?;
    }
    Ok(())
}

#[cfg(any(feature = "graph-age", feature = "graph-neo4j"))]
async fn cleanup_graph<B: GraphBackend + ?Sized>(
    backend: &B,
) -> Result<(), dripstone::graph::GraphError> {
    let (nodes, _) = make_sample_data();
    for node in &nodes {
        let _ = backend.delete_node(&node.id).await;
    }
    for i in 0..NUM_COMPANIES {
        let _ = backend.delete_node(&format!("company-{}", i)).await;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    #[cfg(not(any(feature = "graph-age", feature = "graph-neo4j")))]
    {
        anyhow::bail!(
            "Build with at least one graph backend: --features graph-age or --features graph-neo4j"
        );
    }

    #[cfg(any(feature = "graph-age", feature = "graph-neo4j"))]
    {
        let neo4j_uri = std::env::var("NEO4J_URI").ok();

        let backend: Arc<dyn QueryableGraphBackend> = if let Some(ref uri) = neo4j_uri {
            println!("Using Neo4j at {} ...", uri);
            let user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string());
            let password =
                std::env::var("NEO4J_PASSWORD").context("Set NEO4J_PASSWORD for Neo4j")?;
            let config = Neo4jConfig {
                uri: uri.clone(),
                username: user,
                password,
                database: None,
                max_connections: 8,
                schema_cache_ttl_secs: 300,
            };
            let b = Neo4jBackend::new(config)
                .await
                .context("Connect to Neo4j (is docker-compose neo4j up?)")?;
            Arc::new(b)
        } else {
            let age_url = std::env::var("AGE_URL").context(
                "Set AGE_URL for Apache AGE, or set NEO4J_URI and NEO4J_PASSWORD for Neo4j",
            )?;
            println!("Using Apache AGE at {} ...", age_url);
            let config = AgeConfig {
                graph_name: "agent_example".to_string(),
                schema_cache_ttl_secs: 300,
                auto_create_graph: true,
            };
            let b = AgeBackend::new(&age_url, config)
                .await
                .context("Connect to AGE (is docker-compose age up?)")?;
            Arc::new(b)
        };

        println!("\n--- Seeding graph ---\n");
        seed_graph(backend.as_ref()).await.context("Seed graph")?;
        let n = backend.count_nodes().await?;
        let e = backend.count_edges().await?;
        println!("Inserted {} nodes, {} edges\n", n, e);

        let mut agent = AgentBuilder::new()
            .provider(caliche::llm::ProviderType::OpenAI)
            .model("gpt-4o".to_string())
            .with_graph_backend(backend.clone())
            .enable_graph_tools()
            .build()
            .await
            .context("Build agent (set OPENAI_API_KEY or adjust provider/model)")?;

        println!("--- Chat: ask for schema ---\n");
        let r = agent
            .chat("Use the graph_query tool with operation get_schema and tell me what vertex and edge types exist.")
            .await?;
        println!("Response: {}\n", r.text());

        println!("--- Chat: find Person nodes ---\n");
        let r = agent
            .chat("Use graph_query with operation find_nodes, node_type Person, limit 5. Summarize what you see.")
            .await?;
        println!("Response: {}\n", r.text());

        println!("--- Chat: neighbors of person-0 ---\n");
        let r = agent
            .chat("Use graph_query with operation get_neighbors and node_id person-0. Who is connected to person-0?")
            .await?;
        println!("Response: {}\n", r.text());

        println!("--- Cleaning up graph ---\n");
        cleanup_graph(backend.as_ref()).await?;
        let after = backend.count_nodes().await?;
        println!("After cleanup: {} nodes\n", after);
    }

    println!("Done.");
    Ok(())
}
