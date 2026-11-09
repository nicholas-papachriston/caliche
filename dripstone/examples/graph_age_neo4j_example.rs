//! Example using Apache AGE and Neo4j graph backends: insert data, run operations, clean up.
//!
//! Requires AGE and Neo4j. Start them with:
//!   docker compose -f docker-compose.example.yml --profile graph up -d
//!
//! Run with:
//!   cargo run --example graph_age_neo4j_example --features "graph-age,graph-neo4j"
//!
//! Environment (optional):
//!   AGE_URL       - Postgres connection for AGE (default: postgres://postgres:postgres@127.0.0.1:5455/age)
//!   NEO4J_URI     - Bolt URI (default: bolt://127.0.0.1:7687)
//!   NEO4J_USER    - default neo4j
//!   NEO4J_PASSWORD - default password

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use std::sync::Arc;

use dripstone::graph::age::{AgeBackend, AgeConfig};
use dripstone::graph::neo4j::{Neo4jBackend, Neo4jConfig};
use dripstone::graph::{Edge, GraphBackend, Node, QueryableGraphBackend};

const NUM_PERSONS: usize = 30;
const NUM_COMPANIES: usize = 15;
const NUM_KNOWS: usize = 40;
const NUM_WORKS_AT: usize = 35;

fn make_nodes() -> (Vec<Node>, Vec<Edge>) {
    let mut nodes: Vec<Node> = Vec::with_capacity(NUM_PERSONS + NUM_COMPANIES);
    for i in 0..NUM_PERSONS {
        nodes.push(
            Node::new(format!("person-{}", i), "Person".to_string())
                .with_property(
                    "name".to_string(),
                    serde_json::Value::String(format!("Person {}", i)),
                )
                .with_property(
                    "age".to_string(),
                    serde_json::Value::Number((25 + (i % 40)).into()),
                ),
        );
    }
    for i in 0..NUM_COMPANIES {
        nodes.push(
            Node::new(format!("company-{}", i), "Company".to_string())
                .with_property(
                    "name".to_string(),
                    serde_json::Value::String(format!("Company {}", i)),
                )
                .with_property(
                    "industry".to_string(),
                    serde_json::Value::String(format!("Industry {}", i % 5)),
                ),
        );
    }

    let mut edges: Vec<Edge> = Vec::with_capacity(NUM_KNOWS + NUM_WORKS_AT);
    for _ in 0..NUM_KNOWS {
        let a = (0..NUM_PERSONS).fold(0usize, |acc, _| (acc + 7) % NUM_PERSONS);
        let b = (a + 1 + (a % 5)) % NUM_PERSONS;
        if a != b {
            edges.push(Edge::new(
                format!("person-{}", a),
                format!("person-{}", b),
                "KNOWS".to_string(),
            ));
        }
    }
    for i in 0..NUM_WORKS_AT {
        let p = i % NUM_PERSONS;
        let c = (i * 3) % NUM_COMPANIES;
        edges.push(Edge::new(
            format!("person-{}", p),
            format!("company-{}", c),
            "WORKS_AT".to_string(),
        ));
    }

    (nodes, edges)
}

async fn insert_data<B: GraphBackend + ?Sized>(
    backend: &B,
) -> Result<(), dripstone::graph::GraphError> {
    let (nodes, edges) = make_nodes();
    for node in &nodes {
        backend.add_node(node.clone()).await?;
    }
    for edge in &edges {
        backend.add_edge(edge.clone()).await?;
    }
    Ok(())
}

async fn run_operations<B: GraphBackend + QueryableGraphBackend + ?Sized>(
    name: &str,
    backend: &B,
) -> Result<(), dripstone::graph::GraphError> {
    let nodes = backend.count_nodes().await?;
    let edges_count = backend.count_edges().await?;
    outputln!("  [{}] nodes: {}, edges: {}", name, nodes, edges_count);

    let schema = backend.get_schema().await?;
    outputln!("  [{}] schema summary:\n{}", name, schema.to_summary());

    let person = backend.get_node("person-0").await?;
    if let Some(ref n) = person {
        outputln!(
            "  [{}] get_node(person-0): {} {:?}",
            name,
            n.node_type,
            n.id
        );
    }

    let neighbors = backend.get_neighbors("person-0", None).await?;
    outputln!(
        "  [{}] get_neighbors(person-0): {} neighbors",
        name,
        neighbors.len()
    );

    let path = backend.find_path("person-0", "company-0", 5).await?;
    if let Some(ref p) = path {
        outputln!(
            "  [{}] find_path(person-0 -> company-0): {} nodes, {} edges",
            name,
            p.nodes.len(),
            p.edges.len()
        );
    }

    let result = backend
        .execute_query(
            "MATCH (n:Person) RETURN n LIMIT 3",
            &std::collections::HashMap::new(),
        )
        .await?;
    outputln!(
        "  [{}] execute_query MATCH (n:Person): {} records",
        name,
        result.records.len()
    );

    let by_type = backend.get_nodes_by_type("Company").await?;
    outputln!(
        "  [{}] get_nodes_by_type(Company): {} nodes",
        name,
        by_type.len()
    );

    Ok(())
}

async fn cleanup<B: GraphBackend + ?Sized>(
    backend: &B,
) -> Result<(), dripstone::graph::GraphError> {
    let (nodes, _) = make_nodes();
    for node in &nodes {
        let _ = backend.delete_node(&node.id).await;
    }
    for i in 0..NUM_COMPANIES {
        let _ = backend.delete_node(&format!("company-{}", i)).await;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let age_url = std::env::var("AGE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@127.0.0.1:5455/age".to_string());
    let neo4j_uri =
        std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://127.0.0.1:7687".to_string());
    let neo4j_user = std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string());
    let neo4j_password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".to_string());

    outputln!("Connecting to AGE at {} ...", age_url);
    let age_config = AgeConfig {
        graph_name: "example_graph".to_string(),
        schema_cache_ttl_secs: 300,
        auto_create_graph: true,
    };
    let age_backend = AgeBackend::new(&age_url, age_config)
        .await
        .map_err(|e| format!("AGE connection failed: {}", e))?;
    let age_backend: Arc<dyn QueryableGraphBackend> = Arc::new(age_backend);

    outputln!("Connecting to Neo4j at {} ...", neo4j_uri);
    let neo4j_config = Neo4jConfig {
        uri: neo4j_uri.clone(),
        username: neo4j_user,
        password: neo4j_password,
        database: None,
        max_connections: 8,
        schema_cache_ttl_secs: 300,
    };
    let neo4j_backend = Neo4jBackend::new(neo4j_config)
        .await
        .map_err(|e| format!("Neo4j connection failed: {}", e))?;
    let neo4j_backend: Arc<dyn QueryableGraphBackend> = Arc::new(neo4j_backend);

    outputln!(
        "\n--- Inserting data ({} persons, {} companies, knows + works_at edges) ---\n",
        NUM_PERSONS,
        NUM_COMPANIES
    );

    outputln!("Inserting into AGE ...");
    insert_data(age_backend.as_ref()).await?;
    outputln!("Inserting into Neo4j ...");
    insert_data(neo4j_backend.as_ref()).await?;

    outputln!("\n--- Running operations ---\n");

    outputln!("AGE:");
    run_operations("AGE", age_backend.as_ref()).await?;
    outputln!("\nNeo4j:");
    run_operations("Neo4j", neo4j_backend.as_ref()).await?;

    outputln!("\n--- Cleaning up ---\n");

    outputln!("Deleting nodes from AGE ...");
    cleanup(age_backend.as_ref()).await?;
    outputln!("Deleting nodes from Neo4j ...");
    cleanup(neo4j_backend.as_ref()).await?;

    let age_after = age_backend.count_nodes().await?;
    let neo_after = neo4j_backend.count_nodes().await?;
    outputln!(
        "After cleanup: AGE nodes = {}, Neo4j nodes = {}",
        age_after,
        neo_after
    );

    outputln!("\nDone.");
    Ok(())
}
