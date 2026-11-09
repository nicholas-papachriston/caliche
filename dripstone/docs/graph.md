# Dripstone Graph Backends

Dripstone provides graph backends for nodes, edges, schema inspection, neighbor queries, path search, and raw Cypher execution.

## Apache AGE

Apache AGE is a PostgreSQL extension that adds Cypher support.

```rust
use std::sync::Arc;
use dripstone::graph::age::{AgeBackend, AgeConfig};
use dripstone::graph::QueryableGraphBackend;

let config = AgeConfig {
    graph_name: "my_graph".to_string(),
    schema_cache_ttl_secs: 300,
    auto_create_graph: true,
};

let backend: Arc<dyn QueryableGraphBackend> = Arc::new(
    AgeBackend::new("postgres://postgres:postgres@127.0.0.1:5455/age", config).await?
);
```

Requires `graph-age` and `sqlx`.

## Neo4j

```rust
use std::sync::Arc;
use dripstone::graph::neo4j::{Neo4jBackend, Neo4jConfig};
use dripstone::graph::QueryableGraphBackend;

let config = Neo4jConfig {
    uri: "bolt://127.0.0.1:7687".to_string(),
    username: "neo4j".to_string(),
    password: "password".to_string(),
    database: None,
    max_connections: 8,
    schema_cache_ttl_secs: 300,
};

let backend: Arc<dyn QueryableGraphBackend> = Arc::new(
    Neo4jBackend::new(config).await?
);
```

Requires `graph-neo4j` and the Neo4j Rust driver.

## Operations

Both backends support:

- `add_node`
- `add_edge`
- `get_node`
- `get_neighbors`
- `get_nodes_by_type`
- `get_edges_by_type`
- `find_path`
- `execute_query`
- `get_schema`
- `count_nodes`
- `count_edges`
- bulk delete

Query results include node and edge properties.

## Local Services

Start Apache AGE and Neo4j from the example Compose file:

```bash
docker compose -f docker-compose.example.yml --profile graph up -d
```

Common environment variables:

```bash
AGE_URL=postgres://postgres:postgres@127.0.0.1:5455/age
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

Run the comparison example:

```bash
cargo run -p dripstone --example graph_age_neo4j_example --features "graph-age,graph-neo4j"
```
