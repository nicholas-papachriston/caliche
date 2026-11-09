//! Apache AGE graph backend using sqlx (raw SQL).
//!
//! Uses the same sqlx pool as PostgresBackend for connection sharing with pgvector.
//! AGE Cypher is executed via ag_catalog.cypher() wrapped in SQL.

mod backend;
mod graph_ops;
mod parse;
mod query;

pub use backend::{AgeBackend, AgeConfig};

#[cfg(test)]
mod age_query_params_tests {
    use super::AgeBackend;
    use crate::graph::{GraphError, QueryParams};

    #[test]
    fn agtype_map_contains_build_map_and_keys() {
        let mut p = QueryParams::new();
        p.insert("name".to_string(), serde_json::json!("alice"));
        p.insert("n".to_string(), serde_json::json!(42));
        let sql = AgeBackend::query_params_to_agtype_map_sql(&p).expect("sql");
        assert!(sql.contains("ag_catalog.agtype_build_map"));
        assert!(sql.contains("'name'"));
        assert!(sql.contains("'n'"));
        assert!(sql.contains("42"));
    }

    #[test]
    fn agtype_map_rejects_bad_key() {
        let mut p = QueryParams::new();
        p.insert("bad-key".to_string(), serde_json::json!(1));
        assert!(AgeBackend::query_params_to_agtype_map_sql(&p).is_err());
    }

    #[test]
    fn agtype_map_rejects_object_value() {
        let mut p = QueryParams::new();
        p.insert("x".to_string(), serde_json::json!({ "a": 1 }));
        assert!(matches!(
            AgeBackend::query_params_to_agtype_map_sql(&p),
            Err(GraphError::Unsupported(_))
        ));
    }

    #[test]
    fn agtype_out_projection_avoids_scalar_text_cast() {
        let sql =
            AgeBackend::agtype_out_projection("(graph_id agtype, label agtype, props agtype)")
                .expect("projection");
        assert!(sql.contains("ag_catalog.agtype_out(graph_id)::text"));
        assert!(sql.contains("ag_catalog.agtype_out(props)::text"));
        assert!(!sql.contains("props::text"));
    }

    #[test]
    fn agtype_path_annotations_parse_as_alternating_array() {
        let raw = r#"[{"id": 1, "label": "Person", "properties": {"id": "person-0", "name": "Person 0"}}::vertex, {"id": 2, "label": "WORKS_AT", "start_id": 1, "end_id": 3, "properties": {}}::edge, {"id": 3, "label": "Company", "properties": {"id": "company-0"}}::vertex]::path"#;
        let val = AgeBackend::parse_agtype_cell(raw);
        let path = AgeBackend::path_from_agtype(&val).expect("path");
        assert_eq!(path.nodes.len(), 2);
        assert_eq!(path.edges.len(), 1);
        assert_eq!(path.nodes[0].id, "person-0");
        assert_eq!(path.nodes[1].node_type, "Company");
        assert_eq!(path.edges[0].edge_type, "WORKS_AT");
    }

    #[test]
    fn properties_map_from_agtype_out_is_object() {
        let raw = r#"{"id": "person-0", "name": "Person 0", "age": 25}"#;
        let val = AgeBackend::parse_agtype_cell(raw);
        assert_eq!(val["id"], "person-0");
        assert_eq!(val["age"], 25);
    }

    #[test]
    fn age_count_text_accepts_numeric_suffix() {
        assert_eq!(super::backend::parse_age_count_text("45"), Some(45));
        assert_eq!(super::backend::parse_age_count_text("\"75\""), Some(75));
        assert_eq!(
            super::backend::parse_age_count_text("45::numeric"),
            Some(45)
        );
    }

    #[test]
    fn agtype_map_u64_beyond_i64() {
        let mut p = QueryParams::new();
        p.insert(
            "big".to_string(),
            serde_json::Value::Number(serde_json::Number::from(9_223_372_036_854_775_808u64)),
        );
        let sql = AgeBackend::query_params_to_agtype_map_sql(&p).expect("sql");
        assert!(sql.contains("9223372036854775808"));
    }
}
