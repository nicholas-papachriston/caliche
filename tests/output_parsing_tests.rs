/// Tests for output parsing
#[cfg(feature = "output-parsing")]
mod parsing_tests {
    use caliche::output::{
        EnumOutputParser, JsonOutputParser, ListOutputParser, OutputParser, StructuredOutputParser,
    };
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Person {
        name: String,
        age: u32,
        email: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Address {
        street: String,
        city: String,
        country: String,
    }

    #[tokio::test]
    async fn test_json_parser_basic() {
        let parser = JsonOutputParser::new();
        let json_str = r#"{"name": "Alice", "age": 30}"#;

        let result = parser
            .parse(json_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result["name"], "Alice");
        assert_eq!(result["age"], 30);
    }

    #[tokio::test]
    async fn test_json_parser_with_markdown() {
        let parser = JsonOutputParser::new();
        let output = r#"```json
{"name": "Bob", "age": 25}
```"#;

        let result = parser.parse(output).await.expect("Parsing should succeed");
        assert_eq!(result["name"], "Bob");
        assert_eq!(result["age"], 25);
    }

    #[tokio::test]
    async fn test_json_parser_with_backticks() {
        let parser = JsonOutputParser::new();
        let output = r#"
```
{"name": "Charlie", "age": 35}
```
"#;

        let result = parser.parse(output).await.expect("Parsing should succeed");
        assert_eq!(result["name"], "Charlie");
        assert_eq!(result["age"], 35);
    }

    #[tokio::test]
    async fn test_json_parser_array() {
        let parser = JsonOutputParser::new();
        let json_str = r#"[1, 2, 3, 4, 5]"#;

        let result = parser
            .parse(json_str)
            .await
            .expect("Parsing should succeed");
        assert!(result.is_array());
        assert_eq!(
            result.as_array().expect("Result should be an array").len(),
            5
        );
    }

    #[tokio::test]
    async fn test_json_parser_nested() {
        let parser = JsonOutputParser::new();
        let json_str = r#"{"person": {"name": "Alice", "age": 30}, "city": "NYC"}"#;

        let result = parser
            .parse(json_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result["person"]["name"], "Alice");
        assert_eq!(result["city"], "NYC");
    }

    #[tokio::test]
    async fn test_json_parser_invalid() {
        let parser = JsonOutputParser::new();
        let invalid_json = "This is not JSON";

        let result = parser.parse(invalid_json).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_json_parser_empty() {
        let parser = JsonOutputParser::new();
        let result = parser.parse("").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_json_parser_with_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        });

        let parser = JsonOutputParser::with_schema(schema)
            .expect("Parser creation with schema should succeed");
        let valid_json = r#"{"name": "Alice", "age": 30}"#;

        let result = parser.parse(valid_json).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_json_parser_schema_validation_fail() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        });

        let parser = JsonOutputParser::with_schema(schema)
            .expect("Parser creation with schema should succeed");
        let invalid_json = r#"{"name": "Alice"}"#; // Missing required 'age'

        let result = parser.parse(invalid_json).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_structured_parser_basic() {
        let parser = StructuredOutputParser::<Person>::new();
        let json_str = r#"{"name": "Alice", "age": 30, "email": "alice@example.com"}"#;

        let result = parser
            .parse(json_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result.name, "Alice");
        assert_eq!(result.age, 30);
        assert_eq!(result.email, Some("alice@example.com".to_string()));
    }

    #[tokio::test]
    async fn test_structured_parser_optional_field() {
        let parser = StructuredOutputParser::<Person>::new();
        let json_str = r#"{"name": "Bob", "age": 25}"#;

        let result = parser
            .parse(json_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result.name, "Bob");
        assert_eq!(result.age, 25);
        assert_eq!(result.email, None);
    }

    #[tokio::test]
    async fn test_structured_parser_with_markdown() {
        let parser = StructuredOutputParser::<Person>::new();
        let output = r#"```json
{"name": "Charlie", "age": 35}
```"#;

        let result = parser.parse(output).await.expect("Parsing should succeed");
        assert_eq!(result.name, "Charlie");
        assert_eq!(result.age, 35);
    }

    #[tokio::test]
    async fn test_structured_parser_missing_required() {
        let parser = StructuredOutputParser::<Person>::new();
        let json_str = r#"{"name": "Alice"}"#; // Missing 'age'

        let result = parser.parse(json_str).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_structured_parser_wrong_type() {
        let parser = StructuredOutputParser::<Person>::new();
        let json_str = r#"{"name": "Alice", "age": "thirty"}"#; // Wrong type for age

        let result = parser.parse(json_str).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_structured_parser_nested() {
        #[derive(Debug, Serialize, Deserialize)]
        struct PersonWithAddress {
            name: String,
            address: Address,
        }

        let parser = StructuredOutputParser::<PersonWithAddress>::new();
        let json_str = r#"{
            "name": "Alice",
            "address": {
                "street": "123 Main St",
                "city": "NYC",
                "country": "USA"
            }
        }"#;

        let result = parser
            .parse(json_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result.name, "Alice");
        assert_eq!(result.address.city, "NYC");
    }

    #[tokio::test]
    async fn test_list_parser_basic() {
        let parser = ListOutputParser::new();
        let list_str = "apple, banana, cherry";

        let result = parser
            .parse(list_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "apple");
        assert_eq!(result[1], "banana");
        assert_eq!(result[2], "cherry");
    }

    #[tokio::test]
    async fn test_list_parser_with_whitespace() {
        let parser = ListOutputParser::new();
        let list_str = "apple , banana  ,  cherry";

        let result = parser
            .parse(list_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "apple");
        assert_eq!(result[1], "banana");
        assert_eq!(result[2], "cherry");
    }

    #[tokio::test]
    async fn test_list_parser_custom_separator() {
        let parser = ListOutputParser::new().with_separator(";");
        let list_str = "apple;banana;cherry";

        let result = parser
            .parse(list_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn test_list_parser_single_item() {
        let parser = ListOutputParser::new();
        let list_str = "apple";

        let result = parser
            .parse(list_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "apple");
    }

    #[tokio::test]
    async fn test_list_parser_empty() {
        let parser = ListOutputParser::new();
        let list_str = "";

        let result = parser
            .parse(list_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_list_parser_with_newlines() {
        let parser = ListOutputParser::new().with_separator("\n");
        let list_str = "apple\nbanana\ncherry";

        let result = parser
            .parse(list_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn test_enum_parser_basic() {
        let parser = EnumOutputParser::new(vec![
            "positive".to_string(),
            "negative".to_string(),
            "neutral".to_string(),
        ]);

        let result = parser
            .parse("positive")
            .await
            .expect("Parsing should succeed");
        assert_eq!(result, "positive");
    }

    #[tokio::test]
    async fn test_enum_parser_case_handling() {
        // By default, EnumOutputParser is case-insensitive
        let parser = EnumOutputParser::new(vec!["positive".to_string(), "negative".to_string()]);

        // Test case-insensitive matching - returns the choice from list
        let result = parser
            .parse("POSITIVE")
            .await
            .expect("Parsing should succeed");
        assert_eq!(result, "positive");

        // Test with case-sensitive mode
        let parser_sensitive =
            EnumOutputParser::new(vec!["positive".to_string(), "POSITIVE".to_string()])
                .with_case_sensitive(true);

        let result = parser_sensitive
            .parse("POSITIVE")
            .await
            .expect("Parsing should succeed");
        assert_eq!(result, "POSITIVE");
    }

    #[tokio::test]
    async fn test_enum_parser_invalid_choice() {
        let parser = EnumOutputParser::new(vec!["positive".to_string(), "negative".to_string()]);

        let result = parser.parse("unknown").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_enum_parser_with_whitespace() {
        let parser = EnumOutputParser::new(vec!["positive".to_string(), "negative".to_string()]);

        let result = parser
            .parse("  positive  ")
            .await
            .expect("Parsing should succeed");
        assert_eq!(result, "positive");
    }

    #[tokio::test]
    async fn test_enum_parser_empty_choices() {
        let parser = EnumOutputParser::new(vec![]);
        let result = parser.parse("anything").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_json_parser_pretty_print() {
        let parser = JsonOutputParser::new().with_pretty_print(true);
        let json_str = r#"{"name":"Alice","age":30}"#;

        let result = parser
            .parse(json_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result["name"], "Alice");
    }

    #[tokio::test]
    async fn test_json_parser_whitespace_handling() {
        let parser = JsonOutputParser::new();
        let json_str = r#"
        
        {"name": "Alice", "age": 30}
        
        "#;

        let result = parser
            .parse(json_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result["name"], "Alice");
    }

    #[tokio::test]
    async fn test_structured_parser_array() {
        let parser = StructuredOutputParser::<Vec<Person>>::new();
        let json_str = r#"[
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]"#;

        let result = parser
            .parse(json_str)
            .await
            .expect("Parsing should succeed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "Alice");
        assert_eq!(result[1].name, "Bob");
    }

    #[tokio::test]
    async fn test_list_parser_with_trailing_comma() {
        let parser = ListOutputParser::new();
        let list_str = "apple, banana, cherry,";

        let result = parser
            .parse(list_str)
            .await
            .expect("Parsing should succeed");
        // Should handle trailing comma gracefully
        assert!(result.len() >= 3);
    }

    #[tokio::test]
    async fn test_enum_parser_partial_match() {
        let parser = EnumOutputParser::new(vec!["positive".to_string(), "negative".to_string()]);

        // Should not match partial strings
        let result = parser.parse("pos").await;
        assert!(result.is_err());
    }
}
