#![expect(
    clippy::expect_used,
    reason = "static Regex/LazyLock literals are reviewed constants; failure is a programmer bug"
)]

use crate::agent::tools::common::params;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::utils::json_merge::{json_merge_mode_from_str, merge_json_value};
use anyhow::{Result, bail};
use async_trait::async_trait;
use regex::Regex;
use serde_json::json;
use std::sync::LazyLock;

/// Matches simple XML tags: `<tag>content</tag>` or self-closing `<tag/>`.
static RE_XML_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<([^/>]+)>([^<]*)</[^>]+>|<([^/>]+)/>").expect("RE_XML_TAG"));

const NAME: &str = "data_manipulation";
const DESCRIPTION: &str = "Manipulates and transforms data in various formats (JSON, XML, CSV). \
Supports parsing, stringifying, transformation, validation, schema checking, and format conversion. \
Operations: parse_json, stringify_json, parse_xml, stringify_xml, parse_csv, stringify_csv, \
validate_json_schema, convert_format, extract_json_path, transform_json, merge_json.";

#[derive(Debug)]
pub struct DataManipulationTool;

impl ToolMeta for DataManipulationTool {
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
                    "enum": [
                        "parse_json", "stringify_json", "parse_xml", "stringify_xml",
                        "parse_csv", "stringify_csv", "validate_json_schema",
                        "convert_format", "extract_json_path", "transform_json", "merge_json"
                    ],
                    "description": "Operation to perform on the data"
                },
                "data": {
                    "description": "Input data (string for parse operations, object/array for stringify/transform operations)"
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "xml", "csv"],
                    "description": "Target format for conversion operations (required for convert_format)"
                },
                "json_path": {
                    "type": "string",
                    "description": "JSONPath expression to extract data (e.g., '$.users[0].name' or '$.items[*].id')"
                },
                "transformation": {
                    "type": "object",
                    "description": "Transformation rules as key-value pairs where keys are output paths and values are JSONPath expressions or functions"
                },
                "schema": {
                    "type": "object",
                    "description": "JSON Schema object for validation (required for validate_json_schema)"
                },
                "csv_headers": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Column headers for CSV (optional, auto-detected if not provided)"
                },
                "csv_delimiter": {
                    "type": "string",
                    "description": "CSV delimiter character (default: ',')",
                    "default": ","
                },
                "merge_mode": {
                    "type": "string",
                    "enum": ["deep", "shallow", "replace"],
                    "description": "Merge mode for merge_json: 'deep' merges nested objects, 'shallow' only top level, 'replace' replaces conflicting keys (default: deep)",
                    "default": "deep"
                }
            },
            "required": ["operation"]
        })
    }
}

#[async_trait]
impl Tool for DataManipulationTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let operation = match params::extract_string(&arguments, "operation", None) {
            Ok(op) => op,
            Err(e) => return ToolResult::error(e),
        };

        match operation.as_str() {
            "parse_json" => Self::parse_json(&arguments),
            "stringify_json" => Self::stringify_json(&arguments),
            "parse_xml" => Self::parse_xml(&arguments),
            "stringify_xml" => Self::stringify_xml(&arguments),
            "parse_csv" => Self::parse_csv(&arguments),
            "stringify_csv" => Self::stringify_csv(&arguments),
            "validate_json_schema" => Self::validate_json_schema(&arguments),
            "convert_format" => Self::convert_format(&arguments),
            "extract_json_path" => Self::extract_json_path(&arguments),
            "transform_json" => Self::transform_json(&arguments),
            "merge_json" => Self::merge_json(&arguments),
            _ => ToolResult::error(format!("Unknown operation: {operation}")),
        }
    }
}

macro_rules! parse_operation {
    ($name:ident, $param_err:expr, $format_label:expr, $convert_fn:path) => {
        fn $name(args: &serde_json::Value) -> ToolResult {
            let data_str = match args.get("data").and_then(|v| v.as_str()) {
                Some(s) => s,
                None => return ToolResult::error($param_err.to_string()),
            };
            match $convert_fn(data_str) {
                Ok(parsed) => ToolResult::success(format!(
                    "Successfully parsed {}:\n{}",
                    $format_label,
                    serde_json::to_string_pretty(&parsed).unwrap_or_else(|_| parsed.to_string())
                )),
                Err(e) => ToolResult::error(format!("Failed to parse {}: {}", $format_label, e)),
            }
        }
    };
}

impl DataManipulationTool {
    fn parse_json_impl(data_str: &str) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::from_str(data_str)
    }
    parse_operation!(
        parse_json,
        "parse_json requires 'data' parameter as a string",
        "JSON",
        Self::parse_json_impl
    );

    fn stringify_json(args: &serde_json::Value) -> ToolResult {
        let data = match args.get("data") {
            Some(d) => d,
            None => {
                return ToolResult::error("stringify_json requires 'data' parameter");
            }
        };

        match serde_json::to_string_pretty(data) {
            Ok(json_str) => ToolResult::success(json_str),
            Err(e) => ToolResult::error(format!("Failed to stringify JSON: {e}")),
        }
    }

    parse_operation!(
        parse_xml,
        "parse_xml requires 'data' parameter as a string",
        "XML",
        Self::xml_to_json
    );

    fn stringify_xml(args: &serde_json::Value) -> ToolResult {
        let data = match args.get("data") {
            Some(d) => d,
            None => {
                return ToolResult::error("stringify_xml requires 'data' parameter");
            }
        };

        match Self::json_to_xml(data) {
            Ok(xml_str) => ToolResult::success(xml_str),
            Err(e) => ToolResult::error(format!("Failed to stringify XML: {e}")),
        }
    }

    fn parse_csv(args: &serde_json::Value) -> ToolResult {
        let data_str = match args.get("data").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return ToolResult::error(
                    "parse_csv requires 'data' parameter as a string".to_string(),
                );
            }
        };

        let delimiter = params::extract_string_or(args, "csv_delimiter", ",");
        let delimiter_char = delimiter.chars().next().unwrap_or(',');

        let headers = args
            .get("csv_headers")
            .and_then(|v| v.as_array())
            .map(|headers_arr| {
                headers_arr
                    .iter()
                    .filter_map(|v| v.as_str().map(std::string::ToString::to_string))
                    .collect::<Vec<_>>()
            });

        match Self::csv_to_json(data_str, delimiter_char, headers) {
            Ok(json_value) => ToolResult::success(format!(
                "Successfully parsed CSV:\n{}",
                serde_json::to_string_pretty(&json_value)
                    .unwrap_or_else(|_| json_value.to_string())
            )),
            Err(e) => ToolResult::error(format!("Failed to parse CSV: {e}")),
        }
    }

    fn stringify_csv(args: &serde_json::Value) -> ToolResult {
        let data = match args.get("data") {
            Some(d) => d,
            None => {
                return ToolResult::error("stringify_csv requires 'data' parameter");
            }
        };

        let delimiter = params::extract_string_or(args, "csv_delimiter", ",");
        let delimiter_char = delimiter.chars().next().unwrap_or(',');

        match Self::json_to_csv(data, delimiter_char) {
            Ok(csv_str) => ToolResult::success(csv_str),
            Err(e) => ToolResult::error(format!("Failed to stringify CSV: {e}")),
        }
    }

    fn validate_json_schema(args: &serde_json::Value) -> ToolResult {
        let data = match args.get("data") {
            Some(d) => d,
            None => {
                return ToolResult::error(
                    "validate_json_schema requires 'data' parameter".to_string(),
                );
            }
        };

        let schema = match args.get("schema") {
            Some(s) => s,
            None => {
                return ToolResult::error(
                    "validate_json_schema requires 'schema' parameter".to_string(),
                );
            }
        };

        // Subset of JSON Schema: `type`, `required`, and nested `properties`.
        match Self::basic_schema_validation(data, schema) {
            Ok(()) => ToolResult::success("Data validates against schema"),
            Err(e) => ToolResult::error(format!("Schema validation failed: {e}")),
        }
    }

    fn convert_format(args: &serde_json::Value) -> ToolResult {
        let data = match args.get("data") {
            Some(d) => d,
            None => {
                return ToolResult::error("convert_format requires 'data' parameter");
            }
        };

        let format = match params::extract_string(args, "format", None) {
            Ok(f) => f,
            Err(e) => return ToolResult::error(e),
        };

        match format.as_str() {
            "json" => {
                if data.is_string() {
                    // Try to detect and parse current format
                    let data_str = match data.as_str() {
                        Some(s) => s,
                        None => {
                            return ToolResult::error(
                                "Data parameter must be a valid string".to_string(),
                            );
                        }
                    };
                    if data_str.trim_start().starts_with('<') {
                        // Looks like XML - avoid unnecessary json! wrapper
                        Self::parse_xml(&json!({"data": data_str}))
                    } else if data_str.contains(',') && data_str.contains('\n') {
                        // Looks like CSV - avoid unnecessary json! wrapper
                        Self::parse_csv(&json!({"data": data_str}))
                    } else {
                        ToolResult::error(String::from(
                            "Unable to detect input format. Please use parse_xml or parse_csv first.",
                        ))
                    }
                } else {
                    // Pass data directly without wrapping in json! macro
                    Self::stringify_json(&json!({"data": data}))
                }
            }
            "xml" => Self::stringify_xml(&json!({"data": data})),
            "csv" => Self::stringify_csv(&json!({"data": data})),
            _ => ToolResult::error(format!("Unsupported target format: {format}")),
        }
    }

    fn extract_json_path(args: &serde_json::Value) -> ToolResult {
        let data = match args.get("data") {
            Some(d) => d,
            None => {
                return ToolResult::error(
                    "extract_json_path requires 'data' parameter".to_string(),
                );
            }
        };

        let json_path = match params::extract_string(args, "json_path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };

        match Self::evaluate_json_path(data, &json_path) {
            Ok(result) => ToolResult::success(format!(
                "Extracted value:\n{}",
                serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string())
            )),
            Err(e) => ToolResult::error(format!("Failed to extract JSON path: {e}")),
        }
    }

    fn transform_json(args: &serde_json::Value) -> ToolResult {
        let data = match args.get("data") {
            Some(d) => d,
            None => {
                return ToolResult::error("transform_json requires 'data' parameter");
            }
        };

        let transformation = match args.get("transformation").and_then(|v| v.as_object()) {
            Some(t) => t,
            None => {
                return ToolResult::error(
                    "transform_json requires 'transformation' parameter as an object".to_string(),
                );
            }
        };

        match Self::apply_transformation(data, transformation) {
            Ok(result) => ToolResult::success(format!(
                "Transformed data:\n{}",
                serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string())
            )),
            Err(e) => ToolResult::error(format!("Transformation failed: {e}")),
        }
    }

    fn merge_json(args: &serde_json::Value) -> ToolResult {
        let data1 = match args.get("data") {
            Some(d) => d,
            None => {
                return ToolResult::error(
                    "merge_json requires 'data' parameter (first object)".to_string(),
                );
            }
        };

        // For merge, we expect data to be an array of objects or two separate data objects
        // Let's support both: if data is array, merge all; if data2 exists, merge data and data2
        let merge_mode = params::extract_string_or(args, "merge_mode", "deep");

        if let Some(data2) = args.get("data2") {
            match Self::merge_objects(data1, data2, &merge_mode) {
                Ok(result) => ToolResult::success(format!(
                    "Merged data:\n{}",
                    serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string())
                )),
                Err(e) => ToolResult::error(format!("Merge failed: {e}")),
            }
        } else if let Some(data_arr) = data1.as_array() {
            // Merge all objects in array
            let mut result = json!({});
            for item in data_arr {
                match Self::merge_objects(&result, item, &merge_mode) {
                    Ok(merged) => result = merged,
                    Err(e) => {
                        return ToolResult::error(format!("Merge failed: {e}"));
                    }
                }
            }
            ToolResult::success(format!(
                "Merged data:\n{}",
                serde_json::to_string_pretty(&result).unwrap_or_else(|_| result.to_string())
            ))
        } else {
            ToolResult::error(
                "merge_json requires 'data2' parameter or 'data' as an array of objects"
                    .to_string(),
            )
        }
    }

    // Helper functions

    fn xml_to_json(xml_str: &str) -> Result<serde_json::Value> {
        // Flat tag→text map for simple XML; nested structure is not preserved.
        let mut result = json!({});

        for cap in RE_XML_TAG.captures_iter(xml_str) {
            if let Some(tag) = cap.get(1).or_else(|| cap.get(3)) {
                let tag_name = tag.as_str().split_whitespace().next().unwrap_or("");
                let content = cap.get(2).map_or("", |m| m.as_str().trim());

                if !tag_name.is_empty() {
                    result[tag_name] = if content.is_empty() {
                        json!(null)
                    } else {
                        json!(content)
                    };
                }
            }
        }

        if result.is_object()
            && let Some(obj) = result.as_object()
            && obj.is_empty()
        {
            bail!("No valid XML tags found");
        }

        Ok(result)
    }

    fn json_to_xml(data: &serde_json::Value) -> Result<String> {
        let mut xml = String::new();
        Self::json_value_to_xml(data, "", &mut xml, 0)?;
        Ok(xml)
    }

    fn json_value_to_xml(
        value: &serde_json::Value,
        tag_name: &str,
        xml: &mut String,
        indent: usize,
    ) -> Result<()> {
        let indent_str = "  ".repeat(indent);

        match value {
            serde_json::Value::Object(obj) => {
                // Pre-allocate buffer for opening tag
                let mut opening_tag = String::with_capacity(indent_str.len() + tag_name.len() + 4);
                use std::fmt::Write;
                if writeln!(opening_tag, "{indent_str}<{tag_name}>").is_err() {
                    bail!("Failed to write XML opening tag");
                }
                xml.push_str(&opening_tag);

                for (key, val) in obj {
                    Self::json_value_to_xml(val, key, xml, indent + 1)?;
                }

                // Pre-allocate buffer for closing tag
                let mut closing_tag = String::with_capacity(indent_str.len() + tag_name.len() + 5);
                if writeln!(closing_tag, "{indent_str}</{tag_name}>").is_err() {
                    bail!("Failed to write XML closing tag");
                }
                xml.push_str(&closing_tag);
            }
            serde_json::Value::Array(arr) => {
                // Pre-allocate buffer for item tag
                let mut item_tag = String::with_capacity(tag_name.len() + 6);
                item_tag.push_str(tag_name);
                item_tag.push_str("_item");
                for item in arr {
                    Self::json_value_to_xml(item, &item_tag, xml, indent)?;
                }
            }
            _ => {
                let content = match value {
                    serde_json::Value::String(s) => s.as_str(), // Use reference instead of clone
                    serde_json::Value::Number(n) => {
                        // Store in temporary for formatting
                        let num_str = n.to_string();
                        // We need owned String for format!, so keep as is
                        return {
                            let mut tag_content = String::with_capacity(
                                indent_str.len() * 2 + tag_name.len() * 2 + num_str.len() + 10,
                            );
                            use std::fmt::Write;
                            if writeln!(
                                tag_content,
                                "{indent_str}<{tag_name}>{num_str}</{tag_name}>"
                            )
                            .is_err()
                            {
                                bail!("Failed to write XML number tag");
                            }
                            xml.push_str(&tag_content);
                            Ok(())
                        };
                    }
                    serde_json::Value::Bool(b) => {
                        let bool_str = b.to_string();
                        let mut tag_content = String::with_capacity(
                            indent_str.len() * 2 + tag_name.len() * 2 + bool_str.len() + 10,
                        );
                        use std::fmt::Write;
                        if writeln!(
                            tag_content,
                            "{indent_str}<{tag_name}>{bool_str}</{tag_name}>"
                        )
                        .is_err()
                        {
                            bail!("Failed to write XML boolean tag");
                        }
                        xml.push_str(&tag_content);
                        return Ok(());
                    }
                    serde_json::Value::Null => "",
                    _ => {
                        let val_str = value.to_string();
                        let mut tag_content = String::with_capacity(
                            indent_str.len() * 2 + tag_name.len() * 2 + val_str.len() + 10,
                        );
                        use std::fmt::Write;
                        if writeln!(
                            tag_content,
                            "{indent_str}<{tag_name}>{val_str}</{tag_name}>"
                        )
                        .is_err()
                        {
                            bail!("Failed to write XML value tag");
                        }
                        xml.push_str(&tag_content);
                        return Ok(());
                    }
                };
                // For String case, use reference
                let mut tag_content = String::with_capacity(
                    indent_str.len() * 2 + tag_name.len() * 2 + content.len() + 10,
                );
                use std::fmt::Write;
                if writeln!(
                    tag_content,
                    "{indent_str}<{tag_name}>{content}</{tag_name}>"
                )
                .is_err()
                {
                    bail!("Failed to write XML string tag");
                }
                xml.push_str(&tag_content);
            }
        }
        Ok(())
    }

    fn csv_to_json(
        csv_str: &str,
        delimiter: char,
        headers: Option<Vec<String>>,
    ) -> Result<serde_json::Value> {
        let lines: Vec<&str> = csv_str.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return Ok(json!([]));
        }

        let header_row = if let Some(ref h) = headers {
            h.clone()
        } else {
            lines[0]
                .split(delimiter)
                .map(|s| s.trim().to_string())
                .collect()
        };

        let start_idx = usize::from(!headers.is_some());
        let mut rows = Vec::new();

        for line in lines.iter().skip(start_idx) {
            let values: Vec<&str> = line.split(delimiter).map(str::trim).collect();
            let mut row = json!({});
            for (idx, header) in header_row.iter().enumerate() {
                if idx < values.len() {
                    row[header] = json!(values[idx]);
                }
            }
            rows.push(row);
        }

        Ok(json!(rows))
    }

    fn json_to_csv(data: &serde_json::Value, delimiter: char) -> Result<String> {
        let mut csv = String::new();

        if let Some(arr) = data.as_array() {
            if arr.is_empty() {
                return Ok(csv);
            }

            // Get headers from first object
            if let Some(first_obj) = arr[0].as_object() {
                let headers: Vec<String> = first_obj.keys().cloned().collect();
                csv.push_str(&headers.join(&delimiter.to_string()));
                csv.push('\n');

                for obj in arr {
                    if let Some(obj_map) = obj.as_object() {
                        let values: Vec<String> = headers
                            .iter()
                            .map(|h| {
                                obj_map
                                    .get(h)
                                    .map(|v| v.to_string().trim_matches('"').to_string())
                                    .unwrap_or_default()
                            })
                            .collect();
                        csv.push_str(&values.join(&delimiter.to_string()));
                        csv.push('\n');
                    }
                }
            } else {
                bail!("CSV conversion requires array of objects");
            }
        } else if let Some(obj) = data.as_object() {
            // Single object - header row + data row
            let headers: Vec<String> = obj.keys().cloned().collect();
            csv.push_str(&headers.join(&delimiter.to_string()));
            csv.push('\n');
            let values: Vec<String> = headers
                .iter()
                .map(|h| {
                    obj.get(h)
                        .map(|v| v.to_string().trim_matches('"').to_string())
                        .unwrap_or_default()
                })
                .collect();
            csv.push_str(&values.join(&delimiter.to_string()));
        } else {
            bail!("CSV conversion requires object or array of objects");
        }

        Ok(csv)
    }

    fn basic_schema_validation(data: &serde_json::Value, schema: &serde_json::Value) -> Result<()> {
        crate::utils::json_schema_lite::validate_basic_schema(data, schema)
    }

    fn evaluate_json_path(data: &serde_json::Value, path: &str) -> Result<serde_json::Value> {
        if path != "$" && !path.starts_with('$') {
            bail!("JSONPath must start with $");
        }
        crate::utils::json_path::evaluate(data, path)
    }

    fn apply_transformation(
        data: &serde_json::Value,
        transformation: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let mut result = json!({});

        for (output_key, transform_value) in transformation {
            if let Some(json_path) = transform_value.as_str() {
                // Extract value using JSONPath
                let extracted = Self::evaluate_json_path(data, json_path)?;
                result[output_key] = extracted;
            } else {
                // Use value directly
                result[output_key] = transform_value.clone();
            }
        }

        Ok(result)
    }

    fn merge_objects(
        obj1: &serde_json::Value,
        obj2: &serde_json::Value,
        mode: &str,
    ) -> Result<serde_json::Value> {
        match (obj1.as_object(), obj2.as_object()) {
            (Some(_), Some(_)) => {
                let merge_mode = json_merge_mode_from_str(mode)?;
                let mut result = obj1.clone();
                merge_json_value(&mut result, obj2.clone(), merge_mode);
                Ok(result)
            }
            _ => bail!("Both values must be objects for merging"),
        }
    }
}
