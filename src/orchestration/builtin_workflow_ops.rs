//! Built-in workflow transform and custom-merge operations shared by CLI and API executors.

use crate::orchestration::ExecutionContext;
use anyhow::Result;
use serde_json::Value;

#[inline]
fn json_numeric_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|i| i as f64))
        .or_else(|| v.as_u64().map(|u| u as f64))
}

/// Built-in transform functions for [`super::WorkflowNodeExecutor::transform`].
pub fn builtin_workflow_transform(
    function: &str,
    _context: &ExecutionContext,
    last_output: Option<&Value>,
) -> Result<Value> {
    match function {
        "identity" => Ok(last_output.cloned().unwrap_or(Value::Null)),
        "to_string" => Ok(Value::String(
            last_output
                .as_ref()
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".to_string()),
        )),
        "to_number" => transform_to_number(last_output),
        "to_bool" => transform_to_bool(last_output),
        _ if function.starts_with("set:") => {
            let key = function.strip_prefix("set:").unwrap_or("").trim();
            if key.is_empty() {
                anyhow::bail!("Transform 'set:' requires a variable name");
            }
            Ok(last_output.cloned().unwrap_or(Value::Null))
        }
        _ => anyhow::bail!("Unknown transform function '{}'", function),
    }
}

fn transform_to_number(last_output: Option<&Value>) -> Result<Value> {
    let Some(output) = last_output else {
        return Ok(Value::Number(0.into()));
    };
    if let Some(number) = output.as_i64() {
        return Ok(Value::Number(number.into()));
    }
    if let Some(number) = output.as_u64() {
        return Ok(Value::Number(number.into()));
    }
    if let Some(number) = output.as_f64() {
        return json_number_from_f64(number);
    }
    if let Some(text) = output.as_str() {
        let number = text
            .parse::<f64>()
            .map_err(|_| anyhow::anyhow!("Cannot convert '{text}' to number"))?;
        return json_number_from_f64(number);
    }
    anyhow::bail!("Cannot convert value to number")
}

fn transform_to_bool(last_output: Option<&Value>) -> Result<Value> {
    let Some(output) = last_output else {
        return Ok(Value::Bool(false));
    };
    if let Some(value) = output.as_bool() {
        return Ok(Value::Bool(value));
    }
    if let Some(text) = output.as_str() {
        return match text.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Ok(Value::Bool(true)),
            "false" | "0" | "no" | "off" => Ok(Value::Bool(false)),
            _ => anyhow::bail!("Cannot convert '{text}' to boolean"),
        };
    }
    anyhow::bail!("Cannot convert value to boolean")
}

/// Built-in custom merge strategies for [`super::WorkflowNodeExecutor::merge_custom`].
pub fn builtin_workflow_merge_custom(
    name: &str,
    inputs: &Value,
    _context: &ExecutionContext,
) -> Result<Value> {
    match name {
        "concat" => merge_concat(inputs),
        "join" => merge_join(inputs),
        "average" => merge_average(inputs),
        "sum" => merge_sum(inputs),
        "first" => merge_first(inputs),
        "last" => merge_last(inputs),
        _ => anyhow::bail!(
            "Unknown custom merge strategy '{}'. Supported: concat, join, average, sum, first, last",
            name
        ),
    }
}

fn json_number_from_f64(value: f64) -> Result<Value> {
    Ok(Value::Number(
        serde_json::Number::from_f64(value).ok_or_else(|| anyhow::anyhow!("Invalid number"))?,
    ))
}

fn merge_concat(inputs: &Value) -> Result<Value> {
    if let Some(arr) = inputs.as_array() {
        let strings: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
        Ok(Value::String(strings.join(" ")))
    } else if let Some(obj) = inputs.as_object() {
        let strings: Vec<String> = obj.values().map(|v| v.to_string()).collect();
        Ok(Value::String(strings.join(" ")))
    } else {
        Ok(Value::String(inputs.to_string()))
    }
}

fn merge_join(inputs: &Value) -> Result<Value> {
    if let Some(arr) = inputs.as_array() {
        let strings: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
        Ok(Value::String(strings.join("\n")))
    } else {
        Ok(Value::String(inputs.to_string()))
    }
}

fn average_of_numbers(numbers: Vec<f64>) -> Result<Value> {
    if numbers.is_empty() {
        anyhow::bail!("No numeric values to average");
    }
    let sum: f64 = numbers.iter().sum();
    let avg = sum / numbers.len() as f64;
    json_number_from_f64(avg)
}

fn merge_average(inputs: &Value) -> Result<Value> {
    if let Some(arr) = inputs.as_array() {
        average_of_numbers(arr.iter().filter_map(json_numeric_f64).collect())
    } else if let Some(obj) = inputs.as_object() {
        average_of_numbers(obj.values().filter_map(json_numeric_f64).collect())
    } else {
        anyhow::bail!("Cannot calculate average of non-array/non-object value")
    }
}

fn merge_sum(inputs: &Value) -> Result<Value> {
    if let Some(arr) = inputs.as_array() {
        json_number_from_f64(arr.iter().filter_map(json_numeric_f64).sum())
    } else if let Some(obj) = inputs.as_object() {
        json_number_from_f64(obj.values().filter_map(json_numeric_f64).sum())
    } else {
        anyhow::bail!("Cannot sum non-array/non-object value")
    }
}

fn merge_first(inputs: &Value) -> Result<Value> {
    if let Some(arr) = inputs.as_array() {
        return Ok(arr.first().cloned().unwrap_or(Value::Null));
    }
    if let Some(obj) = inputs.as_object() {
        return Ok(object_edge_value(obj, true));
    }
    Ok(inputs.clone())
}

fn merge_last(inputs: &Value) -> Result<Value> {
    if let Some(arr) = inputs.as_array() {
        return Ok(arr.last().cloned().unwrap_or(Value::Null));
    }
    if let Some(obj) = inputs.as_object() {
        return Ok(object_edge_value(obj, false));
    }
    Ok(inputs.clone())
}

/// Pick one object value by key order.
///
/// JSON objects have no order of their own, and `serde_json` keeps either
/// sorted keys or insertion order depending on crate features. Compare the
/// keys as strings so `first` and `last` stay the same in every build.
fn object_edge_value(obj: &serde_json::Map<String, Value>, first: bool) -> Value {
    let mut selected: Option<(&str, &Value)> = None;
    for (key, value) in obj {
        let take = match selected {
            None => true,
            Some((best, _)) if first => key.as_str() < best,
            Some((best, _)) => key.as_str() > best,
        };
        if take {
            selected = Some((key.as_str(), value));
        }
    }
    selected
        .map(|(_, value)| value.clone())
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx() -> ExecutionContext {
        ExecutionContext::new("wf_test".to_string())
    }

    #[test]
    fn transform_identity_and_to_string() {
        let c = ctx();
        assert_eq!(
            builtin_workflow_transform("identity", &c, None).unwrap(),
            Value::Null
        );
        assert_eq!(
            builtin_workflow_transform("identity", &c, Some(&json!("x"))).unwrap(),
            json!("x")
        );
        let s = builtin_workflow_transform("to_string", &c, Some(&json!({"a": 1}))).unwrap();
        assert_eq!(s, Value::String(r#"{"a":1}"#.to_string()));
    }

    #[test]
    fn transform_to_number() {
        let c = ctx();
        assert_eq!(
            builtin_workflow_transform("to_number", &c, None).unwrap(),
            json!(0)
        );
        assert_eq!(
            builtin_workflow_transform("to_number", &c, Some(&json!(42))).unwrap(),
            json!(42)
        );
        assert_eq!(
            builtin_workflow_transform("to_number", &c, Some(&json!("3.5"))).unwrap(),
            json!(3.5)
        );
        assert!(builtin_workflow_transform("to_number", &c, Some(&json!("nope"))).is_err());
        assert!(builtin_workflow_transform("to_number", &c, Some(&json!([1]))).is_err());
    }

    #[test]
    fn transform_to_bool() {
        let c = ctx();
        assert_eq!(
            builtin_workflow_transform("to_bool", &c, None).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            builtin_workflow_transform("to_bool", &c, Some(&json!(true))).unwrap(),
            Value::Bool(true)
        );
        for (s, b) in [
            ("true", true),
            ("TRUE", true),
            ("1", true),
            ("yes", true),
            ("on", true),
            ("false", false),
            ("0", false),
            ("no", false),
            ("off", false),
        ] {
            assert_eq!(
                builtin_workflow_transform("to_bool", &c, Some(&json!(s))).unwrap(),
                Value::Bool(b),
                "str {s}"
            );
        }
        assert!(builtin_workflow_transform("to_bool", &c, Some(&json!("maybe"))).is_err());
    }

    #[test]
    fn transform_set_prefix() {
        let c = ctx();
        assert!(builtin_workflow_transform("set:", &c, Some(&json!(1))).is_err());
        assert!(builtin_workflow_transform("set:  ", &c, Some(&json!(1))).is_err());
        assert_eq!(
            builtin_workflow_transform("set: out_var", &c, Some(&json!([2]))).unwrap(),
            json!([2])
        );
    }

    #[test]
    fn transform_unknown() {
        let c = ctx();
        assert!(builtin_workflow_transform("nope", &c, None).is_err());
    }

    #[test]
    fn merge_concat_join() {
        let c = ctx();
        let arr = json!(["a", "b"]);
        assert_eq!(
            builtin_workflow_merge_custom("concat", &arr, &c).unwrap(),
            Value::String(r#""a" "b""#.to_string())
        );
        assert_eq!(
            builtin_workflow_merge_custom("join", &arr, &c).unwrap(),
            Value::String(
                r#""a"
"b""#
                    .to_string()
            )
        );
    }

    #[test]
    fn merge_average_sum() {
        let c = ctx();
        assert_eq!(
            builtin_workflow_merge_custom("average", &json!([2, 4, 6]), &c).unwrap(),
            json!(4.0)
        );
        assert_eq!(
            builtin_workflow_merge_custom("sum", &json!([1, 2, 3]), &c).unwrap(),
            json!(6.0)
        );
        assert!(builtin_workflow_merge_custom("average", &json!([]), &c).is_err());
        assert!(builtin_workflow_merge_custom("average", &json!(["x"]), &c).is_err());
    }

    #[test]
    fn merge_first_last_object_order() {
        let c = ctx();
        let obj = json!({"z": 1, "a": 2});
        // Object edges follow key text, so "a" is first and "z" is last.
        let first = builtin_workflow_merge_custom("first", &obj, &c).unwrap();
        let last = builtin_workflow_merge_custom("last", &obj, &c).unwrap();
        assert_eq!(first, json!(2)); // "a" -> 2 is first alphabetically
        assert_eq!(last, json!(1)); // "z" -> 1 is last alphabetically
    }

    #[test]
    fn merge_unknown() {
        let c = ctx();
        assert!(builtin_workflow_merge_custom("unknown", &json!([]), &c).is_err());
    }
}
