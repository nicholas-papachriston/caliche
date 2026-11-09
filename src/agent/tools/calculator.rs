use crate::agent::tools::common::params;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use async_trait::async_trait;
use serde_json::json;

const NAME: &str = "calculator";
const DESCRIPTION: &str =
    "Performs mathematical operations including arithmetic, trigonometry, logarithms, and more.";
#[derive(Debug)]
pub struct CalculatorTool;

impl ToolMeta for CalculatorTool {
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
                        "add", "subtract", "multiply", "divide", "power", "modulo",
                        "sqrt", "abs", "sin", "cos", "tan", "asin", "acos", "atan",
                        "log", "ln", "factorial", "min", "max", "round", "floor", "ceil"
                    ],
                    "description": "The mathematical operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number (or the only number for single-operand operations)"
                },
                "b": {
                    "type": "number",
                    "description": "Second number (required only for two-operand operations)"
                }
            },
            "required": ["operation", "a"]
        })
    }
}

#[async_trait]
impl Tool for CalculatorTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let operation = params::extract_string_or(&arguments, "operation", "");

        // Validate operation parameter
        if operation.is_empty() {
            tracing::warn!(
                "Calculator called with empty operation. Arguments: {:?}",
                arguments
            );
            return ToolResult::error(
                "Missing required parameter 'operation'. Please specify the mathematical operation to perform.",
            );
        }

        let a = params::extract_float_or(&arguments, "a", 0.0);
        let b = arguments.get("b").and_then(serde_json::Value::as_f64);

        let result = match calculate(&operation, a, b) {
            Ok(value) => value,
            Err(CalcError::Unknown) => {
                tracing::warn!(
                    "Unknown calculator operation '{}'. Arguments: {:?}",
                    operation,
                    arguments
                );
                return ToolResult::error(CalcError::Unknown.message(&operation));
            }
            Err(error) => return ToolResult::error(error.message(&operation)),
        };

        // Format result nicely
        let formatted = if result.fract() == 0.0 && result.abs() < 1e15 {
            (result as i64).to_string()
        } else {
            result.to_string()
        };

        ToolResult::success(format!("Result: {formatted}"))
    }
}

enum CalcError {
    MissingOperand(&'static str),
    Domain(&'static str),
    Unknown,
}

impl CalcError {
    fn message(self, operation: &str) -> String {
        match self {
            Self::MissingOperand(name) => format!("Operation '{name}' requires parameter 'b'"),
            Self::Domain(message) => message.to_owned(),
            Self::Unknown => format!(
                "Unknown operation: '{operation}'. Supported operations: add, subtract, multiply, divide, power, modulo, sqrt, abs, sin, cos, tan, asin, acos, atan, log, ln, factorial, min, max, round, floor, ceil"
            ),
        }
    }
}

enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Modulo,
    Min,
    Max,
}

enum UnaryOp {
    Sqrt,
    Abs,
    Round,
    Floor,
    Ceil,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Ln,
    Log,
    Factorial,
}

fn calculate(operation: &str, a: f64, b: Option<f64>) -> Result<f64, CalcError> {
    if let Some(op) = parse_binary(operation) {
        return eval_binary(op, a, b);
    }
    if let Some(op) = parse_unary(operation) {
        return eval_unary(op, a);
    }
    Err(CalcError::Unknown)
}

fn parse_binary(operation: &str) -> Option<BinaryOp> {
    Some(match operation {
        "add" => BinaryOp::Add,
        "subtract" => BinaryOp::Subtract,
        "multiply" => BinaryOp::Multiply,
        "divide" => BinaryOp::Divide,
        "power" => BinaryOp::Power,
        "modulo" => BinaryOp::Modulo,
        "min" => BinaryOp::Min,
        "max" => BinaryOp::Max,
        _ => return None,
    })
}

fn parse_unary(operation: &str) -> Option<UnaryOp> {
    Some(match operation {
        "sqrt" => UnaryOp::Sqrt,
        "abs" => UnaryOp::Abs,
        "round" => UnaryOp::Round,
        "floor" => UnaryOp::Floor,
        "ceil" => UnaryOp::Ceil,
        "sin" => UnaryOp::Sin,
        "cos" => UnaryOp::Cos,
        "tan" => UnaryOp::Tan,
        "asin" => UnaryOp::Asin,
        "acos" => UnaryOp::Acos,
        "atan" => UnaryOp::Atan,
        "ln" => UnaryOp::Ln,
        "log" => UnaryOp::Log,
        "factorial" => UnaryOp::Factorial,
        _ => return None,
    })
}

fn require_operand(operation: &'static str, b: Option<f64>) -> Result<f64, CalcError> {
    b.ok_or(CalcError::MissingOperand(operation))
}

fn eval_binary(operation: BinaryOp, a: f64, b: Option<f64>) -> Result<f64, CalcError> {
    match operation {
        BinaryOp::Add => Ok(a + require_operand("add", b)?),
        BinaryOp::Subtract => Ok(a - require_operand("subtract", b)?),
        BinaryOp::Multiply => Ok(a * require_operand("multiply", b)?),
        BinaryOp::Divide => divide(a, require_operand("divide", b)?),
        BinaryOp::Power => Ok(a.powf(require_operand("power", b)?)),
        BinaryOp::Modulo => modulo(a, require_operand("modulo", b)?),
        BinaryOp::Min => Ok(a.min(require_operand("min", b)?)),
        BinaryOp::Max => Ok(a.max(require_operand("max", b)?)),
    }
}

fn divide(a: f64, divisor: f64) -> Result<f64, CalcError> {
    if divisor == 0.0 {
        return Err(CalcError::Domain("Division by zero"));
    }
    Ok(a / divisor)
}

fn modulo(a: f64, divisor: f64) -> Result<f64, CalcError> {
    if divisor == 0.0 {
        return Err(CalcError::Domain("Modulo by zero"));
    }
    Ok(a % divisor)
}

fn eval_unary(operation: UnaryOp, a: f64) -> Result<f64, CalcError> {
    match operation {
        UnaryOp::Sqrt => {
            if a < 0.0 {
                return Err(CalcError::Domain(
                    "Cannot take square root of negative number",
                ));
            }
            Ok(a.sqrt())
        }
        UnaryOp::Abs => Ok(a.abs()),
        UnaryOp::Round => Ok(a.round()),
        UnaryOp::Floor => Ok(a.floor()),
        UnaryOp::Ceil => Ok(a.ceil()),
        UnaryOp::Sin => Ok(a.sin()),
        UnaryOp::Cos => Ok(a.cos()),
        UnaryOp::Tan => Ok(a.tan()),
        UnaryOp::Asin => inverse_trig(
            a,
            f64::asin,
            "asin domain error: input must be between -1 and 1",
        ),
        UnaryOp::Acos => inverse_trig(
            a,
            f64::acos,
            "acos domain error: input must be between -1 and 1",
        ),
        UnaryOp::Atan => Ok(a.atan()),
        UnaryOp::Ln => positive_log(a, f64::ln, "Natural log requires positive number"),
        UnaryOp::Log => positive_log(a, f64::log10, "Log base 10 requires positive number"),
        UnaryOp::Factorial => factorial(a),
    }
}

fn inverse_trig(a: f64, function: fn(f64) -> f64, message: &'static str) -> Result<f64, CalcError> {
    if !(-1.0..=1.0).contains(&a) {
        return Err(CalcError::Domain(message));
    }
    Ok(function(a))
}

fn positive_log(a: f64, function: fn(f64) -> f64, message: &'static str) -> Result<f64, CalcError> {
    if a <= 0.0 {
        return Err(CalcError::Domain(message));
    }
    Ok(function(a))
}

fn factorial(a: f64) -> Result<f64, CalcError> {
    if a < 0.0 || a.fract() != 0.0 {
        return Err(CalcError::Domain("Factorial requires non-negative integer"));
    }
    if a > 170.0 {
        return Err(CalcError::Domain("Factorial input too large (max 170)"));
    }
    let n = a as u64;
    let mut result = 1u128;
    for i in 2..=n {
        result *= u128::from(i);
    }
    Ok(result as f64)
}

#[cfg(test)]
mod tests {
    use super::{CalcError, calculate};

    #[test]
    fn binary_ops_require_the_second_operand() {
        assert!(matches!(
            calculate("add", 1.0, None),
            Err(CalcError::MissingOperand("add"))
        ));
    }

    #[test]
    fn unknown_operation_is_a_typed_error() {
        assert!(matches!(
            calculate("nope", 1.0, Some(2.0)),
            Err(CalcError::Unknown)
        ));
        let message = CalcError::Unknown.message("nope");
        assert!(message.contains("Unknown operation: 'nope'"));
    }

    #[test]
    fn divide_by_zero_is_a_domain_error() {
        assert!(matches!(
            calculate("divide", 1.0, Some(0.0)),
            Err(CalcError::Domain("Division by zero"))
        ));
    }

    #[test]
    fn sqrt_of_four_is_two() {
        assert!(matches!(
            calculate("sqrt", 4.0, None),
            Ok(value) if (value - 2.0).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn negative_sqrt_is_a_domain_error() {
        assert!(matches!(
            calculate("sqrt", -1.0, None),
            Err(CalcError::Domain(
                "Cannot take square root of negative number"
            ))
        ));
    }

    #[test]
    fn missing_operand_names_the_operation() {
        assert_eq!(
            CalcError::MissingOperand("add").message("add"),
            "Operation 'add' requires parameter 'b'"
        );
    }

    #[test]
    fn factorial_of_five_is_one_hundred_twenty() {
        assert!(matches!(
            calculate("factorial", 5.0, None),
            Ok(value) if (value - 120.0).abs() < f64::EPSILON
        ));
    }
}
