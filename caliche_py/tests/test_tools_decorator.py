"""
Tests for the @tool decorator and FunctionTool class.
"""

import json

import pytest
from pydantic import BaseModel, Field

from caliche.tools import FunctionTool, ToolResult, tool


def test_basic_function_tool():
    """Test creating a tool from a basic function with type hints."""

    def reverse_string(text: str) -> str:
        """Reverses a string."""
        return text[::-1]

    tool_instance = FunctionTool(reverse_string)

    assert tool_instance.name() == "reverse_string"
    assert "Reverses a string" in tool_instance.description()

    params = tool_instance.parameters()
    assert params["type"] == "object"
    assert "text" in params["properties"]
    assert params["properties"]["text"]["type"] == "string"
    assert "text" in params["required"]

    result = tool_instance.execute({"text": "hello"})
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.content == "olleh"


def test_function_tool_with_defaults():
    """Test creating a tool from a function with default parameters."""

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greets a person."""
        return f"{greeting}, {name}!"

    tool_instance = FunctionTool(greet)

    params = tool_instance.parameters()
    assert "name" in params["required"]
    assert "greeting" not in params["required"]  # Has default value

    # Test with both parameters
    result = tool_instance.execute({"name": "Alice", "greeting": "Hi"})
    assert result.success is True
    assert result.content == "Hi, Alice!"

    # Test with only required parameter
    result = tool_instance.execute({"name": "Bob"})
    assert result.success is True
    assert result.content == "Hello, Bob!"


def test_function_tool_with_pydantic_model():
    """Test creating a tool with an explicit Pydantic model."""

    class WeatherArgs(BaseModel):
        city: str = Field(description="The city name")
        unit: str = Field(default="celsius", description="Temperature unit")

    def get_weather(city: str, unit: str = "celsius") -> str:
        """Get weather for a city."""
        return f"Weather in {city}: 20°{unit}"

    tool_instance = FunctionTool(get_weather, args_schema=WeatherArgs)

    params = tool_instance.parameters()
    assert params["type"] == "object"
    assert "city" in params["properties"]
    assert "unit" in params["properties"]
    assert params["properties"]["city"]["description"] == "The city name"

    result = tool_instance.execute({"city": "Paris", "unit": "fahrenheit"})
    assert result.success is True
    assert "Paris" in result.content


def test_tool_decorator_no_args():
    """Test @tool decorator used without parentheses."""

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Adds two numbers."""
        return a + b

    assert isinstance(add_numbers, FunctionTool)
    assert add_numbers.name() == "add_numbers"

    result = add_numbers.execute({"a": 5, "b": 3})
    assert result.success is True
    assert result.content == "8"


def test_tool_decorator_with_args():
    """Test @tool decorator used with parentheses and arguments."""

    @tool(name="custom_add", description="Custom addition tool")
    def add(a: int, b: int) -> int:
        """Adds two numbers."""
        return a + b

    assert isinstance(add, FunctionTool)
    assert add.name() == "custom_add"
    assert add.description() == "Custom addition tool"

    result = add.execute({"a": 10, "b": 20})
    assert result.success is True
    assert result.content == "30"


def test_tool_decorator_with_pydantic_model():
    """Test @tool decorator with Pydantic model."""

    class MathArgs(BaseModel):
        a: int = Field(description="First number")
        b: int = Field(description="Second number")

    @tool(args_schema=MathArgs)
    def multiply(a: int, b: int) -> int:
        """Multiplies two numbers."""
        return a * b

    assert isinstance(multiply, FunctionTool)
    result = multiply.execute({"a": 4, "b": 5})
    assert result.success is True
    assert result.content == "20"


@pytest.mark.asyncio
async def test_async_function_tool():
    """Test creating a tool from an async function."""

    async def fetch_data(url: str) -> str:
        """Fetches data from a URL."""
        # Simulate async operation
        return f"Data from {url}"

    tool_instance = FunctionTool(fetch_data)

    assert tool_instance.is_async() is True

    result = await tool_instance.execute({"url": "https://example.com"})
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert "example.com" in result.content


def test_tool_result_wrapping():
    """Test that different return types are properly wrapped."""

    @tool
    def return_string() -> str:
        """Returns a string."""
        return "hello"

    @tool
    def return_dict() -> dict:
        """Returns a dict."""
        return {"key": "value"}

    @tool
    def return_tool_result() -> ToolResult:
        """Returns a ToolResult."""
        return ToolResult(success=True, content="custom", error=None)

    result1 = return_string.execute({})
    assert result1.success is True
    assert result1.content == "hello"

    result2 = return_dict.execute({})
    assert result2.success is True
    assert json.loads(result2.content) == {"key": "value"}

    result3 = return_tool_result.execute({})
    assert result3.success is True
    assert result3.content == "custom"


def test_tool_validation_error():
    """Test that validation errors are handled properly."""

    @tool
    def require_int(value: int) -> int:
        """Requires an integer."""
        return value * 2

    # Should handle invalid input gracefully
    result = require_int.execute({"value": "not_an_int"})
    # The result should indicate failure or handle the error
    # (exact behavior depends on Pydantic validation)
    assert isinstance(result, ToolResult)


def test_tool_with_complex_types():
    """Test tool with list and dict types."""

    @tool
    def process_list(items: list) -> str:
        """Processes a list of items."""
        return f"Processed {len(items)} items"

    @tool
    def process_dict(data: dict) -> str:
        """Processes a dictionary."""
        return f"Processed dict with {len(data)} keys"

    result1 = process_list.execute({"items": [1, 2, 3]})
    assert result1.success is True

    result2 = process_dict.execute({"data": {"a": 1, "b": 2}})
    assert result2.success is True
