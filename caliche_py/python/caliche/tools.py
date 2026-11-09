"""
Tool base class and examples for creating custom tools in Python.

Custom tools allow you to extend the agent's capabilities with domain-specific
functionality. Tools can be synchronous or asynchronous.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import Any, Callable, Dict, Optional, Type, Union, overload

from caliche import ToolResult

try:
    from pydantic import BaseModel, Field, create_model
    from pydantic._internal._model_construction import ModelMetaclass

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None  # type: ignore[assignment,misc]
    create_model = None  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment,misc]
    ModelMetaclass = None  # type: ignore[assignment,misc]


class Tool(ABC):
    """
    Abstract base class for custom tools.

    To create a custom tool, inherit from this class and implement:
    - name(): Returns the tool name
    - description(): Returns a description of what the tool does
    - parameters(): Returns a JSON schema dict describing the tool's parameters
    - execute(arguments): Executes the tool with the given arguments

    The execute method can be synchronous or asynchronous (async def).
    It should return a ToolResult, a dict with 'success', 'content', and optionally 'error',
    or a plain string (which will be wrapped in a successful ToolResult).

    Example:
        class ReverseStringTool(Tool):
            def name(self) -> str:
                return "reverse_string"

            def description(self) -> str:
                return "Reverses a string"

            def parameters(self) -> Dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to reverse"
                        }
                    },
                    "required": ["text"]
                }

            def execute(self, arguments: Dict[str, Any]) -> ToolResult:
                text = arguments.get("text", "")
                reversed_text = text[::-1]
                return ToolResult(
                    success=True,
                    content=json.dumps({"original": text, "reversed": reversed_text}),
                    error=None
                )
    """

    @abstractmethod
    def name(self) -> str:
        """Return the tool name (must be unique)."""
        pass

    @abstractmethod
    def description(self) -> str:
        """Return a description of what the tool does."""
        pass

    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        Return a JSON schema dict describing the tool's parameters.

        Should follow the JSON Schema format, e.g.:
        {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param_name"]
        }
        """
        pass

    @abstractmethod
    def execute(
        self, arguments: Dict[str, Any]
    ) -> Union[
        ToolResult,
        Dict[str, Any],
        str,
        Awaitable[ToolResult],
        Awaitable[Dict[str, Any]],
        Awaitable[str],
    ]:
        """
        Execute the tool with the given arguments.

        This method can be synchronous or asynchronous (async def).
        If async, it should return an awaitable that resolves to the result.

        Args:
            arguments: Dict containing the tool arguments

        Returns:
            ToolResult, dict, str, or an awaitable of any of these.
            If dict, should have 'success', 'content', and optionally 'error' keys.
            If str, will be wrapped in a successful ToolResult.
        """
        pass

    def is_async(self) -> bool:
        """Check if the execute method is async.

        Returns:
            True if execute is an async method, False otherwise.
        """
        return asyncio.iscoroutinefunction(self.execute)


class ReverseStringTool(Tool):
    """Example tool that reverses a string."""

    def name(self) -> str:
        return "reverse_string"

    def description(self) -> str:
        return "Reverses a string. Takes a 'text' parameter and returns the reversed text."

    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The text to reverse"}},
            "required": ["text"],
        }

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        text = arguments.get("text", "")
        if not text:
            return ToolResult(success=False, content="", error="Missing 'text' parameter")

        reversed_text = text[::-1]
        result = {"original": text, "reversed": reversed_text}
        return ToolResult(success=True, content=json.dumps(result), error=None)


class WordCountTool(Tool):
    """Example tool that counts words, characters, and lines in text."""

    def name(self) -> str:
        return "count_words"

    def description(self) -> str:
        return "Counts the number of words, characters, and lines in a text string."

    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The text to analyze"}},
            "required": ["text"],
        }

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        text = arguments.get("text", "")
        if not text:
            return ToolResult(success=False, content="", error="Missing 'text' parameter")

        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.splitlines())

        result = {"text": text, "words": word_count, "characters": char_count, "lines": line_count}
        return ToolResult(success=True, content=json.dumps(result), error=None)


class RandomNumberTool(Tool):
    """Example tool that generates a random number between min and max."""

    def name(self) -> str:
        return "random_number"

    def description(self) -> str:
        return "Generates a random number between min and max (inclusive)."

    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "min": {"type": "number", "description": "Minimum value (inclusive)"},
                "max": {"type": "number", "description": "Maximum value (inclusive)"},
            },
            "required": ["min", "max"],
        }

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        import random

        try:
            min_val = arguments.get("min")
            max_val = arguments.get("max")

            if min_val is None or max_val is None:
                return ToolResult(
                    success=False, content="", error="Missing 'min' or 'max' parameter"
                )

            min_val = float(min_val)
            max_val = float(max_val)

            if min_val > max_val:
                return ToolResult(
                    success=False, content="", error="min must be less than or equal to max"
                )

            random_num = random.randint(int(min_val), int(max_val))

            result = {"min": min_val, "max": max_val, "result": random_num}
            return ToolResult(success=True, content=json.dumps(result), error=None)
        except (ValueError, TypeError) as e:
            return ToolResult(success=False, content="", error=f"Invalid parameters: {e}")


class FunctionTool(Tool):
    """
    A Tool implementation that wraps a Python function with automatic schema generation.

    This class automatically generates a JSON schema from function type hints or a provided
    Pydantic model, making it easy to create tools from regular Python functions.

    Example:
        def get_weather(city: str, unit: str = "celsius") -> str:
            \"\"\"Get the weather for a city.\"\"\"
            return f"Weather in {city}: 20°{unit}"

        tool = FunctionTool(get_weather)
        # Or with explicit Pydantic model:
        from pydantic import BaseModel
        class WeatherArgs(BaseModel):
            city: str
            unit: str = "celsius"
        tool = FunctionTool(get_weather, args_schema=WeatherArgs)
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None,
    ):
        """
        Initialize a FunctionTool.

        Args:
            func: The function to wrap as a tool
            name: Optional tool name (defaults to func.__name__)
            description: Optional tool description (defaults to func.__doc__)
            args_schema: Optional Pydantic model for argument validation
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError(
                "pydantic is required for FunctionTool. Install it with: pip install pydantic>=2.0.0"
            )

        self._func = func
        self._name = name or getattr(func, "__name__", "unknown")
        self._description = description or (func.__doc__ or "").strip()
        self._args_schema = args_schema
        self._is_async = asyncio.iscoroutinefunction(func)

        # Generate schema if not provided
        if args_schema is None:
            self._args_schema = self._create_schema_from_signature(func)

    def _create_schema_from_signature(self, func: Callable) -> Type[BaseModel]:
        """Create a Pydantic model from function signature and type hints."""
        sig = inspect.signature(func)
        fields = {}
        required_fields = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            param_default = param.default if param.default != inspect.Parameter.empty else ...

            # Convert common types to Pydantic-compatible types
            pydantic_type: Any
            if param_type == str or param_type is str:
                pydantic_type = str
            elif param_type == int or param_type is int:
                pydantic_type = int
            elif param_type == float or param_type is float:
                pydantic_type = float
            elif param_type == bool or param_type is bool:
                pydantic_type = bool
            elif param_type == list or param_type is list:
                pydantic_type = list
            elif param_type == dict or param_type is dict:
                pydantic_type = dict
            else:
                # Default to str for unknown types
                pydantic_type = str

            if param_default is ...:
                fields[param_name] = (pydantic_type, ...)
                required_fields.append(param_name)
            else:
                fields[param_name] = (pydantic_type, Field(default=param_default))

        # Create Pydantic model dynamically
        model_name = f"{getattr(func, '__name__', 'unknown')}Args"
        return create_model(model_name, **fields)  # type: ignore[call-overload]

    def is_async(self) -> bool:
        """Return whether the wrapped function is async."""
        return self._is_async

    def name(self) -> str:
        """Return the tool name."""
        return str(self._name)

    def description(self) -> str:
        """Return the tool description."""
        return str(self._description)

    def parameters(self) -> Dict[str, Any]:
        """Return JSON schema for tool parameters."""
        if self._args_schema is None:
            return {
                "type": "object",
                "properties": {},
                "required": [],
            }

        schema = self._args_schema.model_json_schema()
        # Ensure required fields are present
        if "required" not in schema:
            schema["required"] = []
        return schema

    def execute(self, arguments: Dict[str, Any]) -> Union[ToolResult, Awaitable[ToolResult]]:
        """
        Execute the tool with validated arguments.

        Args:
            arguments: Dict containing tool arguments

        Returns:
            ToolResult or awaitable ToolResult
        """
        try:
            # Validate arguments using Pydantic model
            if self._args_schema:
                validated_args = self._args_schema(**arguments)
                # Convert Pydantic model to dict
                args_dict = validated_args.model_dump()
            else:
                args_dict = arguments

            # Call the function
            if self._is_async:
                # Return coroutine that will be awaited
                async def async_execute():
                    result = await self._func(**args_dict)
                    return self._wrap_result(result)

                return async_execute()
            else:
                result = self._func(**args_dict)
                return self._wrap_result(result)

        except Exception as e:
            return ToolResult(success=False, content="", error=f"Tool execution failed: {str(e)}")

    def _wrap_result(self, result: Any) -> ToolResult:
        """Wrap function result in ToolResult."""
        if isinstance(result, ToolResult):
            return result
        elif isinstance(result, dict):
            if "success" in result and "content" in result:
                return ToolResult(
                    success=result.get("success", True),
                    content=result.get("content", ""),
                    error=result.get("error"),
                )
            else:
                return ToolResult(
                    success=True,
                    content=json.dumps(result),
                    error=None,
                )
        elif isinstance(result, str):
            return ToolResult(success=True, content=result, error=None)
        else:
            # Try to serialize to JSON
            try:
                return ToolResult(
                    success=True,
                    content=json.dumps(result),
                    error=None,
                )
            except (TypeError, ValueError):
                return ToolResult(
                    success=True,
                    content=str(result),
                    error=None,
                )


@overload
def tool(func: Callable[..., Any]) -> FunctionTool: ...


@overload
def tool(
    func: None = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
) -> Union[FunctionTool, Callable[[Callable[..., Any]], FunctionTool]]:
    """
    Decorator to convert a function into a Tool.

    Can be used with or without parentheses:

    @tool
    def my_function(text: str) -> str:
        \"\"\"Does something.\"\"\"
        return text.upper()

    @tool(name="custom_name", description="Custom description")
    def my_function(text: str) -> str:
        return text.upper()

    @tool(args_schema=MyPydanticModel)
    def my_function(args: MyPydanticModel) -> str:
        return args.text.upper()

    Args:
        func: The function to decorate (when used without parentheses)
        name: Optional tool name
        description: Optional tool description
        args_schema: Optional Pydantic model for argument validation

    Returns:
        FunctionTool instance or decorator function
    """

    def decorator(f: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(
            func=f,
            name=name,
            description=description,
            args_schema=args_schema,
        )

    if func is None:
        # Called with parentheses: @tool(...)
        return decorator
    # Called without parentheses: @tool
    return decorator(func)


def create_calculator_tool():
    """Create the built-in calculator tool."""
    from caliche._runtime import load_runtime

    return load_runtime().create_calculator_tool()


def create_fs_tool(mutation_style=None):
    """Create the built-in filesystem tool (optional MutationStyle)."""
    from caliche._runtime import load_runtime

    runtime = load_runtime()
    if mutation_style is None:
        return runtime.create_fs_tool()
    return runtime.create_fs_tool(mutation_style)


def create_apply_patch_tool():
    """Create the Codex-style apply_patch tool."""
    from caliche._runtime import load_runtime

    return load_runtime().create_apply_patch_tool()


def create_http_tool():
    """Create the built-in HTTP tool."""
    from caliche._runtime import load_runtime

    return load_runtime().create_http_tool()


def create_code_execution_tool():
    """Create the built-in Python code execution tool."""
    from caliche._runtime import load_runtime

    return load_runtime().create_code_execution_tool()


__all__ = [
    "Tool",
    "FunctionTool",
    "tool",
    "ReverseStringTool",
    "WordCountTool",
    "RandomNumberTool",
    "create_calculator_tool",
    "create_fs_tool",
    "create_apply_patch_tool",
    "create_http_tool",
    "create_code_execution_tool",
]
