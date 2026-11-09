"""
Structured output helpers for Caliche.

This module provides utilities for working with structured outputs using
Pydantic models and JSON schemas.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar

try:
    from pydantic import BaseModel

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[assignment,misc]  # Dummy for type hints when pydantic missing

T = TypeVar("T", bound=BaseModel)


@dataclass
class JsonSchemaDefinition:
    """JSON Schema definition for structured output."""

    name: str
    schema: Dict[str, Any]
    description: Optional[str] = None
    strict: bool = True

    @classmethod
    def from_pydantic(cls, model: Type[BaseModel]) -> JsonSchemaDefinition:
        """Create schema from Pydantic model."""
        if not PYDANTIC_AVAILABLE:
            raise ImportError(
                "Pydantic is required for from_pydantic. Install with: pip install pydantic"
            )

        return cls(
            name=model.__name__,
            schema=model.model_json_schema(),
            description=model.__doc__,
            strict=True,
        )


async def chat_structured(
    agent: Any, message: str, response_model: Type[T], max_retries: int = 2
) -> T:
    """
    Chat with structured output, returning a typed Pydantic model.

    Args:
        agent: Agent instance from AgentBuilder
        message: User message to send
        response_model: Pydantic model class for the expected response
        max_retries: Maximum number of retries on validation failure

    Returns:
        Parsed response as the specified Pydantic model

    Example:
        ```python
        from pydantic import BaseModel
        from caliche import AgentBuilder, ProviderType
        from caliche.structured import chat_structured

        class Person(BaseModel):
            name: str
            age: int

        agent = (
            AgentBuilder()
            .provider(ProviderType.OpenAI)
            .model("gpt-4o")
            .build()
        )

        person = await chat_structured(
            agent,
            "Extract: John is 30 years old",
            Person
        )
        print(person.name)  # Typed!
        ```
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "Pydantic is required for chat_structured. Install with: pip install pydantic"
        )

    import asyncio

    schema_def = JsonSchemaDefinition.from_pydantic(response_model)

    # Call the Rust method (which blocks, so run in thread pool)
    response_json, usage = await asyncio.to_thread(
        agent.chat_structured, message, schema_def.schema
    )

    # Parse and validate with Pydantic
    return response_model.model_validate_json(response_json)


def structured_output(model: Type[BaseModel]) -> Callable[..., Any]:
    """
    Decorator to enforce structured output on agent methods.

    Example:
        ```python
        from pydantic import BaseModel
        from caliche.structured import structured_output

        class Person(BaseModel):
            name: str
            age: int

        @structured_output(Person)
        async def extract_person(agent, text: str) -> Person:
            return await agent.chat(f"Extract person from: {text}")
        ```
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "Pydantic is required for structured_output decorator. Install with: pip install pydantic"
        )

    from functools import wraps

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(agent: Any, *args: Any, **kwargs: Any) -> Any:
            # Temporarily set schema on agent
            original_format = getattr(agent, "_response_format", None)
            agent.json_schema(model.__name__, model.model_json_schema())

            try:
                response = await func(agent, *args, **kwargs)
                if isinstance(response, str):
                    # If function returns a string (like chat), parse it
                    return model.model_validate_json(response)
                return response
            finally:
                # Restore original format
                if original_format is not None:
                    agent._response_format = original_format
                elif hasattr(agent, "_response_format"):
                    delattr(agent, "_response_format")

        return wrapper

    return decorator
