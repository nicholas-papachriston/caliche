"""
Example demonstrating native structured output with schema enforcement

Run with:
```bash
uv run examples/structured_output.py
```
"""

import asyncio
import os
import json
import re
from typing import Optional

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("Warning: Pydantic not available. Install with: pip install pydantic")
    print("   Examples will show basic usage without Pydantic models\n")

from caliche import AgentBuilder, ProviderType
from caliche.structured import chat_structured, JsonSchemaDefinition


class Person(BaseModel):
    """A person's information"""
    name: str = Field(description="Full name")
    age: int = Field(ge=0, le=150, description="Age in years")
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$", description="Email address")


async def main():
    print("=== Native Structured Output Examples ===\n")

    # Check if we have API credentials
    has_api_key = (
        os.getenv("OPENAI_API_KEY") or 
        os.getenv("ANTHROPIC_API_KEY") or
        os.getenv("AGENT_PROVIDER")
    )

    # Example 1: Using json_mode() for any valid JSON
    print("1. JSON Mode (any valid JSON):")
    try:
        agent1 = (
            AgentBuilder()
            .provider(ProviderType.OpenAI)
            .model("gpt-4o")
            .json_mode()
            .build()
        )

        response, usage, _ = agent1.chat("Return a JSON object with name and age")
        print(f"   Response: {response}")
        
        # Extract JSON from markdown code blocks if present
        import json
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        
        try:
            parsed = json.loads(response)
            print(f"   Parsed: {parsed}")
        except json.JSONDecodeError as e:
            print(f"   Warning: Failed to parse JSON: {e}")
    except Exception as e:
        print(f"   Error: {e}")
        if not has_api_key:
            print("   (Set OPENAI_API_KEY to test this feature)")

    print()

    # Example 2: Using json_schema() for schema-enforced output
    print("2. JSON Schema Mode (schema-enforced):")
    try:
        person_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "age", "email"]
        }

        agent2 = (
            AgentBuilder()
            .provider(ProviderType.OpenAI)
            .model("gpt-4o")
            .json_schema("Person", person_schema)
            .build()
        )

        response, usage, _ = agent2.chat(
            "Extract person info: John Doe is 30 years old, email john@example.com"
        )
        
        print(f"   Response: {response}")
        
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        
        try:
            person = json.loads(response)
            print(f"   Parsed Person: {person}")
            print(f"   Name: {person['name']}, Age: {person['age']}, Email: {person['email']}")
        except json.JSONDecodeError as e:
            print(f"   Warning: Failed to parse JSON: {e}")
            print("   (The model may not have followed the JSON schema format)")
    except Exception as e:
        print(f"   Error: {e}")
        if not has_api_key:
            print("   (Set OPENAI_API_KEY to test this feature)")

    print()

    # Example 3: Using chat_structured() with Pydantic (if available)
    if PYDANTIC_AVAILABLE:
        print("3. Using chat_structured() with Pydantic models:")
        try:
            agent3 = (
                AgentBuilder()
                .provider(ProviderType.OpenAI)
                .model("gpt-4o")
                .build()
            )

            person = await chat_structured(
                agent3,
                "Extract person info: Jane Smith is 25 years old, email jane@example.com",
                Person
            )
            
            print(f"   Parsed Person: {person.name} is {person.age} years old")
            print(f"   Email: {person.email}")
            print(f"   Type-safe access: person.name = '{person.name}'")
        except Exception as e:
            print(f"   Error: {e}")
            if not has_api_key:
                print("   (Set OPENAI_API_KEY to test this feature)")
    else:
        print("3. Using chat_structured() with Pydantic models:")
        print("   (Skipped - Pydantic not available)")
        print("   Install with: pip install pydantic")

    print("\nExamples completed!")
    print("\nNote: The agent will automatically retry if the response")
    print("      does not match the schema (up to 2 retries by default)")


if __name__ == "__main__":
    asyncio.run(main())
