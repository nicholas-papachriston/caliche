"""
Caliche Prompt Templates Examples

Demonstrates how to use prompt templates for reusable prompts with variable substitution.

uv run examples/prompts_example.py
"""

from caliche.prompts import PromptTemplate


def example_basic_template():
    """Basic prompt template example."""
    print("=== Basic Prompt Template ===\n")

    template = PromptTemplate("Translate the following text to {language}: {text}")

    # Format with variables
    formatted = template.format({
        "language": "Spanish",
        "text": "Hello, how are you?"
    })

    print(f"Template: {template.template}")
    print(f"Variables: {template.variables()}")
    print(f"Formatted: {formatted}")
    print()


def example_factory_templates():
    """Factory template methods example."""
    print("=== Factory Template Methods ===\n")

    # Summarize template
    summarize_template = PromptTemplate.summarize()
    print(f"Summarize template: {summarize_template.template}")
    print(f"Variables: {summarize_template.variables()}")
    print()

    # Analyze template
    analyze_template = PromptTemplate.analyze()
    print(f"Analyze template: {analyze_template.template}")
    print(f"Variables: {analyze_template.variables()}")
    print()

    # Translate template
    translate_template = PromptTemplate.translate()
    print(f"Translate template: {translate_template.template}")
    print(f"Variables: {translate_template.variables()}")
    print()


def example_complex_template():
    """Complex template with multiple variables example."""
    print("=== Complex Template with Multiple Variables ===\n")

    template = PromptTemplate(
        "You are a {role} helping with {task}.\n\n"
        "Context: {context}\n\n"
        "Instructions: {instructions}\n\n"
        "Please provide a {format} response."
    )

    formatted = template.format({
        "role": "data analyst",
        "task": "analyzing sales data",
        "context": "Q4 sales increased by 15% compared to Q3",
        "instructions": "Identify key trends and provide insights",
        "format": "structured summary"
    })

    print("Formatted prompt:")
    print(formatted)
    print()


def main():
    example_basic_template()
    example_factory_templates()
    example_complex_template()


if __name__ == "__main__":
    main()

