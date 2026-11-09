#!/usr/bin/env python3
"""
Templates Example

This example demonstrates:
- Template composition and chaining
- Template search and browsing
- Template variables and instantiation
- Template storage (filesystem, S3, R2)
- Prompt template storage

Best practices shown:
- Composing complex prompts from simple templates
- Template organization and discovery
- Parameterized template usage

Run with:
```bash
uv run examples/templates.py
```
"""

import caliche
import json


def template_composition():
    """Example 1: Template Composition"""
    print("=== Example 1: Template Composition ===\n")
    
    try:
        # Basic template chaining
        analyze = caliche.PromptTemplate("Analyze this data: {data}")
        summarize = caliche.PromptTemplate("Summarize the analysis: {text}")
        
        # Chain templates
        composed = analyze.then(summarize)
        prompt = composed.format({
            "data": "Sales figures Q1-Q4: $1M, $1.2M, $1.5M, $1.8M"
        })
        
        print("Chained Template:")
        print(f"{prompt}\n")
        
        # Multi-step composition
        step1 = caliche.PromptTemplate("Extract key points from: {text}")
        step2 = caliche.PromptTemplate("Categorize these points: {text}")
        step3 = caliche.PromptTemplate("Create action items from: {text}")
        
        workflow = step1.then(step2).then(step3)
        prompt = workflow.format({
            "text": "Meeting notes: Discussed Q4 goals, budget approval, team expansion"
        })
        
        print("Multi-Step Composition:")
        print(f"{prompt}\n")
        
        # Template with partial variables
        base = caliche.PromptTemplate("You are a {role}. {instruction}")
        base = base.with_partial("role", "helpful assistant")
        
        prompt1 = base.format({"instruction": "Explain quantum computing"})
        prompt2 = base.format({"instruction": "Write a haiku"})
        
        print("Template Reuse:")
        print(f"1. {prompt1}")
        print(f"2. {prompt2}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def template_search():
    """Example 2: Template Search"""
    print("=== Example 2: Template Search ===\n")
    
    try:
        registry = caliche.TemplateRegistry()
        registry.add_template_dir("templates")
        count = registry.load_templates()
        print(f"Loaded {count} templates\n")
        
        # Search by query
        print("Searching for 'semantic_search'...")
        filters = caliche.TemplateSearchFilters(query="semantic_search")
        results = registry.search(filters)
        if not results:
            print("   No results found (create templates/ directory)")
        else:
            for template in results:
                print(f"  - {template.name} (v{template.version})")
        print()
        
        # Search by category
        print("Searching by category 'customer-service'...")
        filters = caliche.TemplateSearchFilters(category="customer-service")
        results = registry.search(filters)
        if not results:
            print("   No results found")
        else:
            for template in results:
                print(f"  - {template.name} (v{template.version})")
        print()
    except Exception as e:
        print(f"(Skipped: {e})\n")


def template_variables():
    """Example 3: Template Variables"""
    print("=== Example 3: Template Variables ===\n")
    
    try:
        loader = caliche.TemplateLoader()
        
        # Try to load a template (may not exist)
        template = loader.load_template("templates/customer_support.yaml")
        
        print(f"Template: {template.name} (v{template.version})")
        print(f"Description: {template.description}")
        
        if template.variables:
            print("\nTemplate Variables:")
            for name, definition in template.variables.items():
                print(f"  - {name}: {definition}")
        
        values = {
            "company_name": "Acme Corp",
            "tone": "friendly"
        }
        
        builder = template.instantiate_with_variables(values)
        agent = builder.build()
        
        print("\nAgent created:")
        print(f"  Provider: {agent.provider}")
        print(f"  Model: {agent.model}\n")
    except Exception as e:
        print(f"(Skipped: Create templates/customer_support.yaml to see this example)\n")


def template_storage():
    """Example 4: Template Storage"""
    print("=== Example 4: Template Storage ===\n")
    
    try:
        # Filesystem storage
        print("1. Filesystem Storage:")
        fs_config = caliche.StorageConfig.filesystem("./template_storage")
        fs_config = fs_config.with_cache(True)
        fs_config = fs_config.with_cache_ttl(3600)
        
        registry = caliche.TemplateRegistry.with_storage(fs_config)
        registry.load_templates()
        print("   Loaded templates from filesystem\n")
        
        # AWS S3 storage (if available)
        print("2. AWS S3 Storage:")
        print("   (Requires S3 feature and credentials)")
        print("   s3_config = StorageConfig.aws_s3('bucket', 'region')")
        print("   registry = TemplateRegistry.with_storage(s3_config)\n")
        
        # Cloudflare R2 storage (if available)
        print("3. Cloudflare R2 Storage:")
        print("   (Requires R2 feature and credentials)")
        print("   r2_config = StorageConfig.cloudflare_r2(...)")
        print("   registry = TemplateRegistry.with_storage(r2_config)\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def prompt_storage():
    """Example 5: Prompt Template Storage"""
    print("=== Example 5: Prompt Template Storage ===\n")
    
    try:
        # Filesystem storage
        print("1. Filesystem Storage:")
        fs_config = caliche.StorageConfig.filesystem("./prompt_storage")
        fs_config = fs_config.with_cache(True)
        
        store = caliche.PromptTemplateStore(fs_config)
        
        # Create and save a template
        template = caliche.PromptTemplate("Summarize this text: {text}")
        template = template.with_partial("style", "concise")
        
        store.save("summarize", template)
        print("   Saved 'summarize' template")
        
        # Load the template
        loaded = store.load("summarize")
        formatted = loaded.format({"text": "Long article here..."})
        print(f"   Loaded template and formatted: {formatted}")
        print(f"   Variables: {loaded.variables()}\n")
        
        # List all templates
        templates = store.list()
        print(f"   Available templates: {templates}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def main():
    """Main execution"""
    print("=== Templates Examples ===\n")
    
    template_composition()
    template_search()
    template_variables()
    template_storage()
    prompt_storage()
    
    print("=== Examples Complete ===")
    print("\nTips:")
    print("- Use template composition to build complex prompts")
    print("- Search templates by query, category, or tags")
    print("- Use template variables for parameterization")
    print("- Store templates in filesystem, S3, or R2")
    print("- Use prompt template storage for reusable prompts")


if __name__ == "__main__":
    main()

