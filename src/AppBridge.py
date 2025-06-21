#!/usr/bin/env python3
# scripts/AppBridge.py
import sys
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class ModuleInstance:
    """Represents a module instance with metadata and configuration."""
    id: str
    name: str
    content_type: str = "python"
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


@dataclass(frozen=True)
class ContentModule:
    """Represents a content module with metadata and wrapped content.
    'content' is non-python source code and multi-media; the knowledge base."""
    original_path: Path
    module_name: str
    content: str
    is_python: bool

    def generate_module_content(self) -> str:
        """Generate the Python module content with self-invoking functionality."""
        if self.is_python:
            return self.content
            
        # Use proper immediate execution pattern
        return f'''"""
Original file: {self.original_path}
Auto-generated content module
"""

ORIGINAL_PATH = "{self.original_path}"
CONTENT = """{self.content}"""

def get_content() -> str:
    """Returns the original content."""
    return CONTENT

def get_metadata() -> dict:
    """Metadata for the original file."""
    return {{
        "original_path": ORIGINAL_PATH,
        "is_python": False,
        "module_name": "{self.module_name}"
    }}

def display_content():
    """Display the content immediately."""
    print(CONTENT)
    return True

# Execute immediately when module is imported
_result = display_content()
'''


def create_module(module_name: str, module_code: str, main_module_path: str) -> Optional[ModuleType]:
    """
    Dynamically creates a module with the specified name, injects code into it,
    and adds it to sys.modules.

    Args:
        module_name: Name of the module to create.
        module_code: Source code to inject into the module.
        main_module_path: File path of the main module.

    Returns:
        The dynamically created module, or None if an error occurs.
    """
    dynamic_module = ModuleType(module_name)
    dynamic_module.__file__ = main_module_path or "runtime_generated"
    dynamic_module.__package__ = module_name.rpartition('.')[0] or None
    dynamic_module.__path__ = None
    dynamic_module.__doc__ = f"Dynamically generated module: {module_name}"

    try:
        exec(module_code, dynamic_module.__dict__)
        sys.modules[module_name] = dynamic_module
        return dynamic_module
    except Exception as e:
        print(f"Error injecting code into module {module_name}: {e}", file=sys.stderr)
        return None


def validate_instance(instance: Dict[str, Any]) -> bool:
    """Validate instance JSON schema."""
    required_keys = ["id", "name"]
    return (
        isinstance(instance, dict) and
        all(key in instance for key in required_keys) and
        all(isinstance(instance[key], str) for key in required_keys)
    )


def create_instance_module(instance: ModuleInstance) -> Optional[ModuleType]:
    """Create a dynamic module from a ModuleInstance."""
    module_name = f"morphological.instance_{instance.id}"
    
    # Generate module code based on instance
    if instance.content_type == "python":
        module_code = textwrap.dedent(f'''
            """Instance module for {instance.name} (ID: {instance.id})"""
            
            INSTANCE_ID = "{instance.id}"
            INSTANCE_NAME = "{instance.name}"
            METADATA = {instance.metadata}
            
            def greet():
                print(f"Hello from instance {{INSTANCE_ID}} ({{INSTANCE_NAME}}) in module: {module_name}")
                return True
            
            def get_info():
                return {{
                    "id": INSTANCE_ID,
                    "name": INSTANCE_NAME,
                    "module": "{module_name}",
                    "metadata": METADATA
                }}
        ''')
    else:
        # Handle non-Python content
        content_module = ContentModule(
            original_path=Path(f"instance_{instance.id}"),
            module_name=module_name,
            content=f"Instance: {instance.name}",
            is_python=False
        )
        module_code = content_module.generate_module_content()
    
    main_module_path = getattr(sys.modules.get('__main__'), '__file__', 'runtime_generated')
    return create_module(module_name, module_code, main_module_path)


def process_json_input(json_input: str) -> int:
    """Process JSON input and create module instance."""
    try:
        data = json.loads(json_input)
        if not validate_instance(data):
            print("Error: Invalid instance schema - requires 'id' and 'name' fields", file=sys.stderr)
            return 1

        instance = ModuleInstance(
            id=data["id"],
            name=data["name"],
            content_type=data.get("content_type", "python"),
            metadata=data.get("metadata", {})
        )

        dynamic_module = create_instance_module(instance)
        if not dynamic_module:
            print(f"Error: Failed to create module for instance {instance.id}", file=sys.stderr)
            return 1

        # Execute the module's main function
        if hasattr(dynamic_module, 'greet'):
            dynamic_module.greet()
        
        return 0
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """
    Main entry point. Can process JSON from stdin or from command line arguments.
    Returns an exit code (0 for success, 1 for failure).
    """
    try:
        # Check if we have command line arguments
        if len(sys.argv) > 1:
            # Use command line argument as JSON input
            json_input = sys.argv[1]
        else:
            # Check if stdin has data available (non-blocking)
            if sys.stdin.isatty():
                # No piped input, provide example
                print("Usage: python AppBridge.py '<json>' or echo '<json>' | python AppBridge.py")
                print("Example JSON: {'id': 'test123', 'name': 'Test Instance'}")
                return 1
            else:
                # Read from stdin
                json_input = sys.stdin.read().strip()
                if not json_input:
                    print("Error: No input provided", file=sys.stderr)
                    return 1

        return process_json_input(json_input)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())