"""
Utilities for converting project tool definitions into OpenAI chat-tool format.
"""

from copy import deepcopy
from typing import Any, Dict, List

from common.models import TOOLS_DESCRIPTIONS


TOOL_PARAMETER_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "add_breakpoint": {
        "breakpoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "File path where breakpoint should be set"},
                    "lines": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Line numbers to break on"
                    }
                },
                "required": ["file", "lines"]
            },
            "minItems": 1
        }
    },
    "remove_breakpoint": {
        "breakpoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "File path whose breakpoints should be removed"},
                    "lines": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Line numbers to clear"
                    }
                },
                "required": ["file", "lines"]
            },
            "minItems": 1
        }
    },
    "list_code": {
        "start": {"type": "integer"},
        "end": {"type": "integer"}
    },
    "print_variable": {
        "name": {"type": "string"}
    },
    "run_expression": {
        "expression": {"type": "string"}
    },
    "bash": {
        "command": {"type": "string"}
    },
    "start_debugger": {
        "file_path": {"type": "string"},
        "test": {"type": "string"}
    }
}

TOOL_REQUIRED_PARAMS: Dict[str, List[str]] = {
    "bash": ["command"],
    "add_breakpoint": ["breakpoints"],
    "remove_breakpoint": ["breakpoints"],
    "print_variable": ["name"],
    "run_expression": ["expression"]
}


def get_openai_tools() -> List[Dict[str, Any]]:
    """
    Convert TOOLS_DESCRIPTIONS to the OpenAI chat tool schema used in training.

    Returns:
        List of tool definitions suitable for tokenizer.apply_chat_template(..., tools=...)
    """
    tools = []
    for tool_name, tool_desc in TOOLS_DESCRIPTIONS.items():
        required_params = TOOL_REQUIRED_PARAMS.get(tool_name, [])
        tool_schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_desc['description'],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": list(required_params)
                }
            }
        }

        if tool_desc.get("parameters"):
            for param_name, param_desc in tool_desc["parameters"].items():
                schema_override = TOOL_PARAMETER_SCHEMAS.get(tool_name, {}).get(param_name)
                if schema_override:
                    parameter_schema = deepcopy(schema_override)
                    parameter_schema.setdefault("description", param_desc)
                else:
                    parameter_schema = {
                        "type": "string",
                        "description": param_desc
                    }

                tool_schema["function"]["parameters"]["properties"][param_name] = parameter_schema

        if not tool_schema["function"]["parameters"]["required"]:
            tool_schema["function"]["parameters"].pop("required")

        tools.append(tool_schema)

    return tools
