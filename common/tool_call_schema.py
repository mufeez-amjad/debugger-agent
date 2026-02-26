"""Build JSON schemas for tool calls and vLLM structural-tag decoding."""

from __future__ import annotations

import json
from typing import Any, Dict

from common.models import TOOLS_DESCRIPTIONS, TOOL_NAMES


def _per_tool_object_schema(tool_name: str, tool_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build the per-tool object schema with `name` as const and exact arguments."""
    params = tool_info.get("parameters") or {}
    arg_props: Dict[str, Any] = {
        param: {"type": "string"} for param in params
    }
    return {
        "type": "object",
        "properties": {
            "name": {"const": tool_name},
            "arguments": {
                "type": "object",
                "properties": arg_props,
                "required": list(params.keys()),
                "additionalProperties": False,
            },
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }


def build_tool_call_schema(strict_per_tool: bool = False) -> Dict[str, Any]:
    """Constrain tool names and require an arguments object.

    strict_per_tool also constrains argument names, but treats all values as
    strings. It defaults to False because some tools accept arrays or numbers."""
    if strict_per_tool:
        return {
            "oneOf": [
                _per_tool_object_schema(name, info)
                for name, info in TOOLS_DESCRIPTIONS.items()
            ]
        }
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": list(TOOL_NAMES)},
            "arguments": {"type": "object"},
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }


def build_structural_tag(strict_per_tool: bool = False) -> Dict[str, Any]:
    """Wrap the tool-call schema in vLLM’s structural_tag format."""
    # Preserve the newlines used in training to avoid forcing an unlikely prefix.
    return {
        "type": "structural_tag",
        "structures": [
            {
                "begin": "<tool_call>\n",
                "schema": build_tool_call_schema(strict_per_tool=strict_per_tool),
                "end": "\n</tool_call>",
            }
        ],
        "triggers": ["<tool_call>"],
    }


def structural_tag_json(strict_per_tool: bool = False) -> str:
    """Serialized spec — vLLM's ``StructuredOutputsParams.structural_tag`` is a string."""
    return json.dumps(build_structural_tag(strict_per_tool=strict_per_tool))
