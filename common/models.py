"""Tool-call models and tool definitions shared by the harness and agent."""

import re
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ToolCall:
    """A tool invocation with its reasoning and result."""
    name: str
    parameters: Dict[str, Any]
    reasoning: str
    output: str = ""

    @property
    def raw_string(self) -> str:
        """Reconstruct the raw tool call string from name and parameters (Qwen format)."""
        tool_call_json = json.dumps({"name": self.name, "arguments": self.parameters})
        return f'<tool_call>\n{tool_call_json}\n</tool_call>'


@dataclass
class DebuggingStep:
    """A model response and the tool calls it produced."""
    step_id: int
    tool_calls: List[ToolCall]
    response: str


TOOLS_DESCRIPTIONS = {
    # These descriptions are the single home of the tool usage rules (the
    # system prompt has no tool prose; see common/prompts.py). Keep them
    # short: they render into the <tools> block of every step's prompt.
    'bash': {
        'description': 'Run a bash command. Output is capped — pipe large output through `head -n 50` or `grep -m 50`. Read files with `sed -n \'1,150p\' file` (smaller windows for large files); check size with `wc -l`. To find a function/method, locate it FIRST with `grep -n "def <name>" file.py`, then read exactly that range — do not guess line numbers or rescan growing windows. Use edit_file/write_file for writing files, not heredoc.',
        'parameters': {
            'command': 'Bash command to run'
        }
    },
    'start_debugger': {
        'description': 'Start pdb session; if one is already running, it is restarted from the beginning (breakpoints re-applied). IMPORTANT: always debug the TEST file (it calls the library, so breakpoints in library code will be hit). Pass the failing test as "test" — otherwise every test in the file runs in order and a breakpoint stops in the FIRST one that reaches it, which is usually a PASSING test.',
        'parameters': {
            'file_path': 'Path to the TEST file to debug',
            'test': 'Name of the single failing test to run, e.g. "test_settle_levels_a_lopsided_rota". Omit to run the whole file.'
        }
    },
    'add_breakpoint': {
        'description': 'Set one or MORE breakpoints in a single call (pass several lines/files at once instead of one call per breakpoint); optional per-breakpoint condition.',
        'parameters': {
            'breakpoints': 'Objects with "file", "lines" (list of line numbers), optional "condition" (Python expression)'
        }
    },
    'remove_breakpoint': {
        'description': 'Remove breakpoints',
        'parameters': {
            'breakpoints': 'Objects with "file" and "lines"'
        }
    },
    'clear_breakpoints': {
        'description': 'Clear all breakpoints',
        'parameters': {}
    },
    'continue_execution': {
        'description': 'Continue to next breakpoint, if it exists, else continue until program finishes (shows a small source window around the stop position)',
        'parameters': {}
    },
    'step_over': {
        'description': 'Step over current line (shows a small source window around the new position)',
        'parameters': {}
    },
    'step_into': {
        'description': 'Step into the FIRST function called on the current line — check the reported function name; if it is not the one you wanted, prefer add_breakpoint inside the target function + continue_execution',
        'parameters': {}
    },
    'step_out': {
        'description': 'Step out of the current function, returning to the caller',
        'parameters': {}
    },
    'print_variable': {
        'description': 'Print variable value',
        'parameters': {
            'name': 'Name of the variable to print'
        }
    },
    'examine_locals': {
        'description': 'Show local variables',
        'parameters': {}
    },
    'examine_globals': {
        'description': 'Show global variables',
        'parameters': {}
    },
    'list_code': {
        'description': 'Show code lines, labeled with the file they came from. Pass an optional "file" to read another file fresh from disk (also refreshes after edits).',
        'parameters': {
            'start': 'Starting line number',
            'end': 'Ending line number'
        }
    },
    'where_am_i': {
        'description': 'Show current location in debugger session',
        'parameters': {}
    },
    'inspect_stack': {
        'description': 'Show the call stack with local variables at each frame.',
        'parameters': {
            'max_frames': 'Max frames to inspect (default: all)'
        }
    },
    'run_expression': {
        'description': 'Evaluate a Python expression in the current debug context (NOT shell commands — run pytest etc. via bash)',
        'parameters': {
            'expression': 'Python expression to evaluate'
        }
    },
    # There is no read_file tool. Files are read through bash
    # (`sed -n 'A,Bp' file`, `grep -n`).
    'edit_file': {
        'description': 'Replace text in a file. Copy old_string VERBATIM from prior tool output (never from memory), including enough surrounding lines that it appears exactly once in the file.',
        'parameters': {
            'file_path': 'Path to the file to edit',
            'old_string': 'Exact text to replace — must appear exactly once in the file; include neighboring lines to make it unique',
            'new_string': 'Replacement text'
        }
    },
    'write_file': {
        'description': 'Write content to a file (create or overwrite)',
        'parameters': {
            'file_path': 'Path to the file to write',
            'content': 'Full file contents'
        }
    },
    'revert_file': {
        'description': 'Restore a file to the state it had when this session started, discarding your edits to it. Use to undo a change that made things worse.',
        'parameters': {
            'file_path': 'Path to the file to revert'
        }
    },
    'done': {
        'description': 'Finish the task. Only call after making at least one edit AND re-running the failing tests to verify they now pass; if the tests already pass before you change anything, call done immediately.',
        'parameters': {}
    }
}

TOOL_NAMES = list(TOOLS_DESCRIPTIONS.keys())

NON_DEBUGGING_TOOL_NAMES = [
    'bash', 'edit_file', 'write_file', 'revert_file', 'done'
]

NON_DEBUGGING_TOOLS = {name: TOOLS_DESCRIPTIONS[name] for name in NON_DEBUGGING_TOOL_NAMES}
DEBUGGING_TOOLS = {name: TOOLS_DESCRIPTIONS[name] for name in TOOL_NAMES if name not in NON_DEBUGGING_TOOL_NAMES}

# Shared tool groups for dispatch and observation hints.
EDIT_TOOL_NAMES = frozenset({"edit_file", "write_file"})

# Read-only debugger inspections that surface RUNTIME STATE (variable values,
# expression results, frame locals).
STATE_INSPECTION_TOOL_NAMES = frozenset(
    {
        "print_variable",
        "examine_locals",
        "examine_globals",
        "run_expression",
        "inspect_stack",
    }
)

# All read-only inspections inside a debugger session: runtime state plus
# code/position reads (list_code / where_am_i).
DEBUGGER_INSPECTION_TOOL_NAMES = STATE_INSPECTION_TOOL_NAMES | frozenset(
    {"list_code", "where_am_i"}
)

# Ops that advance program execution under the debugger.
DEBUGGER_EXECUTION_TOOL_NAMES = frozenset(
    {"continue_execution", "step_over", "step_into", "step_out"}
)

# Session scaffolding ops (structural, not reads).
DEBUGGER_STRUCTURAL_TOOL_NAMES = frozenset(
    {"start_debugger", "add_breakpoint", "remove_breakpoint", "clear_breakpoints"}
)

# Invariant: the three debugger subsets exactly partition DEBUGGING_TOOLS.
assert (
    DEBUGGER_INSPECTION_TOOL_NAMES
    | DEBUGGER_EXECUTION_TOOL_NAMES
    | DEBUGGER_STRUCTURAL_TOOL_NAMES
) == frozenset(DEBUGGING_TOOLS), "debugger tool subsets drifted from DEBUGGING_TOOLS"

def create_debugging_tools(use_debugging_tools: bool = True) -> List[Dict]:
    """Build chat-template tool schemas, optionally excluding debugger tools."""
    tool_descriptions = TOOLS_DESCRIPTIONS if use_debugging_tools else NON_DEBUGGING_TOOLS
    tools = []
    for tool_name, tool_info in tool_descriptions.items():
        tool_schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_info["description"],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

        if tool_info.get("parameters"):
            for param_name, param_desc in tool_info["parameters"].items():
                tool_schema["function"]["parameters"]["properties"][param_name] = {
                    "type": "string",
                    "description": param_desc
                }
                tool_schema["function"]["parameters"]["required"].append(param_name)

        tools.append(tool_schema)

    return tools


def _fix_json_control_chars(s: str) -> str:
    """Escape literal newlines/tabs inside JSON string values (common LLM output issue)."""
    out = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
                out.append(ch)
            elif ch == '\\':
                esc = True
                out.append(ch)
            elif ch == '"':
                in_str = False
                out.append(ch)
            elif ch == '\n':
                out.append('\\n')
            elif ch == '\r':
                out.append('\\r')
            elif ch == '\t':
                out.append('\\t')
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return ''.join(out)


def parse_xml_tool_calls(content: str) -> List[Dict]:
    """Parse JSON inside <tool_call> blocks into OpenAI tool-call dictionaries."""
    tool_calls = []

    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)

    for tool_call_raw in matches:
        call_id = f"call_{uuid.uuid4().hex[:8]}"

        raw = tool_call_raw.strip()

        try:
            fixed = _fix_json_control_chars(raw)
            tool_data = json.loads(fixed)
        except json.JSONDecodeError as exc:
            raise InvalidToolCallError(
                f"Invalid tool call format: {raw[:100]}"
            ) from exc

        function_name = tool_data.get("name")
        if not function_name or function_name == "unknown":
            raise UnknownToolCallError("Missing or invalid tool name in tool call")
        if function_name not in TOOLS_DESCRIPTIONS:
            raise UnknownToolCallError(f"Unknown tool name: {function_name}")

        arguments = tool_data.get("arguments", {})
        if isinstance(arguments, dict):
            arguments_str = json.dumps(arguments)
        elif isinstance(arguments, str):
            arguments_str = arguments
        else:
            arguments_str = "{}"

        tool_calls.append({
            "id": call_id,
            "type": "function",
            "function": {"name": function_name, "arguments": arguments_str}
        })

    return tool_calls

class UnknownToolCallError(Exception):
    """Exception raised when an unknown tool call is encountered."""
    pass

class InvalidToolCallError(Exception):
    """Exception raised when no valid tool call is found in the action."""
    pass
