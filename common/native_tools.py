"""Single source of truth for native tool-calling message construction.

The model's JSON wire format is ``<tool_call>{json}</tool_call>``. Tool schemas
are injected via ``tools=`` and rendered by ``common/chat_template_json.jinja``
(a Qwen2.5/hermes-style template). Assistant turns carry structured
``tool_calls``, and tool results come back as ``role:"tool"`` messages.

Every surface that builds a conversation to render through the chat template,
including the ``debugger_agent`` CLI, MUST construct assistant and tool messages
through the helpers here. Then the rendered prompt is identical to the one the
model saw in training and the one the server renders. Two rules matter for fidelity, both
verified by rendering ``chat_template_json.jinja``:

  1. The assistant ``content`` MUST have the ``<tool_call>…</tool_call>`` block
     stripped out. The template re-emits the call from the structured
     ``tool_calls`` field, so leaving it in ``content`` double-renders it.
  2. ``arguments`` may be a JSON string (as ``parse_xml_tool_calls`` returns) or a
     dict — the template handles both. No dict-coercion is required.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.models import parse_xml_tool_calls

# Canonical JSON-rendering chat template. Wired into every ``apply_chat_template``
# call site (via :func:`load_json_chat_template`) and passed to vLLM as
# ``--chat-template`` so serving and the client render identically.
JSON_CHAT_TEMPLATE_PATH = Path(__file__).with_name("chat_template_json.jinja")

# Matches a full ``<tool_call>…</tool_call>`` block (JSON or ``<function=>`` body).
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def load_json_chat_template() -> str:
    """Return the canonical JSON chat-template source.

    Assign to ``tokenizer.chat_template`` (or pass as ``chat_template=`` to
    ``apply_chat_template``) at every render site so the checkpoint's shipped
    ``<function=>`` template is never used.
    """
    return JSON_CHAT_TEMPLATE_PATH.read_text()


def strip_tool_call_block(content: str) -> str:
    """Remove ``<tool_call>…</tool_call>`` blocks, leaving only reasoning text.

    Rule (1) above: the structured ``tool_calls`` field carries the call, so the
    literal block must not remain in ``content`` or the template renders it twice.
    """
    return _TOOL_CALL_BLOCK_RE.sub("", content).strip()


def build_assistant_message(raw_output: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build a native assistant message from a raw model emission.

    Returns ``(message, tool_calls)`` where ``message`` is the dict to append to
    the conversation (``content`` = reasoning with the tool-call block stripped,
    plus a structured ``tool_calls`` field when a call was emitted) and
    ``tool_calls`` is the parsed OpenAI-shape list (empty when none / unparseable).
    The caller uses ``tool_calls`` to drive execution and to read the id/name for
    the paired :func:`build_tool_message`.
    """
    try:
        tool_calls = parse_xml_tool_calls(raw_output)
    except Exception:  # noqa: BLE001 — malformed call: treat as no structured call
        tool_calls = []

    message: Dict[str, Any] = {"role": "assistant", "content": strip_tool_call_block(raw_output)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, tool_calls


def build_tool_message(
    content: str,
    *,
    tool_call_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a ``role:"tool"`` result message.

    ``content`` is the raw observation WITHOUT a ``<tool_response>`` wrapper — the
    template adds ``<tool_response>…</tool_response>``. ``tool_call_id``/``name`` are
    carried for the OpenAI-compatible serve/eval path; the chat template itself
    only renders ``content``.
    """
    message: Dict[str, Any] = {"role": "tool", "content": content}
    if tool_call_id is not None:
        message["tool_call_id"] = tool_call_id
    if name is not None:
        message["name"] = name
    return message
