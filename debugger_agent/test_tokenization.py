"""Tests for exact prompt-token counting in the debug CLI."""

import json
from pathlib import Path

import pytest

from common.context_utils import truncate_messages
from debugger_agent.agent import _INITIAL_CHARS_PER_TOKEN, _WireSizeEstimator
from debugger_agent.tokenization import (
    _CHAT_TEMPLATE_PATH,
    _resolve_tokenizer_file,
    load_tokenizer,
)


def _snapshot_dir():
    """A cached HF snapshot with tokenizer_config.json, for the transformers gate."""
    matches = sorted(
        Path.home().glob(
            ".cache/huggingface/hub/models--moofeez--qwen3-coder-30b-a3b-debugger/snapshots/*"
        )
    )
    return matches[0] if matches else None


requires_tokenizer = pytest.mark.skipif(
    _resolve_tokenizer_file(None) is None,
    reason="no tokenizer.json in the local HF cache",
)


@pytest.fixture
def tokenizer():
    tok = load_tokenizer()
    if tok is None:
        pytest.skip("no exact tokenizer available in this checkout")
    return tok


def _messages():
    """A payload shaped like a real debug turn: tool_calls, a tool result, angle
    brackets and quoted paths in the arguments (the `tojson` escaping trap)."""
    return [
        {"role": "system", "content": "You are a debugger. Prefer a < b over a>b."},
        {"role": "user", "content": "tests/test_reconciliation.py::test_preview fails."},
        {
            "role": "assistant",
            "content": "Reproduce the failure.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": "pytest 'tests/a.py' 2>&1 | head -5 && [ $? -lt 2 ]"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "E  assert 4 == 3\n  where 4 = len([Entry(...)])",
        },
    ]


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a command. Use <cmd> when a < b and b > c.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]


@requires_tokenizer
def test_tokenizer_loads_in_this_checkout():
    """Regression guard for silent degradation to estimation.

    Every failure inside `load_tokenizer` is swallowed by design, so a broken
    import (e.g. `jinja2.ext` not being bound by a bare `import jinja2`) costs
    a third of the context window without raising anything. If the cache holds
    a tokenizer, the repo venv must actually produce one.
    """
    tok = load_tokenizer()
    assert tok is not None
    assert tok.source.endswith("tokenizer.json")


@requires_tokenizer
def test_render_and_token_ids_match_transformers(tokenizer):
    """The count is only meaningful if it equals what the server computes.

    vLLM renders with transformers' `apply_chat_template` over the same
    template file, so transformers is the oracle here — not an alternative
    implementation to agree with roughly.
    """
    transformers = pytest.importorskip("transformers")
    snapshot = _snapshot_dir()
    if snapshot is None:
        pytest.skip("no cached snapshot with tokenizer_config.json")

    messages, tools = _messages(), _tools()
    hf = transformers.AutoTokenizer.from_pretrained(str(snapshot))
    expected = hf.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        chat_template=_CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"),
    )

    rendered = tokenizer.apply_chat_template(messages, tools=tools)
    assert rendered == expected
    assert list(tokenizer.encode(rendered)) == list(
        hf.encode(expected, add_special_tokens=False)
    )


@requires_tokenizer
def test_tojson_does_not_html_escape(tokenizer):
    """Jinja's built-in `tojson` escapes `< > & '`; transformers overrides it.

    Inheriting the built-in would inflate every tool schema and every tool_call
    argument blob — `\\u003c` is six characters where `<` is one — so the count
    would run high exactly where this prompt is heaviest.
    """
    rendered = tokenizer.apply_chat_template(_messages(), tools=_tools())
    assert "\\u003c" not in rendered
    assert "\\u0026" not in rendered
    assert "a < b" in rendered


@requires_tokenizer
def test_counts_below_the_char_estimator(tokenizer):
    """The point of the exercise: recover window the estimator gives away.

    `_INITIAL_CHARS_PER_TOKEN` is deliberately pessimistic, which overstates
    the prompt and drops history earlier than necessary.
    """
    # The SHIPPED tool set, not a toy one. The estimator serializes messages +
    # schemas as JSON while the exact count also carries the template's fixed
    # preamble, so on a tiny payload the estimator can come out lower; the
    # claim only holds at the scale the CLI actually runs at (20 schemas,
    # ~2k tokens — a third of the window).
    from common.tools import get_openai_tools

    messages, tools = _messages(), get_openai_tools()
    exact = len(tokenizer.encode(tokenizer.apply_chat_template(messages, tools=tools)))

    estimator = _WireSizeEstimator(_INITIAL_CHARS_PER_TOKEN, tools)
    estimated = len(
        estimator.encode(estimator.apply_chat_template(messages, tools=tools))
    )
    assert exact < estimated


@requires_tokenizer
def test_satisfies_the_truncate_messages_contract(tokenizer):
    """`truncate_messages` falls back to a chars//6 heuristic on any tokenizer
    exception, so a contract break degrades silently rather than raising."""
    # Roles must alternate: truncate_messages preserves the whole trailing run
    # of user/tool messages, so an all-user history is pinned end to end and
    # nothing can be dropped.
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "the original task description"},
    ]
    for i in range(40):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"filler observation {i} " + "x" * 400})

    result = truncate_messages(messages, max_tokens=1200, tokenizer=tokenizer, tools=_tools())

    assert len(result) < len(messages)
    assert result[0] == messages[0]
    assert result[-1] == messages[-1]
    fitted = len(
        tokenizer.encode(tokenizer.apply_chat_template(result, tools=_tools()))
    )
    assert fitted <= 1200


@pytest.mark.parametrize(
    "spec",
    ["/nonexistent/path/tokenizer.json", "not-a-real-org/not-a-real-model"],
)
def test_unusable_spec_falls_back_to_none(spec, monkeypatch):
    """A bad spec must degrade to estimation, never fail the debug session."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert load_tokenizer(spec) is None


@requires_tokenizer
def test_explicit_file_spec_is_honored():
    cached = _resolve_tokenizer_file(None)
    tok = load_tokenizer(cached)
    assert tok is not None and tok.source == cached
