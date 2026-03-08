"""Recover when the server reports a smaller context window."""

from __future__ import annotations

import pytest

from debugger_agent.agent import AgentConfig, DebugAgent, _ContextOverflow

SERVED = 7168
BODY = (
    "This model's maximum context length is 7168 tokens. However, you requested "
    "1024 output tokens and your prompt contains at least 6145 input tokens, for "
    "a total of at least 7169 tokens."
)


def _agent(tmp_path, **kw):
    return DebugAgent(str(tmp_path), "fix it", AgentConfig(**kw))


def test_client_adopts_the_smaller_window_and_retries(tmp_path):
    agent = _agent(tmp_path)
    assert agent._prompt_budget() > SERVED, "precondition: budget exceeds the server"

    calls = []

    def fake_post(messages, tools):
        calls.append(agent._prompt_budget())
        if agent._context_limit > SERVED:
            raise _ContextOverflow(BODY, input_tokens=6145, max_context=SERVED)
        return {"role": "assistant", "content": "ok"}

    agent._post_completion = fake_post
    assert agent._call_model([{"role": "user", "content": "hi"}])["content"] == "ok"
    assert len(calls) == 2, "one rejected attempt, then one that fits"
    assert agent._context_limit == SERVED
    assert agent._prompt_budget() < SERVED


def test_the_adopted_limit_persists_for_later_turns(tmp_path):
    """The bug was per-session, not per-request: every later turn died too."""
    agent = _agent(tmp_path)
    agent._adopt_server_context_limit(SERVED)
    assert agent._context_limit == SERVED
    # A second overflow at the same size must not re-trigger a refit.
    assert agent._adopt_server_context_limit(SERVED) is False


def test_a_larger_reported_window_is_not_adopted(tmp_path):
    """Growing into a window the caller did not ask for is not ours to decide."""
    agent = _agent(tmp_path, context_limit=8192)
    assert agent._adopt_server_context_limit(32768) is False
    assert agent._context_limit == 8192


def test_a_body_without_the_window_still_raises(tmp_path):
    """No number to believe means the existing error paths handle it, and may give up."""
    agent = _agent(tmp_path)

    def fake_post(messages, tools):
        raise _ContextOverflow("overflowed", input_tokens=None, max_context=None)

    agent._post_completion = fake_post
    with pytest.raises(_ContextOverflow):
        agent._call_model([{"role": "user", "content": "hi"}])
