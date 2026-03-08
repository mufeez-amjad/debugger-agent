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
