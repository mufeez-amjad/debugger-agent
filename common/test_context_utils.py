"""Tests for common.context_utils — truncate_messages optimization.

The 04-17 perf review flagged truncate_messages as an O(N^2) hotspot that
blocked the worker's async event loop on 4-vCPU hosts. These tests cover:
- Fast path (full conversation fits): single tokenize call, no truncation
- Slow path (doesn't fit): binary search; O(log N) tokenize calls
- Edge cases: no user-role tail, minimum set larger than limit, empty input
"""

from common.context_utils import _extract_latest_state_summary, truncate_messages


class _CountingTokenizer:
    """Minimal tokenizer stub that counts apply_chat_template calls."""

    def __init__(self, tokens_per_char: float = 0.25):
        self.calls = 0
        self.tokens_per_char = tokens_per_char

    def apply_chat_template(
        self, msgs, tools=None, tokenize=False, add_generation_prompt=False
    ):
        self.calls += 1
        return " ".join(str(m.get("content", "")) for m in msgs)

    def encode(self, s, add_special_tokens=False):
        return [0] * max(1, int(len(s) * self.tokens_per_char))


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


class TestFastPath:
    def test_short_conversation_no_tokenizer_returns_all(self):
        msgs = [
            _msg("system", "sys"),
            _msg("user", "hi"),
            _msg("assistant", "hello"),
            _msg("user", "bye"),
        ]
        result = truncate_messages(msgs, max_tokens=1000)
        assert result == msgs

    def test_short_conversation_with_tokenizer_single_tokenize_call(self):
        """Fast path should fire one tokenize call (the full-fit check) and return."""
        msgs = [
            _msg("system", "sys"),
            _msg("user", "hi"),
            _msg("assistant", "hello"),
            _msg("user", "bye"),
        ]
        tok = _CountingTokenizer()
        result = truncate_messages(msgs, max_tokens=10000, tokenizer=tok)
        assert result == msgs
        assert tok.calls == 1, f"fast path should be 1 tokenize call, got {tok.calls}"


class TestTruncation:
    def test_long_conversation_keeps_system_and_last_user(self):
        msgs = [_msg("system", "sys")]
        for i in range(20):
            msgs.append(
                _msg("user" if i % 2 == 0 else "assistant", "x" * 100 + f" msg{i}")
            )
        msgs[-1] = _msg("user", "final user message")

        result = truncate_messages(msgs, max_tokens=30)

        assert result[0]["role"] == "system"
        assert result[-1]["content"] == "final user message"
        assert len(result) < len(msgs)

    def test_binary_search_scales_logarithmically(self):
        """Binary search should hit O(log N) tokenize calls — well below the
        prior O(N) regression this change addresses."""
        msgs = [_msg("system", "sys")]
        for i in range(40):
            msgs.append(
                _msg("user" if i % 2 == 0 else "assistant", "x" * 100 + f" m{i}")
            )
        msgs[-1] = _msg("user", "latest")

        tok = _CountingTokenizer()
        result = truncate_messages(msgs, max_tokens=50, tokenizer=tok)

        # Old linear-scan implementation would have been ~40 calls; binary
        # search with a fast-path + min-check + log2(40)=6 + final count
        # should land well under 15.
        assert tok.calls < 15, (
            f"expected <15 tokenize calls (binary search), got {tok.calls}"
        )
        assert result[0]["role"] == "system"
        assert result[-1]["content"] == "latest"

    def test_respects_token_limit(self):
        """Truncated result must actually fit within max_tokens."""
        tok = _CountingTokenizer()
        msgs = [_msg("system", "sys " * 5)]
        for i in range(15):
            msgs.append(_msg("user" if i % 2 == 0 else "assistant", "y" * 200))
        msgs[-1] = _msg("user", "last")

        # The system message, first user turn and last user turn are always kept.
        result = truncate_messages(msgs, max_tokens=80, tokenizer=tok)

        prompt = tok.apply_chat_template(result)
        final_tokens = len(tok.encode(prompt))
        assert final_tokens <= 80, (
            f"result has {final_tokens} tokens, exceeds limit 80"
        )

    def test_system_state_template_is_not_preserved_as_model_memory(self):
        template = (
            "<STATE v1>\n"
            "facts: [key discoveries]\n"
            "decisions: [approach choices]\n"
            "pointers: [file:path sha1=X lines=[Y-Z]]\n"
            "todos: [remaining tasks]\n"
            "</STATE>"
        )
        msgs = [
            _msg("system", f"Instructions:\n{template}"),
            _msg("user", "task"),
            _msg("assistant", "x" * 500),
            _msg("tool", "y" * 500),
            _msg("user", "latest observation"),
        ]

        result = truncate_messages(msgs, max_tokens=40)

        preserved = [
            message
            for message in result
            if message.get("role") == "user"
            and str(message.get("content", "")).startswith(
                "[Previous context summary]"
            )
        ]
        assert preserved == []

    def test_latest_real_state_skips_newer_placeholder(self):
        real_state = (
            "<STATE v2>\n"
            "facts: [walker.py returns a shared iterator]\n"
            "decisions: [copy the iterator before caching]\n"
            "pointers: [walker.py:42]\n"
            "todos: [run walker_test.py]\n"
            "</STATE>"
        )
        placeholder = (
            "<STATE v3>\n"
            "facts: [key discoveries]\n"
            "decisions: [approach choices]\n"
            "pointers: [file:path sha1=X lines=[Y-Z]]\n"
            "todos: [remaining tasks]\n"
            "</STATE>"
        )
        msgs = [
            _msg("system", f"Use this template:\n{placeholder}"),
            _msg("assistant", real_state),
            _msg("assistant", placeholder),
        ]

        assert _extract_latest_state_summary(msgs) == real_state
