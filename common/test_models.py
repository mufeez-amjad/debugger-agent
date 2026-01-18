"""Tests for common.models — tool call parsing and data structures."""

import json
import pytest

from common.models import (
    ToolCall,
    parse_xml_tool_calls,
    UnknownToolCallError,
    InvalidToolCallError,
    TOOL_NAMES,
    create_debugging_tools,
)


class TestToolCall:
    def test_raw_string_roundtrip(self):
        tc = ToolCall(name="bash", parameters={"command": "ls"}, reasoning="list files")
        raw = tc.raw_string
        assert "<tool_call>" in raw
        assert "</tool_call>" in raw
        parsed = json.loads(raw.split("<tool_call>")[1].split("</tool_call>")[0].strip())
        assert parsed["name"] == "bash"
        assert parsed["arguments"] == {"command": "ls"}

    def test_raw_string_empty_params(self):
        tc = ToolCall(name="done", parameters={}, reasoning="finished")
        raw = tc.raw_string
        parsed = json.loads(raw.split("<tool_call>")[1].split("</tool_call>")[0].strip())
        assert parsed["arguments"] == {}


class TestParseXmlToolCalls:
    def test_valid_single_tool_call(self):
        content = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'
        result = parse_xml_tool_calls(content)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "bash"
        assert json.loads(result[0]["function"]["arguments"]) == {"command": "ls"}
        assert result[0]["id"].startswith("call_")

    def test_valid_multiple_tool_calls(self):
        content = (
            '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>\n'
            'Some text\n'
            '<tool_call>\n{"name": "done", "arguments": {}}\n</tool_call>'
        )
        result = parse_xml_tool_calls(content)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "bash"
        assert result[1]["function"]["name"] == "done"

    def test_empty_input(self):
        assert parse_xml_tool_calls("") == []

    def test_no_tool_calls(self):
        assert parse_xml_tool_calls("just some regular text") == []

    def test_malformed_json_raises_invalid(self):
        content = '<tool_call>\n{not valid json}\n</tool_call>'
        with pytest.raises(InvalidToolCallError):
            parse_xml_tool_calls(content)

    def test_unknown_tool_name_raises(self):
        content = '<tool_call>\n{"name": "nonexistent_tool", "arguments": {}}\n</tool_call>'
        with pytest.raises(UnknownToolCallError):
            parse_xml_tool_calls(content)

    def test_missing_name_raises(self):
        content = '<tool_call>\n{"arguments": {"command": "ls"}}\n</tool_call>'
        with pytest.raises(UnknownToolCallError):
            parse_xml_tool_calls(content)

    def test_name_unknown_raises(self):
        content = '<tool_call>\n{"name": "unknown", "arguments": {}}\n</tool_call>'
        with pytest.raises(UnknownToolCallError):
            parse_xml_tool_calls(content)

    def test_arguments_as_string(self):
        content = '<tool_call>\n{"name": "bash", "arguments": "{\\"command\\": \\"ls\\"}"}\n</tool_call>'
        result = parse_xml_tool_calls(content)
        assert len(result) == 1
        assert result[0]["function"]["arguments"] == '{"command": "ls"}'

    def test_arguments_missing_defaults_to_empty(self):
        content = '<tool_call>\n{"name": "done"}\n</tool_call>'
        result = parse_xml_tool_calls(content)
        assert len(result) == 1
        assert json.loads(result[0]["function"]["arguments"]) == {}

    def test_arguments_non_dict_non_string(self):
        content = '<tool_call>\n{"name": "done", "arguments": 42}\n</tool_call>'
        result = parse_xml_tool_calls(content)
        assert result[0]["function"]["arguments"] == "{}"


class TestCreateDebuggingTools:
    def test_all_tools_included(self):
        tools = create_debugging_tools(use_debugging_tools=True)
        names = {t["function"]["name"] for t in tools}
        assert set(TOOL_NAMES) == names

    def test_non_debugging_only(self):
        tools = create_debugging_tools(use_debugging_tools=False)
        names = {t["function"]["name"] for t in tools}
        assert "bash" in names
        assert "step_over" not in names

    def test_openai_format(self):
        tools = create_debugging_tools()
        for t in tools:
            assert t["type"] == "function"
            assert "name" in t["function"]
            assert "parameters" in t["function"]
