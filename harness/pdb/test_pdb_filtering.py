"""Pdb output keeps user data and filters Python internals."""

import pytest

from common.models import ToolCall
from harness.pdb.pdb import PdbHarness, PdbSession


FORBIDDEN_PATTERNS = [
    "'ValueError': <class 'ValueError'>",
    "'TypeError': <class 'TypeError'>",
    "'ZeroDivisionError': <class 'ZeroDivisionError'>",
    "'RuntimeError': <class 'RuntimeError'>",
    "'KeyError': <class 'KeyError'>",
    "'AttributeError': <class 'AttributeError'>",
    "'bool': <class 'bool'>",
    "'int': <class 'int'>",
    "'str': <class 'str'>",
    "'list': <class 'list'>",
    "'dict': <class 'dict'>",
    "'tuple': <class 'tuple'>",
    "'abs': <built-in function",
    "'len': <built-in function",
    "'max': <built-in function",
    "'min': <built-in function",
    "'__debug__': True",
    "This module provides direct access to all 'built-in'",
    "identifiers of Python; for example, builtins.len",
]


@pytest.mark.asyncio
async def test_pdb_filtering(tmp_path):
    test_file = tmp_path / "program.py"
    test_file.write_text(
        'def calculate_sum(a, b):\n'
        '    x = 10\n'
        '    y = 20\n'
        '    message = "hello world"\n'
        '    data = [1, 2, 3, 4, 5]\n'
        '    config = {"timeout": 30, "retries": 3}\n'
        '    return a + b\n'
        '\n'
        'print(calculate_sum(5, 7))\n'
    )
    harness = PdbHarness(working_directory=str(tmp_path))
    commands = [
        ("start_debugger", {"file_path": str(test_file)}, "Debugging session started"),
        ("add_breakpoint", {"line": 7}, "Breakpoint set"),
        ("continue_execution", {}, "calculate_sum"),
        ("examine_locals", {}, "hello world"),
        ("examine_globals", {}, "calculate_sum"),
        ("print_variable", {"name": "x"}, "10"),
        ("where_am_i", {}, "calculate_sum"),
        ("list_code", {"start": 1, "end": 9}, "return a + b"),
    ]
    try:
        for name, parameters, expected in commands:
            output = harness.execute_tool_call(ToolCall(name, parameters, reasoning=""))
            assert expected in output, (name, output)
            for pattern in FORBIDDEN_PATTERNS:
                assert pattern not in output, (name, pattern, output)
            if name.startswith("examine_"):
                assert len(output) <= 5000, (name, output)
    finally:
        await harness.cleanup_async()


@pytest.mark.parametrize(
    "raw, expected, forbidden",
    [
        pytest.param("", [], [], id="empty"),
        pytest.param(
            "'ValueError': <class 'ValueError'>, 'bool': <class 'bool'>",
            ["<All output filtered"],
            ["ValueError", "bool"],
            id="builtins",
        ),
        pytest.param(
            "{'x': 42,\n'ValueError': <class 'ValueError'>,\n'y': 'test'}",
            ["x", "42", "y", "test"],
            ["ValueError"],
            id="mixed",
        ),
        pytest.param(
            "'__builtins__': {'ValueError': <class 'ValueError'>, 'len': <built-in function len>}",
            [],
            ["ValueError", "len", "<built-in function"],
            id="builtins-expansion",
        ),
        pytest.param(
            "\n".join([
                "'ArithmeticError': <class 'ArithmeticError'>",
                "'AssertionError': <class 'AssertionError'>",
                "'AttributeError': <class 'AttributeError'>",
                "'BaseException': <class 'BaseException'>",
                "'BlockingIOError': <class 'BlockingIOError'>",
            ]),
            ["<All output filtered"],
            ["ArithmeticError", "AttributeError"],
            id="consecutive-builtins",
        ),
        pytest.param(
            "{'my_error': CustomError('test'), 'value': 123}",
            ["my_error", "CustomError", "value", "123"],
            [],
            id="user-exceptions",
        ),
    ],
)
def test_filter_edge_cases(raw, expected, forbidden):
    output = PdbSession("/dev/null")._filter_verbose_output(raw)
    if not raw:
        assert output == ""
    for pattern in expected:
        assert pattern in output
    for pattern in forbidden:
        assert pattern not in output
