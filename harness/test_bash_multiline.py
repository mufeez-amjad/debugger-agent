"""Regression tests for multiline commands through the real PTY wrapper."""

import shlex
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.bash import BashSession


@pytest.fixture
def session(tmp_path):
    shell = BashSession(tmp_path, timeout=5)
    try:
        yield shell
    finally:
        shell.close()


@pytest.mark.parametrize("indent", ["    ", "\t"])
def test_quoted_multiline_python_preserves_indentation_and_comments(session, indent):
    source = """
# A comment must end at its original newline.
def total(values):
    result = 0
    for value in values:
        result += value
    return result

print(total([10, 80, 30, 120]))
"""
    source = source.replace("    ", indent)
    out = session.execute(f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}")
    assert out == "240\n\nExit code: 0"


def test_heredoc_preserves_literal_shell_syntax(session):
    out = session.execute("cat <<'EOF'\n$HOME $(echo substituted) 'quoted' \\n\n  indented\nEOF")
    assert out == "$HOME $(echo substituted) 'quoted' \\n\n  indented\n\nExit code: 0"


def test_multiline_commands_wait_for_all_output_and_preserve_state(session):
    out = session.execute(
        "MULTILINE_VALUE='persisted'\n"
        "multiline_function() {\n"
        "    printf '%s\\n' \"$MULTILINE_VALUE\"\n"
        "}\n"
        "printf 'first\\n'\n"
        "sleep 0.1\n"
        "multiline_function # trailing comment"
    )
    assert out == "first\npersisted\n\nExit code: 0"
    assert session.execute("multiline_function") == "persisted\n\nExit code: 0"


def test_quoted_text_is_not_expanded_by_transport(session):
    source = "print(\"$HOME $(echo substituted) `echo substituted` café\"); print(r\"\\n\")"
    out = session.execute(f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}")
    assert out == "$HOME $(echo substituted) `echo substituted` café\n\\n\n\nExit code: 0"


def test_multiline_failure_reports_final_exit_code(session):
    out = session.execute("printf 'before\\n'\nfalse # trailing comment")
    assert out == "before\n\nExit code: 1"
    assert session.execute("false | cat") == "Command executed with no output\n\nExit code: 1"


@pytest.mark.skipif(sys.platform == "darwin", reason="bash 3.2 reports exit code 0 here")
def test_syntax_error_reports_exit_code_and_shell_remains_usable(session):
    out = session.execute("if then")
    assert "syntax error" in out
    assert "Exit code: 2" in out
    assert session.execute("echo recovered") == "recovered\n\nExit code: 0"
