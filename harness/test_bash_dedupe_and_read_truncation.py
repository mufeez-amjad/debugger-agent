"""
Tests for BashSession output handling:

  1. Duplicate-output compression: re-running a byte-identical command
     returns a short "nothing changed" note instead of the full output.
  2. Head-first truncation for pure file reads: a `sed -n '1,150p'` style
     read keeps a contiguous head plus a continue-from-here hint, not
     middle omission (which can hide the buggy function).
  3. Failed-undo redirect — `git ...` in the git-less workspace and `cp` of
     nonexistent backups get a pointer at the revert_file tool.
"""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.bash import BashSession, _file_read_target
from harness.debug_session import DebugSession
from common.models import ToolCall


# --------------------------------------------------------------------------- #
# 1. duplicate-output compression
# --------------------------------------------------------------------------- #

def test_identical_rerun_is_compressed():
    with tempfile.TemporaryDirectory() as d:
        s = BashSession(Path(d).resolve(), timeout=5)
        try:
            first = s.execute("seq 1 40")
            second = s.execute("seq 1 40")
            assert "byte-identical" not in first
            assert "byte-identical" in second
            assert "repeat #1" in second
            # tail lines stay visible for grounding
            assert "40" in second
            assert "Exit code: 0" in second
            third = s.execute("seq 1 40")
            assert "repeat #2" in third
        finally:
            s.close()


def test_changed_output_is_not_compressed(tmp_path):
    source = tmp_path / "data.txt"
    source.write_text("\n".join(f"line {i}" for i in range(40)) + "\n")
    session = BashSession(tmp_path.resolve(), timeout=5)
    try:
        session.execute("cat data.txt")
        assert "byte-identical" in session.execute("cat data.txt")
        source.write_text(source.read_text() + "new content\n")
        changed = session.execute("cat data.txt")
        assert "new content" in changed
        assert "byte-identical" not in changed
    finally:
        session.close()


def test_small_outputs_pass_through():
    with tempfile.TemporaryDirectory() as d:
        s = BashSession(Path(d).resolve(), timeout=5)
        try:
            first = s.execute("echo hello")
            second = s.execute("echo hello")
            assert first == second
            assert "byte-identical" not in second
        finally:
            s.close()


def test_pytest_style_rerun_keeps_summary_tail():
    with tempfile.TemporaryDirectory() as d:
        s = BashSession(Path(d).resolve(), timeout=5)
        try:
            cmd = "for i in $(seq 1 30); do echo line$i; done; echo '2 failed, 3 passed in 0.5s'"
            # compound commands are dedupe-eligible (only file READS are
            # special-cased in truncation, not in dedupe)
            s.execute(cmd)
            second = s.execute(cmd)
            assert "byte-identical" in second
            assert "2 failed, 3 passed" in second  # summary survives in the tail
        finally:
            s.close()


# --------------------------------------------------------------------------- #
# 2. head-first truncation for pure file reads
# --------------------------------------------------------------------------- #

def test_file_read_target_detection():
    assert _file_read_target("cat foo.py") == ("foo.py", 1)
    assert _file_read_target("sed -n '1,150p' foo.py") == ("foo.py", 1)
    assert _file_read_target("sed -n '30,200p' pkg/mod.py") == ("pkg/mod.py", 30)
    assert _file_read_target("tail -n +71 foo.py") == ("foo.py", 71)
    assert _file_read_target("cat foo.py | grep x") is None
    assert _file_read_target("python -m pytest foo_test.py") is None
    assert _file_read_target("sed -i 's/a/b/' foo.py") is None


def test_file_read_truncates_head_first_with_continue_hint():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        (root / "big.py").write_text(
            "\n".join(f"line_{i}" for i in range(1, 301)) + "\n"
        )
        s = BashSession(root, timeout=5, max_output_lines=100)
        try:
            out = s.execute("cat big.py")
            assert "<TOOL_OUTPUT_TRUNCATED:" in out
            assert "trailing lines omitted" in out
            assert "middle lines omitted" not in out
            # contiguous head: line_1 .. line_100 all present, line_150 absent
            assert "line_1\n" in out
            assert "line_100" in out
            assert "line_150" not in out
            # continue-from-here hint names the file and the next range
            assert "sed -n '101,200p' big.py" in out
            assert "Exit code: 0" in out
        finally:
            s.close()


def test_range_read_hint_uses_file_line_numbers():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        (root / "big.py").write_text(
            "\n".join(f"line_{i}" for i in range(1, 401)) + "\n"
        )
        s = BashSession(root, timeout=5, max_output_lines=100)
        try:
            out = s.execute("sed -n '30,250p' big.py")
            assert "trailing lines omitted" in out
            # 100 output lines starting at file line 30 → next chunk starts at 130
            assert "sed -n '130,229p' big.py" in out
        finally:
            s.close()


def test_pytest_output_still_uses_middle_omission():
    with tempfile.TemporaryDirectory() as d:
        s = BashSession(Path(d).resolve(), timeout=5, max_output_lines=100)
        try:
            out = s.execute("seq 1 300")
            assert "middle lines omitted" in out
            assert "trailing lines omitted" not in out
        finally:
            s.close()


# --------------------------------------------------------------------------- #
# 3. failed-undo redirect to revert_file
# --------------------------------------------------------------------------- #

def _bash(session: DebugSession, command: str) -> str:
    return session.execute_tool_call(
        ToolCall(name="bash", parameters={"command": command}, reasoning="t")
    )
