"""BashSession hardening tests.

All tests use a real PTY-backed BashSession with a short timeout so the
timeout-recovery path runs in ~1s instead of the production 30s.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.bash import BashSession


def _first_line(result: str) -> str:
    return result.split("\n", 1)[0]


# --------------------------------------------------------------------------- #
# cwd resets to the workspace root before every command
# --------------------------------------------------------------------------- #

def test_cd_does_not_persist_across_commands():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        (root / "sub").mkdir()
        s = BashSession(root, timeout=5)
        try:
            s.execute("cd sub")
            pwd = _first_line(s.execute("pwd")).strip()
            assert Path(pwd).resolve() == root, pwd
        finally:
            s.close()


def test_cd_within_compound_command_still_works():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        (root / "sub").mkdir()
        s = BashSession(root, timeout=5)
        try:
            pwd = _first_line(s.execute("cd sub && pwd")).strip()
            assert Path(pwd).resolve() == (root / "sub").resolve(), pwd
        finally:
            s.close()


# --------------------------------------------------------------------------- #
# history expansion disabled ('!' does not raise "event not found")
# --------------------------------------------------------------------------- #

def test_bang_does_not_trigger_history_expansion():
    with tempfile.TemporaryDirectory() as d:
        s = BashSession(Path(d).resolve(), timeout=5)
        try:
            out = s.execute('echo "hi!"')
            assert "hi!" in out
            assert "event not found" not in out
        finally:
            s.close()


# --------------------------------------------------------------------------- #
# range-read past EOF gets an explicit line-count note
# --------------------------------------------------------------------------- #

def test_sed_range_past_eof_gets_note():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        (root / "small.py").write_text("a\nb\nc\n")
        s = BashSession(root, timeout=5)
        try:
            out = s.execute("sed -n '71,74p' small.py")
            assert "has 3 lines" in out
            assert "beyond EOF" in out
        finally:
            s.close()


def test_tail_plus_range_past_eof_gets_note():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        (root / "small.py").write_text("a\nb\n")
        s = BashSession(root, timeout=5)
        try:
            out = s.execute("tail -n +50 small.py")
            assert "has 2 lines" in out
        finally:
            s.close()


def test_sed_range_within_file_gets_no_note():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        (root / "small.py").write_text("a\nb\nc\nd\n")
        s = BashSession(root, timeout=5)
        try:
            out = s.execute("sed -n '1,2p' small.py")
            assert "beyond EOF" not in out
            assert "a" in out and "b" in out
        finally:
            s.close()


def test_plain_no_output_command_unaffected():
    with tempfile.TemporaryDirectory() as d:
        s = BashSession(Path(d).resolve(), timeout=5)
        try:
            out = s.execute("true")
            assert "Command executed with no output" in out
            assert "beyond EOF" not in out
        finally:
            s.close()


# --------------------------------------------------------------------------- #
# middle truncation keeps head AND tail (first traceback + summary)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# the shell self-heals after being bricked / after a timeout
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
