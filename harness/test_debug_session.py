"""
Tests for debug_session.py, specifically bash output truncation.
"""
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.debug_session import DebugSession
from common.models import ToolCall


def test_bash_output_truncation_by_lines():
    """Truncation is middle-out: head AND tail survive so pytest's final summary stays visible."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = DebugSession(tmpdir)

        # Outputs 150 lines, over the 100-line limit
        tool_call = ToolCall(
            name='bash',
            parameters={'command': 'seq 1 150'},
            reasoning='test'
        )

        result = session.execute_tool_call(tool_call)

        assert '<TOOL_OUTPUT_TRUNCATED' in result, "Expected truncation marker in output"
        assert 'middle lines omitted' in result, "Expected middle-omission marker"

        lines = result.split('\n')
        assert lines[0] == '1', "Head of output must survive truncation"
        assert '150' in result, "Tail of output (final summary line) must survive truncation"
        assert len(lines) <= 105, f"Expected ~100 lines plus marker, got {len(lines)}"

        print("✓ Line truncation test passed")


def test_bash_output_truncation_by_bytes():
    """Test that bash output is truncated when exceeding byte limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = DebugSession(tmpdir)

        # 60 lines x ~200 bytes = ~12KB, over the 8192-byte limit but under the
        # 100-line limit, so this exercises the byte-overflow path specifically.
        long_text = 'A' * 200
        tool_call = ToolCall(
            name='bash',
            parameters={'command': f'for i in {{1..60}}; do echo "$i {long_text}"; done'},
            reasoning='test'
        )

        result = session.execute_tool_call(tool_call)

        assert '<TOOL_OUTPUT_TRUNCATED' in result, "Expected truncation marker for large output"
        assert 'bytes=' in result, "Expected byte count in truncation marker"
        assert '60 ' in result, "Final line must survive byte truncation (tail preserved)"

        # Result should stay near the 8KB budget (plus marker/exit-code overhead)
        assert len(result) < 9216, f"Expected truncated output, got {len(result)} bytes"

        print("✓ Byte truncation test passed")


def test_bash_no_truncation_for_small_output():
    """Test that small bash outputs are not truncated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = DebugSession(tmpdir)

        tool_call = ToolCall(
            name='bash',
            parameters={'command': 'echo "hello world"'},
            reasoning='test'
        )

        result = session.execute_tool_call(tool_call)

        # Should NOT contain truncation marker
        assert '<TOOL_OUTPUT_TRUNCATED' not in result, "Small output should not be truncated"
        assert 'hello world' in result, "Expected actual output"

        print("✓ No truncation for small output test passed")


def test_truncation_marker_format():
    """Test that truncation marker has correct format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = DebugSession(tmpdir)

        tool_call = ToolCall(
            name='bash',
            parameters={'command': 'seq 1 100'},
            reasoning='test'
        )

        result = session.execute_tool_call(tool_call)

        # Check marker format
        assert '<TOOL_OUTPUT_TRUNCATED' in result
        assert 'command=' in result
        assert 'lines=' in result
        assert 'bytes=' in result
        assert 'sha1=' in result

        print("✓ Truncation marker format test passed")


def test_bash_context_env_persists_but_cwd_resets():
    """Shell variables persist across calls, but cwd resets to the workspace root
    each call (a stray `cd` must not poison file-tool/debugger path resolution).
    `cd` still works WITHIN a single compound command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = DebugSession(tmpdir)

        session.execute_tool_call(ToolCall(
            name='bash',
            parameters={'command': 'export SWEEP_FIX_PROBE=persisted && mkdir -p subdir && cd subdir'},
            reasoning='setup'
        ))

        env_result = session.execute_tool_call(ToolCall(
            name='bash',
            parameters={'command': 'echo $SWEEP_FIX_PROBE'},
            reasoning='verify env persistence'
        ))
        assert 'persisted' in env_result

        pwd_result = session.execute_tool_call(ToolCall(
            name='bash',
            parameters={'command': 'pwd'},
            reasoning='verify cwd reset'
        ))
        first_line = pwd_result.strip().split('\n')[0]
        assert Path(first_line).resolve() == Path(tmpdir).resolve(), \
            f"cwd should reset to workspace root, got {first_line}"

        compound = session.execute_tool_call(ToolCall(
            name='bash',
            parameters={'command': 'cd subdir && pwd'},
            reasoning='verify cd works within one command'
        ))
        assert compound.strip().split('\n')[0].endswith('subdir')

        print("✓ Bash env-persists / cwd-resets test passed")

def test_bash_strips_ansi_sequences():
    """Ensure ANSI escape sequences are removed from bash output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = DebugSession(tmpdir)

        tool_call = ToolCall(
            name='bash',
            parameters={'command': r"printf '\033[1mHELLO\033[0m\n'"},
            reasoning='test ansi strip'
        )

        result = session.execute_tool_call(tool_call)
        assert "\x1b" not in result
        assert "HELLO" in result


def test_edit_file_rejects_noop_replacement():
    """edit_file should error if old_string == new_string."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "noop.txt"
        target.write_text("hello", encoding="utf-8")
        session = DebugSession(tmpdir)

        tool_call = ToolCall(
            name='edit_file',
            parameters={'file_path': 'noop.txt', 'old_string': 'hello', 'new_string': 'hello'},
            reasoning='test noop edit'
        )

        result = session.execute_tool_call(tool_call)
        assert result.startswith("Error:")
        assert "old_string == new_string" in result
        assert target.read_text(encoding="utf-8") == "hello"


if __name__ == '__main__':
    print("Running bash truncation tests...")
    test_bash_output_truncation_by_lines()
    test_bash_output_truncation_by_bytes()
    test_bash_no_truncation_for_small_output()
    test_truncation_marker_format()
    print("\nAll tests passed! ✓")
