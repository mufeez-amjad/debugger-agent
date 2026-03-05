"""The input matrix the observation-hint golden test replays.

Kept apart from the test so the capture script and the assertion read the same
cases. Each case names the branch it pins; adding a branch to
``common.observation_hints`` means adding a case here and re-capturing.

``pdb_sequence`` cases drive the revisit tracker, which is stateful: the list is
replayed in order against ONE hinter and every step's return value is recorded,
so the window trimming and the visit counter are pinned, not just the final
message.
"""

from __future__ import annotations

WORKSPACE = "/workspace/defect"

# (case_id, tool_name, result, tool_args, workflow_state)
ERROR_MESSAGE_CASES = [
    (
        "bash_cd_absolute_missing",
        "bash",
        "cd: /root/src: No such file or directory",
        {"command": "cd /root/src && ls"},
        {},
    ),
    (
        "bash_cd_relative_missing_not_matched",
        "bash",
        "cd: nope: No such file or directory",
        {"command": "cd nope"},
        {},
    ),
    (
        "pytest_collection_error",
        "bash",
        "ImportError while importing test module 'engine_test.py'.\n"
        "engine.py:12: SyntaxError: invalid syntax\n"
        "!!!!!! Interrupted: 1 error during collection !!!!!!\n"
        "1 error in 0.11s",
        {"command": "python -m pytest engine_test.py -x -q"},
        {"turn_last_pytest": 3, "turn_last_edit": 5},
    ),
    (
        "pytest_collected_nothing",
        "bash",
        "collected 0 items\n\nno tests ran in 0.01s",
        {"command": "python -m pytest tests/engine_test.py -x -q"},
        {},
    ),
    (
        "pytest_failed_rerun_without_edit",
        "bash",
        "1 failed, 42 passed in 0.30s",
        {"command": "python -m pytest engine_test.py -q"},
        {"turn_last_pytest": 6, "turn_last_edit": 4},
    ),
    (
        "pytest_failed_first_run",
        "bash",
        "1 failed, 42 passed in 0.30s",
        {"command": "python -m pytest engine_test.py -q"},
        {"turn_last_pytest": 0, "turn_last_edit": 0},
    ),
    (
        "pytest_failed_after_edit",
        "bash",
        "1 failed, 42 passed in 0.30s",
        {"command": "python -m pytest engine_test.py -q"},
        {"turn_last_pytest": 4, "turn_last_edit": 6},
    ),
    (
        "pytest_all_passing",
        "bash",
        "43 passed in 0.29s",
        {"command": "python -m pytest engine_test.py -q"},
        {"turn_last_pytest": 6, "turn_last_edit": 4},
    ),
    (
        "sed_no_output",
        "bash",
        "(no output)",
        {"command": "sed -n '200,260p' engine.py"},
        {},
    ),
    (
        "sed_with_output",
        "bash",
        "def allocate(order):\n    return None",
        {"command": "sed -n '1,20p' engine.py"},
        {},
    ),
    (
        "bash_clean_result_no_error",
        "bash",
        "engine.py\nengine_test.py\nstrategy.py",
        {"command": "ls"},
        {},
    ),
    (
        "file_not_found_quoted",
        "edit_file",
        "Error: [Errno 2] No such file or directory: 'strategy.py'",
        {"file_path": "strategy.py"},
        {},
    ),
    (
        "file_not_found_unnamed",
        "edit_file",
        "Error: FileNotFoundError raised while opening the target",
        {"file_path": "strategy.py"},
        {},
    ),
    (
        "edit_string_not_found",
        "edit_file",
        "Error: string not found in file. Closest region:\n  12 | def rank(lots):",
        {"file_path": "strategy.py", "old_string": "def rank(lot)"},
        {},
    ),
    (
        "attribute_error_pdb_in_init",
        "print_variable",
        "*** Error: AttributeError: 'Report' object has no attribute 'flags'",
        {"name": "self.flags"},
        {},
    ),
    (
        "attribute_error_pdb_outside_init",
        "print_variable",
        "*** Error: AttributeError: 'Report' object has no attribute 'flags'",
        {"name": "self.flags"},
        {},
    ),
    (
        # Pins that the stepping ops count as frame sources: the frame this
        # hint keys on arrives on a step_over result, not on an inspection.
        "attribute_error_frame_from_step_over",
        "print_variable",
        "*** Error: AttributeError: 'Report' object has no attribute 'flags'",
        {"name": "self.flags"},
        {},
    ),
    (
        "attribute_error_pdb_not_self",
        "print_variable",
        "*** Error: AttributeError: 'Report' object has no attribute 'flags'",
        {"name": "report.flags"},
        {},
    ),
    (
        "attribute_error_non_pdb_tool",
        "bash",
        "AttributeError: 'Catalog' object has no attribute 'to_each'",
        {"command": "python -c 'import catalog'"},
        {},
    ),
    (
        "attribute_error_no_name",
        "bash",
        "Error: AttributeError raised during call",
        {"command": "python -c 'import catalog'"},
        {},
    ),
    (
        "syntax_error_heredoc",
        "bash",
        "  File \"<stdin>\", line 3\n    SyntaxError: invalid syntax",
        {"command": "cat > patch.py <<'EOF'\ndef f(:\nEOF"},
        {},
    ),
    (
        "syntax_error_plain",
        "bash",
        "  File \"engine.py\", line 30\n    SyntaxError: invalid syntax",
        {"command": "python -c 'import engine'"},
        {},
    ),
    (
        "syntax_error_from_pdb",
        "run_expression",
        "*** SyntaxError: invalid syntax",
        {"expression": "lot.expiry ="},
        {},
    ),
    (
        "timeout_interactive_pdb",
        "bash",
        "Error: tool execution timed out; the pdb child was left interactive",
        {"command": "python -m pdb engine.py"},
        {},
    ),
    (
        "timeout_generic",
        "bash",
        "Error: command timed out after 30s",
        {"command": "python -m pytest"},
        {},
    ),
    (
        "timeout_word_in_source_not_a_timeout",
        "bash",
        "def retry(fn, timeout=30):\n    # Error handling below\n    pass",
        {"command": "sed -n '1,10p' retry.py"},
        {},
    ),
    (
        "unrecognised_error_passthrough",
        "bash",
        "Error: something went sideways in a way nothing matches",
        {"command": "true"},
        {},
    ),
]

# Cases whose expected output depends on the tracked pdb frame. The capture
# feeds the named frame-bearing observation through the tracker first, using the
# tool named alongside it — which tool reported the frame is itself part of the
# contract (see INSPECTION_DEBUG_TOOLS).
FRAME_PRIMERS = {
    "attribute_error_pdb_in_init": (
        "start_debugger",
        "> /workspace/defect/report.py(18)__init__()\n-> self.flags = frozenset()",
    ),
    "attribute_error_pdb_outside_init": (
        "start_debugger",
        "> /workspace/defect/report.py(52)render()\n-> return self.flags",
    ),
    "attribute_error_frame_from_step_over": (
        "step_over",
        "> /workspace/defect/report.py(18)__init__()\n-> self.flags = frozenset()",
    ),
}

# Replayed in order against one hinter; every step's hint is recorded.
PDB_SEQUENCE = [
    ("step_over", "> /workspace/defect/engine.py(49)allocate()"),
    ("step_over", "> /workspace/defect/engine.py(50)allocate()"),
    ("step_over", "> /workspace/defect/engine.py(57)_reserve()"),
    ("step_over", "> /workspace/defect/engine.py(58)_reserve()"),
    ("step_over", "> /workspace/defect/engine.py(49)allocate()"),
    ("step_over", "> /workspace/defect/engine.py(50)allocate()"),
    ("continue_execution", "> /workspace/defect/engine.py(49)allocate()"),
    ("step_into", "> /workspace/defect/engine.py(49)allocate()"),
    ("examine_locals", "> /workspace/defect/engine.py(49)allocate()"),
    ("step_over", "no frame header in this result"),
    ("step_over", "> /workspace/defect/engine.py(101)finish()"),
]

# (case_id, tool_name, result, tool_args, hints_enabled). The first two differ
# only in `hints_enabled`, so they pin that the raw arm really does suppress a
# hint the hinted arm emits.
FORMAT_CASES = [
    (
        "format_hinted",
        "bash",
        "cd: /root: No such file or directory",
        {"command": "cd /root"},
        True,
    ),
    (
        "format_raw",
        "bash",
        "cd: /root: No such file or directory",
        {"command": "cd /root"},
        False,
    ),
    ("format_truncated", "bash", "x" * 9000, {"command": "cat big.py"}, True),
    ("format_short", "bash", "ok", {"command": "true"}, True),
]
