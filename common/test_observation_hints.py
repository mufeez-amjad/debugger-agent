"""Byte-identical gate on the observation-hint layer.

The model saw this wording in training, so a change to a hint must be deliberate:
re-run ``python -m common.capture_hint_golden`` and say so in the commit.
"""

from __future__ import annotations

import json
from pathlib import Path

from common.capture_hint_golden import GOLDEN_PATH, HintAdapter, capture


def _golden():
    return json.loads(Path(GOLDEN_PATH).read_text())


def test_shared_hinter_matches_golden():
    assert capture(HintAdapter()) == _golden()


def test_revisit_hint_stays_disabled_across_sessions():
    """Revisiting a line should not append an observation nudge."""
    from common.observation_hints import ObservationHinter

    hinter = ObservationHinter(workspace_path="/workspace/defect")
    frame = "> /workspace/defect/engine.py(49)allocate()"
    assert hinter.revisit_hint("step_over", frame) == ""
    assert hinter.revisit_hint("step_over", frame) == ""
    hinter.reset()
    assert hinter.revisit_hint("step_over", frame) == ""


def test_unwrapped_format_omits_the_tool_header():
    """The CLI carries the tool name in its own message envelope."""
    from common.observation_hints import ObservationHinter

    hinter = ObservationHinter(workspace_path="/repo")
    wrapped = hinter.format_observation("bash", "ok", {"command": "true"})
    bare = hinter.format_observation("bash", "ok", {"command": "true"}, wrap=False)
    assert wrapped == "Tool: bash\nResult: ok"
    assert bare == "ok"


def test_failed_pytest_result_is_not_replaced_by_a_rerun_hint():
    """A diagnostic rerun should return its result regardless of edit history."""
    from common.observation_hints import ObservationHinter

    hinter = ObservationHinter(workspace_path="/repo")
    args = {"command": "pytest -q"}
    result = "1 failed, 42 passed"
    assert hinter.improve_error_message("bash", result, args) == result
    hinter.set_progress(turn_last_pytest=6, turn_last_edit=4)
    assert hinter.improve_error_message("bash", result, args) == result
    hinter.set_progress(turn_last_pytest=4, turn_last_edit=6)
    assert hinter.improve_error_message("bash", result, args) == result
