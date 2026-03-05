"""Record the observation-hint layer's output to common/hint_golden.json.

Run this only to re-baseline after a deliberate change to the hint text.
``common/test_observation_hints.py`` replays the same matrix and demands a
byte-identical match, because the model saw this wording in training.

    python -m common.capture_hint_golden
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from common.hint_cases import (
    ERROR_MESSAGE_CASES,
    FORMAT_CASES,
    FRAME_PRIMERS,
    PDB_SEQUENCE,
    WORKSPACE,
)

GOLDEN_PATH = Path(__file__).with_name("hint_golden.json")


class HintAdapter:
    """Drive ``common.observation_hints.ObservationHinter``."""

    def __init__(self) -> None:
        from common.observation_hints import ObservationHinter

        self._hinter = ObservationHinter(workspace_path=WORKSPACE)

    def reset(self) -> None:
        self._hinter.reset()

    def set_workflow_state(self, state: Dict[str, Any]) -> None:
        self._hinter.set_progress(
            turn_last_pytest=int(state.get("turn_last_pytest", 0) or 0),
            turn_last_edit=int(state.get("turn_last_edit", 0) or 0),
        )

    def track(self, tool_name: str, result: str) -> None:
        self._hinter.track_frame(tool_name, result)

    def improve(self, tool_name, result, tool_args) -> str:
        return self._hinter.improve_error_message(tool_name, result, tool_args)

    def revisit(self, tool_name, result) -> str:
        return self._hinter.revisit_hint(tool_name, result)

    def format(self, tool_name, result, tool_args, hints_enabled) -> str:
        return self._hinter.format_observation(
            tool_name, result, tool_args, hints_enabled=hints_enabled
        )


def capture(adapter) -> Dict[str, Any]:
    out: Dict[str, Any] = {"improve": {}, "revisit": [], "format": {}}

    for case_id, tool, result, args, wf_state in ERROR_MESSAGE_CASES:
        adapter.reset()
        adapter.set_workflow_state(wf_state)
        primer = FRAME_PRIMERS.get(case_id)
        if primer:
            adapter.track(*primer)
        out["improve"][case_id] = adapter.improve(tool, result, args)

    adapter.reset()
    adapter.set_workflow_state({})
    for tool, result in PDB_SEQUENCE:
        out["revisit"].append([tool, adapter.revisit(tool, result)])

    for case_id, tool, result, args, enabled in FORMAT_CASES:
        adapter.reset()
        adapter.set_workflow_state({})
        out["format"][case_id] = adapter.format(tool, result, args, enabled)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(GOLDEN_PATH))
    args = parser.parse_args()
    payload = capture(HintAdapter())
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
