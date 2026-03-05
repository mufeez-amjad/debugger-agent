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
