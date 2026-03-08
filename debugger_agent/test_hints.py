"""The CLI applies the same observation hints the model saw in training.

Serving the model without them leaves a gap between training and inference on
the mechanism that moves a stuck model off a dead static-reading path, so these
tests pin the wiring: the hint text itself is gated by ``common/test_observation_hints.py``.
"""

from __future__ import annotations

from debugger_agent.agent import AgentConfig, DebugAgent


class _FakeSession:
    """Stands in for DebugSession; returns a scripted result per call."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def execute_tool_call(self, call):
        self.calls.append((call.name, call.parameters))
        return self._results.pop(0)


def _agent(tmp_path, results, hints=True):
    agent = DebugAgent(str(tmp_path), "fix it", AgentConfig(hints=hints))
    agent.debug_session = _FakeSession(results)
    return agent


def test_missing_path_gets_the_workspace_reminder(tmp_path):
    agent = _agent(tmp_path, ["cd: /root/src: No such file or directory"])
    out = agent._execute("bash", {"command": "cd /root/src && ls"}, step=0)
    assert "Reminder: your workspace is" in out
    assert str(tmp_path.resolve()) in out


def test_an_edit_between_runs_clears_the_nag(tmp_path):
    agent = _agent(
        tmp_path,
        [
            "1 failed, 42 passed in 0.3s",
            "Successfully replaced text in engine.py",
            "1 failed, 42 passed in 0.3s",
        ],
    )
    cmd = {"command": "python -m pytest engine_test.py -q"}
    agent._execute("bash", cmd, step=0)
    agent._execute("edit_file", {"file_path": "engine.py"}, step=1)
    out = agent._execute("bash", cmd, step=2)
    assert "still failing" not in out


def test_a_run_that_collected_nothing_is_not_a_test_observation(tmp_path):
    """Wrong path first, then the right one — the right one must not be nagged."""
    agent = _agent(
        tmp_path,
        [
            "collected 0 items\n\nno tests ran in 0.01s",
            "1 failed, 42 passed in 0.3s",
        ],
    )
    bad = agent._execute(
        "bash", {"command": "pytest tests/engine_test.py -q"}, step=0
    )
    assert "pytest ran 0 tests" in bad
    good = agent._execute("bash", {"command": "pytest engine_test.py -q"}, step=1)
    assert "still failing" not in good


def test_observations_are_capped_at_the_training_size(tmp_path):
    agent = _agent(tmp_path, ["x" * 9000])
    out = agent._execute("bash", {"command": "cat big.py"}, step=0)
    assert len(out) <= 4000
    assert "chars omitted" in out


def test_no_hints_flag_leaves_the_result_bare(tmp_path):
    agent = _agent(tmp_path, ["cd: /root: No such file or directory"], hints=False)
    out = agent._execute("bash", {"command": "cd /root"}, step=0)
    assert out == "cd: /root: No such file or directory"


def test_capping_still_applies_with_hints_off(tmp_path):
    """The cap is distribution fidelity, not a hint — disabling hints keeps it."""
    agent = _agent(tmp_path, ["x" * 9000], hints=False)
    out = agent._execute("bash", {"command": "cat big.py"}, step=0)
    assert len(out) <= 4000
