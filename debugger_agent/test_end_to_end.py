"""One full session: a scripted server, the real harness, and a real test run."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from debugger_agent import events
from debugger_agent.agent import AgentConfig, DebugAgent

BUGGY = "def add(a, b):\n    return a - b\n"
TEST = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def _call(name, **arguments):
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


@pytest.fixture
def server():
    """An OpenAI-compatible stub that replays tool calls and records requests."""
    pytest_cmd = f"{sys.executable} -B -m pytest -q -p no:cacheprovider"
    script = [
        _call("bash", command=pytest_cmd),
        _call("edit_file", file_path="calc.py", old_string="a - b", new_string="a + b"),
        _call("bash", command=pytest_cmd),
        _call("done"),
    ]
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, dict(self.headers), json.loads(body)))
            message = {"role": "assistant", "content": "", "tool_calls": [script.pop(0)]}
            payload = json.dumps({"choices": [{"message": message}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", requests
    httpd.shutdown()


def test_session_fixes_the_bug_and_reports_success(tmp_path, server):
    url, requests = server
    (tmp_path / "calc.py").write_text(BUGGY)
    (tmp_path / "test_calc.py").write_text(TEST)

    config = AgentConfig(model_url=url, api_key="secret", tokenizer=str(tmp_path))
    stream = list(DebugAgent(str(tmp_path), "fix add", config).run())

    final = stream[-1]
    assert isinstance(final, events.SessionComplete)
    assert final.success is True
    assert final.steps_taken == 4
    assert "a + b" in (tmp_path / "calc.py").read_text()

    path, headers, body = requests[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert body["model"] == "llm-debugger"
    assert {t["function"]["name"] for t in body["tools"]} >= {"bash", "start_debugger", "done"}
    assert body["response_format"]["type"] == "structural_tag"
