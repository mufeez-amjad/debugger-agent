"""Serverless hosting of the debugger model on Modal.

The adapter is a rank-2 LoRA over every MoE expert projection. `prepare_model`
caches the base model and adapter on a Volume. vLLM loads the bf16 base weights
and applies the adapter at inference time.

The server is vLLM's OpenAI-compatible API on one A100-80GB. It scales to zero:
you pay while a request is in flight, plus SCALEDOWN_WINDOW of idle time.

The endpoint is transport-public. Every request must carry the vLLM API key
(Authorization: Bearer <key>) from the Modal secret `debugger-agent-vllm-key`.

Setup (one time):
    modal secret create debugger-agent-vllm-key VLLM_API_KEY=<random hex>
    modal secret create debugger-agent-hf-token HF_TOKEN=<read token>
    modal run --detach inference_server/modal_app.py::prepare_model
    modal deploy inference_server/modal_app.py

Use --detach for prepare_model: a plain `modal run` cancels the function when
the local process exits, and the Volume keeps nothing until the final commit.
"""

import os
import subprocess
import time
from pathlib import Path

import modal

BASE_REPO = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
ADAPTER_REPO = "moofeez/qwen3-coder-30b-a3b-debugger"
ADAPTER_REVISION = "0e4e349c80fa48bee674580e204a7b4ce6c1d406"
SERVED_NAME = "llm-debugger"

HF_CACHE = "/root/.cache/huggingface"

GPU = "A100-80GB"  # 57 GiB of bf16 weights plus the KV cache
VLLM_PORT = 8000
MINUTES = 60
# A debug run sends requests seconds apart, so it stays warm by itself. This
# window covers the gaps between runs, when a person reads the diff and retries.
SCALEDOWN_WINDOW = 10 * MINUTES
MAX_MODEL_LEN = 8192

# Training rendered tool calls as JSON with this template. The base model
# carries the stock Qwen XML template, so --chat-template below is required.
# The client counts tokens with this file too.
CHAT_TEMPLATE_LOCAL = Path(__file__).parent.parent / "common" / "chat_template_json.jinja"
CHAT_TEMPLATE_REMOTE = "/root/chat_template_json.jinja"

app = modal.App("debugger-agent-inference")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "hf_transfer")
    .env(
        {
            # The Xet backend hangs on large model repos here.
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_LOG_STATS_INTERVAL": "1",
        }
    )
    .add_local_file(CHAT_TEMPLATE_LOCAL, CHAT_TEMPLATE_REMOTE)
)

hf_cache_vol = modal.Volume.from_name("debugger-agent-hf-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    volumes={HF_CACHE: hf_cache_vol},
    secrets=[modal.Secret.from_name("debugger-agent-hf-token")],
    cpu=8,
    memory=32 * 1024,
    timeout=2 * 60 * MINUTES,
)
def prepare_model():
    """Cache the base model and pinned adapter on the Volume. No GPU needed."""
    from huggingface_hub import snapshot_download

    base = snapshot_download(BASE_REPO)
    adapter = snapshot_download(ADAPTER_REPO, revision=ADAPTER_REVISION)
    hf_cache_vol.commit()
    print(f"cached base: {base}\nadapter: {adapter}")


def _local(path: str) -> str:
    return f"http://127.0.0.1:{VLLM_PORT}{path}"


@app.server(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=15 * MINUTES,
    volumes={HF_CACHE: hf_cache_vol},
    secrets=[modal.Secret.from_name("debugger-agent-vllm-key")],
    port=VLLM_PORT,
    target_concurrency=8,
    max_containers=1,  # handful of users; hard cost cap
    unauthenticated=True,  # transport-public; vLLM enforces the API key below
)
class Server:
    @modal.enter()
    def start_vllm(self):
        """Boot vLLM and wait until it answers health checks."""
        import requests
        from huggingface_hub import snapshot_download

        base = snapshot_download(BASE_REPO, local_files_only=True)
        adapter = snapshot_download(
            ADAPTER_REPO, revision=ADAPTER_REVISION, local_files_only=True
        )
        cmd = [
            "vllm",
            "serve",
            base,
            "--served-model-name",
            BASE_REPO,
            "--enable-lora",
            "--lora-modules",
            f"{SERVED_NAME}={adapter}",
            "--max-loras",
            "1",
            # vLLM's smallest supported capacity above the adapter's rank 2.
            "--max-lora-rank",
            "8",
            # Passed as an argv element, so it appears in this container's
            # process list and in vLLM's own startup log. vLLM also honours the
            # VLLM_API_KEY env var (already injected by the Modal secret), which
            # would keep it out of both — verify against the pinned vLLM version
            # before switching, since a silent no-op here removes all auth.
            "--api-key",
            os.environ["VLLM_API_KEY"],
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--dtype",
            "bfloat16",
            "--trust-remote-code",
            # The hermes parser turns <tool_call>{json}</tool_call> output into
            # structured message.tool_calls for debugger_agent.
            "--chat-template",
            CHAT_TEMPLATE_REMOTE,
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--uvicorn-log-level=info",
        ]
        # Redact the key: container logs are retained and readable in the Modal
        # dashboard, so echoing the raw command leaks the bearer token that
        # guards this endpoint. (vLLM also prints it in its own "non-default
        # args" line at INFO — see the note on --api-key above.)
        _secret = os.environ["VLLM_API_KEY"]
        print("launching:", " ".join(
            "***REDACTED***" if part == _secret else part for part in cmd
        ))
        self.process = subprocess.Popen(cmd)

        deadline = time.time() + 14 * MINUTES
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"vllm exited with {self.process.returncode}")
            try:
                if requests.get(_local("/health"), timeout=5).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.time() > deadline:
                raise TimeoutError("vllm did not become healthy")
            time.sleep(5)
        print("vllm ready")
