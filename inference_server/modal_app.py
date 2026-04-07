"""Serverless hosting of the debugger model on Modal.

The adapter is a rank-2 LoRA over every MoE expert projection. `prepare_model`
caches the base model and adapter on a Volume. vLLM loads the bf16 base weights
and applies the adapter at inference time.

The server is vLLM's OpenAI-compatible API on one A100-80GB. It scales to zero:
you pay while a request is in flight, plus SCALEDOWN_WINDOW of idle time.

Cold starts use GPU memory snapshots. The first boot starts vLLM, warms it up,
puts the engine to sleep, and Modal checkpoints the container. Later boots
restore the snapshot and wake the engine, which takes seconds. The first boots
after a deploy still take the slow path while Modal records the snapshot.

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
            # exposes /sleep and /wake_up, needed for the snapshot lifecycle
            "VLLM_SERVER_DEV_MODE": "1",
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
    # The recording boot takes about 6.5 minutes in vLLM, then about 10 minutes
    # while Modal writes the snapshot. A kill near the end would loop forever.
    startup_timeout=30 * MINUTES,
    # No Volume for /root/.cache/vllm here: compile artifacts must live in the
    # container filesystem so the GPU snapshot captures them. Mounting them on
    # a Volume breaks snapshot restore (uncommitted 9p paths vanish across
    # boots: "failed to walk .../torch_compile_cache").
    volumes={HF_CACHE: hf_cache_vol},
    secrets=[modal.Secret.from_name("debugger-agent-vllm-key")],
    port=VLLM_PORT,
    target_concurrency=8,
    max_containers=1,  # handful of users; hard cost cap
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    unauthenticated=True,  # transport-public; vLLM enforces the API key below
)
class Server:
    @modal.enter(snap=True)
    def start_vllm_and_sleep(self):
        """Slow path, runs only when no snapshot exists: boot vLLM, warm it up,
        then put the engine to sleep so Modal checkpoints a quiescent state."""
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
            # sleep/wake support; CUDA-graph capture is kept (no --enforce-eager)
            # since the snapshot amortizes it across all future boots
            "--enable-sleep-mode",
            # The HF cache Volume is a 9p mount, which vLLM doesn't recognize
            # as a network FS, so it defaults to sequential shard reads
            # (~0.8 GiB/s, 70 s for the 57 GiB checkpoint). Force the parallel
            # prefetcher it uses on NFS/Lustre.
            "--safetensors-load-strategy=prefetch",
        ]
        # Redact the key: container logs are retained and readable in the Modal
        # dashboard, so echoing the raw command leaks the bearer token that
        # guards this endpoint. (vLLM also prints it in its own "non-default
        # args" line at INFO — see the note on --api-key above.)
        _secret = os.environ["VLLM_API_KEY"]
        print("launching:", " ".join(
            "***REDACTED***" if part == _secret else part for part in cmd
        ))
        # Both models are cached on the Volume, so serve offline. This skips about
        # 20 s of Hub metadata requests per boot. The setting applies to this
        # process only because prepare_model shares the image and needs the Hub.
        self.process = subprocess.Popen(cmd, env={**os.environ, "HF_HUB_OFFLINE": "1"})

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
                raise TimeoutError("vllm did not become healthy before snapshot")
            time.sleep(5)

        # Warm up so tokenizer/kernels/caches are hot inside the snapshot.
        for _ in range(3):
            requests.post(
                _local("/v1/chat/completions"),
                headers={"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"},
                json={
                    "model": SERVED_NAME,
                    "messages": [{"role": "user", "content": "warmup"}],
                    "max_tokens": 8,
                },
                timeout=120,
            ).raise_for_status()

        # Level 1 sleep: KV cache freed, weights offloaded to host RAM — the
        # state Modal captures in the GPU memory snapshot.
        requests.post(_local("/sleep"), params={"level": "1"}, timeout=120).raise_for_status()
        print("vllm asleep; ready for snapshot")

    @modal.enter(snap=False)
    def wake_vllm(self):
        """Fast path, runs on every boot from a restored snapshot."""
        import requests

        requests.post(_local("/wake_up"), timeout=300).raise_for_status()
        requests.get(_local("/health"), timeout=30).raise_for_status()
        print("vllm awake")
