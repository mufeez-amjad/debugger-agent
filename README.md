# debugger-agent

A post-trained model that debugs failing Python tests with a real `pdb` session.

Accompanying post: [Training an LLM to use a Debugger](https://mufeezamjad.com/blog/training-llm-to-debug).

<br>

- `inference_server/`: Modal + vLLM server
- `harness/`, `common/`: bash, pdb, and file tools
- `debugger_agent/`, `bin/debug`: the CLI

## Model

[`moofeez/qwen3-coder-30b-a3b-debugger`](https://huggingface.co/moofeez/qwen3-coder-30b-a3b-debugger):
a rank-2 LoRA on the MoE experts of
`Qwen/Qwen3-Coder-30B-A3B-Instruct`. vLLM serves the bf16 base model with the
adapter loaded natively as `llm-debugger`.

## Deploy

```bash
uv sync --no-dev --extra serve
modal secret create debugger-agent-vllm-key VLLM_API_KEY=$(openssl rand -hex 32)
modal secret create debugger-agent-hf-token HF_TOKEN=<token>
modal run --detach inference_server/modal_app.py::prepare_model
modal deploy inference_server/modal_app.py
```

## Use

```bash
uv sync --no-dev
export LLM_DEBUGGER_URL=https://<endpoint> VLLM_API_KEY=<key>
cd <project> && /path/to/debugger-agent/bin/debug -p 'test_allocation fails'
```

The agent runs commands and edits files on your machine. See `debug --help`.
