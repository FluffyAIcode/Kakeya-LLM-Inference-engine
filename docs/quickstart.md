# Kakeya Quickstart — 10-minute walkthrough

This guide takes you from "fresh checkout" to "first generated token via
the gRPC SDK". Targets v0.3.0; PyPI / npm / GHCR-image paths are flagged
where they apply (and where they don't, yet).

## Time budget

| Step | Mac M4 (warm cache) | Mac M4 (cold) | Linux x86 CPU |
| --- | --- | --- | --- |
| Clone + checkout | <30 s | <30 s | <30 s |
| Setup script + dependencies | ~1 min | ~5 min | ~2 min |
| HF cache warm (Qwen3-0.6B, ~1.2 GB) | already done | ~5-10 min | ~2-5 min |
| Start gRPC server | ~3 s | ~3 s | ~5 s |
| First SDK call | <1 s | <1 s | <1 s |
| **Total** | **~2 min** | **~10-15 min** | **~5 min** |

## Step 1 — Clone + checkout v0.3.0

```bash
git clone https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine
cd Kakeya-LLM-Inference-engine
git checkout v0.3.0   # pinned to GA tag
```

Why pin to a tag: `main` may carry in-flight v0.4 work; the v0.3.0 tag is
the binding release point.

## Step 2 — Install dependencies

### Mac M4 (recommended primary platform)

```bash
bash scripts/setup_mac.sh
```

What this does:

1. Creates `.venv-mac/` with Python 3.12+ from Homebrew or system.
2. Installs `requirements.txt` (`torch>=2.4`, `transformers>=4.45,<5.0`,
   `mlx`, `grpcio`, `fastapi`, `pytest` family).
3. Probes `huggingface.co` connectivity. If you're behind a firewall or
   in mainland China, set the mirror **before** running setup:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   bash scripts/setup_mac.sh
   ```
4. Pre-downloads the v0.3 default verifier (`Qwen/Qwen3-0.6B`, ~1.2 GB
   bf16). For larger / quantized variants, set
   `KAKEYA_VERIFIER_IDS=mlx-community/Qwen3-1.7B-4bit,...` before
   running setup.

After setup, every subsequent shell needs `source .venv-mac/bin/activate`
or the script's helper `source scripts/activate_mac.sh`.

### Linux x86 CPU

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Pre-warm Qwen3-0.6B
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
"
```

### Linux x86 with NVIDIA GPU

```bash
bash scripts/setup_cuda.sh
```

Same steps as Mac, plus a `torch` build matching your CUDA toolkit.

### Common pitfalls

- **`ModuleNotFoundError: dllm`**: the legacy diffusion proposer (v0.2)
  references a `dllm` package that v0.3 doesn't need but transformers'
  static imports flag. Fix:
  ```bash
  python3 -c "import site, os; \
      p = os.path.join(site.getusersitepackages(), 'dllm'); \
      os.makedirs(p, exist_ok=True); \
      open(os.path.join(p, '__init__.py'), 'a').close()"
  ```
  `setup_mac.sh` and `setup_cuda.sh` do this automatically.
- **Connection refused to `huggingface.co`**: set `HF_ENDPOINT` before
  setup (see above) or pre-warm offline.

## Step 3 — Start the gRPC runtime server

In one terminal:

```bash
PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \
    --backend cpu \
    --verifier-id Qwen/Qwen3-0.6B \
    --bind 127.0.0.1:50051 \
    --capacity 1 \
    --sink 4 --window 64
```

Expected output:

```
[grpc-server] loading verifier backend=cpu id=Qwen/Qwen3-0.6B sink=4 window=64
[grpc-server] verifier dims: layers=28 kv_heads=8 head_dim=128 capacity=68
[grpc-server] kakeya gRPC RuntimeService listening on 127.0.0.1:50051
```

Flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--backend` | `cpu` | `cpu` (Linux/Mac) or `mlx` (Apple Silicon only — uses Metal acceleration). |
| `--verifier-id` | `Qwen/Qwen3-0.6B` | Any HF model id with a Qwen3-family tokenizer. Larger = slower; 0.6B is enough for development. |
| `--bind` | `127.0.0.1:50051` | Loopback by default for security. Set `--bind 0.0.0.0:50051` to expose to other hosts (and run behind a real reverse proxy). |
| `--capacity` | `4` | Max concurrent sessions. Each session reserves one slab worth of KV bookkeeping. |
| `--sink` | `4` | Sink-token KV cache size per session ([ADR 0001](adr/0001-proposer-sizing-and-alignment.md)). |
| `--window` | `64` | Sliding-window KV cache size per session. Total per-session KV bound: `(sink + window) * num_layers * num_kv_heads * head_dim * 2 (K+V) * dtype_bytes`. |
| `--max-concurrent-rpcs` | unset | Cap on simultaneous in-flight RPCs at the gRPC layer. Defaults to grpc.aio's default. |

## Step 4 — Talk to the runtime via the Python SDK

In another terminal:

```bash
PYTHONPATH=.:sdks/python python3 - <<'PY'
from kakeya import Client

with Client("127.0.0.1:50051") as client:
    with client.create_session() as session:
        # Prefill with a synthetic short token sequence.
        session.append([1, 2, 3, 4, 5])

        # Stream up to 16 generated tokens.
        emitted = []
        for token_id in session.generate(max_tokens=16):
            emitted.append(token_id)
        print("emitted token ids:", emitted)

        # Inspect server-side state.
        info = session.info()
        print(f"history_length={info.history_length}")
        print(f"kv_live_bytes={info.kv_live_bytes}")
        print(f"idle_seconds={info.idle_seconds:.3f}")
PY
```

For real text input, encode via the verifier's tokenizer:

```bash
PYTHONPATH=.:sdks/python python3 - <<'PY'
from kakeya import Client
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
prompt_ids = tok.apply_chat_template(
    [{"role": "user", "content": "Reply with exactly: OK."}],
    add_generation_prompt=True, tokenize=True, return_dict=False,
    enable_thinking=False,
)

with Client("127.0.0.1:50051") as client:
    eos = [int(tok.eos_token_id)] if tok.eos_token_id else []
    with client.create_session(eos_token_ids=eos) as session:
        session.append(prompt_ids)
        emitted = list(session.generate(max_tokens=32))
        print("response:", tok.decode(emitted, skip_special_tokens=True))
PY
```

## Step 5 — Multi-turn conversation (where session-bound runtime shines)

The point of v0.3's session-bound architecture: each turn only sends the
**new** user message; the server keeps the running KV cache.

```python
from kakeya import Client
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
eos = [int(tok.eos_token_id)] if tok.eos_token_id else []

with Client("127.0.0.1:50051") as client:
    with client.create_session(eos_token_ids=eos) as session:
        for user_message in [
            "Hi.",
            "What's your name?",
            "Tell me a one-line joke.",
        ]:
            new_tokens = tok.encode(user_message, add_special_tokens=False)
            session.append(new_tokens)              # O(new_message), NOT O(history)
            response_ids = list(session.generate(max_tokens=64))
            print(">", user_message)
            print("<", tok.decode(response_ids, skip_special_tokens=True))
            print()
```

This is the architectural property the 4-h Mac M4 bench validates: 9 ms
latency drift over 14400 s, 480 turns. Compare to the deprecated HTTP
shim where every turn re-prefills the full conversation.

## Step 6 — Stop the runtime

`Ctrl-C` in the gRPC server's terminal triggers a graceful shutdown
(default 5 s grace for in-flight RPCs). To stop programmatically from
another process:

```bash
# Find the PID and SIGTERM it
pkill -TERM -f start_grpc_runtime_server
```

The server logs `kakeya gRPC RuntimeService stopped cleanly` on a clean
exit.

## Troubleshooting

### `kv_live_bytes` reports 0 after generating

Confirm you're on v0.3.0 (or `main` post-`6399546`). Pre-PR-E1c the
`kv_live_bytes` field was always 0 because the slab placeholder didn't
sync from the verifier. PR-E1c fixed this; v0.3.0 reports real bytes.

### `ResourceExhaustedError` on `create_session`

Hit the server's `--capacity` limit. Either (a) increase capacity if
your hardware allows, or (b) close completed sessions promptly. The
SDK's `Session` context manager auto-closes on `__exit__`.

### `SessionNotFoundError` mid-conversation

Server-side LRU / TTL eviction kicked in. Either bump `--capacity`
or tune the eviction policy (advanced — see
[`inference_engine.session.store`](../inference_engine/session/store.py)).

### Different model

Any HF model with a Qwen3-family tokenizer works. For other tokenizers,
you'd also need to adjust the chat template and EOS id resolution. The
plumbing supports it; v0.3 ships tested against Qwen3 0.6B / 1.7B.

```bash
# bf16 Qwen3-1.7B (CPU; ~3.4 GB resident)
python3 scripts/start_grpc_runtime_server.py \
    --backend cpu --verifier-id Qwen/Qwen3-1.7B \
    --bind 127.0.0.1:50051 --capacity 1 --sink 4 --window 64

# 4-bit MLX Qwen3-1.7B (Apple Silicon only; ~1 GB resident)
python3 scripts/start_grpc_runtime_server.py \
    --backend mlx --verifier-id mlx-community/Qwen3-1.7B-4bit \
    --bind 127.0.0.1:50051 --capacity 2 --sink 4 --window 128
```

### Authentication / API keys

The gRPC server v0.3 binds to loopback by default and assumes trusted
local clients. For exposed deployments, terminate TLS + auth at a real
reverse proxy (nginx, envoy, traefik) in front of the gRPC server. The
deprecated HTTP shim has Bearer-token auth via `--api-key` flags;
v0.4 will surface gRPC-native auth.

### Mainland-China networking

Set `HF_ENDPOINT=https://hf-mirror.com` before any `pip install` or
HF download command. The setup scripts route everything through it
when the variable is set.

## Next steps

- **Build an agent on top.** The Python SDK's `Session` is OpenAI-style
  but session-bound. Plug it into LangChain / CrewAI / AutoGen as a
  custom LLM client. Examples queued in `docs/integrations/`.
- **Run the long-session bench yourself.**
  ```bash
  PYTHONPATH=.:sdks/python python3 \
      scripts/bench_agentic/bench_session_long_run.py \
      --grpc-address 127.0.0.1:50051 \
      --tokenizer-id Qwen/Qwen3-0.6B \
      --duration-s 1800 --turn-spacing-s 30 \
      --output results/platform-tests/my_smoke_$(date +%s).json
  ```
  30-minute smoke; 14400 s for the full 4-h GA-gate evidence run.
- **Read [ADR 0008](adr/0008-session-bound-runtime-and-grpc-protocol.md)**
  to understand the architectural decisions. The session-bound contract
  is the load-bearing invariant; everything else follows from it.

## Common patterns

### Reset a long-running session

The cheap way: close + create new.

```python
with client.create_session() as session:
    # ... 100 turns ...
    pass   # __exit__ closes

with client.create_session() as session:
    # fresh K/V cache
    pass
```

### Keep one session for the whole process

```python
client = Client("127.0.0.1:50051")
session = client.create_session()
try:
    while user_input := get_next_user_message():
        new_tokens = tokenizer.encode(user_input)
        session.append(new_tokens)
        for tok in session.generate(max_tokens=64):
            print(tokenizer.decode([tok], skip_special_tokens=True), end="", flush=True)
        print()
finally:
    session.close()
    client.close()
```

### Use the deprecated HTTP shim (not recommended)

```bash
PYTHONPATH=.:sdks/python python3 scripts/serve.py \
    --backend cpu --verifier-id Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 8000

curl -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"any","messages":[{"role":"user","content":"hi"}]}'
```

The HTTP shim returns `Deprecation: true` + `Sunset` headers in every
response. New deployments should migrate to gRPC for v0.3's full perf
story (see [ADR 0008 §2.7](adr/0008-session-bound-runtime-and-grpc-protocol.md)).

## What's next for v0.3.x and v0.4

| Coming in | What |
| --- | --- |
| **v0.3.1** | `pip install kakeya-inference` + `pip install kakeya` (Python SDK), `npm install @kakeya/runtime`, `docker pull ghcr.io/fluffyaicode/kakeya:0.3.1`, `kakeya prewarm` + `kakeya chat` CLIs |
| **v0.4** | Speculative decoding restored (proposer-back-in), alignment training (ADR 0004 Stages 2-4), cross-request KV reuse on gRPC |

If you want a bug filed or a feature requested, open an issue at
[github.com/FluffyAIcode/Kakeya-LLM-Inference-engine/issues](https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine/issues).
