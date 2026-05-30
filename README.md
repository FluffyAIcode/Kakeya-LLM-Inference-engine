# DLM Proposer + AR Verifier — runnable KV-cache-saving framework

[![CI](https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine/actions/workflows/ci.yaml)
[![Release](https://img.shields.io/badge/release-v0.1.0-blue)](https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine/releases/tag/v0.1.0)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey)](docs/local-inference-engine.md)
[![ADRs](https://img.shields.io/badge/ADRs-0001%20%7C%200002%20%7C%200003%20%7C%200006-green)](docs/adr/)

Runs the speculative-decoding architecture designed in the prior product
discussion using **real, public** weights:

| Role     | Model                                                   | Params | Tokenizer    |
| -------- | ------------------------------------------------------- | ------ | ------------ |
| Proposer | [`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`][p]          | 0.75 B | Qwen3 family |
| Verifier | [`Qwen/Qwen3-1.7B`][v] (closest public stand-in for "Qwen 3.6") | 1.72 B | Qwen3 family |

[p]: https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1
[v]: https://huggingface.co/Qwen/Qwen3-1.7B

> **Note on the verifier choice**: at the time of this writing, no public
> "Qwen 3.6" checkpoint exists. We use `Qwen/Qwen3-1.7B` because it is the
> closest publicly-available autoregressive Qwen-3 model that (a) shares the
> proposer's tokenizer (the prompt encodes to identical token ids — verified
> at startup) and (b) is large enough to make KV-cache savings non-trivial.
> Swapping in an actual Qwen 3.5/3.6 checkpoint requires only changing
> `--verifier-id`. Note that Qwen 3.5/3.6's hybrid attention design carries
> KV on only 16/64 layers, so its baseline KV/token would be **smaller** than
> Qwen3-1.7B's 114 KB/token (closer to ~65 KB/token); compression *ratios*
> against that smaller baseline would be correspondingly smaller, but the
> framework code is unchanged.

### Quantized verifiers (4-bit MLX)

Per [ADR 0002](docs/adr/0002-verifier-selection-and-quantization.md), the
engine selects bf16 below 4 B params and 4-bit MLX above. The current
v0.1.0 baseline is `Qwen/Qwen3-1.7B` at bf16; the 4-bit path is opt-in
and works without code changes — just point the verifier id at a
quantized checkpoint:

```bash
# 4-bit Qwen3-1.7B (~1 GB resident vs ~3.4 GB at bf16)
PYTHONPATH=. python3 scripts/chat.py --backend mlx \
    --verifier-id mlx-community/Qwen3-1.7B-4bit

# bf16 vs 4-bit side-by-side bench (writes JSON to results/)
PYTHONPATH=. python3 scripts/bench_mlx_verifier_quant.py
```

Pre-download quantized checkpoints by exporting `KAKEYA_VERIFIER_IDS`
before running setup:

```bash
export KAKEYA_VERIFIER_IDS="mlx-community/Qwen3-1.7B-4bit,mlx-community/Qwen3-8B-4bit"
./scripts/setup_mac.sh
```

`MLXSinkWindowVerifier.quantization` exposes a `QuantizationInfo` record
(bits, group_size, effective bits per parameter, byte breakdown) for any
loaded model; bench scripts and future engine telemetry consume it. See
the module docstring of
[`inference_engine/backends/mlx/quantization.py`](inference_engine/backends/mlx/quantization.py)
for the detection algorithm.

## Memory accounting and what we measure

The metric is **Net Bytes per Token**, defined as:

    Net Bytes per Token (KV-only) =
            verifier_KV_per_token
          + proposer_KV_per_token
          + proposer_weight_bytes / (B * S)

where `B` is concurrent-request batch size and `S` is per-request sequence
length (both at production operating point).

**Activation peak is *not* in Net Bytes per Token.** A transient activation
tensor is allocated when `model(...)` starts, freed when `model(...)`
returns; it does not accumulate across forwards and does not scale
per-session. It is a GPU **capacity constraint** (the forward must fit in
HBM), not a per-token cost. We report it separately.

> ⚠️ **Earlier metric was wrong.** A previous version of `metrics.py`
> amortized `peak_activation / (B * L_block)` into Net Bytes per Token.
> This conflated a transient peak with persistent memory and inflated the
> metric by 30,000+ B/token in the long-context regime, making compression
> appear at 3.5× when it should have been ~600×. The fix is in
> `metrics.py` and the new report shape; the design-stage formula in the
> project notes had the same error and is corrected accordingly.

## Architecture

```
┌──────────────────┐     L tokens      ┌────────────────────────┐
│  DLM Proposer    │ ────────────────► │ AR Verifier            │
│  Qwen3-0.6B-MDLM │                   │ Qwen3-1.7B             │
│  K diffusion     │ ◄──────────────── │ DynamicCache trimmed   │
│  steps / block   │  accept / reject  │ to sink+window slots   │
└──────────────────┘                   └────────────────────────┘
```

* `proposer.py` — masked-diffusion block generator faithful to the model card's reference (low-confidence remasking, deterministic at temperature 0). The proposer in this build re-encodes the full prefix per block; it does **not** maintain a persistent KV cache, so its persistent memory contribution to Net Bytes per Token is zero.
* `verifier.py` — `SinkWindowVerifier` slices each `DynamicCache` layer's K/V tensors after every step; new queries always use the **global** RoPE position (so RoPE on new K/Q is correct), and evicted tokens drop out of attention's view (StreamingLLM-style). Layer-shape invariants raise on mismatch.
* `speculative.py` — greedy speculative-decoding loop with rejection sampling. When `sink + window >= full_seq_len`, output is **bit-equivalent** to greedy AR — verified at runtime; the demo exits with code 2 on mismatch.
* `baseline.py` — reference greedy AR with full `DynamicCache`.
* `metrics.py` — KV byte counting; KV-only Net-Bytes-per-Token formula; capacity-constraint report; projection table to canonical operating points.

## Project layout

```
kv_cache_proposer/
├── proposer.py        # DLM Proposer (masked-diffusion block generator)
├── verifier.py        # AR Verifier with sink+window DynamicCache
├── speculative.py     # Greedy speculative-decoding loop
├── baseline.py        # Reference greedy AR with full DynamicCache
├── metrics.py         # KV byte counting + Net-Bytes-per-Token + projection table
├── run_demo.py        # End-to-end demo + JSON results
└── __init__.py
scripts/
└── smoke_test.py      # Component smoke tests on real weights
results/                # Logs and JSON outputs from runs
requirements.txt
```

## How to run

> **Network requirement**: tests load real Qwen3 weights from the
> HuggingFace cache. The setup scripts (`scripts/setup_mac.sh` /
> `scripts/setup_cuda.sh`) probe `huggingface.co` and download both
> required snapshots (~5 GB total) before tests run. **If you're in
> mainland China or behind a firewall**, set the mirror endpoint
> first:
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```
>
> The setup scripts will then route all downloads through it. If the
> initial connectivity probe fails, the script exits with a clear
> remediation message rather than producing cascading test failures.

```bash
pip install -r requirements.txt
# One-time fix: the dllm-hub modeling file references the broken `dllm`
# package inside an `if __name__ == "__main__":` block; transformers'
# static check_imports flags it. Install a no-op stub at the user's
# site-packages directory (Python-version portable):
python3 -c "import site, os; \
    p = os.path.join(site.getusersitepackages(), 'dllm'); \
    os.makedirs(p, exist_ok=True); \
    open(os.path.join(p, '__init__.py'), 'a').close()"

# Smoke test: tokenizer agreement, model loading, cache invariants
PYTHONPATH=. python3 scripts/smoke_test.py

# Equivalence regime: window >= sequence length => bit-identical to baseline
PYTHONPATH=. python3 -m kv_cache_proposer.run_demo \
    --max-new-tokens 32 \
    --block-size 8 --num-diffusion-steps 8 \
    --sink-size 4 --window-size 64 \
    --batch-size-for-amortization 8 \
    --prompt "Reply with exactly: OK."

# Compression regime: window << sequence => real KV eviction observed
PYTHONPATH=. python3 -m kv_cache_proposer.run_demo \
    --max-new-tokens 64 \
    --block-size 16 --num-diffusion-steps 16 \
    --sink-size 4 --window-size 24 \
    --batch-size-for-amortization 64 \
    --prompt "Write a one-paragraph explanation of why prime numbers are infinite, suitable for a high school student." \
    --results-json results/run_compress.json
```

## Results from the included CPU runs

### 1. Equivalence-regime test (sink+window covers full sequence)

```
prompt  : "Reply with exactly: OK."
config  : sink=4, window=64, block_size=8, K=8

baseline (full KV)        : "OK.<|im_end|>"   (3 tokens, peak KV = 3,584 KB)
speculative (sink+window) : "OK.<|im_end|>"   (3 tokens, peak KV = 3,696 KB)
exact match               : True              <- "no intelligence loss" verified
acceptance rate           : 0.375
```

Self-check passes: `sink+window=68 >= full_seq_len=33`, output bit-identical
to the verifier's own greedy decode. The math of speculative decoding +
no-eviction reduces to "verifier emits its argmax everywhere", exactly
what the baseline computes.

### 2. Compression-regime test (window << sequence)

```
prompt   : "Write a one-paragraph explanation of why prime numbers are infinite ..."
config   : sink=4, window=24, block_size=16, K=16, B=64 (for amortization)
S        : 108 tokens (44 prompt + 64 generated)

Persistent (in Net Bytes per Token):
  verifier KV (full DynamicCache, baseline)     =  12.10 MB total =  114,688 B/token
  verifier KV (sink+window,  speculative)       =   3.06 MB total =   29,734 B/token
                                                                     ── 3.86x verifier-side
  proposer KV                                   =   0 B            (recomputed per block)
  proposer weights amortized at B=64,S=108      = 172,468 B/token  (small-S dominates here)
  Net Bytes per Token (KV-only) at this scale   = 202,202 B/token  (compression 0.57x)

Capacity (separate, NOT counted in Net Bytes per Token):
  proposer peak activation (single forward)     =  31.30 MB
  verifier peak activation (single forward)     =  12.75 MB
```

Net Bytes per Token < baseline only kicks in once `B*S` is large enough
that proposer weights amortize away. The framework reports projected Net
Bytes per Token at canonical operating points using the **empirically
measured per-slot KV** and **actual measured weight bytes** (no
extrapolation beyond reusing the slot constant):

```
  per-slot verifier KV measured = 114,688 B; cache_budget = 28 slots; proposer KV = 0
  --------------------------------------------------------------------------
     B           S     Net Bytes per Token   compression
  --------------------------------------------------------------------------
     1       8,192               145,912.0         0.79x  ← single-request, weights dominate
     8       8,192                18,582.0         6.17x
     8      32,768                 4,645.5        24.69x
     8     131,072                 1,161.4        98.75x
     8   1,048,576                   145.2       790.02x
    32     131,072                   308.7       371.50x
    64     131,072                   166.6       688.36x  ← B=64, S=128k production point
    64   1,048,576                    20.8      5506.92x  ← B=64, S=1M
  --------------------------------------------------------------------------
```

These numbers are consistent with the design analysis: at small `B*S` the
proposer's weight bytes dominate; at large `B*S` the only persistent cost
is the bounded `sink+window` KV (28 slots × 114,688 B = 3.06 MB total,
amortized over `S` tokens → ≈25 B/token at S=128k).

## Honest caveats

1. **Verifier model**: Qwen3-1.7B (28 layers, all carrying KV) stands in
   for the still-unreleased Qwen 3.6 (16 of 64 layers carrying KV). Against
   a real Qwen 3.5/3.6 baseline of ~65 KB/token, the *absolute* compression
   ratios above would be lower by a factor of about 1.75; the framework
   code is unchanged.
2. **Acceptance rate is low (~0.12)**. The proposer was trained with masked
   diffusion on Nemotron-SFT-Code by a different research group; it is *not*
   Repr-Align-aligned to Qwen3-1.7B's representation geometry. With a same-
   family Repr-Align proposer (the design's recommended choice), reported
   acceptance rates are 0.6–0.85. **Low acceptance does not break
   correctness** — it costs throughput, not memory.
3. **Proposer activation memory** is dominated by the dense logits buffer
   (`[1, T, V_vocab]`). The included implementation does not use the standard
   "compute logits only at masked positions" optimization — its peak is
   `T * V * 2` bytes per forward. At long contexts this would not fit in
   HBM and the optimization is mandatory; **the activation peak we report
   is therefore the value of `T * V * 2` at the run's actual context
   length, not a long-context projection**. The capacity number is real for
   what we ran; engineering for S=128k requires the masked-positions
   optimization (a few-line change). The Net-Bytes-per-Token numbers are
   independent of this optimization (activation is not in the metric).
4. **CPU runs**. The repository runs end-to-end on a 4-core, 15 GB-RAM CPU
   environment in tens of seconds. GPU runs would just change wall-clock,
   not byte accounting; the Net-Bytes-per-Token numbers are deterministic
   functions of model shapes and the cache budget.
5. **No fallback**. If anything in the pipeline becomes inconsistent
   (cache layout, tokenizer drift, mask leakage from the proposer) the
   code raises immediately. There is no path that silently degrades to
   "just call the verifier".

## What is and isn't being demonstrated

- **Demonstrated**: KV-cache memory bound is enforced and measured (the
  cache really stays at sink+window=28 slots throughout 108-token
  generation); the speculative loop is greedily distribution-equivalent to
  the verifier (in the equivalence regime); the Net-Bytes-per-Token
  trade-off curve crosses unity at the predicted operating regime.
- **Not demonstrated** (out of scope for a single CPU runnable demo):
  multi-target verifier routing (Qwen / Gemma / DeepSeek), session-affinity
  scheduling, OTA, federated self-learning. Those are platform-level
  components from the design discussion that need separate plumbing.

## Where this is going — local inference engine

The next layer up is a Mac/Ubuntu local inference engine that wraps the
algorithmic core in this repo with continuous batching, async
proposer/verifier pipelining, NF4 KV quantization, and a fixed-slab
KV pool sized for sink+window. Architecture and phased build plan are
in [`docs/local-inference-engine.md`](docs/local-inference-engine.md).

Short version of why the engine **does not use PagedAttention**: the
sink+window invariant turns each session's KV cache into a constant-size
object, so all three problems PagedAttention solves (fragmentation,
prefix sharing, non-contiguous KV) cease to apply. A 30-line fixed-slab
pool replaces it and runs ~5–15% faster because attention kernels see
contiguous memory.

## HTTP serving (E2 + E4 integrated)

The engine exposes an OpenAI-compatible HTTP API over Server-Sent
Events. Every request flows through the
[continuous batching scheduler](inference_engine/scheduler/) for
admission control, fair queuing, and per-session lifecycle tracking.

Routes:

- `GET /healthz` — liveness probe
- `GET /v1/models` — model list
- `POST /v1/chat/completions` — chat completions, streaming or JSON

```bash
# launch the server (MLX backend, bf16 verifier, single-user mode)
PYTHONPATH=. python3 scripts/serve.py --backend mlx \
    --verifier-id Qwen/Qwen3-1.7B \
    --host 127.0.0.1 --port 8000

# multi-user mode: 4 concurrent sessions with FIFO queue
PYTHONPATH=. python3 scripts/serve.py --backend mlx \
    --max-concurrent 4 --admission-policy queue --queue-max-wait-s 30

# non-streaming chat completion
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{
        "model": "kakeya-v1",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": false
    }'

# streaming (SSE)
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{
        "model": "kakeya-v1",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true
    }'
```

When the scheduler's slab pool is full under the default `reject`
policy, new requests get HTTP **429 Too Many Requests** immediately.
Switch to `--admission-policy queue` to make them wait up to
`--queue-max-wait-s` seconds for a slot.

### Observability

`GET /metrics` exposes Prometheus-compatible metrics on every running
server instance. The scrape includes:

- HTTP traffic (`http_requests_total{method,path,status}`,
  `http_request_duration_seconds`)
- Inference outcomes (`inference_completions_total{finish_reason}`,
  `inference_completion_tokens`, `inference_acceptance_rate`)
- Scheduler state (`scheduler_active_sessions`, `scheduler_pool_in_use`,
  `scheduler_pool_total`, `scheduler_pending`, `scheduler_admission_total`)

```bash
curl -sS http://127.0.0.1:8000/metrics | head -20
```

### Authentication

API-key auth is opt-in. Pass `--api-key` one or more times (or set
the CSV env var `KAKEYA_API_KEYS`) and every `/v1/*` route requires
a matching `Authorization: Bearer <key>` header. The public routes
`/healthz` and `/metrics` always remain unauthenticated so liveness
probes and Prometheus scrapers don't need credentials.

```bash
PYTHONPATH=. python3 scripts/serve.py --backend mlx \
    --api-key sk-test-abc --api-key sk-prod-xyz

curl -sS http://127.0.0.1:8000/v1/chat/completions \
    -H 'authorization: Bearer sk-test-abc' \
    -H 'content-type: application/json' \
    -d '{"model":"kakeya-v1","messages":[{"role":"user","content":"hi"}]}'
```

Errors follow the OpenAI envelope (`{"error":{"message","type","code","param"}}`)
so off-the-shelf OpenAI clients surface meaningful messages.

The streaming response follows the OpenAI chunk schema and terminates
with the literal `data: [DONE]` sentinel that the OpenAI Python and
JavaScript SDKs recognize. Sampling parameters (`temperature`,
`top_p`, `stop`) are accepted in the request body for OpenAI-client
compatibility but **not applied** — the underlying decoder is greedy
temperature-0 by design ([ADR 0001 §2.2](docs/adr/0001-proposer-sizing-and-alignment.md)).

The FastAPI lifespan handler calls `scheduler.shutdown()` on
SIGTERM / SIGINT, which cancels active sessions, rejects queued
admissions, and releases all slabs before the process exits.

Configuration is via env vars (all prefixed `KAKEYA_*`): see the
docstring of [`inference_engine/server/config.py`](inference_engine/server/config.py).

## Docker

A CPU-only Docker image is published from the [`Dockerfile`](Dockerfile)
at the repo root. Apple Silicon contributors should run locally
(MLX does not work in Linux containers); CUDA users should mount
their NVIDIA toolkit and use a separate `Dockerfile.cuda` (deferred
to a future PR per ADR 0002).

```bash
# build
docker build -t kakeya:latest .

# run with HF cache mounted so weights persist across restarts
docker run --rm -p 8000:8000 \
    -v "$HOME/.cache/huggingface:/home/kakeya/.cache/huggingface" \
    kakeya:latest

# pass extra serve.py flags after the image name
docker run --rm -p 8000:8000 \
    -v "$HOME/.cache/huggingface:/home/kakeya/.cache/huggingface" \
    kakeya:latest \
    --max-concurrent 4 --admission-policy queue --queue-max-wait-s 30
```

The image runs as non-root user `kakeya` (uid 10001), exposes
port 8000, and registers `/healthz` as a Docker `HEALTHCHECK` so
orchestrators detect a hung process without parsing logs. CI builds
the image on every push to verify the Dockerfile stays buildable.

## Continuous integration

Every push to `main` and every PR runs the platform-neutral test
suite on GitHub Actions ([`.github/workflows/ci.yaml`](.github/workflows/ci.yaml)),
enforcing **100% line coverage** on the shipping library modules:

```
inference_engine.server  inference_engine.memory  inference_engine.scheduler
inference_engine.pipeline  inference_engine.proposer  training.repr_align
```

Tests that need real Qwen3 weights (`tests/core/`, `tests/system/`,
`tests/inference_engine/proposer/`) are run locally on hosts with the
HuggingFace cache populated; backend-specific suites
(`tests/backends/mlx/test_{verifier,proposer,cache,torch_bridge}.py`)
run on Apple Silicon contributors' machines via
`scripts/run_platform_tests.sh --backend mlx`. The CI workflow
guards the platform-neutral surface so a regression there cannot
land on `main`.

## Architecture Decision Records

Design decisions that the rest of the codebase depends on are recorded
in [`docs/adr/`](docs/adr/). New contributors and agents should read the
ADR index before changing proposer / verifier / training code; the ADRs
explain *why* a particular design was chosen and which alternatives were
explicitly rejected.

- [ADR 0001 — Proposer sizing, alignment strategy, and verifier
  decoupling](docs/adr/0001-proposer-sizing-and-alignment.md): the
  load-bearing decision behind why we keep the proposer in a fixed
  0.25–1 B band, treat EAGLE-3 representation alignment as the canonical
  training recipe, and design verifier swaps to be data-and-fine-tune
  operations rather than re-architecture operations.
- [ADR 0002 — Verifier selection, quantization, and the
  open-vs-closed-weight constraint](docs/adr/0002-verifier-selection-and-quantization.md):
  the v1/v2 ship sequence (Qwen3-1.7B bf16 → Qwen3-8B 4-bit), the 60 %
  memory rule for choosing bf16 vs 4-bit, and why closed-weight APIs
  (GPT/Claude/Gemini) cannot be aligned with EAGLE-3 and are out of
  scope for v1 / v2.
- [ADR 0003 — Verifier ↔ slab pool integration: deferred refactor +
  intermediate step](docs/adr/0003-verifier-slab-pool-integration.md):
  why the full "slab tensors hold the real KV" refactor is deferred
  to v0.3 (correctness fragility without a bit-equivalence harness)
  and what intermediate step ships in v0.2 — `PooledVerifier`
  wrapper that makes pool memory accounting accurate without
  touching the model forward.
- [ADR 0006 — Project positioning as local agent
  infrastructure](docs/adr/0006-local-agent-infrastructure-positioning.md):
  the strategic positioning decision that Kakeya is **local agent
  infrastructure for Mac**, not a generic chat-acceleration engine.
  Reframes v0.3+ release notes around multi-agent / long-session /
  personalized usage, commits to shipping `docs/integrations/`
  examples (LangChain / CrewAI / AutoGen / Cursor) and an agentic
  benchmark suite (`scripts/bench_agentic/`), and explicitly declines
  to compete with llama.cpp on chat speedup or with vLLM on
  data-center serving.
