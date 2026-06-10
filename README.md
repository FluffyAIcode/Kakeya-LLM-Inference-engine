# Kakeya — local agent runtime with session-bound gRPC

[![CI](https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine/actions/workflows/ci.yaml)
[![Release](https://img.shields.io/badge/release-v0.3.0-blue)](https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine/releases/tag/v0.3.0)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20%7C%20Linux--CPU-lightgrey)](docs/quickstart.md)
[![Architecture](https://img.shields.io/badge/architecture-ADR%200008-green)](docs/adr/0008-session-bound-runtime-and-grpc-protocol.md)
[![License](https://img.shields.io/badge/license-MIT-lightblue)](LICENSE)

Kakeya is a **local agent runtime**: a long-running inference server that
holds session state on the server side, exposes a gRPC `RuntimeService`
(plus a deprecated OpenAI-compatible HTTP shim), and bounds memory + per-
turn latency on long conversations.

The v0.3 architectural arc landed in June 2026 ([ADR 0008](docs/adr/0008-session-bound-runtime-and-grpc-protocol.md));
the headline result is **9 ms latency drift over a 4-hour, 480-turn run on
Mac M4 24 GB** — vs +39.74 s on the previous (HTTP-only) architecture, a
4400× improvement.

```
┌────────────────────┐    gRPC bidi-stream     ┌────────────────────────┐
│  Your Python /     │ ─────────────────────►  │  Kakeya Runtime        │
│  TypeScript SDK    │   AppendTokens          │  ┌──────────────────┐  │
│                    │   Generate              │  │ SessionStore +   │  │
│  Holds session_id, │   GetSessionInfo        │  │ AppendTokens     │  │
│  retries, errors   │   CloseSession          │  │ Coordinator +    │  │
│                    │ ◄─────────────────────  │  │ Generation       │  │
└────────────────────┘    TokenEvent stream    │  │ Coordinator      │  │
                                               │  └────────┬─────────┘  │
                                               │           ▼            │
                                               │  ┌──────────────────┐  │
                                               │  │ SinkWindow       │  │
                                               │  │ Verifier         │  │
                                               │  │ (Qwen3-0.6B,     │  │
                                               │  │  CPU / MLX)      │  │
                                               │  └──────────────────┘  │
                                               └────────────────────────┘
```

## Quickstart (5 minutes on Mac M4 / Linux x86)

> **Status — v0.3.0 GA.** PyPI + npm + GHCR Docker image are queued for
> v0.3.1; today the runtime ships from source. The flow below works on
> a clean checkout against the v0.3.0 tag.

```bash
# 1. Clone + check out v0.3.0
git clone https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine
cd Kakeya-LLM-Inference-engine
git checkout v0.3.0

# 2. Install (Mac M4 -- handles HF cache + dependencies + venv)
bash scripts/setup_mac.sh
#    Linux x86 with NVIDIA: bash scripts/setup_cuda.sh
#    Linux x86 CPU-only:    pip install -r requirements.txt

# 3. Start the gRPC runtime in one terminal
PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \
    --backend cpu --verifier-id Qwen/Qwen3-0.6B \
    --bind 127.0.0.1:50051 --capacity 1 --sink 4 --window 64

# 4. In another terminal, talk to it via the Python SDK
PYTHONPATH=.:sdks/python python3 - <<'PY'
from kakeya import Client

with Client("127.0.0.1:50051") as client:
    with client.create_session() as session:
        session.append([1, 2, 3, 4, 5])
        for token_id in session.generate(max_tokens=16):
            print(token_id, end=" ", flush=True)
        print()
        info = session.info()
        print(f"history={info.history_length} kv_live={info.kv_live_bytes}B")
PY
```

For a full 10-minute walkthrough — Mac vs Linux setup, troubleshooting,
HuggingFace cache pre-warm, mainland-China mirror routing, gRPC SDK
patterns — see [`docs/quickstart.md`](docs/quickstart.md).

## What's in the v0.3 architecture

| Component | What it does | Where |
| --- | --- | --- |
| `RuntimeService` (gRPC) | `CreateSession` / `AppendTokens` / `Generate` (server-streaming) / `GetSessionInfo` / `CloseSession`. Wire-stable; protobuf in [`proto/kakeya/v1/runtime.proto`](proto/kakeya/v1/runtime.proto). | `inference_engine.server.grpc_app` |
| `SessionStore` | In-memory session registry, server-issued IDs, append-only history, INV-1 / INV-2 enforcement, slab pool ownership. | `inference_engine.session.store` |
| `AppendTokensCoordinator` | Drives the verifier through `prefill` (cold) or `forward_block` + `commit_or_truncate` (incremental). Mirrors verifier state into the session. | `inference_engine.session.coordinator` |
| `GenerationCoordinator` | Greedy session-aware decode. Yields `TokenEvent` / `HistoryTruncatedEvent` / `DoneEvent`. | `inference_engine.session.generator` |
| `SinkWindowVerifier` | Real Qwen3 (0.6B / 1.7B / 4-bit MLX). Sink+window K/V trim per ADR 0001 / 0002. | `kv_cache_proposer.verifier` (CPU), `inference_engine.backends.mlx.verifier` (Apple Silicon) |
| Python SDK | `kakeya.Client`, `kakeya.Session`. Sync gRPC. | [`sdks/python/kakeya/`](sdks/python/kakeya/) |
| TypeScript SDK | `@kakeya/runtime` `Client`, `Session`. Node 20+ via `@grpc/grpc-js`. | [`sdks/typescript/`](sdks/typescript/) |
| HTTP shim (deprecated) | OpenAI-compatible `/v1/chat/completions`. Pure-AR (no speculative decoding); single-shot session per request. `Deprecation` + `Sunset` headers. | `inference_engine.server.app` |

## v0.3 GA evidence

The integration suite under [`tests/integration/`](tests/integration/) is
the binding correctness gate. Mac M4 evidence on `main`:

| Gate | Metric | Result |
| --- | --- | --- |
| Memory bounded ([ADR 0006 §2.3.a](docs/adr/0006-local-agent-infrastructure-positioning.md)) | `agg.kv_bounded` | True |
| **Prefill bounded** ([ADR 0008 §7 G2](docs/adr/0008-session-bound-runtime-and-grpc-protocol.md)) | `latency_drift_p50_s` over 14400 s | **+0.0093 s** (vs +39.74 s on v0.2.0 HTTP shim — **4400×** improvement) |
| INV-3 byte-exact greedy decoding ([ADR 0008 §7 G3](docs/adr/0008-session-bound-runtime-and-grpc-protocol.md)) | All chunkings produce identical token streams | Pass |
| Throughput | Turns over 4 h | 480 (vs 58 on v0.2.0 — 8.3× more sustained) |
| Latency | p50 / p95 over 4 h | 1.829 s / 1.853 s |

Raw artifacts: [`results/platform-tests/bench_session_4h_1780332893.json`](results/platform-tests/) (4-h evidence) and the v0.3.0 GA tag's smoke run committed at `6399546`.

## SDKs

### Python — `sdks/python/kakeya`

```python
from kakeya import Client

with Client("127.0.0.1:50051") as client:
    with client.create_session(eos_token_ids=[151645]) as session:
        # Tokenize the new user message ONLY (the session keeps history)
        new_tokens = my_tokenizer.encode("hi")
        session.append(new_tokens)

        # Stream generated tokens
        for token_id in session.generate(max_tokens=64):
            ...
```

Typed exception surface: `KakeyaError` base; `SessionNotFoundError`,
`InvalidArgumentError`, `InvariantViolationError`, `ResourceExhaustedError`,
`UnimplementedError`, `RpcCancelledError`, `SessionClosedError`. Errors map
1:1 from gRPC status codes per
[ADR 0008 §2.10](docs/adr/0008-session-bound-runtime-and-grpc-protocol.md).

### TypeScript — `sdks/typescript/@kakeya/runtime`

```typescript
import { Client } from "@kakeya/runtime";

const client = new Client("127.0.0.1:50051");
const session = await client.createSession();
await session.append([1, 2, 3]);
for await (const tokenId of session.generate({ maxTokens: 16 })) {
  console.log(tokenId);
}
await session.close();
```

Built for Node.js 20+ / Electron / Bun (not browser — uses `@grpc/grpc-js`
not gRPC-Web). Browser clients can use the deprecated HTTP shim while
gRPC-Web support is queued.

### Why session-bound matters

The previous (HTTP-only) architecture re-prefilled the entire conversation
on every turn; latency grew linearly with history. The session-bound model
caches `(history_token_ids, K, V)` server-side, so each `AppendTokens` call
processes only the **new tokens you send**. The gRPC SDKs preserve this
property end-to-end:

```python
with client.create_session() as session:
    session.append(tokens_turn_1)              # O(turn_1) prefill
    list(session.generate(max_tokens=64))
    session.append(tokens_turn_2_new_only)     # O(turn_2) prefill, NOT O(turn_1+turn_2)
    list(session.generate(max_tokens=64))
```

That's where the 4400× latency-drift improvement comes from.

## Multi-host: capability exchange + distributed spec decode (v0.5-M1)

Per [ADR 0009](docs/adr/0009-mlx-distributed-spec-decode-and-capability-exchange.md),
Kakeya nodes on one LAN (Mac minis, plus Linux CPU hosts) can now
**gossip capability cards** — which models each node has warmed, in
which role (verifier / proposer), with how much unified memory — and
**trade work**: an AR verifier on one node drives speculative decoding
with draft blocks served by a proposer on another node. The greedy
accept rule runs locally and is unchanged, so remote drafts can change
throughput but never tokens.

```bash
# Node B (proposer host)
PYTHONPATH=. python3 scripts/demo_distributed_spec_decode.py \
    --role proposer-node --bind 0.0.0.0:50061 --node-id node-b

# Node A (verifier host) — discovers B, plans placement, decodes
PYTHONPATH=. python3 scripts/demo_distributed_spec_decode.py \
    --role verifier-node --bind 0.0.0.0:50060 --node-id node-a \
    --peer <node-b-ip>:50061 --verifier-id Qwen/Qwen3-0.6B
```

The production runtime joins a fleet with the same flags on
`scripts/start_grpc_runtime_server.py` (`--node-id`, `--peer`,
`--serve-ngram-proposer`). `mlx.distributed` rings are advertised on
capability cards (`ring_address`) as the bulk-tensor data plane for
the K3 hidden-state flows; the control plane is pure gRPC and needs no
MLX. Design details: [agent capability exchange platform](docs/design/agent-capability-exchange-platform.md).

## Deprecated HTTP shim

The OpenAI-compatible HTTP API at `/v1/chat/completions` still works for
backward compatibility but is **feature-frozen** per
[ADR 0008 §2.7](docs/adr/0008-session-bound-runtime-and-grpc-protocol.md):

- Every response carries `Deprecation: true` + `Sunset` + a `Link` header
  pointing to ADR 0008.
- Each request becomes a single-shot session (no session reuse across
  requests on the HTTP path).
- Speculative decoding is **not** applied; pure AR. Migrate to gRPC for
  v0.3's full perf story.

```bash
# Deprecated — only when you really need OpenAI-API compat
PYTHONPATH=.:sdks/python python3 scripts/serve.py \
    --backend cpu --verifier-id Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 8000

curl -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"any","messages":[{"role":"user","content":"hi"}]}'
```

For a curl-friendly path with the v0.3 architecture's full speed, use a
Python or TypeScript SDK against the gRPC server.

---

# Architecture & Background

## How speculative decoding worked in v0.2

> **Note**: in v0.3, speculative decoding is queued for the v0.4
> proposer-back-in PR ([roadmap](#roadmap)). The v0.2 architecture is
> documented here for historical context and because most of the
> mathematical analysis still applies.

v0.2 ran a DLM (diffusion language model) proposer in front of an AR
verifier with a sink+window KV cache:

```
┌──────────────────┐     L tokens      ┌────────────────────────┐
│  DLM Proposer    │ ────────────────► │ AR Verifier            │
│  Qwen3-0.6B-MDLM │                   │ Qwen3-1.7B             │
│  K diffusion     │ ◄──────────────── │ DynamicCache trimmed   │
│  steps / block   │  accept / reject  │ to sink+window slots   │
└──────────────────┘                   └────────────────────────┘
```

Memory accounting metric: **Net Bytes per Token** (KV-only):

```
Net Bytes per Token = verifier_KV_per_token
                    + proposer_KV_per_token
                    + proposer_weight_bytes / (B * S)
```

where `B` is concurrent-request batch size and `S` is per-request sequence
length. Activation peak is **not** in Net Bytes per Token (it's a transient
GPU capacity constraint, not a per-token cost).

### Compression-regime measurements (CPU)

```
prompt   : "Write a one-paragraph explanation of why prime numbers are infinite ..."
config   : sink=4, window=24, block_size=16, K=16, B=64 (for amortization)
S        : 108 tokens (44 prompt + 64 generated)

  per-slot verifier KV measured = 114,688 B; cache_budget = 28 slots; proposer KV = 0
  --------------------------------------------------------------------------
     B           S     Net Bytes per Token   compression
  --------------------------------------------------------------------------
     1       8,192               145,912.0         0.79x  ← single-request, weights dominate
     8       8,192                18,582.0         6.17x
     8     131,072                 1,161.4        98.75x
    64     131,072                   166.6       688.36x  ← B=64, S=128k production point
    64   1,048,576                    20.8      5506.92x  ← B=64, S=1M
  --------------------------------------------------------------------------
```

These numbers are deterministic functions of model shapes and the cache
budget. At small B×S the proposer's weight bytes dominate; at large B×S
the only persistent cost is the bounded `sink+window` KV.

### Quantized verifiers (4-bit MLX, Apple Silicon)

Per [ADR 0002](docs/adr/0002-verifier-selection-and-quantization.md), the
engine selects bf16 below 4 B params and 4-bit MLX above:

```bash
# 4-bit Qwen3-1.7B (~1 GB resident vs ~3.4 GB at bf16)
PYTHONPATH=. python3 scripts/chat.py --backend mlx \
    --verifier-id mlx-community/Qwen3-1.7B-4bit
```

`MLXSinkWindowVerifier.quantization` exposes a `QuantizationInfo` record
(bits, group_size, effective bits per parameter, byte breakdown) for any
loaded model.

## Project layout

```
inference_engine/
├── server/             # gRPC + (deprecated) HTTP shim, FastAPI app
│   ├── grpc_app.py     # RuntimeService implementation
│   ├── app.py          # OpenAI HTTP shim (deprecated, single-shot session)
│   └── proto_gen/      # Generated Python protobuf stubs
├── session/            # Session-bound runtime
│   ├── store.py        # SessionStore (server-issued IDs, INV-1/2)
│   ├── coordinator.py  # AppendTokensCoordinator
│   └── generator.py    # GenerationCoordinator
├── memory/             # SlabPool + KVSlab (session-bookkeeping placeholders)
├── scheduler/          # Pre-v0.3 admission scheduler (still used by HTTP shim path)
├── pipeline/           # Producer/consumer abstractions
└── backends/mlx/       # Apple Silicon verifier
kv_cache_proposer/      # CPU verifier + (legacy) proposer
sdks/
├── python/kakeya/      # Python SDK
└── typescript/         # @kakeya/runtime
proto/kakeya/v1/        # Protobuf source-of-truth
tests/
├── integration/        # Real Qwen3-0.6B; Mac M4 GA gate
├── inference_engine/   # Linux CI gate (verifier-independent)
└── sdk/python/         # SDK error-mapping unit tests
docs/
├── quickstart.md       # 10-min walkthrough
├── adr/                # Architecture Decision Records
└── ops/                # Operator runbooks (Mac M4 self-hosted runner, etc.)
scripts/
├── start_grpc_runtime_server.py  # gRPC entrypoint
├── serve.py                       # HTTP shim entrypoint (deprecated)
├── bench_agentic/                 # Long-session perf bench harnesses
├── setup_mac.sh / setup_cuda.sh   # First-time setup
└── review_pr_*_on_mac.sh          # Mac M4 reviewer aids per PR
```

## Roadmap

| Milestone | Status | Description |
| --- | --- | --- |
| **v0.3.0 GA** | ✅ shipped | Session-bound gRPC runtime, Python + TS SDKs, HTTP shim deprecated, no test doubles in Linux CI gate, Mac M4 self-hosted integration workflow |
| v0.3.1 deployment polish | queued | PyPI + npm publishing (`pip install kakeya-inference`, `npm install @kakeya/runtime`), GHCR Docker image, `kakeya prewarm` CLI, `kakeya chat` REPL |
| v0.4 proposer-back-in | designing | Wire `SparseLogitsProposer` into the session-bound coordinator; restores speculative decoding to both gRPC and HTTP paths |
| v0.4 alignment training | designing | [ADR 0004](docs/adr/0004-alignment-training-data-preparation-policy.md) Stage 2-4: data prep → training → ship aligned proposer |
| v0.4 cross-request KV reuse | designing | Sessions survive across requests on gRPC; turns 9 ms intra-session drift into 0 ms inter-request drift |
| **v0.5-M1 multi-host milestone** | ✅ landed | [ADR 0009](docs/adr/0009-mlx-distributed-spec-decode-and-capability-exchange.md): agent **capability exchange** between Mac mini hosts (gossip `CapabilityService`, TTL registry, deterministic placement) + **distributed speculative decoding** (remote `ProposerService` drafts, local greedy verification, byte-identical output) + optional `mlx.distributed` ring probe for bulk-tensor flows. See [design doc](docs/design/agent-capability-exchange-platform.md) and `scripts/demo_distributed_spec_decode.py` |
| v0.5 GA multi-host hardening | queued | mTLS node identity, Bonjour seed discovery, K3 DFlash hidden-state flow over the mlx.distributed ring |

## Continuous integration

Two-tier gating model:

| Tier | Workflow | Coverage | Trigger |
| --- | --- | --- | --- |
| Linux gate | [`ci.yaml`](.github/workflows/ci.yaml) | Verifier-independent code, 100 % | Every PR; non-optional |
| Mac M4 gate | [`integration.yaml`](.github/workflows/integration.yaml) | Verifier-dependent code (runtime + SDK + proto + integration tests) | PRs labelled `needs-mac-m4` (auto-applied) |

Linux CI runs in <2 min on a github-hosted ubuntu runner against the
verifier-independent boundary (server route handlers, memory pool,
scheduler config / session, pipeline, session store, SDK error mapping,
training utilities). The Mac M4 self-hosted runner exercises the full
integration suite against real Qwen3-0.6B in ~90 s; setup runbook at
[`docs/ops/mac-m4-runner-setup.md`](docs/ops/mac-m4-runner-setup.md).

## Architecture Decision Records

Design decisions that the rest of the codebase depends on are recorded
in [`docs/adr/`](docs/adr/). Read these before changing core machinery:

- [ADR 0001 — Proposer sizing, alignment strategy, verifier
  decoupling](docs/adr/0001-proposer-sizing-and-alignment.md): why the
  proposer stays in a fixed 0.25–1 B band, EAGLE-3 representation
  alignment, verifier swaps as data-and-fine-tune operations.
- [ADR 0002 — Verifier selection, quantization, open-vs-closed-weight
  constraint](docs/adr/0002-verifier-selection-and-quantization.md):
  v1/v2 ship sequence (Qwen3-1.7B bf16 → Qwen3-8B 4-bit), 60 % memory
  rule for bf16 vs 4-bit, why closed-weight APIs can't be EAGLE-3-aligned.
- [ADR 0003 — Verifier ↔ slab pool integration](docs/adr/0003-verifier-slab-pool-integration.md):
  why the full "slab tensors hold the real KV" refactor is deferred,
  what intermediate step ships in v0.2 (`PooledVerifier`), now retired
  by ADR 0008.
- [ADR 0004 — Alignment training data preparation
  policy](docs/adr/0004-alignment-training-data-preparation-policy.md):
  v0.3 alignment training data + LoRA + masking + per-slice eval policy,
  Nemotron-informed `o_proj`-only LoRA at rank 128 / α 512, 7-domain
  prompt pool, block-aligned hidden state capture, position-dependent
  masking, per-slice acceptance gates.
- [ADR 0006 — Project positioning as local agent
  infrastructure](docs/adr/0006-local-agent-infrastructure-positioning.md):
  Kakeya is **local agent infrastructure for Mac**, not a generic
  chat-acceleration engine. Multi-agent / long-session / personalized
  usage as the headline framing.
- **[ADR 0008 — Session-bound runtime + gRPC protocol](docs/adr/0008-session-bound-runtime-and-grpc-protocol.md)**
  *(load-bearing for v0.3)*: the architectural pivot from automatic
  prefix matching (ADR 0007, superseded) to an explicit server-issued
  `session_id`. Frames the v0.3 deliverables: gRPC primary protocol,
  Python + TypeScript SDKs, HTTP shim deprecated single-shot path,
  no-doubles cleanup boundary, Mac M4 GA gate.

## License

MIT. See [`LICENSE`](LICENSE).
