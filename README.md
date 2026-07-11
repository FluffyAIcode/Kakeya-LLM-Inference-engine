# Kakeya Inference engine — memory bounded local inference engine

[![CI](https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine/actions/workflows/ci.yaml)
[![Release](https://img.shields.io/badge/release-v0.5--cuda%20%7C%20v0.4-blue)](https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine/tags)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20(MLX)%20%7C%20CUDA%20%7C%20Linux--CPU-lightgrey)](docs/quickstart.md)
[![Architecture](https://img.shields.io/badge/architecture-ADR%200008%20%7C%200014-green)](docs/adr/0014-agent-connection-capacity-and-cross-host-topology-tests.md)
[![License](https://img.shields.io/badge/license-MIT-lightblue)](LICENSE)

Kakeya is a **memory-bounded local agent runtime**: a long-running inference
server that holds session state on the server side, exposes a gRPC
`RuntimeService`, and bounds **KV memory + per-turn latency** on long
conversations — its KV footprint **does not grow with the conversation** (see
[Kakeya Attention](#how-this-differs--kakeya-attention-vs-pagedattention--radixattention)).

**v0.4** pairs a frozen **AR verifier** (Gemma-4 26B-A4B) with a **dLLM proposer**
(DFlash) and a trained projection **f_θ**: a sliding-window-bounded KV cache plus
**K/V restoration** reconstructs evicted context on demand, so memory is bounded
**without trading away recall, throughput, or context length**. It ships per
platform — **`v0.4-mac`** (Apple-Silicon MLX) and **`v0.4-cuda`** (NVIDIA) — atop
the session-bound gRPC runtime whose foundation landed in June 2026
([ADR 0008](docs/adr/0008-session-bound-runtime-and-grpc-protocol.md): 9 ms
latency drift over a 4-hour, 480-turn Mac M4 run; bounded memory).

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
                                               │  │ Restored verifier│  │
                                               │  │ Gemma-4 26B (AR) │  │
                                               │  │ + DFlash proposer│  │
                                               │  │ + f_θ / S5       │  │
                                               │  │ bounded sink+win │  │
                                               │  │ per-session bind │  │
                                               │  └──────────────────┘  │
                                               └────────────────────────┘
```

> The verifier slot is pluggable: a small **Qwen3 (CPU/MLX)** sink+window
> verifier for lightweight serving, or the **restored Gemma-4 26B** path
> (proposer + f_θ/S5) for the memory-bounded, recall-preserving engine below.

## Distributed Prefill KV Cache Network

Kakeya can use trusted peer Mac minis as an **immutable prefill-cache tier**.
Every node advertises model/cache compatibility through the existing P2P
`CapabilityService`; a cold inference node queries local and remote caches in
parallel, imports the longest valid token-prefix snapshot once, computes only
the missing suffix, and keeps autoregressive decode entirely local.

This is not remote attention and not coherent shared RAM:

```text
tokenize + chained prefix hashes
        │
        ├── local lookup ──────────────┐
        └── P2P lookup over gossip ────┤ choose longest compatible prefix
                                       ▼
                         stream one immutable KV snapshot
                                       ▼
                         local suffix prefill → local decode
```

Key properties:

- exact model/tokenizer/quantization/RoPE/cache-format compatibility;
- longest **contiguous** prefix reuse — arbitrary holes are never reused;
- memory-bounded LRU storage with leases and cache epochs;
- point-to-point chunked gRPC publish/fetch with SHA-256 validation;
- failure-safe fallback to local prefill;
- Thunderbolt/LAN/Tailscale endpoint priority;
- node registration, inference groups, token accounting and topology UI.

Live product dashboard: **[https://kakeya.ai](https://kakeya.ai)**.

Two-Mac measured evidence (Gemma 26B MLX 4-bit, 93-token prompt):

```text
cold local prefill     5.926 s
remote Thunderbolt hit 0.061 s
observed speedup       ≈97×
```

Architecture and operations:

- [ADR 0016 — Distributed Prefill KV Cache Network](docs/adr/0016-distributed-prefill-kv-cache-network.md)
- [Two-Mac live report](docs/reports/distributed-prefill-kv-mac-thunderbolt.md)
- [Operator runbook](docs/ops/distributed-prefill-kv-network.md)

## Quickstart (5 minutes on Mac M4 / Linux x86)

> **Status — v0.4** (`v0.4-mac` / `v0.4-cuda` tags). Ships from source; PyPI +
> npm + GHCR packaging is queued. The lightweight CPU flow below works on any
> checkout; the memory-bounded Gemma-4 restored engine (recall-preserving
> spec-decode, multi-tenant) is documented in the sections that follow.

```bash
# 1. Clone + check out the v0.4 tag for your platform
git clone https://github.com/FluffyAIcode/Kakeya-LLM-Inference-engine
cd Kakeya-LLM-Inference-engine
git checkout v0.4-mac     # Apple Silicon (MLX);  or: git checkout v0.4-cuda

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

## What's in the v0.4 architecture

| Component | What it does | Where |
| --- | --- | --- |
| `RuntimeService` (gRPC) | `CreateSession` / `AppendTokens` / `Generate` (server-streaming) / `GetSessionInfo` / `CloseSession`. Wire-stable; protobuf in [`proto/kakeya/v1/runtime.proto`](proto/kakeya/v1/runtime.proto). | `inference_engine.server.grpc_app` |
| `SessionStore` | In-memory session registry, server-issued IDs, append-only history, INV-1 / INV-2 enforcement, slab pool ownership. | `inference_engine.session.store` |
| **Restored verifier** | Gemma-4 26B-A4B (AR) + DFlash dLLM proposer + trained **f_θ**; **S5** keeps 5 full-attention layers exact, restores sliding layers → bounded resident KV, recall preserved. | `inference_engine.v04` (CUDA), `inference_engine.backends.mlx` (Apple Silicon) |
| `SinkWindowVerifier` | Lightweight path: Qwen3 (0.6B / 1.7B / 4-bit MLX), sink+window K/V trim (ADR 0001 / 0002). | `kv_cache_proposer.verifier` (CPU), `inference_engine.backends.mlx.verifier` |
| **Per-session binding** | `PerSessionVerifierRegistry` + coordinator resolver: each session owns isolated KV (shared weights) — true multi-tenant serving (PR-A3c, `--multi-tenant`). | `inference_engine.session.verifier_registry` |
| **Batched scheduler** | Fuses a cohort's decode steps into one batched forward — **8.45× served throughput** at 8 sessions, recall 1.0. **CUDA-only**: on Apple-Silicon MLX, `v0.4-mac` multi-tenant is **serial-only** (batched `B>1` decode is unsupported — upstream MLX `B>1, L=1` quantized-kernel bug, [ADR 0014](docs/adr/0014-agent-connection-capacity-and-cross-host-topology-tests.md)). | `inference_engine.session.batch_scheduler` |
| `AppendTokens` / `Generation` coordinators | Drive prefill / incremental forward / greedy decode; route per-session (multi-tenant) or single. | `inference_engine.session.{coordinator,generator}` |
| Python / TypeScript SDKs | `kakeya.Client` / `Session` (sync gRPC); `@kakeya/runtime` (Node 20+). | [`sdks/`](sdks/) |
| HTTP shim (deprecated) | OpenAI-compatible `/v1/chat/completions`; `Deprecation` + `Sunset` headers. | `inference_engine.server.app` |
| **Distributed Prefill KV Cache** | P2P capability gossip, exact compatibility locks, chained longest-prefix lookup, chunked snapshot publish/fetch, local suffix prefill and local decode. | `inference_engine.distributed.prefill_cache*`, `inference_engine.network` |

## Runtime evidence (foundational, carried into v0.4)

The integration suite under [`tests/integration/`](tests/integration/) is
the binding correctness gate. Mac M4 evidence:

| Gate | Metric | Result |
| --- | --- | --- |
| Memory bounded ([ADR 0006 §2.3.a](docs/adr/0006-local-agent-infrastructure-positioning.md)) | `agg.kv_bounded` | True |
| **Prefill bounded** ([ADR 0008 §7 G2](docs/adr/0008-session-bound-runtime-and-grpc-protocol.md)) | `latency_drift_p50_s` over 14400 s | **+0.0093 s** (vs +39.74 s on the prior HTTP-shim architecture — **4400×** improvement) |
| INV-3 byte-exact greedy decoding ([ADR 0008 §7 G3](docs/adr/0008-session-bound-runtime-and-grpc-protocol.md)) | All chunkings produce identical token streams | Pass |
| Throughput | Turns over 4 h | 480 (vs 58 on the prior architecture — 8.3× more sustained) |
| Latency | p50 / p95 over 4 h | 1.829 s / 1.853 s |

Raw artifacts: [`results/platform-tests/bench_session_4h_1780332893.json`](results/platform-tests/) (4-h evidence). The v0.4 memory / throughput / recall / multi-tenant results are in the [beta scorecards](#beta-scorecards--kakeya-vs-the-standalone-model-main--9d5e6b4) below and [ADR 0014](docs/adr/0014-agent-connection-capacity-and-cross-host-topology-tests.md).

## Design philosophy — AR verifier + dLLM proposer, KV restoration for a memory-bounded Gemma-4 26B

Kakeya pairs a frozen **autoregressive (AR) verifier** — `Gemma-4 26B-A4B-it` — with
a **diffusion-LM (dLLM) proposer** (`z-lab` DFlash, 0.4 B). The proposer's *first*
role is not "drafter" but **history reconstructor**: a dLLM carries **no KV cache**
and can emit transient K/V for *any* past position, so it can restore the verifier's
**evicted** K/V on demand. A small trained projection **f_θ** maps proposer hidden
states → verifier K/V; on Gemma-4 the **S5** strategy keeps the 5 full-attention
layers exact and restores the sliding-window layers.

The whole architecture is built around one inequality:

> Make `Gemma-4 26B-A4B-it` **memory-bounded** *without* trading away model
> **intelligence** (recall), **token throughput**, or **context length**.

**KV restoration is the mechanism.** The verifier only ever keeps a **bounded
sink+window** of its own K/V resident (constant **~17 MB** on CUDA / **~133 MB** in
the Mac S5 config), while the *effective* attention context — the full
multi-thousand-token history — is **reconstructed on demand** by the proposer + f_θ.
Because the restored/spec-decoded K/V is **byte-checked** against the AR cache, the
**output is identical to the standalone AR model** (recall **1.0**): the memory win
costs **zero intelligence**. Throughput and context length are held at **parity**
(Mac) or **improved** (CUDA spec-decode **1.79× AR**) — never sacrificed. This is the
inversion of the usual quantize/evict trade-off: instead of *cheaper, dumber, shorter*,
KV restoration buys *bounded memory at full fidelity*. See
[ADR 0012](docs/adr/0012-proposer-verifier-value-proposition.md) (value realised on
the **memory axis** all-platform + **throughput** on CUDA) and
[ADR 0013](docs/adr/0013-distributed-inference-topology.md).

### Kakeya Attention — the attention algorithm

**Kakeya Attention** is an LLM attention compute + KV-management algorithm:
**sliding-window bound (sink + window) + f_θ KV-projection + dLLM-proposer
restoration, taken as one primitive.** It is a peer of — and drop-in replacement
for — the attention layer in today's engines: eager attention, **FlashAttention**,
vLLM **PagedAttention**, and SGLang **RadixAttention**. Where those keep the
**whole** KV history (memory grows with the conversation) and differ only in
*how* the full cache is computed or laid out, Kakeya Attention bounds *how much*
is resident: evicted context is **reconstructed on demand** by the proposer+f_θ,
so the resident footprint does not grow with the session.

| Algorithm | Layer it replaces | Mechanism | Memory vs conversation length |
| --- | --- | --- | --- |
| eager attention | compute | materialise full `QKᵀ` scores | grows (O(T²) compute, full KV) |
| **FlashAttention** | compute | tiled/online-softmax, no score materialisation | grows — still full KV |
| **vLLM** PagedAttention | storage | OS-style **paged** KV blocks | grows — still full KV |
| **SGLang** RadixAttention | storage | **radix-tree** KV, prefix reuse | grows — still full KV |
| **Kakeya Attention** | **compute + storage** | sink+window bound + **f_θ + dLLM-proposer restoration** | **bounded** — provision for the **peak window**, not the history |

The orthogonality matters: FlashAttention makes the compute cheaper, Paged/Radix
make the *same total* KV cheaper to allocate or share — **Kakeya Attention makes
the total itself bounded**, and is **composable** with all of them (a flash
kernel computes a Kakeya window; a paged/radix store holds it). The cost is the
restoration compute (a proposer forward), quantified below (recall 1.0;
~AR-parity / 1.79–2.06× on CUDA; ~4× more concurrent agents per GB,
[ADR 0014 §3.4](docs/adr/0014-agent-connection-capacity-and-cross-host-topology-tests.md)).

**North star — a product-grade engine that replaces vLLM.** Kakeya Attention is
the native algorithm of a **product-grade inference engine whose goal is to
replace vLLM** — not a technique bolted onto HuggingFace transformers, and not
"vLLM with a different cache". The engine is designed **bounded-KV-native**: the
full history is never resident, admission/scheduling sizes sessions by their
**peak window** (not total tokens), and restoration is fused into prefill/decode.
Graph-captured decode, fused-MoE and efficient masking are table stakes built *in
service of* that design, not a port of vLLM's full-KV pipeline
([ADR 0015](docs/adr/0015-kakeya-attention-and-engine-substrate.md)). The
eager-transformers numbers in the comparison reports are **feasibility probes**,
not "Kakeya performance"; the vLLM-beating demonstration runs on a
**full-attention** verifier, where restoration is load-bearing (on gemma-4 its
native sliding window already bounds 25/30 layers, so it is not the showcase).

**Where the bounded-KV win is large (and where it isn't).** The advantage scales
with the model's **full-attention fraction**. On **gemma-4-26B-A4B** only 5 of 30
layers are full-attention (25 are natively sliding-window) — so vLLM already
bounds 25/30 layers, the 5 full layers dominate long-context KV in **both**
engines, and Kakeya's resident-KV edge is only **~7 % at 62k**. On a
**full-attention** model (no native sliding, e.g. Qwen/Llama) vLLM keeps all
layers full while Kakeya bounds all-but-exact → a **~6×** resident-KV edge. The
long-context concurrency "sweet spot" is therefore **architecture-dependent** —
see [the long-context report](docs/reports/kakeya-vs-vllm-longcontext-h200.md).

### Beta scorecards — Kakeya vs the standalone model (`main` @ `9d5e6b4`)

Both betas run the *same* `Gemma-4 26B-A4B-it` verifier, `z-lab` DFlash proposer, and
`f_theta_v5_s5_sliding`. "Standalone model" = the same Gemma-4 run **without** Kakeya
(`mlx_lm` AR oracle on Mac; HuggingFace bf16 AR on CUDA) — i.e. the honest *"what does
the engine cost vs just running the model?"* baseline.

**Mac (MLX) — Kakeya vs `mlx_lm` AR oracle** · Mac mini M4 · 4-bit verifier:

| Axis | Kakeya | MLX-only | Result |
| --- | --- | --- | --- |
| **Memory** (resident KV @ 5 810 tok) | **132.92 MB** (S5) | 1 308.88 MB | **89.8 % saved** (20 vs 220 KB/tok, 11× slower growth) |
| **Context length** | 4 406–5 810 tok handled, **recall 1.0** | recall 1.0 | byte-identical output |
| **Throughput** (code, 128-tok decode) | 21.68 tok/s | 23.26 tok/s | **0.93×** (≈ parity) |

*Raw scorecard report — Mac MLX (reproducible evidence):*

```
Kakeya Inference Engine (MLX beta, main @ 9d5e6b4 / PR #117) vs MLX-only
Gemma-4 26B-A4B-it 4-bit, Mac mini M4, verifier=gemma-4-26B-A4B-it-mlx-4bit,
drafter=z-lab DFlash, f_theta=v5_s5_sliding, S5 (5 exact full-attn layers).

================ 1) MEMORY BOUNDED  (NIAH ctx280, T=5810 tok) ================
                         Kakeya (S5)     MLX-only (naive full-KV)
resident KV @5810 tok    132.92 MB       1308.88 MB        -> 89.8% saved
KV growth per token       20.0 KB/tok      220.0 KB/tok     -> 11x slower
exact full-attn layers    5,11,17,23,29 hold all 5810 pos (full recall)
sliding layers            bounded to 68 resident positions

================ 2) CONTEXT LENGTH  (NIAH ctx280) ===========================
prompts handled          4406 - 5810 tokens
recall (Kakeya)          1.0  (5/5)   == MLX-only oracle 1.0 (5/5)  byte-identical
verifier attention ctx   full 5810-tok window kept EXACT on 5 full-attn layers
                         while sliding layers stay window-bounded

================ 3) TOKEN THROUGHPUT  (code workload, 128-tok decode) ========
                         Kakeya fused    MLX-only AR        ratio
long-sample mean (e2e)   21.68 tok/s     23.26 tok/s        0.93x  (~parity)
decode-only (long)       ~24-27 tok/s    --                 best 0.99x
recall                   1.0 (8/8)       1.0 (8/8)          byte-identical

Net: Kakeya delivers bounded memory (~90% KV saving) + full-context recall at
MLX-only-identical output, at ~AR-parity throughput on Mac (the 26B verify(L)
compute per block is the throughput floor; >AR remains CUDA-favored: H200 1.79x).
```

**CUDA (H200) — Kakeya vs standalone Gemma-4 26B AR** · bf16:

| Axis | Kakeya | AR | Result |
| --- | --- | --- | --- |
| **Memory** (resident KV @ 3 238 / 6 438 tok) | **constant 16.71 MB** | 733.06 / 1 453.96 MB | **43.9× / 87.0× saving** |
| **Context length** | 68-tok window ↦ 3 254 / 6 454 tok, **recall 1.0** | recall 1.0 | **47.9× / 94.9× compression** |
| **Throughput** (fused spec-decode, block-16) | **28.94 tok/s** | 16.13 tok/s | **1.79× AR** (accept-len 3.32) |

*Raw scorecard report — CUDA H200 (reproducible evidence):*

```
Kakeya Inference Engine (GPU beta, main @ 9d5e6b4 / #107+#117) vs standalone AR
NVIDIA H200 · Gemma-4 26B-A4B-it (bf16) · verifier=google/gemma-4-26B-A4B-it
drafter=z-lab DFlash · f_theta=v5_s5_sliding · S5 (5 exact full-attn layers)
"AR" = standalone Gemma-4 26B AR model (GPU analog of "mlx-only").

================ 1) MEMORY BOUNDED  (resident KV) ===========================
context rung      AR full-KV      Kakeya restored     saving
3238-tok prompt   733.06 MB       16.71 MB            43.9x
6438-tok prompt   1453.96 MB      16.71 MB            87.0x
-> Kakeya KV is CONSTANT 16.71 MB (68-tok sink+window) regardless of context;
   AR KV grows linearly. Saving scales with context length.

================ 2) CONTEXT LENGTH  (window vs effective) ===================
context rung      resident window   effective ctx      compression   recall
3238-tok prompt   68 tok            3254 tok           47.9x         1.0 == AR
6438-tok prompt   68 tok            6454 tok           94.9x         1.0 == AR
-> 68-token bounded window reconstructs full multi-thousand-token context
   via f_theta/S5 restoration, with recall identical to AR.

================ 3) TOKEN THROUGHPUT  (decode tok/s, 3238-tok prompt) ========
path                         tok/s     vs AR     recall
standalone AR                16.125    1.00x     1.0
restored per-token (Gap A)   16.297    1.01x     1.0   (restoration is free)
Kakeya FUSED spec-decode     28.937    1.79x     1.0   (block-16, accept_len 3.32)
-> On GPU the fused spec-decode delivers 1.79x AR at byte-identical output,
   because verify-batch is cheap (vs Mac ~0.93x where 26B verify(L) dominates).

Net (GPU): bounded memory (44-87x KV saving, constant 16.71 MB) + full-context
recall (48-95x compression, recall 1.0) + 1.79x AR throughput, all at
AR-identical correctness. This is the platform where spec-decode value lands.
```

Both platforms hold **recall 1.0 / byte-identical output**. The fork is on the
throughput axis only: CUDA's cheap verify-batch turns spec-decode into a **1.79×**
win, while on Mac the **26 B `verify(L)` compute per block** is the floor, so the
engine lands at **≈ AR parity** — the memory + context wins are platform-independent.
Reproduce with `scripts/research/k3_e2e_gpu_bench.py` + `k3_specdecode_gpu_bench.py`
(CUDA) and the `k3-beta-scorecard` / `k3-fused-allmlx-code-trim` Mac-bridge presets.

### Agent-connection capacity & cross-host topology ([ADR 0014](docs/adr/0014-agent-connection-capacity-and-cross-host-topology-tests.md))

**Agent connections (gRPC `RuntimeService`, Mac mini M4).** A connection load
test (`scripts/research/grpc_agent_capacity_loadtest.py`, preset
`agent-capacity-loadtest`) ramps N concurrent agents — an independent gRPC
channel + session each — against one runtime:

| | result |
| --- | --- |
| Max concurrent agents | **256 / 256, zero errors** (the configured capacity — a clean floor, not a failure point) |
| Per-session resident KV | **bounded** (sink+window; ~7.8 MB @ window 64, ~30 MB @ window 256) |
| Node KV upper bound | **capacity × per-session bound** (≈2.0 GB @ cap 256) — independent of context length / churn |
| Server RSS vs agents | **flat** (3825 → 3850 MB across 1 → 256) — adding agents costs ~0 memory |

Note on tenancy: this connection sweep ran the **single-tenant** admission path
(one shared verifier), so "256" is the max concurrent connections *served*, not
parallel inferences. **v0.4 adds per-session binding (PR-A3c)** — each session
owns isolated KV (shared weights) — making serving truly multi-tenant, and a
**batched scheduler** fuses the cohort into one forward for **8.45× throughput**
at 8 sessions with **per-session recall 1.0** (see the multi-tenant results
below / [ADR 0014 §3.4–3.7](docs/adr/0014-agent-connection-capacity-and-cross-host-topology-tests.md)
and the [detailed report](docs/reports/pr-a3c-multitenant-serving-test-report.md)).
**Platform scope:** the batched/parallel cohort path is **CUDA-only**. On
Apple-Silicon **MLX, `v0.4-mac` multi-tenant is serial-only** — per-session
binding still gives isolated, recall-preserving sessions, but they are served
**one at a time**; batched `B>1` decode is blocked by an upstream MLX
quantized-kernel bug (`B>1, L=1` → per-session recall collapses to 0.125, while
serialized stays 1.0; confirmed on the latest published `mlx 0.31.2 / mlx-lm
0.31.3` — [ADR 0014 §3.4](docs/adr/0014-agent-connection-capacity-and-cross-host-topology-tests.md)).
Pushing the connection sweep further (preset `agent-capacity-stress`, the
open-file-descriptor limit `RLIMIT_NOFILE` raised to 100k / hard unlimited on
the Mac — each connection uses one descriptor) shows the true ceilings: **the
open-file-descriptor limit is not the constraint**; **memory** scales with
`capacity × window` (capacity 2048 @ window 256 → ~11 GB RSS, theoretical node
bound ~61 GB > 24 GB RAM, so capacity must be sized to RAM); and with a
per-agent **context** prefill the binding constraint was **serialization** on
that single-tenant path (concurrent heavy-prefill agents serialize and time out
well before any file-descriptor / connection limit) — which is exactly what
v0.4's per-session binding + batched scheduler remove. Bounded memory is
structural: light-session agent count does **not** grow RSS; the memory lever is
the resident **window**, not the number of agents.

**Cross-host proposer/verifier.** A GPU proposer ⇄ Mac verifier *token-level
draft* data plane is **design-only** (no `CapabilityService` / `ProposeBlock` /
gossip) **and** ruled out by the WAN latency budget — now **measured** on real
H200 compute by injecting one proposer↔verifier round-trip per block:

| per-block RTT | 0 (co-located) | 15 ms (LAN) | 30 ms | 60 ms | 100 ms | 150 ms |
| --- | --- | --- | --- | --- | --- | --- |
| vs AR | **2.20×** | 1.81× | 1.50× | 1.22× | **0.98×** (break-even) | 0.77× (loss) |

**Break-even ≈100 ms/block**: a cloud↔desk WAN (30–150 ms) straddles/exceeds it,
while a LAN (≤15 ms) keeps the 1.8–2.2× win. Confirmed end-to-end with a **real
two-process socket over a real ~102 ms network** (reverse SSH tunnel, real
156 KB/block aux payload): co-located **2.02×** → real-network **0.56× AR** (a
net loss; network was 71 % of wall time). So the realizable split is **WAN =
control + tool plane** (the Mac bridge) and **LAN = co-located data plane**. See
[ADR 0014](docs/adr/0014-agent-connection-capacity-and-cross-host-topology-tests.md)
for the full plan, evidence, and the served-MLX-gemma gap found during testing.

## v0.5 for CUDA — Kakeya Attention on the vLLM runtime

**v0.5-cuda** ships Kakeya Attention's bounded-window (S5) KV management **on top
of the vLLM runtime**, so the three runtime components the engine needs are
inherited unchanged — all **Apache-2.0**:

| component | owner in v0.5-cuda | role |
| --- | --- | --- |
| **Fused MoE Triton kernel** | **vLLM** | grouped-GEMM expert kernel — the dominant ~90 % of the gemma-4-26B-A4B decode forward |
| **CUDA graphs** | **vLLM** | fixed-shape decode capture (`enforce_eager=False`) — removes per-token launch overhead |
| **Continuous-batching scheduler** | **vLLM** | request scheduler + paged KV-manager — drives multi-tenant throughput |
| **Kakeya Attention (bounded window / KV)** | **Kakeya** | bounds resident sliding-layer KV to `sink + window` (S5 = 68) |

This is the **KIE-v2** strategy ([ADR 0015](docs/adr/0015-kakeya-attention-and-engine-substrate.md),
[feasibility](docs/design/kakeya-vllm-backend-feasibility.md)): rather than rebuild
vLLM's fused-MoE + graphs + scheduler (shown blocked in the eager KIE-v1.1.z
attempt), Kakeya Attention runs *on* vLLM and contributes the bounded-KV layer.

```python
from inference_engine.engine import KakeyaVLLM
from vllm import SamplingParams

engine = KakeyaVLLM("google/gemma-4-26b-a4b-it", sink=4, window=64, max_model_len=16384)
out = engine.generate(prompts, SamplingParams(temperature=0.0, max_tokens=128))
```

**Scorecard** ([full report](docs/reports/kakeya-inference-engine-v0.5-cuda.md)),
H200, gemma-4-26B-A4B, recall **1.0**:

| axis | result |
| --- | --- |
| **Token throughput (decode)** | **≥ vLLM**: **1.15–1.23×** vs vLLM default at ctx 16k, N=1..70 (e.g. N=70 **1079 vs 894.9 tok/s**) |
| **Parallel inference** | bounded window on vLLM measured to **N=70** @16k (recall 1.0); eager research engine reached **N=75 @62k** (≈4.8× vLLM concurrency) |
| **Memory saving** | gemma-4 hybrid: **~7 % @ 62k** (vLLM already bounds 25/30 layers; the 5 full layers dominate both); **~6×** edge needs a **full-attention** model + the v0.6 restoration backend |

> **Honest scope.** v0.5-cuda is the **gemma-4 bounded-window** instantiation, and
> it works **without a trained f_θ/proposer**: gemma-4's 5/30 native full-attention
> layers carry recall, so recall is **1.0 at `sliding_window=68` with no restoration**
> (the S5 "free lunch"), delivered via vLLM `hf_overrides`. The gemma-4 throughput /
> concurrency / recall numbers above are the measured KIE-v2 results. The
> `KakeyaVLLM` wrapper *plumbing* was separately smoke-tested on an H200 (builds
> vLLM, window→config, CUDA graphs capture, generate returns) using Qwen3-4B —
> that is a **wrapper smoke test only, not engine validation**: Qwen3 has no
> trained f_θ/proposer, so restoration never ran. The **restoration backend** (f_θ
> + dLLM-proposer training + quantized-exact attention) for **full-attention**
> models — the large ~6× memory differentiator, where a bounded window *without*
> restoration would destroy recall — is the **v0.6** roadmap item.

## v0.4 for Mac — MLX speculative-decode port (the journey to parity)

After the **CUDA** path (f_θ + S5 K/V-restoration verifier, **fused DFlash
spec-decode at 1.79–2.06× AR, recall 1.0 on Gemma-4-26B-A4B / H200**), the engine
was ported to the **Apple-Silicon MLX** backend (`v0.4-mac`). The decode throughput climbed from a
near-total collapse to **≈AR parity** through a sequence of precisely-diagnosed
fixes. This is the baseline record of that journey (all numbers are decode-only
tok/s vs the native `mlx_lm` AR oracle on the same model, measured on a Mac M4 via
the [Mac bridge](#evaluation-environment); ×AR is the ratio).

| Stage | ×AR | Binding problem | Fix |
| --- | --- | --- | --- |
| Naïve restored decode | **~0.09×** | **O(T²) collapse** — the restored verifier did a *full-sequence* forward **per generated token** (`restored_logits`); the Mac harness called it once per token. | **Gap-A incremental decode**: prefill **once**, capture the restored K/V into the model's **native** cache, then decode with `mlx_lm.generate_step` (chunked prefill + `mx.async_eval` pipelined) — O(L)/token, never re-forward the sequence. |
| Hybrid fused spec-decode | **~0.2×** | **Cross-runtime bridge** — MLX verifier + PyTorch/MPS drafter shipped **MB/block of aux-hidden** across runtimes on the critical path; plus a benchmark **forced-over-generation** artifact (`--ignore-turn-stop`) that tanked acceptance. | Recognised the bridge as the bottleneck; moved toward an **all-MLX drafter** (single runtime, zero per-block bridge crossings). |
| All-MLX + sound rollback | **~0.5×** | **Unsound rollback** — `RotatingKVCache` is not trimmable once the sliding ring wraps (`is_trimmable → offset < max_size`), so the loop **rolled the whole block back and re-forwarded** the carried accepted tokens every partial-accept block (~2 verifier forwards/block). | **CUDA-`DynamicCache` parity**: prefill an **all-`KVCache`** layout (sliding too — byte-exact, the window mask applies regardless of cache capacity) so `trim_prompt_cache` is a sound O(1) slice; **keep accepted K/V, trim only the rejected tail**, never re-forward. |
| Block-4 CUDA-trim | **~0.7×** | **Per-block Python graph construction** (`build_s` ≈ 50 ms/block building the 26B lazy graph). | Removing the re-forward (above) closed most of it; block-4 lands at **0.68× AR**. |
| Block-8 tuned | **~1.0×** | **Block size vs the drafter's accept-len plateau.** | Tune to **block-8** (matches the all-MLX drafter's ~4.5 accept-len ceiling); long-code completions reach **~1.0–1.05× AR (parity, best samples just over)**. block-16 is *worse* — `verify(16)` cost is wasted because acceptance plateaus. |

**Honest ceiling & what was *ruled out*.** ≈AR parity is the Mac result on the
spec-decode sweet spot (short-context, naturally-long *code/agent* generation);
**>AR meaningfully remains CUDA-favoured** (H200 **1.79×** fused/block-16 on the
fresh `main` scorecard above; #107 originally reported 1.27×) because the binding
constraint is the **26B `verify(L)` compute per block** — *not* rollback (fixed),
*not* sync count (a one-graph "single-fused" probe ran stably at ~0.16 s/block and
was ≈ equal — the b876 single-fused "143 s" pathology is **large-cache-specific**,
not fundamental), *not* drafter acceptance (a clean ~3–4.5/block on natural
workloads), *not* verifier quantization (4-bit ≥ bf16; the loop is self-consistent),
*not* context length (NIAH ≥ general), and *not* a missing alignment asset
(fc_norms fine-tuning *degraded* held-out acceptance — the base z-lab drafter is
already near its block-4 ceiling). The earlier "low acceptance / 2.13" numbers were
a **forced-over-generation benchmark artifact**, reproduced on a clean full-KV bf16
verifier. The one genuine remaining lever is closing the **drafter accept-len gap
(~4.5 ours → ~7.7 z-lab reference)** — a port-fidelity / alignment residual.

Recall (the architecture's primary deliverable) is **1.0** throughout, with
bounded resident KV (**S5**: ~133 MB vs ~1309 MB naïve at 5.8 k ctx, ~90 % saving;
~48 MB after affine-4). See [ADR 0012](docs/adr/0012-proposer-verifier-value-proposition.md)
(value is realised on the **memory axis** all-platform + **throughput** on CUDA)
and [ADR 0013](docs/adr/0013-distributed-inference-topology.md) (what AR
sequentiality allows for distribution).

### Evaluation environment

The Mac port was developed and benchmarked **remotely from a Linux cloud agent**,
since MLX runs only on Apple Silicon:

- **Mac bridge** (`scripts/mac_bridge/`): a **git-bus** request/response plane — the
  agent pushes an allowlisted-preset request branch, a **self-hosted GitHub Actions
  runner (`kakeya-mac-m4`)** executes it on the Mac and pushes results back. No SSH/
  VPN — only git push. Presets + param bounds are enforced by
  `inference_engine/bridge/manifest.py`; this is itself an instance of the
  multi-host capability plane ([ADR 0009](docs/adr/0009-mlx-distributed-spec-decode-and-capability-exchange.md)).
  Full guide: [`docs/mac-bridge.md`](docs/mac-bridge.md).
- **Evidence gate** (`inference_engine/bench/k3_report_gate.py`): every Mac report is
  machine-validated — rejects fused runs that didn't execute (`blocks=0`), baseline
  bypasses claiming recall/speedup, self-comparison speedups, prefill-variance, and
  decode-token-budget violations — so a number is admissible only if it survives the
  same rules that caught the earlier artifacts.
- **GPU side** (vast.ai H200): alignment-training + acceptance-factor experiments
  (`scripts/research/k3_dflash_alignment_train.py`, `k3_dflash_specdecode_eval.py`)
  used to rule out the non-levers above.

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
- Speculative decoding is **not** applied; pure AR. Migrate to gRPC for the
  full v0.4 perf story.

```bash
# Deprecated — only when you really need OpenAI-API compat
PYTHONPATH=.:sdks/python python3 scripts/serve.py \
    --backend cpu --verifier-id Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 8000

curl -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"any","messages":[{"role":"user","content":"hi"}]}'
```

For a curl-friendly path with the gRPC runtime's full speed, use a
Python or TypeScript SDK against the gRPC server.

---

# Architecture & Background

## Background — the speculative-decoding lineage

> **Note**: v0.4 ships speculative decoding in the restored Gemma-4 engine
> (fused DFlash spec-decode, recall 1.0 — see the scorecards above). This
> section documents the original proposer/verifier formulation for context;
> the memory-accounting analysis still applies.

The lineage runs a DLM (diffusion language model) proposer in front of an AR
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
| Session-bound gRPC runtime | ✅ shipped | Long-running gRPC `RuntimeService`, Python + TS SDKs, bounded memory + prefill (4-h Mac M4 evidence), Mac M4 self-hosted integration gate |
| **v0.4 for Mac (`v0.4-mac`)** | ✅ shipped | MLX restored Gemma-4 26B engine: bounded KV (~90% saved), recall 1.0, ≈AR-parity spec-decode. Multi-tenant is **serial-only** (no batched `B>1` decode — upstream MLX kernel bug, [ADR 0014](docs/adr/0014-agent-connection-capacity-and-cross-host-topology-tests.md)) |
| **v0.4 for CUDA (`v0.4-cuda`)** | ✅ shipped | Restored Gemma-4 26B engine on NVIDIA: fused DFlash spec-decode **1.79–2.06× AR**, 44–87× KV saving, recall 1.0 |
| **v0.4 multi-tenant (PR-A3c)** | ✅ shipped | Per-session binding (isolated KV, shared weights) on both platforms. **CUDA**: batched scheduler → **8.45× served throughput**, per-session recall 1.0. **MLX (Mac): serial-only** (sessions served one at a time; batched parallel decode unsupported upstream) |
| **v0.5-M1 multi-host milestone** | ✅ landed | [ADR 0009](docs/adr/0009-mlx-distributed-spec-decode-and-capability-exchange.md): agent **capability exchange** between Mac mini hosts (gossip `CapabilityService`, TTL registry, deterministic placement) + **distributed speculative decoding** (remote `ProposerService` drafts, local greedy verification, byte-identical output) + optional `mlx.distributed` ring probe for bulk-tensor flows. See [design doc](docs/design/agent-capability-exchange-platform.md) and `scripts/demo_distributed_spec_decode.py` |
| v0.5 GA multi-host hardening | queued | mTLS node identity, Bonjour seed discovery, K3 DFlash hidden-state flow over the mlx.distributed ring |
| Async continuous batching | designing | Dynamic mid-flight arrival + ragged-length cohorts under the async gRPC `Generate` handlers (current batcher is fixed-cohort) |
| Deployment polish | queued | PyPI + npm publishing, GHCR Docker image, `kakeya prewarm` CLI, `kakeya chat` REPL |
| **Distributed Prefill KV reuse** | ✅ live MVP | Cross-node immutable snapshots, chained longest-prefix matching, Thunderbolt gRPC transfer and public fleet dashboard ([ADR 0016](docs/adr/0016-distributed-prefill-kv-cache-network.md)) |

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
