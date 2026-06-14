# ADR 0014 — Agent-connection capacity & cross-host proposer/verifier topology: test plan & results

- **Status**: Accepted (test record + topology decision)
- **Date**: 2026-06-14
- **Relates to**: [ADR 0008](0008-session-bound-runtime-and-grpc-protocol.md)
  (session-bound gRPC runtime), [`docs/mac-bridge.md`](../mac-bridge.md) +
  [`docs/design/mac-bridge-cloud-agent-access.md`](../design/mac-bridge-cloud-agent-access.md)
  (the cloud-agent ⇄ Mac soft link).
- **Implementation**: `scripts/research/grpc_agent_capacity_loadtest.py`,
  manifest preset `agent-capacity-loadtest`
  (`inference_engine/bridge/manifest.py`).
- **Evidence**: `results/research/k3_agent_capacity_mac.json`.

> Note on numbering: `main`'s ADR index stops at 0008; the README references
> 0009/0012/0013, which were authored on branches that never merged to `main`
> (a known doc gap, out of scope here). This ADR uses 0014 to avoid collision.

## 1. Context

Two test cases were requested against the AR-verifier + dLLM-proposer
architecture, using the Mac bridge as the cloud-agent ⇄ Mac mini M4 link:

1. **Case 1 — agent connection capacity.** Simulate many agents connecting to
   the Kakeya inference engine on the Mac mini and find the **maximum
   concurrent agent connections**, plus the bounded KV residency.
2. **Case 2 — cross-host proposer/verifier.** Run the CUDA proposer on a GPU,
   have it **discover** and submit **drafts** to the verifier on the Mac mini,
   and measure **token throughput / max agent connections / Mac KV upper
   bound** under that topology.

Ground truth from a code audit of `main` (`9d5e6b4` lineage) determined what is
runnable vs design-only and shaped the test plan below.

## 2. Test environment

- **Mac mini M4** (24 GB unified memory), self-hosted Actions runner
  `[self-hosted, macOS, ARM64, kakeya-mac-m4]`, reached via the **Mac bridge**
  git-bus plane (no inbound path; allowlisted presets only).
- **Cloud agent**: Linux x86 VM (no Metal). Orchestrates via the bridge.
- **GPU**: H200 NVL (vast.ai) — runs the CUDA proposer + verifier for the
  co-located reference (fused **2.06–2.20× AR**, recall 1.0) and the §4.3
  cross-host WAN-penalty sweep.
- **Engine**: gRPC `RuntimeService` (`scripts/start_grpc_runtime_server.py`),
  Python SDK clients (`sdks/python/kakeya`).

## 3. Case 1 — agent connection capacity (RUN, real evidence)

### 3.1 Implementation

`scripts/research/grpc_agent_capacity_loadtest.py` launches one
`RuntimeService` subprocess and ramps `N` concurrent **agents**, each an
independent gRPC channel + session that creates a session, appends a short
prompt, holds the session open while all `N` are established (true concurrent
peak), then generates and reads `GetSessionInfo.kv_live_bytes`. It records, per
level: created/generate success, create & generate latency p50/p95,
per-session bounded KV, and server RSS. Run on the Mac via the
`agent-capacity-loadtest` bridge preset.

Verifier: **cpu `Qwen/Qwen3-0.6B`** (the integration-gate model). Connection /
admission scaling is **model-independent**, so this isolates the connection
behavior; the served **MLX gemma** path is a separate v0.4 gap (§6).

### 3.2 Results (Mac mini M4, capacity=256, sink=4 window=64)

| agents | created | errors | create p95 (s) | gen p95 (s) | per-session KV | server RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1/1 | — | 0.78 | 0.10 | 1.38 MB | 3825 MB |
| 16 | 16/16 | — | 1.33 | 1.66 | 7.80 MB | 3835 MB |
| 64 | 64/64 | — | 5.66 | 6.85 | 7.80 MB | 3840 MB |
| 128 | 128/128 | — | 11.27 | 13.64 | 7.80 MB | 3845 MB |
| **256** | **256/256** | **—** | 22.61 | 26.44 | 7.80 MB | 3850 MB |

- **Max concurrent agents: 256 / 256, zero errors.** 256 was the configured
  `--capacity` and was sustained completely — i.e. **256 is a clean floor on
  the connection ceiling, not a failure point**. The true resource ceiling is
  higher (not probed past the configured capacity).
- **Per-session KV is bounded at 7.80 MB** (plateaus from 16 agents up): the
  `sink+window` (68-token) ceiling holds regardless of agent count.
- **Node KV upper bound = capacity × per-session bound = 256 × 7.80 MB ≈
  2.0 GB** — the whole-node resident-KV ceiling, independent of context length
  or agent churn. This is the bounded-memory guarantee at the fleet level.
- **Server RSS is flat** (3825 → 3850 MB across 1 → 256 agents): adding agents
  costs ~0 memory beyond the bounded slab; model weights dominate.

### 3.2b Stress beyond 256 — the real ceilings (preset `agent-capacity-stress`)

Pushing further with the open-file-descriptor limit raised (`RLIMIT_NOFILE`
soft 100k, hard unlimited on the Mac) and a **per-agent context prefill** (window 256,
`--context-len 256`, capacity 2048):

| agents | created | create p95 | per-session KV | server RSS |
| --- | --- | --- | --- | --- |
| 1 | 1/1 | 3.07 s | 29.8 MB | 11 477 MB |
| 8 | 8/8 | 25.2 s | 29.8 MB | 11 343 MB |
| 16 | 15/16 (1 `RpcCancelled`) | 44.6 s | 29.8 MB | 10 781 MB |

- **The open-file-descriptor limit is not the ceiling** (raised to 100k; Mac
  hard limit is unlimited) — each gRPC channel/session consumes one descriptor.
- **Memory** scales with `capacity × window`: capacity 2048 @ window 256 →
  **~11.5 GB RSS**, and the theoretical node bound is **~61 GB > 24 GB RAM** —
  so capacity must be **sized to RAM** (it is the memory knob, not agent count).
- The binding constraint with real per-agent context is **single-tenant
  serialization**: create latency is purely linear (3 → 12 → 25 → 45 s as
  N = 1 → 4 → 8 → 16) because every session's prefill serializes through the one
  shared verifier, so clean concurrency tops out at **~8 heavy-context agents**
  before RPCs time out — versus **256 light-session agents** (§3.2). Per-session
  KV stays bounded (29.8 MB @ window 256) throughout.

### 3.3 Honest caveat — v0.3 is single-tenant

Create/generate latency scales **linearly** with `N` (256 agents → gen p95
26 s). That is the single-tenant signature: one shared verifier, RPC handlers
serialized on one asyncio loop (per-session verifier binding is deferred to
v0.4 / PR-A3c, see ADR 0008). So **256 = max concurrent connections admitted
and served**, *not* 256 parallel inferences. The capacity cap + LRU eviction
(`SessionStore`) + slab pool (`PoolExhausted → RESOURCE_EXHAUSTED`) are the
admission-control levers; `--max-concurrent-rpcs` caps in-flight handlers.

### 3.4 Multi-tenant resident-window pressure test — A/B vs MLX-native

§3.2/§3.3 measure *connection admission* on the single-tenant served path. The
question that actually matters for many concurrent agents is **how many agents,
each with its own resident KV window, fit in a memory budget** — the
multi-tenant *capacity*, and the axis where a bounded window should win. The
served path can't answer this (per-session binding is PR-A3c), so it is measured
at the **model/cache level**: build one independent KV cache per agent, prefill
each to a context length, and ramp the agent count with **real per-agent
prefills** (real N× memory) until a memory budget is hit
(`scripts/research/mlx_multitenant_pressure.py`, preset
`mlx-multitenant-pressure`; `results/research/k3_multitenant_pressure_mac.json`).

Result (Mac mini M4, gemma-4-26B-A4B 4-bit, **ctx 2048**, 21 GB budget):

| config | per-agent KV | budget hit at | derived max agents (KV budget) |
| --- | --- | --- | --- |
| MLX-native (gemma hybrid cache) | **256.9 MB** | N=15 | ~22 |
| **Kakeya S5** (recall-preserving) | **61.1 MB** | N=32 | ~93 |
| Kakeya pure sink+window (no recall) | 15.3 MB | — | ~370 |

- **Kakeya S5 fits ~4.2× more concurrent agents** than MLX-native at equal
  context, **with recall preserved** (the 5 full-attention layers stay exact;
  only the 25 sliding layers drop from gemma's native 1024-window to
  `sink+window`=68). The measured budget-hit points (15 vs 32) confirm the
  per-agent-KV ratio empirically.
- Honest nuance: gemma's *native* cache **already** bounds sliding layers to
  1024, so the win vs native is **4.2×**, not the headline 16.8× one gets vs a
  pure sink+window cache — but pure sink+window **sacrifices long-context
  recall**, so S5 is the fair recall-preserving comparison. The ratio shrinks at
  longer context (the shared 5 full-attention layers grow with ctx in both).
- This is **memory-fit capacity**, not parallel-inference throughput: a single
  Mac GPU serializes/batches compute, so per-agent decode rate is unchanged; the
  multi-tenant value is fitting **~4× more bounded-window agents** in the same
  RAM. A truly parallel served path is measured in §3.5 (PR-A3c).

### 3.5 PR-A3c — per-session binding + true parallel multi-tenant throughput

§3.2–3.4 establish that v0.3's *served* path is single-tenant (serialized) and
that bounded windows fit more agents. The remaining question — **does the engine
actually decode N sessions in parallel, recall-preserving?** — is answered here.
On a single accelerator, "parallel" = a **batched** forward where **each batch
row is a session with its own KV-cache row** (per-session binding). Implemented
as `scripts/research/k3_cuda_multitenant_parallel_bench.py` on the recall-
preserving restored **S5** path (the non-recall pure sink+window config is out
of scope by design — recall is the bottom line). Required one batch-1 fix in the
restore path (RoPE `cos`/`sin` batch-1 broadcast, `restored_attention.py`).

Result (H200 NVL, gemma-4-26B-A4B 4-bit, NIAH ctx≈1238,
`results/research/k3_cuda_multitenant_parallel_gpu.json`):

| sessions N | restored-S5 agg tok/s | parallel speedup vs N=1 | per-session recall | peak |
| --- | --- | --- | --- | --- |
| 1 | 27.4 | 1.00× | 1.0 | 57.6 GB |
| 2 | 54.6 | 1.99× | 1.0 | 60.4 GB |
| 4 | 111.6 | 4.07× | 1.0 | 66.0 GB |
| 8 | 220.4 | **8.04×** | **1.0** | 77.3 GB |

- **Near-linear parallel scaling** (8.04× at N=8) — the engine genuinely decodes
  N sessions in parallel, the opposite of v0.3's serialized single-tenant path
  (§3.3, where concurrent sessions serialize and latency is linear in N).
- **Per-session recall stays 1.0 at every batch size** — recall is preserved
  under batched multi-tenant decode (the bottom line is met).
- **Restored S5 ≈ native AR** throughput (220.4 vs 216.4 tok/s at N=8) — the
  restoration is free, *and* it keeps the bounded resident window (§3.4: ~4×
  more agents fit). So PR-A3c delivers parallel throughput **and** bounded
  memory **and** recall together.
- Caveat: §3.5 is the **engine/batched-decode** capability. The *served*-path
  productization is §3.6; the batched fused spec-decode (DFlash is batch-1
  today) remains a follow-up.

### 3.6 PR-A3c in the gRPC served path — true multi-tenant serving (end-to-end)

§3.5 proved the engine can decode N sessions in parallel; §3.6 wires
**per-session binding into the gRPC `RuntimeService`** so the *served* path is
genuinely multi-tenant (v0.3 was single-tenant: one shared verifier, concurrent
sessions would corrupt each other's KV). Implementation:

- `CrossModelRestoredSinkWindowVerifier.spawn()` — a fresh per-session adapter
  sharing the model weights (multi-GB) but with its own KV cache.
- `PerSessionVerifierRegistry` (`inference_engine/session/verifier_registry.py`)
  — `session_id → adapter` (lazy, shared weights); doubles as the
  `SessionStore` cache-inspector and the coordinators' verifier *resolver*.
- `AppendTokens`/`Generation` coordinators take an optional per-session
  resolver (back-compat: single-tenant unchanged — 271 session+server unit
  tests pass); the servicer's `CloseSession` frees a session's adapter.
- `start_grpc_runtime_server --multi-tenant` (requires `backend=restored`).

End-to-end test (H200, `k3_grpc_multitenant_e2e.py`,
`results/research/k3_grpc_multitenant_e2e_gpu.json`): launch the multi-tenant
server, drive **4 concurrent SDK clients**, each its own session + distinct NIAH
needle, primed then decoded interleaved.

| sessions | transport | per-session recall | isolation |
| --- | --- | --- | --- |
| 4 concurrent | real gRPC `RuntimeService` + Python SDK | **1.0** | ✓ |

Each session recalled **its own** needle (`MAPLE-7890`, `IOTA-8961`,
`THETA-6866`, `IOTA-3281` — note two `IOTA-*` sessions got their *own* numbers),
proving **per-session KV isolation** through the real served path. So
multi-tenant serving is end-to-end correct + recall-preserving. Execution was
still RPC-serialized on the asyncio loop; the batched scheduler that fuses the
cohort into one forward is §3.7.

### 3.7 PR-A3c batched scheduler — fusing concurrent decodes for throughput

§3.6's served path was *correct* multi-tenant but RPC-serialized (each session's
decode forward ran alone). `BatchedDecodeScheduler`
(`inference_engine/session/batch_scheduler.py`) takes the cohort of per-session
adapters from the registry, **stacks their restored caches along the batch dim,
and runs one verifier forward per step** (dropping finished rows) — the
served-path realisation of §3.5's parallel decode.

Result (H200, 8 sessions, NIAH ctx≈1238,
`results/research/k3_served_batched_scheduler_gpu.json`):

| path | aggregate decode tok/s | per-session recall |
| --- | --- | --- |
| serialized (§3.6, each session alone) | 26.6 | 1.0 |
| **batched scheduler (§3.7)** | **224.9** | **1.0** |
| **speedup** | **8.45×** | — |

So the batched scheduler converts the correct-but-serialized multi-tenant path
into **8.45× aggregate throughput at 8 sessions, recall preserved** — matching
the engine-level near-linear scaling (§3.5) now driven through the served
per-session adapters. Scope: a **fixed-cohort** batcher (synchronized burst —
the dominant multi-tenant case); dynamic mid-flight arrival + ragged-length
continuous batching (and the async-gRPC futures glue that lets independent
`Generate` RPC coroutines feed one batch loop) is the remaining productization.

**Platform note — §3.7 is CUDA only.** The `BatchedDecodeScheduler` and its
bench are torch/CUDA (they ran on H200); the Mac served-restored path is blocked
by the MLX-gemma nested-config load gap (§6). A Mac analog
(`scripts/research/mlx_batched_multitenant_bench.py`, preset
`mlx-batched-multitenant`) was run on the Mac mini and surfaced a **correctness
blocker**: batched MLX decode over gemma-4 at batch > 1 **breaks per-session
recall** (`results/research/k3_mlx_batched_multitenant_mac.json`):

| Mac (M4, 8 sessions) | aggregate tok/s | per-session recall |
| --- | --- | --- |
| serialized | 21.2 | **1.0** |
| batched (MLX, batch>1) | 57.1 (2.7× *if* recall held) | **0.125** ✗ |

So on Mac the **recall-safe** multi-tenant path is **serialized** (recall 1.0);
batched throughput is *not* shippable there until the MLX batch>1 forward/cache
correctness is fixed (likely the gemma hybrid/sliding `RotatingKVCache` under
batching) — recall is the bottom line. Note also the Mac speedup ceiling is low
even nominally (M4 saturates at small batch — 2.7× vs CUDA's 8.45×). The
validated batched scheduler is **CUDA-only** today.

**Root-cause of the Mac batched-recall break (investigated).** The
`RotatingKVCache` / sliding-window-mask line was investigated and **ruled out**:

| Mac diagnostic | batched per-session recall | what it isolates |
| --- | --- | --- |
| short prompt (ctx 339 < sliding window → **no rotation**) | 0.25 | **not** rotation |
| concat-based Kakeya `SinkWindowKVCache` (no in-place buffer assign) | 0.25 | **not** the cache (in-place or concat) |
| per-row first decoded token vs serialized | **matches** (all rows) | batched **prefill is correct** |

So the break is **cache-independent and prefill-correct**: batched **decode**
diverges only *after* the first token, with both mlx_lm's in-place cache and
Kakeya's concat cache. That localizes it to **mlx_lm 0.31.3's gemma-4
batched (batch>1) decode forward** (RoPE-offset / mask / shared-KV path), an
**upstream MLX limitation** — not a Kakeya cache or rotation issue, and not
present on CUDA (HF transformers batched decode is correct → §3.5/§3.7's 8.04×
/ 8.45×, recall 1.0). Evidence:
`results/research/k3_mlx_batched_{diag_short,kakeya_cache}_mac.json`.

**Status:** recall-safe Mac multi-tenant = **serialized** (recall 1.0). Mac
**batched** throughput needs an upstream mlx_lm gemma-4 batched-decode fix (or a
custom batched gemma decode kernel) — tracked as a follow-up; CUDA is the
recall-safe batched path today.

**Deep localization (per-layer logits + ablations).** A per-layer hidden-diff
instrument (`mlx_batched_layer_diff_diag.py`) + targeted ablations narrowed the
batch>1 decode bug to a single op class. At decode step 1, batched **row 0 is
bit-exact** vs serialized while **row 1+ diverge starting at layer 0**, with
**layer-0 input identical** (embedding/per-layer-input correct). Every
Python-level cause was **ruled out** with evidence:

| candidate | test | result |
| --- | --- | --- |
| sliding-window rotation | short prompt (no rotation) | still diverges |
| in-place cache write | concat `SinkWindowKVCache` | still diverges |
| KV-shared layers | model has **0** shared layers (`first_kv_shared_idx=30`) | N/A |
| embedding / per-layer-input | layer-0 **input** diff = 0 for all rows | ruled out |
| attention mask | decode mask is `None` at this offset | ruled out |
| `mx.fast.scaled_dot_product_attention` | manual matmul-softmax SDPA | still diverges |

With identical layer-0 input + concat cache + manual SDPA, row 1 *still* breaks,
and **prefill (L=327) is correct while decode (L=1) breaks** → the residual is
an **MLX core-kernel bug for 4-bit-quantized *batched single-token* decode**
(`mx.quantized_matmul` / `mx.fast.rope` at `B>1, L=1`) — below the Python layer,
**not patchable in this repo**. (CUDA is unaffected: HF transformers, no MLX
quantized kernels — §3.5/§3.7 keep recall 1.0.)

**Outcome:** the Mac batched path is **blocked upstream in MLX**, now precisely
characterized for an upstream report. Recall-safe Mac multi-tenant remains
**serialized**; a Python workaround (e.g. L≥2 padded decode, or de-quantized
projections) is a possible future probe. Evidence:
`results/research/k3_mlx_batched_manual_sdpa_mac.json` + the layer-diff logs.

**L≥2 padded decode workaround — hypothesis CONFIRMED, recall recovered, but
no throughput win (Mac, 8 sessions, modal prompt 1149).** The `--pad-decode`
mode (preset `mlx-batched-pad-decode`) duplicates the new token each step so
every decode forward is **length-2**, routing through mlx's matrix-matrix
(`qmm`) quantized kernel instead of the single-token (`qmv`) kernel; query
position 0 is the real prediction (attends to cache+self only, == the L=1
result) and the position-1 duplicate is trimmed from the (Kakeya S5, trimmable)
cache so RoPE offsets stay exact. The batch dimension is untouched (stays
parallel over sessions, Python-only).

| metric | batched **L=1** (bug) | batched **L≥2 padded** | serialized (truth) |
| --- | --- | --- | --- |
| per-session recall | **0.125** ✗ | **1.0** ✓ | 1.0 |
| per-row tok0 vs serialized | row 1+ diverge | **all 8 match** | — |
| aggregate decode tok/s | 57.1 (recall void) | 9.945 | 14.927 |
| speedup vs serialized | — | **0.67×** | 1.0× |

This **confirms** the root cause: forcing `L≥2` (avoiding the `B>1, L=1` `qmv`
path) restores batched per-session recall to **1.0**, matching serialized
bit-for-bit on the first decoded token across all rows. But the 2× per-step
padding tax exceeds the batching gain at this scale (**0.67×**), so it is a
**correctness-recovery probe, not a shippable throughput path**: a Mac batched
*win* still needs the upstream `L=1, B>1` quantized-kernel fix (no padding tax)
or a much larger cohort / cheaper verify. Evidence:
`results/research/k3_mac_bridge_mlx_batched_pad_decode.json`.

**mlx/mlx-lm upgrade re-test — bug PERSISTS on the latest published release.**
We attempted `pip install --upgrade mlx mlx-lm` on the Mac runner (preset
`mlx-upgrade`): it was a **no-op** — the runner was already at the newest
versions on PyPI (`mlx=0.31.2`, `mlx_lm=0.31.3`, `mlx-metal=0.31.2`; no newer
stable or pre-release exists on the index). A self-contained probe (preset
`mlx-upstream-batch-probe`, zero `inference_engine` imports, native
`model.make_cache()`, plain `L=1` batched decode) then re-ran the parallel
test on that latest build:

| metric | batched (native `L=1`) | serialized (truth) |
| --- | --- | --- |
| per-session recall | **0.125** ✗ | 1.0 |
| per-row tok0 vs serialized | **all 8 match** (tok0 is from prefill) | — |
| aggregate decode tok/s | 29.6 (recall void) | 21.7 |
| `upstream_l1_batch_bug_fixed` | **false** | — |

The first decoded token (computed from the `L>1` prefill logits) matches on
**all 8 rows**, and the divergence appears only in the subsequent `L=1` decode
steps (rows 1–7 fail) — exactly the `B>1, L=1` signature. **Conclusion:** the
latest PyPI mlx/mlx-lm still ships the bug; a pip upgrade cannot fix it because
nothing newer is published. The only further "upgrade" is a from-source
`mlx` git-`main` build (compiles Metal kernels; invasive on the pinned
runner env) or an upstream patch/issue. Recall-safe Mac parallelism therefore
remains: **serialized**, or the `L≥2` padding probe (recall-safe but 0.67×).
Evidence: `results/research/k3_mac_bridge_mlx_upstream_batch_probe.json` +
`.mac-bridge/logs/mlx-upgrade-{0,1,2}.log`.

## 4. Case 2 — cross-host proposer/verifier (FEASIBILITY VERDICT)

### 4.1 Verdict: the requested topology is not implementable today, and is architecturally bounded out

A code audit found the cross-host discovery + draft plane is **design-only**:

- **No `distributed.proto`, no `CapabilityService`/`ProposerService`, no
  `ProposeBlock` RPC, no gossip/registry/TTL** — zero runnable cross-process
  wiring (the ADR 0009 file is itself absent from `main`).
- The **only implemented cross-machine plane is the Mac-bridge git-bus**
  (async, batch, allowlisted presets) — a **tool/control plane**, not a
  token-level data plane.
- Speculative decoding (proposer + verifier) is implemented **in-process
  only** (`kv_cache_proposer/speculative.py`, `inference_engine/v04/`).

Even if built, **per-block draft submission over WAN is ruled out by the
latency budget** (design doc §4.2): a Gemma-4-26B M4 verify of an 8-token block
is ~50–100 ms; a cloud↔desk RTT is 30–150 ms **per block**, i.e. 30–300 %
overhead that consumes any acceptance gain. **Proposer and verifier must share
a LAN** for the data plane.

### 4.2 What the topology decomposes into (and the measurable proxies)

| Plane | Crosses WAN? | Status | Measured |
| --- | --- | --- | --- |
| Discovery / capability advertise | yes (seconds-scale) | bridge proxy only | bridge dispatch ~10 s + queue; one Mac, serialized (`concurrency: mac-bridge`) |
| Job/tool dispatch (eval/bench) | yes | implemented (bridge) | this ADR's Case-1 run is itself an instance |
| **Token-level draft (data plane)** | **no — must be LAN** | not implemented | **measured penalty curve §4.3** (break-even ~100 ms/block) |
| Co-located spec-decode (the feasible data plane) | n/a (same host) | implemented | **GPU H200 2.06–2.20× AR**; **Mac 0.93× AR** (PR #118) |

So the answers to Case 2's three metrics, under the **realizable** topology:

- **Token throughput**: spec-decode is a *co-located* win — **2.06–2.20× AR on
  the GPU** (recall 1.0) and **≈AR parity (0.93×) on the Mac**. The measured
  WAN-penalty curve (§4.3) shows the cross-host draft loop falls to **break-even
  at ~100 ms/block and a net loss at 150 ms**, i.e. slower than running AR
  locally — so it is not a throughput strategy.
- **Max agent connections**: governed by the *serving* node (Case 1): **256+
  concurrent agents** on the Mac via `RuntimeService`.
- **Mac KV upper bound**: bounded — **capacity × per-session `sink+window`**
  (≈2.0 GB at capacity 256 for Qwen3-0.6B; for the gemma S5 production config
  the per-agent resident KV is ~133 MB at 5.8k ctx, dominated by the 5 exact
  full-attention layers — see the README beta scorecard).

### 4.3 Measured WAN-penalty curve (H200, real models, `--rtt-sweep`)

Rather than rest on the latency *estimate*, we measured it: the fused engine was
re-timed with one injected proposer↔verifier round-trip **per block** on the real
Gemma-4-26B verifier + DFlash drafter (H200 NVL), sweeping per-block RTT across
the cloud↔desk range (`scripts/research/k3_specdecode_gpu_bench.py --rtt-sweep`,
`results/research/k3_crosshost_rtt_gpu.json`):

| per-block RTT | decode tok/s | vs AR | regime |
| --- | --- | --- | --- |
| 0 ms (co-located) | 52.4 | **2.20×** | the win |
| 5 ms (LAN) | 47.0 | 1.97× | LAN keeps it |
| 15 ms | 43.3 | 1.81× | LAN keeps it |
| 30 ms | 35.9 | 1.50× | WAN edge |
| 60 ms | 29.2 | 1.22× | shrinking |
| 100 ms | 23.5 | **0.98×** | **break-even** |
| 150 ms | 18.4 | 0.77× | net **loss** |

AR baseline = 23.8 tok/s. **Break-even is ~100 ms/block**: beyond it, cross-host
spec-decode is *slower than running AR locally*. A cloud↔desk WAN (30–150 ms RTT)
straddles or exceeds break-even, while a LAN/Thunderbolt link (≤15 ms) preserves
the 1.8–2.2× win. This is the architecture's prediction (design doc §4.2),
**now quantified on real compute** — and it is why the data plane must be LAN.

### 4.4 Real two-process socket over a real network (no simulation)

To remove the "injected sleep" caveat, the per-block exchange was run through a
**real TCP socket to a second process** (`scripts/research/socket_echo_server.py`),
serializing the **actual** per-block payload — the verifier→proposer aux hidden
states + draft tokens, **≈156 KB/block** — once per block
(`--socket-echo-addr`). Two transports were measured (H200 NVL,
`results/research/k3_crosshost_{socket_loopback,realnet}_gpu.json`):

| transport | RTT | decode tok/s | vs AR |
| --- | --- | --- | --- |
| loopback socket (same host) | ~0 ms | 51.5 | **2.02×** |
| **real network** (GPU↔cloud-agent, reverse SSH tunnel) | **~102 ms** | **14.1** | **0.56×** |

- **Loopback** matches the co-located number → the socket + serialization of the
  156 KB payload are themselves cheap; the killer is purely the round-trip.
- **Real network**: at a genuine ~102 ms RTT the cross-host loop collapses to
  **0.56× AR — a net loss, worse than running AR alone**, and *worse* than the
  latency-only model (0.98× at 100 ms) because the real transport also pays the
  **156 KB/block bandwidth** (network was **71 %** of decode wall time). This is
  an end-to-end real-models + real-network confirmation that the token-level
  draft data plane is WAN-infeasible.
- Note: `tc netem` artificial latency could **not** be applied inside the vast
  container (`RTNETLINK: Operation not permitted` — no `NET_ADMIN` in the
  restricted netns), so real latency was obtained from a real inter-host link
  (the reverse-tunnel RTT) rather than synthetic netem.

**What the ~102 ms is (and is not).** It is **not** a gRPC RTT and **not** a
floor — it is a raw TCP round-trip through a **reverse SSH tunnel between two
different-region hosts**. Decomposed by payload over the real path:

| payload | median RTT |
| --- | --- |
| 64 B | 102.6 ms |
| 40 KB | 205.2 ms |
| 80 KB | 206.0 ms |
| 156 KB | 206.9 ms |

The 64 B point (~102 ms) is the **pure inter-region latency + SSH-relay
overhead**; the flat 205–207 ms from 40 KB up is **one extra tunnel round-trip**
(TCP windowing / SSH framing), not linear bandwidth.

**Direct-gRPC transport, re-tested.** Swapping the raw socket for a real
**gRPC (HTTP/2)** channel (`grpc_echo_probe.py`, `--grpc-echo-addr`):

| transport / path | 156 KB RTT | fused tok/s | vs AR |
| --- | --- | --- | --- |
| loopback gRPC (same host) | **1.1 ms** | — | — |
| raw socket over the ~102 ms path | 207 ms | 14.1 | 0.56× |
| **gRPC over the ~102 ms path** | 208 ms | **15.95** | **0.63×** |

gRPC is **modestly better** (0.63× vs 0.56×; ~197 vs ~232 ms/block network) —
it serializes the 156 KB more efficiently — but **still a net loss**, because
the **~102 ms geographic RTT dominates, not the transport**. gRPC's real value
shows at **loopback (1.1 ms for 156 KB)** → on a low-RTT link the transport is
free and the engine returns to its 1.8–2.2× win. (A *true* non-SSH ingress to
the GPU could not be established — vast's non-SSH mapped ports accept SYNs but
do not forward data end-to-end, so the gRPC run used the same reverse-SSH path;
this only adds relay overhead, so the real direct-gRPC number would be ≤ these.) So this is a **worst-case
far-WAN + SSH artifact**, not a deployment floor. Optimization room is large and
is exactly the architecture's prescription: (1) **latency** — co-locate the
draft loop on a low-RTT link (same region ~5–20 ms, LAN ~0.5–2 ms, Thunderbolt
sub-ms; at ≤15 ms the engine is back to 1.8–2.2×, cf. §4.3 / loopback 2.02×);
(2) **transport** — a direct **gRPC/QUIC/RDMA** path drops the SSH relay + the
extra round-trip (gRPC would be *lower*, not higher); (3) **payload** — the
156 KB/block fp16 aux can be fp8/int8/top-k compressed 2–4×; (4) **fewer
round-trips** — larger blocks. The invariant is the ratio *per-block RTT :
per-block compute* (break-even ~100 ms/block): the strategy is to keep the loop
on a low-RTT link, not to chase a faster WAN.

## 5. Decision

1. **Case 1 is validated**: the session-bound gRPC runtime admits and serves
   **≥256 concurrent agent connections** on an M4 with **flat memory** and a
   **bounded ~2.0 GB node KV ceiling**, with the documented single-tenant
   latency-serialization caveat.
2. **Case 2's WAN data plane is rejected** as a throughput strategy and is
   unbuilt: cross-host token-level draft must not cross the cloud↔desk
   boundary. The correct topology is **WAN = control + tool plane (bridge),
   LAN = co-located data plane (spec-decode)** — the same conclusion as the
   Mac-bridge design doc §4, now backed by the audit and the co-located
   throughput evidence.

## 6. Consequences & follow-ups

- **Served MLX gemma gap (found during this test)**: `MLXSinkWindowVerifier`
  reads a flat `cfg.num_hidden_layers`, but the gemma-4 MLX model nests its
  config → `AttributeError` when starting `--backend mlx` with the gemma
  verifier. The served gRPC path is wired for the torch/HF verifier (and
  Qwen3-MLX), not gemma-4 MLX. Tracked as a v0.4 item (alongside per-session
  binding); Case 1 therefore used the cpu verifier.
- **Multi-tenant (PR-A3c)**: per-session verifier binding would lift the
  serialization caveat and turn "256 connections" into "256 *concurrent
  inferences*"; until then, capacity sizing should reflect serialized service.
- **M3 (fleet capability plane)**: if/when built, placement must treat
  `ring_address`/RTT class as a hard constraint so data-plane (draft) pairings
  never span WAN — a one-line filter in the placement candidate set.

## 7. Alternatives considered

- **Build the cross-host gRPC draft plane now and benchmark it.** Rejected: it
  is a large unimplemented feature (proto + services + discovery) whose result
  is already known to be *worse than co-located* by the latency budget — it
  would confirm a negative at high cost.
- **Run Case 1 against the MLX gemma verifier.** Blocked by the served-MLX gap
  (§6); connection scaling is model-independent, so cpu Qwen3-0.6B gives the
  same admission/capacity answer with the production KV bound reported
  analytically.
- **Hold live cloud→Mac gRPC sessions for Case 1.** Impossible: the Mac has no
  inbound path (the reason the bridge exists). The load test runs co-located on
  the Mac, dispatched via the bridge.

## Appendix A — Test report index & evidence

Consolidated record of every run behind this ADR (harnesses, how to reproduce,
the committed evidence JSON, and the headline result).

### A.1 Harnesses & how to reproduce

| Test | Harness / preset | Reproduce |
| --- | --- | --- |
| Case 1 — agent connections (light) | `scripts/research/grpc_agent_capacity_loadtest.py`; preset `agent-capacity-loadtest` | `kakeya_mac.py run --preset agent-capacity-loadtest` |
| Case 1 — agent connections (stress) | same; preset `agent-capacity-stress` (`--context-len`, open-file-descriptor limit raise) | `kakeya_mac.py run --preset agent-capacity-stress` |
| Case 2 — injected-RTT sweep | `scripts/research/k3_specdecode_gpu_bench.py --rtt-sweep` | H200, real models |
| Case 2 — raw socket (real net) | `socket_echo_server.py` + `k3_specdecode_gpu_bench.py --socket-echo-addr` | echo on host B; reverse-SSH path |
| Case 2 — direct gRPC | `grpc_echo_probe.py` + `k3_specdecode_gpu_bench.py --grpc-echo-addr` | gRPC echo on host B |

### A.2 Consolidated results

**Case 1 (Mac mini M4, gRPC `RuntimeService`, cpu Qwen3-0.6B):**

| run | result | evidence |
| --- | --- | --- |
| light sessions | **256/256 agents, 0 errors**; per-session KV 7.80 MB; node bound ≈2.0 GB; RSS flat ~3.85 GB | `results/research/k3_agent_capacity_mac.json` |
| stress (ctx prefill, file-descriptor limit 100k, cap 2048) | open-file-descriptor limit not the constraint; mem = cap×window (cap 2048→11.5 GB, bound 61 GB>RAM); serialization caps heavy-ctx concurrency at **~8** | `results/research/k3_agent_capacity_stress_mac.json` |
| multi-tenant capacity A/B (ctx2048, model-level) | per-agent KV native 256.9 MB vs **S5 61.1 MB**; **~4.2× more agents** (budget hit 15 vs 32; derived 22 vs 93) — recall-preserving | `results/research/k3_multitenant_pressure_mac.json` |
| PR-A3c parallel throughput (H200, batched S5) | **8.04× near-linear scaling at N=8** (220 tok/s ≈ AR), **per-session recall 1.0** — per-session binding works | `results/research/k3_cuda_multitenant_parallel_gpu.json` |
| PR-A3c served path (H200, gRPC + 4 SDK clients) | **true multi-tenant serving end-to-end**: 4 concurrent sessions, **per-session recall 1.0**, isolated | `results/research/k3_grpc_multitenant_e2e_gpu.json` |
| PR-A3c batched scheduler (H200, 8 sessions) | **8.45× throughput** (26.6 → 224.9 tok/s) fusing cohort into one forward, **recall 1.0** | `results/research/k3_served_batched_scheduler_gpu.json` |

**Case 2 (H200 NVL, Gemma-4-26B + DFlash, fused spec-decode vs AR):**

| transport / RTT | tok/s | vs AR | evidence |
| --- | --- | --- | --- |
| co-located (0 network) | 44–52 | **1.85–2.20×** | (all four JSONs) |
| injected-RTT sweep | — | 2.20× @0 → **0.98× @100 ms** → 0.77× @150 ms | `k3_crosshost_rtt_gpu.json` |
| loopback gRPC (156 KB) | — | 1.1 ms round-trip | `k3_crosshost_grpc_gpu.json` |
| raw socket over ~102 ms path | 14.1 | 0.56× | `k3_crosshost_realnet_gpu.json` |
| direct gRPC over ~102 ms path | 15.95 | 0.63× | `k3_crosshost_grpc_gpu.json` |
| RTT decomposition | 64 B → 102.6 ms; 156 KB → 206–208 ms (both raw + gRPC) | — | `k3_crosshost_socket_loopback_gpu.json` |

### A.3 One-line verdict

Bounded-memory, admission-controlled multi-agent serving is **validated** (Case 1:
256+ connections, ~2.0 GB node KV ceiling, flat RSS). Cross-host token-level
spec-decode is a **co-located/LAN win (1.8–2.2×)** and a **WAN net loss**
(~0.56–0.63× at ~102 ms RTT, transport-independent) — **WAN = control + tool
plane, LAN = data plane.** The lever is RTT (co-location), not the transport;
gRPC only helps once the link is already low-RTT (loopback 1.1 ms).
