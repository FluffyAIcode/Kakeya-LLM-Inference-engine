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
- **GPU**: H200 (vast.ai) — used for the co-located CUDA reference (PR #119);
  **unavailable at test time** (instance recycled), so Case-2 GPU numbers are
  cited from the prior committed evidence.
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

### 3.3 Honest caveat — v0.3 is single-tenant

Create/generate latency scales **linearly** with `N` (256 agents → gen p95
26 s). That is the single-tenant signature: one shared verifier, RPC handlers
serialized on one asyncio loop (per-session verifier binding is deferred to
v0.4 / PR-A3c, see ADR 0008). So **256 = max concurrent connections admitted
and served**, *not* 256 parallel inferences. The capacity cap + LRU eviction
(`SessionStore`) + slab pool (`PoolExhausted → RESOURCE_EXHAUSTED`) are the
admission-control levers; `--max-concurrent-rpcs` caps in-flight handlers.

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
| **Token-level draft (data plane)** | **no — must be LAN** | not implemented | — (ruled out by §4.1) |
| Co-located spec-decode (the feasible data plane) | n/a (same host) | implemented | **GPU H200 1.79× AR** (PR #119); **Mac 0.93× AR** (PR #118) |

So the answers to Case 2's three metrics, under the **realizable** topology:

- **Token throughput**: spec-decode is a *co-located* win — **1.79× AR on the
  GPU** (28.94 vs 16.13 tok/s, recall 1.0) and **≈AR parity (0.93×) on the
  Mac**. A WAN GPU-proposer→Mac-verifier draft loop would be **slower than
  running either side alone** (§4.1), so it is not a throughput strategy.
- **Max agent connections**: governed by the *serving* node (Case 1): **256+
  concurrent agents** on the Mac via `RuntimeService`.
- **Mac KV upper bound**: bounded — **capacity × per-session `sink+window`**
  (≈2.0 GB at capacity 256 for Qwen3-0.6B; for the gemma S5 production config
  the per-agent resident KV is ~133 MB at 5.8k ctx, dominated by the 5 exact
  full-attention layers — see the README beta scorecard).

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
