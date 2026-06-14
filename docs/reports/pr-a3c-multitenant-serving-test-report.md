# PR-A3c — Multi-tenant serving: detailed test report

Detailed record of the per-session-binding (PR-A3c) work and its end-to-end
tests, from the single-tenant pressure finding through batched parallel
throughput, the gRPC served path, and the batched scheduler. Summarized in
[ADR 0014](../adr/0014-agent-connection-capacity-and-cross-host-topology-tests.md)
§3.4–3.7; this report carries the full methodology, numbers, and evidence index.

## 1. Environment

| Component | Detail |
| --- | --- |
| GPU | NVIDIA **H200 NVL** (143 GB), vast.ai; torch 2.12.0+cu130 |
| Verifier | `google/gemma-4-26B-A4B-it` (bf16, eager attention) |
| Drafter | `z-lab/gemma-4-26B-A4B-it-DFlash` |
| f_θ | `results/research/f_theta_v5_s5_sliding` (S5, 5 exact full-attn layers) |
| Mac | Mac mini M4 (24 GB) via the git-bus Mac bridge (single-tenant pressure, §3.4 source) |
| Recall task | NIAH (needle-in-a-haystack); per-session distinct needle |
| Bottom line | **recall must stay 1.0** — recall-sacrificing configs (pure sink+window) are out of scope |

## 2. Motivation — the single-tenant gap

The first capacity test measured the v0.3 **served** path (`grpc_agent_capacity_loadtest.py`):
256 concurrent agent **connections** admitted, but v0.3 is **single-tenant** — one
shared verifier, RPCs serialized on one asyncio loop, no per-session KV isolation.
So "256" = connections *served*, not parallel inferences; concurrent sessions
would corrupt each other's KV. PR-A3c (per-session binding) is the fix.

## 3. Tests & results

### 3.1 Multi-tenant memory capacity (model-level A/B) — ADR §3.4

`mlx_multitenant_pressure.py` (Mac M4), per-agent KV at ctx2048, 21 GB budget:

| config | per-agent KV | budget hit at | derived max agents |
| --- | --- | --- | --- |
| MLX-native (gemma hybrid) | 256.9 MB | N=15 | ~22 |
| **Kakeya S5** (recall-preserving) | 61.1 MB | N=32 | ~93 |

→ **~4.2× more concurrent agents** at equal context, recall-preserving (the win
vs native, which already bounds sliding layers to 1024; pure sink+window's 16.8×
is excluded — it drops full-attn recall).

### 3.2 Batched parallel throughput (engine level) — ADR §3.5

`k3_cuda_multitenant_parallel_bench.py` (H200), batched restored-S5 decode,
each row = a session with its own KV-cache row:

| sessions N | restored-S5 agg tok/s | parallel speedup | per-session recall |
| --- | --- | --- | --- |
| 1 | 27.4 | 1.00× | 1.0 |
| 2 | 54.6 | 1.99× | 1.0 |
| 4 | 111.6 | 4.07× | 1.0 |
| 8 | 220.4 | **8.04×** | 1.0 |

→ near-linear parallel scaling; restored S5 ≈ native AR (220.4 vs 216.4 @ N=8).
One batch-1 fix was required: RoPE `cos`/`sin` batch-1 broadcast
(`restored_attention.py`). Evidence: `k3_cuda_multitenant_parallel_gpu.json`.

### 3.3 gRPC served path — per-session binding (end-to-end) — ADR §3.6

Implementation: `CrossModelRestoredSinkWindowVerifier.spawn()` (fresh per-session
adapter, shared weights) + `PerSessionVerifierRegistry` (session→adapter; also
the `SessionStore` cache-inspector + coordinator resolver) + coordinator
resolver + servicer `on_session_close` cleanup + `start_grpc_runtime_server
--multi-tenant`. Back-compat: single-tenant unchanged (271 session+server unit
tests pass; `test_verifier_registry.py` proves interleaved-session isolation).

E2E (`k3_grpc_multitenant_e2e.py`, H200): launch the multi-tenant server, 4
concurrent SDK clients, each its own session + distinct needle:

| sessions | transport | per-session recall | isolation |
| --- | --- | --- | --- |
| 4 concurrent | real gRPC `RuntimeService` + Python SDK | **1.0** | ✓ |

Each recalled its own needle (`MAPLE-7890`/`IOTA-8961`/`THETA-6866`/`IOTA-3281` —
two `IOTA-*` sessions got their own numbers) → per-session KV isolation through
the real served path. Evidence: `k3_grpc_multitenant_e2e_gpu.json`.

### 3.4 Batched scheduler — fusing concurrent decodes — ADR §3.7

`BatchedDecodeScheduler` (`inference_engine/session/batch_scheduler.py`) stacks
the cohort's per-session restored caches along the batch dim and runs one
verifier forward per step (drops finished rows). `k3_served_batched_scheduler_bench.py`
(H200, 8 sessions):

| path | aggregate decode tok/s | per-session recall |
| --- | --- | --- |
| serialized (each session alone) | 26.6 | 1.0 |
| **batched scheduler** | **224.9** | **1.0** |
| **speedup** | **8.45×** | — |

→ the served path goes from correct-but-serialized to **8.45× aggregate
throughput at 8 sessions, recall preserved**. Evidence:
`k3_served_batched_scheduler_gpu.json`.

## 4. Net result

PR-A3c delivers, recall-preserving (recall 1.0 throughout), the three multi-tenant
properties together:

- **Bounded memory** — ~4.2× more concurrent agents per GB (§3.1).
- **Parallel throughput** — 8.04× engine-level (§3.2) / **8.45× through the
  served per-session adapters via the batched scheduler** (§3.4).
- **Correct isolation** — true multi-tenant serving end-to-end through gRPC,
  per-session recall 1.0 (§3.3).

## 5. Remaining work (productization)

- **Async continuous batching transport**: wire the batched scheduler under the
  async gRPC streaming `Generate` handlers via per-step futures + a background
  batch loop, so independent RPC coroutines feed one batch — and support
  **dynamic mid-flight arrival + ragged-length** cohorts (this report's
  scheduler is a fixed synchronized cohort, the dominant burst case).
- **Batched fused spec-decode** (DFlash is batch-1 today).
- **Mac served path**: the served MLX gemma verifier can't load gemma-4's nested
  config (`MLXSinkWindowVerifier`) — a v0.4 item; CUDA is the recall-preserving
  served path today.

## 6. Evidence index

| Test | Script | Evidence JSON |
| --- | --- | --- |
| Single-tenant capacity (Mac) | `grpc_agent_capacity_loadtest.py` | `k3_agent_capacity_mac.json`, `k3_agent_capacity_stress_mac.json` |
| Memory A/B (Mac) | `mlx_multitenant_pressure.py` | `k3_multitenant_pressure_mac.json` |
| Parallel throughput (H200) | `k3_cuda_multitenant_parallel_bench.py` | `k3_cuda_multitenant_parallel_gpu.json` |
| Served e2e (H200) | `k3_grpc_multitenant_e2e.py` | `k3_grpc_multitenant_e2e_gpu.json` |
| Batched scheduler (H200) | `k3_served_batched_scheduler_bench.py` | `k3_served_batched_scheduler_gpu.json` |

(All JSON under `results/research/`.)
