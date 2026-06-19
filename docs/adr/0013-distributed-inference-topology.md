# ADR 0013 — Distributed inference topology: what AR sequentiality allows

- **Status**: Accepted (2026-06-13)
- **Date**: 2026-06-13
- **Relates to**: ADR 0009 (mlx.distributed + capability exchange — this ADR
  is its topology companion / clarification), ADR 0008 (session-bound runtime),
  ADR 0001 (proposer sizing), ADR 0012 (value proposition).

## Context

A recurring vision is raised for "distributed inference": *decompose one
inference task into several parallel subtasks, have the proposer coordinate
across multiple verifiers, and win total token throughput* — extrapolated to
**many-to-many** proposer/verifier wiring (one verifier fed by many proposers;
one proposer drafting for many verifiers).

ADR 0009 shipped the *substrate* (capability exchange + remote `ProposeBlock` +
`DistributedSpeculativeDecoder` + an optional `mlx.distributed` data plane) but
did not pin down **which inference topologies are physically achievable**. This
ADR fixes the can/can't-parallelize conclusion so it is not re-derived each time
the idea resurfaces.

## Decision

### The governing constraint

**Single-sequence autoregressive (AR) decoding is inherently sequential**:
token `N+1` depends on the realized value of token `N`. A single sequence's
token chain therefore **cannot** be split into independent parallel subtasks
across multiple verifiers the way a batch / map-reduce job can. This is a
causal-dependency property of AR generation, not an engineering gap.

The only parallelism available to a **single** sequence is:

1. **Intra-forward (model parallelism)** — split *one* verifier's weights/compute
   across hosts via tensor/pipeline parallelism (`mlx.distributed`
   `model.shard`, ADR 0009 §2.1). This is "one verifier across N hosts," not "N
   verifiers." It enables / accelerates a verifier too big for one host;
   throughput scales **sublinearly** (collective-communication bound), and its
   real purpose is fit + latency, not linear throughput multiplication.
2. **Intra-block (speculative decoding)** — the verifier checks `L` drafted
   tokens in **one batched forward**; throughput gain = `acceptance × block`,
   amortizing one verify over many tokens. The `verify(L)` cost is **sublinear**
   in `L` (`results/research/verify_l_sweep.json`: ~4× at L=16), which is the
   headroom that makes blocks and trees pay off.
3. **N:1 tree / multi-candidate speculation** — multiple drafts (many proposers,
   or one proposer emitting a token *tree*) are verified in **one** batched
   forward via tree attention; the longest correct path is accepted. This
   raises **single-request** throughput by exploiting the sublinear `verify(L)`
   headroom from (2).

### The topologies, mapped to feasibility

| Topology | Realizable? | What it is | Status on the ADR-0009 substrate |
|---|---|---|---|
| **Split one sequence across N independent verifiers in parallel** | ❌ No | category error — blocked by AR sequentiality | n/a |
| **Single big verifier sharded across hosts** (1 verifier, N hosts) | ✅ Yes | tensor/pipeline parallel of one model | `mlx.distributed` ring adapter shipped (ADR 0009 §4.4); sharding is mlx-lm `model.shard` |
| **N proposers : 1 verifier** (tree / multi-candidate) | ✅ Yes — **the** path to single-request throughput | parallel candidate verification | **feasible, not built** — current `DistributedSpeculativeDecoder` is single `RemoteProposer` + linear accept; needs tree-attention verify + multi-proposer aggregation |
| **1 proposer : N verifiers** | ✅ Yes (already realized) | a shared proposer capability serves many independent verifier sessions | shipped: `ProposerService` + capability exchange (ADR 0009 §4) |

### What "total throughput advantage" means (two regimes)

- **Single-request throughput**: only (1) intra-verifier model parallelism and
  (3) N:1 tree speculation help. Multiple *independent* verifiers do **not** —
  there is nothing to parallelize across them for one sequence.
- **Fleet / aggregate throughput** (many independent requests): the **1:N**
  proposer-sharing + role placement is the realized win — it raises utilization
  (offload the asymmetrically-cheap 0.25–1 B proposer, free
  `proposer_weight_bytes` on verifier hosts), but does not speed up any single
  request beyond ordinary spec-decode.

## Consequences

- For **single-request throughput**, the correct next investment is **N:1 tree /
  multi-candidate speculation** built on the ADR-0009 capability substrate +
  the sublinear `verify(L)` headroom — **not** "more verifiers." This is tracked
  as a v0.5+ extension to `DistributedSpeculativeDecoder` (territory of the
  ADR-0009 / capability-plane workstream).
- **Multi-host spec-decode trades latency for placement**: F3 (aux hidden
  states) is MB/block on the critical path (ADR 0009 F-flow table); it only pays
  off behind a fast data plane (ring / `jaccl`). Distributing for its own sake
  can regress single-request latency.
- The **Mac bridge** (`scripts/mac_bridge/`) used for dev/eval is an instance of
  the capability plane's **tool plane**, *not* a production inference data
  plane. "The multi-host tool plane is running" must not be extrapolated to "a
  distributed inference data plane is ready."
- Any future "let's parallelize one request across machines" proposal must first
  identify which of the four topologies it is; the "N independent verifiers on
  one sequence" form is closed.

## Alternatives considered

- **"Decompose one sequence into parallel subtasks across N verifiers."**
  Rejected: AR sequentiality (token `N+1` needs token `N`) makes the subtasks
  causally dependent, not independent. No coordination protocol recovers
  independence that the math forbids.
- **"Multiple verifiers vote / ensemble on one sequence for speed."** Rejected
  for throughput: ensembling changes *quality semantics* and still runs each
  verifier over the same sequential chain — it multiplies cost, not speed.
- **"Treat distribution as the throughput lever."** Rejected as the primary
  lever: the realized throughput wins are intra-block (spec-decode, single host)
  and fleet-aggregate (1:N); cross-host distribution's single-request value is
  bounded by F3 latency and is a fit/placement tool, not a linear scaler.

## Evidence pointers

- `verify(L)` sublinearity (the headroom tree-spec would exploit):
  `results/research/verify_l_sweep.json` (3.92× measured @ L=16).
- F-flow latency analysis + `mlx.distributed` data-plane scope: ADR 0009 §2.
- Realized 1:N substrate: ADR 0009 §4 (`CapabilityService`, `ProposerService`,
  `DistributedSpeculativeDecoder`); `inference_engine/distributed/`.
