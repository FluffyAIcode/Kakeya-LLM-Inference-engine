# ADR 0009 — Multi-host milestone: AR-verifier / dLM-proposer on `mlx.distributed`, and the agent capability exchange plane

- **Status**: Accepted
- **Date**: 2026-06-10
- **Relates to**: ADR 0001 (proposer sizing), ADR 0006 (local agent
  infrastructure positioning), ADR 0008 (session-bound runtime + gRPC),
  ADR 0008 §11 (K-series dLM K/V restoration)
- **Companion design doc**:
  [`docs/design/agent-capability-exchange-platform.md`](../design/agent-capability-exchange-platform.md)

## 1. Context

Kakeya is positioned (ADR 0006) as **local agent infrastructure for
Mac**. Through v0.3 every deployment is a single host: one runtime
process, one verifier, sessions bound to one machine. Two pressures
push past one host:

1. **The AR-verifier / dLM-proposer split is naturally asymmetric.**
   The proposer band is fixed at 0.25–1 B params (ADR 0001) while the
   verifier scales with quality targets (Qwen3-1.7B today, Gemma-4-26B
   in the K3 track). On a 16–24 GB Mac mini the verifier wants the
   whole unified-memory budget; evicting the proposer to a second
   Mac mini frees roughly `proposer_weight_bytes + proposer activation
   peak` on the verifier host and lets the proposer run its K diffusion
   steps concurrently with other work.
2. **Agent fleets are appearing on real desks.** Multiple Mac minis on
   one Thunderbolt/10GbE segment, each running Kakeya with different
   models warmed, different quantizations, and different roles. Today
   they cannot discover one another or trade work.

Meanwhile Apple's MLX has grown a real multi-host story,
**`mlx.distributed`**, which we evaluated for this milestone:

- **Backends**: `ring` (TCP/IP, the default; works over Ethernet or
  Thunderbolt bridge at ~10–40 Gb/s) and `jaccl` (RDMA over
  Thunderbolt 5, ~80 Gb/s, macOS 26.2+, TB5-only, full-mesh). MPI is
  also supported where installed.
- **Programming model**: SPMD. `mlx.launch --hostfile …` starts the
  *same* program on every host; ranks coordinate through collectives
  (`all_sum`, `all_gather`, `send`/`recv`) on a static `Group` that is
  fixed at process start.
- **What mlx-lm builds on it**: tensor parallelism via
  `model.shard(group)` and pipeline parallelism for architectures with
  `PipelineMixin` — both for a *single* model too big for one host.
- **Known sharp edges** (June 2026): Metal's ~5 s command-buffer
  timeout fires in distributed settings unless communication is issued
  on `stream=mx.cpu`; long prefills need chunking that `mlx.distributed`
  does not yet do; `jaccl` requires disabling Thunderbolt Bridge and a
  recovery-OS `rdma_ctl enable`; node membership is static — a dead
  rank kills the job.

## 2. Question 1 — should the AR-verifier / dLM-proposer pair run *on*
`mlx.distributed`?

We decompose spec decode traffic into its three flows and evaluate each
against `mlx.distributed`'s strengths:

| Flow | Payload per block (L=16) | Frequency | Latency sensitivity |
| --- | --- | --- | --- |
| F1: committed prefix → proposer | ≤ a few hundred `uint32` ids (~1 KB) | once per block | low — hidden by proposer compute (tens of ms for K diffusion steps) |
| F2: draft block → verifier | L `uint32` ids (64 B) | once per block | low |
| F3: aux hidden states → drafter (K3 DFlash only) | `L_ctx × hidden × n_aux_layers` bf16 — **MBs per block** | once per block | **high** — on the critical path before drafting |

### 2.1 Strengths of `mlx.distributed` for this pair

- **Bandwidth where it matters (F3).** DFlash-style drafters (ADR 0008
  §11, K3) condition on verifier hidden states. At Gemma-4-26B scale
  that is megabytes per block; ring-over-Thunderbolt or `jaccl` RDMA
  moves that 10–50× faster than a gRPC/protobuf hop, and MLX arrays
  cross without serialization into Python objects.
- **Unified memory + native arrays.** No host↔device staging on either
  end; an `mx.array` produced by the verifier's forward is directly
  `send()`-able.
- **Verifier sharding is free riding.** If the verifier itself outgrows
  one Mac mini (Qwen3-32B, Gemma-4-26B bf16), `model.shard(group)`
  tensor-parallelism inside a *verifier sub-group* is the only
  practical option — and it composes with this design (§4).

### 2.2 Weaknesses for this pair

- **SPMD vs. asymmetric roles.** Proposer and verifier are *different
  programs* with different weights, lifecycles, and failure domains.
  Expressing them as ranks of one SPMD job means rank-branching
  (`if rank == 0: verifier_loop() else: proposer_loop()`), one shared
  fate (any rank dying kills generation for every session on the
  fleet), and lock-step launch via `mlx.launch` + static hostfile.
- **No dynamic membership.** Agent fleets churn: a Mac mini sleeps,
  reboots, gets a new model warmed. `mlx.distributed` groups are fixed
  at init; there is no join/leave, no health-check, no re-balance.
- **F1/F2 gain nothing.** Token-id flows are < 1 KB per block. A LAN
  gRPC round trip is ~0.3–1 ms; one proposer block is tens of ms of
  compute and one verifier block forward is similar. Collectives would
  shave microseconds off a millisecond-scale, compute-dominated loop.
- **Operational constraints.** macOS 26.2 + TB5-only for `jaccl`;
  Metal timeout workarounds; no auth story on ring sockets (Kakeya's
  gRPC plane already has an auth path from the HTTP-shim era).
- **Cross-ecosystem reach.** The K3 drafter currently runs in PyTorch
  (MPS) while the verifier runs in MLX — `mlx.distributed` cannot carry
  a PyTorch process; an RPC plane can.

### 2.3 Verdict

`mlx.distributed` is the right **data plane for bulk tensors**
(F3 hidden-state shipping, intra-verifier tensor parallelism) and the
wrong **control plane** (membership, placement, session routing,
failure isolation, F1/F2). Neither a pure-`mlx.distributed` design nor
a pure-gRPC design wins on all flows.

## 3. Question 2 — what does capability exchange between Mac minis need?

Requirements distilled from ADR 0006's agent framing:

- **R1 Discovery**: a node can learn which peers exist, which models
  (verifier/proposer roles, quantization) they have warmed, and how
  much unified memory each has — without a central registry.
- **R2 Liveness**: stale nodes age out (TTL); re-announcing refreshes.
- **R3 Placement**: given "I need verifier X + proposer Y", pick hosts
  deterministically from the exchanged capability set.
- **R4 Work exchange**: actually call the chosen peer (first concrete
  capability: remote `ProposeBlock`).
- **R5 Heterogeneity**: Mac M4 + Linux x86 CPU nodes coexist (our CI
  and dev reality); MLX-only mechanisms exclude half the fleet.

`mlx.distributed` satisfies none of R1–R3 and R5 by construction (SPMD,
static, Apple-only). gRPC + protobuf — already Kakeya's wire contract
per ADR 0008 — satisfies all five, with typed errors, deadlines, and
language-neutral stubs (the TS SDK can render fleet dashboards from the
same proto).

## 4. Decision

**Hybrid, with gRPC as the control plane and `mlx.distributed` as an
optional data plane.** Concretely, this milestone (v0.5-M1) ships:

1. **`kakeya.v1.CapabilityService`** (new proto,
   `proto/kakeya/v1/distributed.proto`): symmetric gossip-style
   `ExchangeCapabilities` — caller pushes its view of the fleet, callee
   merges (last-writer-wins on `announced_at_unix`) and returns its
   merged view. TTL-based expiry. No coordinator, no consensus; the
   registry is a CRDT-ish converging map keyed by `node_id`.
2. **`kakeya.v1.ProposerService`**: `ProposeBlock(committed, L, K) →
   tokens` — the dLM-proposer contract (`DLMProposer.propose_block`,
   ADR 0001) lifted onto the wire, so any node can serve proposals for
   any node's verifier. Token-ids-only on purpose (F1/F2 analysis,
   §2.2): payloads are tiny and the contract is runtime-agnostic
   (PyTorch dLM, MLX dLM, model-free n-gram all serve it).
3. **`DistributedSpeculativeDecoder`**: the v0.2 greedy spec-decode
   loop (`kv_cache_proposer.speculative`) driven by a `RemoteProposer`
   gRPC client instead of an in-process dLM. Bit-equivalence to local
   greedy AR decoding is preserved — the accept rule never changes,
   only where the draft comes from. A draft that arrives late or wrong
   costs throughput, never correctness.
4. **`mlx.distributed` ring adapter** (`inference_engine/distributed/
   mlx_ring.py`): environment probe + group bootstrap mirroring the
   `backends/mlx/env.py` no-fallback pattern. Nodes advertise their
   ring endpoint in their capability card (`ring_address`); when two
   placed roles both have one, bulk-tensor flows (F3, future K3
   integration) can be promoted from gRPC to the ring. Linux nodes
   simply advertise no ring endpoint.

### What we explicitly rejected

- **All-in on `mlx.distributed` (SPMD ranks for proposer/verifier).**
  Rejected for shared fate, static membership, Apple-only fleets, and
  zero benefit on F1/F2 (§2.2). Re-open only if proposer↔verifier
  traffic becomes tensor-dominated *and* fleets are TB5-homogeneous.
- **Central fleet coordinator / etcd-style registry.** Rejected:
  a desk of 2–5 Mac minis does not need consensus infrastructure, and
  a coordinator is one more failure domain. Gossip pairs converge in
  one exchange round per link.
- **mDNS/Bonjour auto-discovery in this milestone.** Deferred, not
  rejected: seed peers are static CLI flags today (`--peer`). The
  registry merge logic is discovery-mechanism-agnostic, so Bonjour can
  later feed the same `merge()`.
- **Carrying logits/hidden states in `ProposeBlockResponse`.**
  Rejected for v0.5-M1: greedy acceptance needs only token ids;
  distribution-level (lossless sampling) acceptance would need draft
  probabilities and is deferred until sampling lands in the session
  path (ADR 0008 OQ-4).

## 5. Consequences

- A second `.proto` module joins the wire contract; `buf` lint/breaking
  gates and the stub-drift CI job extend to it. The capability schema
  is marked **Unstable** until v0.5 GA (same policy as runtime.proto
  pre-v0.3).
- The Linux CI gate gains a fully verifier-independent surface:
  capability registry, merge/TTL semantics, placement planning, the
  gRPC exchange/proposer services (exercised with the model-free
  n-gram proposer — a *real* prompt-lookup implementation, not a test
  double), and the greedy acceptance rule as a pure function.
- The Mac M4 integration gate gains a two-node-on-one-host spec-decode
  equivalence test: remote proposer over loopback gRPC, real verifier,
  output must be byte-identical to local greedy decode.
- Spec decode remains **outside** the gRPC `Generate` session path
  (that wiring is the separate v0.4 proposer-back-in milestone, ADR
  0008). This milestone deliberately lands the distributed machinery at
  the decoder layer where v0.2 spec decode lives, so the two tracks
  compose instead of colliding.
- Security: the capability plane ships on insecure channels bound to
  LAN interfaces, same trust model as v0.3 single-host gRPC. mTLS for
  cross-host channels is queued for v0.5 GA.
