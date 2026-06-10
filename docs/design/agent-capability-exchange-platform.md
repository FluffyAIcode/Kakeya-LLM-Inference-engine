# Design — Agent capability exchange platform across Mac mini hosts (v0.5-M1)

- **Status**: Implemented (v0.5-M1 milestone scope)
- **Decided by**: [ADR 0009](../adr/0009-mlx-distributed-spec-decode-and-capability-exchange.md)
- **Wire contract**: [`proto/kakeya/v1/distributed.proto`](../../proto/kakeya/v1/distributed.proto)
- **Implementation**: [`inference_engine/distributed/`](../../inference_engine/distributed/)

## 1. What "capability exchange" means here

A *capability* is something a node can do for another node's agent
workload: verify with model X at quantization Q, propose draft blocks
with dLM Y, (future) embed, rerank, run a tool. A *capability card*
(`NodeCapability`) is the signed-by-time, TTL-bounded advertisement of
one node's capabilities plus the addresses needed to use them:

```
NodeCapability
├── node_id              stable identity ("mini-attic", uuid, …)
├── grpc_address         host:port where this node's gRPC services live
├── ring_address         mlx.distributed ring endpoint ("" on non-MLX nodes)
├── platform             "mac-m4-24gb", "linux-x86", … (informational)
├── unified_memory_bytes capacity signal for placement
├── mlx_version          "" when MLX unavailable (Linux nodes)
├── models[]             ModelCapability{model_id, role, quantization, tokens_per_second}
├── announced_at_unix    wall-clock freshness stamp (last-writer-wins)
└── ttl_seconds          card expires at announced_at_unix + ttl_seconds
```

Roles are an enum (`CAPABILITY_ROLE_VERIFIER`, `CAPABILITY_ROLE_PROPOSER`;
`EMBEDDER` / `TOOL` reserved). One node usually carries several cards'
worth of models — e.g. a 24 GB M4 mini advertising
`{Qwen3-1.7B-4bit, VERIFIER}` and `{ngram, PROPOSER}` simultaneously.

## 2. Topology and protocol

```
 Mac mini A (verifier host)            Mac mini B (proposer host)
┌──────────────────────────┐          ┌──────────────────────────┐
│ Kakeya runtime           │          │ Kakeya runtime           │
│  RuntimeService  :50051  │          │  RuntimeService  :50051  │
│  CapabilityService ──────┼─gossip──►│  CapabilityService       │
│  ProposerService         │◄─gossip──┼──                        │
│                          │          │  ProposerService ◄───────┼─ ProposeBlock
│  CapabilityRegistry      │          │  CapabilityRegistry      │   from A's
│  {A: card, B: card}      │          │  {A: card, B: card}      │   spec-decode loop
└──────────────────────────┘          └──────────────────────────┘
            │      optional mlx.distributed ring (bulk tensors)   │
            └────────────────── ring_address ─────────────────────┘
```

- **One gRPC server per node**, multiplexing `RuntimeService` (ADR
  0008), `CapabilityService`, and `ProposerService` on the same port.
- **Gossip, not registry.** `ExchangeCapabilities(known_nodes) →
  known_nodes` is symmetric push-pull: the caller sends every card it
  knows (including its own), the callee merges and replies with its
  merged view, the caller merges that. After one round both sides hold
  the union. With seed peers forming a connected graph, the fleet view
  converges in ≤ diameter rounds.
- **Merge rule** (`CapabilityRegistry.merge`): per `node_id`,
  keep the card with the larger `announced_at_unix`; a node's *own*
  card is authoritative locally and never overwritten by gossip.
  Expired cards are dropped on read (`snapshot()`) and on merge.
- **Liveness** is TTL-only in M1 (default 120 s; re-exchange interval
  default 30 s, i.e. ≥3 missed rounds before expiry). No failure
  detector: a card that stops refreshing ages out.

### Why this converges (informally)

The registry is a last-writer-wins element-set keyed by `node_id` with
a totally ordered timestamp per element — merges are commutative,
associative, and idempotent, so exchange order and repetition cannot
diverge replicas. Clock skew between minis shifts freshness by the
skew amount; with TTLs in minutes and desk-LAN NTP skew in
milliseconds this is immaterial (documented limit: TTL must be ≫
max expected skew).

## 3. Placement

`plan_spec_decode_placement(snapshot, …)` turns a converged fleet view
into a `SpecDecodePlacement{verifier_node, verifier_model,
proposer_node, proposer_model, colocated}`:

1. Filter cards to live nodes carrying the requested role (and
   `model_id`, when pinned).
2. Score verifier candidates by `(tokens_per_second, unified_memory_bytes,
   node_id)` — throughput first, memory as tiebreak, id for
   determinism. Same fleet view ⇒ same plan on every node, with no
   coordination round.
3. Score proposer candidates the same way, but **prefer a node other
   than the verifier's** (`prefer_remote_proposer=True`): the point of
   the split (ADR 0009 §1) is to evict proposer weights + activation
   peak from the verifier host. Colocate only when no other live node
   carries the role.
4. No candidates ⇒ raise `PlacementError`. No fallback to silently
   decoding without a proposer — the caller decides (no-fallback
   convention, ADR 0008).

## 4. First exchanged capability: remote block proposal

`ProposerService.ProposeBlock` lifts the ADR 0001 proposer contract
onto the wire unchanged:

```
ProposeBlockRequest { committed_token_ids, block_size, num_steps, model_id }
ProposeBlockResponse { token_ids, diffusion_steps, forward_passes, peak_activation_bytes }
```

- Server side: `ProposerServicer` holds a `{model_id: proposer}` map;
  any object with `propose_block(committed, L, K) → BlockProposal`
  serves (PyTorch dLM, `MLXSparseLogitsProposer`, `NGramProposer`).
  Blocking proposers run via `asyncio.to_thread` so the event loop
  keeps serving capability gossip during a long diffusion.
- Client side: `RemoteProposer` is a drop-in `DLMProposer` substitute —
  same `propose_block` signature, same `ProposerStats` accounting — so
  `SpeculativeDecoder` and `DistributedSpeculativeDecoder.from_placement`
  drive it without modification.
- **Correctness containment**: the verifier-side greedy accept rule
  (`accept_block`) is unchanged, so a wrong/stale/garbage remote draft
  can only lower the acceptance rate, never corrupt output. This is
  what makes it safe to accept drafts from *any* gossip-discovered
  peer: the trust requirement on proposers is availability, not
  integrity.

`NGramProposer` (prompt-lookup decoding: longest-suffix n-gram match
against the committed prefix, copy the historical continuation) ships
as the zero-weight, always-available proposer every node can
advertise. It is a real proposer — on repetitive/agentic text it earns
nonzero acceptance — and it keeps the Linux CI gate and heterogeneous
fleets honest without model weights.

## 5. `mlx.distributed` integration points

Per ADR 0009 the ring is a data-plane *upgrade*, not a dependency:

- `probe_ring_environment()` mirrors `backends/mlx/env.py`: reports
  `RingEnvironment{is_available, rank, world_size, backend,
  failure_reason}` without raising; non-Mac hosts get a structured
  "unavailable" with the reason.
- Nodes launched under `mlx.launch` advertise `ring_address`
  (`hostname:rank`) in their capability card.
- `ring_path_available(a, b)` is the placement-time predicate for
  promoting a flow to the ring; M1 uses it for advertisement and
  diagnostics, the K3 hidden-state flow (F3) is the first planned
  consumer.

## 6. Failure model

| Failure | Behavior |
| --- | --- |
| Peer down at exchange time | `exchange_once` records the error per peer and continues; registry unchanged for that peer; card ages out via TTL |
| Proposer node dies mid-generation | `RemoteProposer.propose_block` raises `RemoteProposerError` (wrapping the gRPC status); the decoder surfaces it — caller may re-plan placement and resume, since the verifier session state is intact |
| Stale capability card (model unloaded) | `ProposeBlock` returns `NOT_FOUND`; caller re-plans |
| Clock skew | bounded staleness, see §2 |
| Two nodes claim same `node_id` | last-writer-wins; documented as operator error (ids must be unique per fleet) |

## 7. Out of scope for M1 (queued)

- Bonjour/mDNS seed discovery (static `--peer` flags today)
- mTLS + node identity keys on cross-host channels
- Session migration between nodes (KV cache is not portable yet)
- Remote *verification* (inverse split) and embed/tool roles
- Wiring spec decode into the gRPC `Generate` session path (v0.4
  proposer-back-in track owns that; the decoder-layer machinery here
  is what that track will call)
