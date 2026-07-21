# ADR 0017 — Primary-decode / distributed-prefill worker orchestration

- **Status:** Accepted / implementation
- **Date:** 2026-07-11
- **Supersedes for product architecture:** DFlash/f_θ/fused speculative-decode
  serving paths
- **Extends:** ADR 0009 (capability gossip), ADR 0016 (distributed prefill KV)

## Context

The product topology is one primary Mac mini that owns RuntimeService, session
state and every autoregressive decode step. Peer Mac minis run the exact same
model but are restricted to prefill work. They retain immutable prefill K/V
snapshots in unified memory and answer lookup/fetch requests from the primary.
Cache-only peers remain useful when maximizing snapshot capacity matters more
than compute.

ADR 0016 implemented the cache half of this design, but its deployed peer is
cache-only: the primary computes every miss and replicates snapshots. It does
not let a peer compute a cold prefix. Peer addresses are also static CLI flags,
and a fetch/import failure can escape instead of falling back to local prefill.

## Decision

Kakeya adopts three explicit fleet roles:

1. **PRIMARY_DECODE** (`VERIFIER` on the wire): serves users and performs all
   decode. No remote RPC is allowed in the token loop.
2. **PREFILL_COMPUTE:** loads the same model, accepts bounded/idempotent prefill
   jobs, writes snapshots to its local `PrefixCacheStore`, and never serves
   user decode.
3. **PREFILL_CACHE:** loads no model; spends RAM on immutable snapshots only.

### Local decode process boundary

Primary decode may run behind the feature flag `--decode-worker`. In this
profile the gRPC/router process does not construct an MLX verifier and does not
load model weights. It starts one child process over a mode-0600 Unix-domain
socket; that child owns the single loaded model and all session K/V adapters.
The transport is protocol-v1 length-prefixed JSON plus an optional opaque
binary payload. It never uses pickle and is not exposed on TCP.

The local protocol has six operations:

- `Init(session_id, token_ids?)` creates/replaces isolated session K/V;
- `ImportSnapshot(session_id, compatibility, payload)` imports the existing
  allens portable snapshot format after exact compatibility validation;
- `Append(session_id, token_ids)` commits an all-accepted block;
- `GenerateStep(session_id)` atomically selects and commits one greedy token;
- `Close(session_id)` releases that session's K/V; and
- `Health()` returns protocol, process, model geometry, session and MLX-memory
  state.

The router serializes requests because the MVP MLX execution stream is shared.
No per-token K/V snapshot crosses IPC. The router retains a proof checkpoint:
the last imported immutable allens snapshot plus only tokens acknowledged
after that boundary (or full acknowledged history when no snapshot exists).
An IPC EOF, child exit, or operation timeout hard-kills the child, starts a new
one, restores that checkpoint, and retries the unacknowledged current
operation once. Because checkpoint state advances only after a correlated
response and `GenerateStep` is atomic in the child, an ambiguous crash cannot
duplicate a committed token. Client cancellation hard-kills an in-flight
worker forward; the next request follows the same restore path.

This process boundary is initially opt-in. The in-process MLX path remains the
rollback path until the Mac acceptance gates pass. It is not a remote decode
service and does not change the rule that no fleet RPC occurs in the token
loop.

### Request flow

For a cold append, the primary:

1. computes tenant-HMAC chained prefix hashes;
2. queries its local cache and compatible live cache/worker cards from gossip;
3. imports the best cache hit when transfer cost beats local recomputation;
4. on a miss, submits an idempotent job to the least-loaded compatible
   `PREFILL_COMPUTE` worker when remote compute+transfer is cheaper;
5. fetches and imports the completed snapshot once;
6. computes any missing suffix locally;
7. decodes entirely locally;
8. publishes snapshots to a deterministic subset of cache peers.

Every remote error (lookup, job, lease, fetch, checksum, decompress, import)
resets the verifier and falls back to full local prefill. Cache availability
must never determine request correctness.

For deployments that require a strictly decode-only primary,
`--prefill-policy remote-required` changes this failure contract: only a
complete cache hit or completed remote worker job is accepted. Partial hits are
not extended on Primary, cost gating is bypassed, and worker failure returns
`UNAVAILABLE` instead of silently running local prefill.

### Discovery and placement

Capability gossip is the only membership source. Static `--cache-peer` and
`--prefill-worker` flags remain emergency/operator overrides. Cards advertise
exact cache compatibility, cache address, compute address, queue depth,
inflight jobs, measured prefill throughput, free RAM and endpoint RTT.

The scheduler is deterministic and cost-aware:

```text
local_ms = missing_tokens / local_prefill_tps * 1000
import_ms = endpoint_rtt_ms + transfer_bytes / link_bytes_per_ms
remote_ms = queue_eta_ms + prompt_tokens / worker_tps * 1000 + import_ms
```

The longest safe hit is preferred only when `import_ms < local_ms`. A worker is
used only when `remote_ms < local_ms`. Unknown metrics use conservative
operator-configured defaults.

### Memory/storage policy

- Decode KV remains local to the primary.
- Peer memory is a pre-decode snapshot tier, not coherent remote attention RAM.
- Cache mounts are exposed as one content-addressed `kv://` namespace for
  management. This virtualizes naming and location only; fetch/import still
  copies the selected snapshot into Primary memory.
- A successful remote import promotes the complete snapshot into Primary's
  bounded hot LRU. Primary eviction removes only that hot copy; the worker's
  cold/offload copy remains available.
- Worker cache capacity is adaptive: physical memory minus active MLX model
  bytes and an operator reserve, bounded by configured minimum and ceiling.
- Snapshot payloads support zlib framing and retain SHA-256 of the uncompressed
  bytes.
- Replication uses rendezvous hashing and a bounded replication factor instead
  of publishing every snapshot to every peer.
- Block size controls checkpoint sparsity. Cache-only nodes should use larger
  blocks for long prompts; delta snapshots remain a future format revision.
- Import checks advertised transfer size against a configurable byte budget
  before allocating/reassembling.

### Security and tenant isolation

Trusted-LAN mode remains available, but production mode uses a fleet PSK:

- request metadata is HMAC-SHA256 signed with timestamp and node/tenant id;
- replay window is bounded;
- prefix hashes are HMACed per tenant, preventing cross-tenant prefix probing;
- tenant namespace is part of `CacheCompatibility`;
- cache/worker services reject unauthenticated requests before allocation.

mTLS can replace PSK transport later without changing the application protocol.

## Correctness gates

Blocking tests cover:

- scheduler decisions, worker job lifecycle and idempotency;
- auth/replay/tenant isolation;
- compression/checksum and replication placement;
- every remote failure falling back to local prefill;
- real MLX local-prefill vs remote-prefill-import continuation logits and
  argmax equivalence;
- zero prefill/cache RPCs during `Generate`.
- protocol correlation/version/size validation for local decode IPC;
- in-process vs worker greedy-token parity;
- cancellation hard-kill and child crash/timeout recovery from both
  full-history and allens-snapshot-plus-proof checkpoints.

## Consequences

Positive:

- the primary dedicates compute to decode;
- peer compute and RAM scale independently;
- repeated system/RAG prefixes survive primary restarts;
- peers join/leave through existing gossip/TTL;
- local privacy is preserved inside the authenticated fleet.

Costs:

- every compute worker duplicates model weights, reducing RAM available for KV;
- first-use remote prefill only wins when worker compute plus transfer beats the
  primary;
- snapshots still duplicate bounded state at checkpoint boundaries;
- MLX worker execution is serialized per loaded verifier in the MVP.
- worker-mode locally computed prefill checkpoints are not republished by the
  router, because export is deliberately absent from the token-loop protocol;
  allens-produced snapshots remain importable and are the preferred durable
  recovery boundary.

## Legacy paths

DFlash, f_θ and fused speculative-decode modules remain temporarily under
`research/legacy` compatibility surfaces for reproducibility. They are not
product completeness criteria and must not be wired into RuntimeService.

