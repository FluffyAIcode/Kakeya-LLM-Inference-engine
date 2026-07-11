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

## Legacy paths

DFlash, f_θ and fused speculative-decode modules remain temporarily under
`research/legacy` compatibility surfaces for reproducibility. They are not
product completeness criteria and must not be wired into RuntimeService.

