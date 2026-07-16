# ADR 0016 — Distributed Prefill KV Cache Network

- **Status:** Proposed / MVP implementation
- **Date:** 2026-07-11
- **Relates to:** ADR 0008 (session runtime), ADR 0009 (capability gossip),
  ADR 0013 (distributed topology), ADR 0015 (engine substrate)

## Context

Kakeya's bounded-memory runtime controls resident KV growth, but long prompt
prefill remains expensive. Multiple Mac minis often run compatible model
revisions and have idle unified memory that cannot become a coherent shared
address space over Thunderbolt. The useful resource is therefore not remote
RAM itself, but immutable prefill state that another node can import once.

The initial rejected interpretation was a hot-path remote KV server queried
during every decoded token. That would add layer-by-layer network latency and
turn remote memory into distributed attention. This ADR instead keeps decode
local and places all network work before it.

## Decision

Kakeya nodes form a symmetric P2P fleet. Every node can be an inference head,
a prefill-cache requester, and a prefill-cache provider. The stored object is a
restorable prefill K/V snapshot; the reuse policy is chained longest-prefix
matching.

Remote cache access happens only before decode:

1. Tokenize the request and compute chained fixed-size block hashes.
2. Query the local cache and compatible live peers concurrently.
3. Select the longest available chained-prefix snapshot whose transfer/import cost is lower
   than local prefill recomputation.
4. Transfer one immutable snapshot at the selected prefix boundary.
5. Import it, compute the missing suffix locally, then decode entirely locally.
6. Publish newly computed prefix-boundary snapshots asynchronously.

There are no per-token remote reads and no coherent shared-memory illusion.

The product name is **Distributed Prefill KV Cache**. “Prefill” names the
stored artifact; “prefix cache” names the chained longest-prefix lookup policy.

## Three layers

### Inference layer

- `PrefixCacheStore` is an immutable, content-addressed, memory-bounded LRU.
- Each chained prefix hash maps to a complete restorable bounded-cache
  checkpoint at that boundary.
- The MLX adapter serializes per-layer K/V, logical position, cached token
  sequence, and next-token logits without pickle.
- `AppendTokensCoordinator` accepts an optional prefill-cache hook for cold
  sessions. Cache failure is always a local-prefill fallback.

### Network layer

- Existing `CapabilityService` gossip remains the decentralized control plane.
- `NodeCapability` advertises interface-specific endpoints and exact cache
  compatibility cards.
- `PrefillCacheService.LookupPrefix` is metadata-only.
- `PrefillCacheService.FetchBlocks` is a point-to-point streaming gRPC data
  plane.
- Thunderbolt endpoints receive higher priority than LAN/Tailscale endpoints.
- Cards expire by TTL. Cache entries are immutable; reads require no
  distributed lock.

### UI and product layer

The inference-network dashboard exposes:

- node registration and expiring pairing tokens;
- online node inventory and coarse region distribution;
- cache discovery and pairing state;
- inference groups;
- completed and KV-assisted token totals;
- cache capacity, hit ratio, and transfer telemetry.

Raw prompts, raw prefix hashes, exact IPs, and hardware serials are not public
UI data.

## Compatibility tuple

Remote K/V is accepted only when all fields match:

- model id and exact weights revision;
- tokenizer/chat-template revision;
- quantization and K/V dtype;
- cache schema version;
- RoPE/position configuration;
- layer geometry;
- token block size.

A mismatch is a cache miss, never a best-effort import.

## Failure model

- peer unavailable / lookup timeout → local prefill;
- stale card → TTL eviction;
- lease expired → local prefill or another lookup;
- incomplete stream / checksum mismatch → reject and local prefill;
- slow transfer → caller may cancel and recompute;
- node restart → cache epoch changes; stale leases are invalid.

Remote cache availability must never determine request correctness.

## Security

The MVP assumes a trusted private fleet. Production pairing must bind `node_id`
to mTLS credentials or a fleet PSK. Prompt-derived block hashes should be HMACed
when membership attacks are in scope.

## Observability

Every node reports:

- capability-card freshness and endpoint RTT;
- cache bytes used/free, entry count and epoch;
- local/remote lookup hit/miss counts;
- snapshot publish/fetch bytes and checksum failures;
- completed inference tokens and KV-assisted prompt tokens;
- fallback reason and local recompute count.

The public dashboard shows aggregate/coarse data. Exact addresses, prompt
hashes and payload metadata remain administrator-only.

## Consequences

Positive:

- expensive prompt prefill is reused across processes, restarts and nodes;
- the remote peer's memory becomes a useful cache tier without changing decode;
- node failure cannot change output correctness;
- the existing capability gossip and tensor codec remain the shared substrate;
- cache-only peers need no loaded model when they receive replicated snapshots.

Costs:

- snapshots are large and must be chunked;
- each model/tokenizer/cache revision creates a separate namespace;
- cache snapshots duplicate state at block boundaries in the MVP;
- peer memory is volatile and cold after restart;
- trusted-LAN deployment precedes production identity/auth hardening.

## Alternatives considered

### Per-token remote KV reads

Rejected. Attention would require remote communication in every transformer
layer, adding latency to the autoregressive critical path.

### Coherent shared memory over Thunderbolt

Rejected. Thunderbolt Bridge exposes IP networking, not a cache-coherent
CPU/GPU address space or GPU-direct remote memory.

### Exact full-prompt cache only

Rejected as the sole policy. It is simpler but misses common-system-prompt and
shared-prefix reuse. Chained block hashes preserve causal correctness while
allowing suffix-only prefill.

### Arbitrary block/hole reuse

Rejected. Later K/V depends on the complete preceding token sequence and
positions. A stored snapshot must cover that complete sequence. Intermediate
checkpoint entries may be absent, however, because the chained hash and final
snapshot already commit and contain the full prefix through that boundary.

### SMB/NFS snapshot files

Rejected for the serving data plane. Files remain useful for offline
checkpoints, but gRPC provides explicit framing, checksums, leases,
backpressure and cancellation.

### Central registry

Rejected. Existing symmetric CapabilityService gossip already provides
eventually-consistent discovery and TTL expiry without a new coordinator.

## Rollout

1. Enable cache services on a private two-Mac Thunderbolt fleet.
2. Run shadow lookup/publish while still computing prefill locally.
3. Compare imported continuation logits against local prefill.
4. Enable remote import with mandatory compatibility/checksum validation.
5. Add launchd supervision, token telemetry and public read-only dashboard.
6. Require mTLS/PSK before expanding beyond trusted private nodes.

The cache feature is disabled by omitting `--enable-prefill-cache`.

## Rollback

Stop cache/dashboard launchd services and restart RuntimeService without the
cache flag. No data migration is needed because all entries are immutable,
volatile optimizations. The direct gateway remains the fallback public origin.

## Evidence

The two-Mac report is:
[`docs/reports/distributed-prefill-kv-mac-thunderbolt.md`](../reports/distributed-prefill-kv-mac-thunderbolt.md).

On Apple M4 Macs over Thunderbolt, a 93-token Gemma 26B prompt measured 5.926 s
cold prefill and 0.061 s after remote snapshot import (approximately 97× for
that prompt). The peer served 36.4 MB across two snapshots with SHA-256
validation.

## Non-goals

- combining physical RAM into one address space;
- remote attention in the per-token loop;
- arbitrary-hole K/V reuse;
- cross-model or cross-tokenizer cache conversion;
- using gossip to carry tensor payloads.
