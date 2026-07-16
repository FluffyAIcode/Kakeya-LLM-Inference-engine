# Distributed Prefill KV Cache Network — Operator Runbook

This runbook deploys one primary decode node, prefill-compute workers and
optional cache-only peers over a private Thunderbolt Bridge. The same services
work over LAN/Tailscale with lower endpoint priority.

## Production surfaces

- dashboard: `https://kakeya.ai/`
- health: `https://kakeya.ai/healthz`
- nodes: `https://kakeya.ai/v1/network/nodes`
- groups: `https://kakeya.ai/v1/network/groups`
- token counters: `https://kakeya.ai/v1/network/tokens`

Public reads expose aliases, coarse regions and aggregate metrics only. Writes
require `X-API-Key`.

## Components

| Component | Primary | Prefill worker | Cache-only peer |
|---|---|---|---|
| Kakeya RuntimeService / decode | `127.0.0.1:51051` | no | no |
| Same MLX model loaded | yes | yes | no |
| PrefillWorkerService | no | `:53051` | no |
| PrefillCacheService | runtime/local LRU | co-located | `:52051` |
| CapabilityService gossip | enabled | enabled | enabled / pull-only |
| Dashboard/API | `127.0.0.1:8090` | no | no |
| Cloudflare public edge | `kakeya.ai/*` Worker | no | no |

## Compatibility lock

All peers in one inference group must match:

```text
model_id
model_revision
tokenizer_revision / chat template
quantization
KV dtype
cache format version
RoPE hash
layer geometry hash
block size
```

Changing any field creates a new cache namespace. Never convert or import a
best-effort mismatch.

## Head runtime

The release launchd asset is:

```text
deploy/launchd/ai.kakeya.grpc-runtime-prefill.plist
```

It runs Gemma 26B MLX on `51051`, queries the peer on `52051`, asynchronously
publishes new snapshots, and reports generated/reused token telemetry to the
network API.

Check:

```bash
launchctl list | grep ai.kakeya.grpc-runtime-prefill
lsof -nP -iTCP:51051 -sTCP:LISTEN
tail -f ~/.kakeya/grpc-runtime-prefill.log
```

## Head dashboard/control node

Install:

```bash
openssl rand -hex 24 > ~/.kakeya/network_api_key
chmod 600 ~/.kakeya/network_api_key
bash deploy/install_prefill_network_launchd.sh
```

Check:

```bash
launchctl list | grep ai.kakeya.prefill-network
curl -fsS http://127.0.0.1:8090/healthz
curl -fsS http://127.0.0.1:8090/v1/network/nodes
```

## Cache peer

Use an isolated venv and copy/sync the repository package. The peer plist is:

```text
deploy/launchd/ai.kakeya.prefill-network-peer.plist
```

The cache-only plist remains available for rollback or additional RAM-only
nodes. In the strict two-Mac decode/prefill profile, allens instead runs the
prefill-worker plist below with a co-located cache.

Check from the head over Thunderbolt:

```bash
ping -c 3 169.254.27.104
nc -vz 169.254.27.104 52051
```

If `nc` works but Python/gRPC outbound calls return `Errno 65`, grant Local
Network access to that Python executable in macOS Privacy & Security. Head→peer
lookup/publish/fetch remains usable while reverse gossip is disabled.

## Prefill-compute worker

The worker loads the exact same MLX model as the primary, accepts queued
prefill-only jobs, writes immutable snapshots into its co-located RAM cache and
never serves user decode.

In the strict two-Mac profile, `allens-mini` runs this role and Primary uses
`--prefill-policy remote-required`. Workers receive canonical token IDs from
the scheduler; they do not construct their own chat template. The worker stores
the resulting snapshots in its co-located content-addressed cache.

Create a fleet PSK once and copy it to every trusted node:

```bash
openssl rand -hex 32 > ~/.kakeya/fleet.psk
chmod 600 ~/.kakeya/fleet.psk
```

Install the worker:

```bash
python -m pip install -r requirements-kakeyalattice.txt
export KAKEYA_WORKER_REPO="$HOME/Kakeya-LLM-Inference-engine"
export KAKEYA_WORKER_PYTHON="$HOME/kakeya-venv/bin/python"
export KAKEYA_WORKER_MODEL="$HOME/kakeya-models/gemma-4-26B-A4B-it-mlx-4bit"
export KAKEYA_CACHE_MODEL_ID="gemma-4-26B-A4B-it-mlx-4bit"
export KAKEYA_MODEL_REVISION="local-4bit-v1"
export KAKEYA_TOKENIZER_REVISION="gemma4-v1"
export KAKEYA_WORKER_NODE_ID="prefill-mini-1"
export KAKEYA_WORKER_BIND="<worker-ip>:53051"
export KAKEYA_WORKER_ADVERTISE="<worker-ip>:53051"
export KAKEYA_LAYER_GEOMETRY_HASH="<same-value-as-primary>"
export KAKEYA_WORKER_SINK="4"
export KAKEYA_WORKER_WINDOW="2048"
export KAKEYA_WORKER_CACHE_GB="8"
export KAKEYA_WORKER_CACHE_MIN_GB="0.25"
export KAKEYA_WORKER_MEMORY_RESERVE_GB="2"
export KAKEYA_WORKER_ADAPTIVE_CACHE="1"
export KAKEYA_CACHE_BLOCK_TOKENS="64"
export KAKEYA_CACHE_FORMAT_VERSION="kakeya-prefill-v3-kl-d4-q38"
export KAKEYA_CACHE_COMPRESSION="kakeyalattice-d4"
export KAKEYA_WORKER_NETWORK="thunderbolt"
export KAKEYA_WORKER_PRIORITY="100"
export KAKEYA_WORKER_RTT_MS="0.55"
export KAKEYA_WORKER_PEER="169.254.187.239:51051"
export KAKEYA_FLEET_PSK_FILE="$HOME/.kakeya/fleet.psk"
export KAKEYA_TENANT_ID="private-fleet"
bash deploy/install_prefill_worker_launchd.sh
```

The primary must use the same compatibility and auth values:

```text
--enable-prefill-cache
--enable-capability-exchange
--peer <worker-ip>:53051
--cache-tenant-id private-fleet
--fleet-psk-file ~/.kakeya/fleet.psk
--cache-compression zlib
--cache-replication-factor 1
```

For bit-packed KakeyaLattice snapshots, replace the primary compression line
with `--cache-compression kakeyalattice-d4` and set
`--cache-format-version kakeya-prefill-v3-kl-d4-q38`. Both nodes must install
the pinned optional dependency from `requirements-kakeyalattice.txt`. D4 Q=38
is lossy; live acceptance must validate output quality as well as byte savings.

`--cache-peer` remains an emergency static override. Normal worker/cache
selection is derived from compatible live capability cards and their TTL/load
metrics.

## Node registration and groups

Create a registration:

```bash
KEY="$(cat ~/.kakeya/network_api_key)"
curl -fsS -X POST http://127.0.0.1:8090/v1/network/nodes/register \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"alias":"allens-mini","address":"169.254.27.104:52051","region":"Private","role":"cache"}'
```

Create a paired group:

```bash
curl -fsS -X POST http://127.0.0.1:8090/v1/network/groups \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"name":"Thunderbolt Pair","node_ids":["head-mini","allens-mini"]}'
```

## Health and acceptance

Expected invariants:

- both node cards appear within two gossip intervals;
- stale nodes disappear after TTL;
- remote lookup returns the longest available chained-prefix snapshot;
- imported snapshot checksum and compatibility fingerprint match;
- remote failure falls back to local prefill;
- no remote RPC occurs in autoregressive decode;
- a cache miss is assigned to a compatible `PREFILL_COMPUTE` worker when its
  queue+compute+transfer estimate beats blocking the primary;
- worker failure/timeout/import rejection resets the verifier and performs full
  local prefill;
- completed and KV-assisted token counters increase after live calls.

Minimal acceptance:

```bash
curl -fsS https://kakeya.ai/healthz
curl -fsS https://kakeya.ai/v1/network/summary
curl -fsS https://kakeya.ai/v1/network/tokens
curl -fsS https://kakeya.ai/v1/network/prefill

PYTHONPATH=.:sdks/python python scripts/verify_remote_prefill_e2e.py \
  --address 127.0.0.1:51051 \
  --dashboard http://127.0.0.1:8090 \
  --tokenizer-id ~/kakeya-models/gemma-4-26B-A4B-it-mlx-4bit
```

By default the verifier accepts a remote cache hit and requires `remote_hits`
and `tokens_reused` to increase. Use `--require-worker` only when testing the
separate Worker A/B/C path; that mode additionally requires `remote_jobs`.
Decode throughput is reported separately because all autoregressive decode
remains on the primary.

The logical cross-node KV namespace is available at:

```bash
curl -fsS http://127.0.0.1:8090/v1/network/kvfs
```

The returned `kv://` URI and mount table virtualize naming and management only.
Payloads remain in each Mac's physical RAM and are copied into Primary once
before decode; `coherent_shared_memory` is always false. Mounts are marked
`hot` for Primary and `cold-offload` for worker/cache peers. Remote imports are
promoted into the hot LRU, while eviction leaves the cold copy untouched.

## Maintenance cache saturation

Enable the bounded, memory-only first-append capture queue on the primary:

```text
--cache-fill-capture-size 256
```

The maintenance endpoints require the network API key even when other read
endpoints are public. Captured token IDs remain in process memory and are
removed when drained; reports contain only salted capture IDs and token counts.

During a maintenance window, start real gRPC chat sessions and run:

```bash
PYTHONPATH=.:sdks/python python scripts/fill_prefill_cache_from_live_grpc.py \
  --tokenizer-id ~/kakeya-models/gemma-4-26B-A4B-it-mlx-4bit \
  --target-one 0.90 \
  --target-two 0.95 \
  --churn-gb 0.9
```

The harness controls head and allens independently, stops on memory pressure,
publish failures, fallbacks, or `/tmp/kakeya-cache-fill.stop`, and never expects
resident fleet usage to exceed the configured 1+8 GiB ceiling. Churn is accepted
when `bytes_evicted` increases while resident bytes remain bounded.

## Three-phase architecture benchmark

Run from Primary; services are started only if missing and remain running:

```bash
bash scripts/run_prefill_architecture_benchmark.sh \
  --output-tokens 32 \
  --report /tmp/kakeya-prefill-benchmark.json
```

The task runs `remote_compute`, `primary_hot_hit`, and
`allens_cold_restore`, recording client-side append latency, TTFT, decode
latency/tokens-per-second, E2E throughput, and server-side hit/promotion deltas.
Reports never persist prompts, token IDs, cache keys, raw addresses, or user
paths.

Benchmark APIs are public reads and API-key writes:

```bash
curl -fsS https://kakeya.ai/v1/network/benchmarks
curl -fsS https://kakeya.ai/v1/network/benchmarks/live
curl -fsS https://kakeya.ai/v1/network/benchmarks/<run-id>
```

The `Benchmarks` dashboard tab shows live progress, phase comparison, history,
and complete redacted stage details.

## Rollback

The cache is an optimization; inference correctness does not depend on it.

1. Stop the cache services:

   ```bash
   launchctl bootout "gui/$(id -u)/ai.kakeya.prefill-network"
   launchctl bootout "gui/$(id -u)/ai.kakeya.grpc-runtime-prefill"
   ```

2. Restart the previous RuntimeService without `--enable-prefill-cache`.
3. Roll back the Cloudflare Worker deployment with Wrangler versions/deployments.
4. `agent.kakeya.ai` remains available as the direct gateway origin.

In-memory cache entries require no migration or cleanup after rollback.

## Security before untrusted fleets

The live MVP assumes trusted private Macs. Before accepting third-party nodes:

- require mTLS or fleet-PSK authentication;
- bind signed node identity to `node_id`;
- HMAC prompt-derived block hashes;
- rate-limit registration, lookup and publish;
- cap block size and stream bytes before allocation;
- maintain revocation and audit logs;
- never expose raw prompts, hashes, IPs or cache payloads in the public UI.
