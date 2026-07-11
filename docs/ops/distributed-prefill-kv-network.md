# Distributed Prefill KV Cache Network — Operator Runbook

This runbook deploys one inference head and one cache peer over a private
Thunderbolt Bridge. The same services work over LAN/Tailscale with lower
endpoint priority.

## Production surfaces

- dashboard: `https://kakeya.ai/`
- health: `https://kakeya.ai/healthz`
- nodes: `https://kakeya.ai/v1/network/nodes`
- groups: `https://kakeya.ai/v1/network/groups`
- token counters: `https://kakeya.ai/v1/network/tokens`

Public reads expose aliases, coarse regions and aggregate metrics only. Writes
require `X-API-Key`.

## Components

| Component | Head | Cache peer |
|---|---|---|
| Kakeya RuntimeService | `127.0.0.1:51051` | optional |
| PrefillCacheService | runtime + `:52051` control node | `169.254.27.104:52051` |
| CapabilityService gossip | enabled | enabled / pull-only if macOS blocks outbound Python sockets |
| Dashboard/API | `127.0.0.1:8090` | no |
| Cloudflare public edge | `kakeya.ai/*` Worker | no |

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

Check from the head over Thunderbolt:

```bash
ping -c 3 169.254.27.104
nc -vz 169.254.27.104 52051
```

If `nc` works but Python/gRPC outbound calls return `Errno 65`, grant Local
Network access to that Python executable in macOS Privacy & Security. Head→peer
lookup/publish/fetch remains usable while reverse gossip is disabled.

## Node registration and groups

Create a registration:

```bash
KEY="$(cat ~/.kakeya/network_api_key)"
curl -fsS -X POST http://127.0.0.1:8090/v1/network/nodes/register \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $KEY" \
  -d '{"alias":"peer-mini","address":"169.254.27.104:52051","region":"Private","role":"cache"}'
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
- remote lookup returns only the longest contiguous prefix;
- imported snapshot checksum and compatibility fingerprint match;
- remote failure falls back to local prefill;
- no remote RPC occurs in autoregressive decode;
- completed and KV-assisted token counters increase after live calls.

Minimal acceptance:

```bash
curl -fsS https://kakeya.ai/healthz
curl -fsS https://kakeya.ai/v1/network/summary
curl -fsS https://kakeya.ai/v1/network/tokens
```

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

