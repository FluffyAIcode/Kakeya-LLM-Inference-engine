# Distributed Prefill KV — two Mac mini live report

Date: 2026-07-11

## Topology

- head Mac mini: Apple M4, 24 GB, `169.254.187.239`
- cache peer Mac mini: Apple M4, 16 GB, `169.254.27.104`
- link: Thunderbolt Bridge, 40 Gb/s physical, measured RTT ≈ 0.55 ms
- runtime model: local Gemma 26B-A4B MLX 4-bit
- cache block boundary: 64 tokens
- peer cache allocation: 2 GiB

## Live result

Prompt length: 93 tokens.

| Run | Local cache | Peer cache | AppendTokens prefill |
|---|---:|---:|---:|
| cold | empty | empty | 5.926 s |
| after runtime restart | empty | 2 remote snapshots (36.4 MB) | 0.061 s |

Observed prefill acceleration: approximately **97×** for this prompt.

The remote hit imported 20,951,040 live KV bytes into the head runtime. The
peer's telemetry reported 93 tokens served. SHA-256 validation passed on
point-to-point publish/lookup/fetch tests.

## Product surfaces

- public dashboard: `https://kakeya.ai/`
- public summary API: `https://kakeya.ai/v1/network/summary`
- node/group API: `/v1/network/nodes`, `/v1/network/groups`
- token accounting: `/v1/network/tokens`

Both cache services and the head Gemma runtime run under launchd with KeepAlive.
Cloudflare Worker `kakeya-inference-network` owns `kakeya.ai/*`; deployed
version at validation time: `e45f67be-721a-413d-804e-33f7e28e80d8`.

## Known constraints

- The allens Miniconda Python process lacks macOS Local Network permission for
  outbound sockets. Head→peer lookup/publish/fetch works over Thunderbolt;
  reverse peer→head gossip is disabled until that permission is granted.
- `agent.kakeya.ai` remains the direct gateway origin and rollback path.
- Cache entries are in-memory and intentionally disappear on peer restart.
- The MVP trusts the private Thunderbolt fleet. Production requires mTLS/PSK
  identity binding before accepting remote tensor payloads.
