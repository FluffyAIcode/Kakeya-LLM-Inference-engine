# Primary decode hardening

The Primary runtime publishes read-only diagnostics on
`http://127.0.0.1:8091` by default:

- `GET /healthz` combines decode liveness and unified-memory health.
- `GET /v1/runtime/liveness` reports phase, session, token index, update
  timestamp, and PID.
- `GET /v1/runtime/memory` reports MLX active/cache/peak bytes, process RSS,
  active sessions, and live KV bytes.

The default memory policy warns at 18 GiB, stops admitting new sessions at
20 GiB, and marks the runtime unhealthy at 21.5 GiB. When the final active
session is removed under memory pressure, cleanup runs in this order:
verifier reset, Python garbage collection, then `mlx.core.clear_cache()`.
Thresholds and the diagnostics port are configurable with the
`--memory-*-gb` and `--runtime-health-*` server flags.

All removal paths (explicit close, client cancellation, LRU, TTL, INV-1, and
INV-2) pass through the `SessionStore` removal hook. This releases the slab
and removes any per-session verifier binding exactly once.

## Decode watchdog

Install the independent per-user LaunchAgent:

```bash
KAKEYA_RUNTIME_REPO=/path/to/repo \
KAKEYA_RUNTIME_PYTHON=/path/to/python \
KAKEYA_RUNTIME_LABEL=ai.kakeya.grpc-runtime-prefill \
deploy/install_decode_watchdog_launchd.sh
```

Every 30 seconds the watchdog reads
`~/.kakeya/primary-decode-liveness.json`. It restarts the configured Primary
LaunchAgent only after observing the same decode token stale for at least
120 seconds twice in succession. The watchdog also recycles a runtime that
has written the 21.5 GiB unhealthy marker. Runtime startup clears a stale
marker.

`Session.generate(inter_token_timeout_s=...)` provides a client-side
notification deadline. It cancels the stream and raises
`InterTokenTimeoutError`; it does not attempt process recovery. Hard recovery
remains the external watchdog's responsibility.
