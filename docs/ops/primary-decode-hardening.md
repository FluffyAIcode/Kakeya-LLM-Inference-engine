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

## Isolated MLX decode worker

Enable the opt-in process boundary by adding these flags to the Primary
runtime command:

```bash
--backend mlx \
--decode-worker \
--decode-worker-timeout-s 120 \
--decode-worker-startup-timeout-s 180
```

The router then loads no MLX model. A private mode-0600 UDS child owns the
model and per-session K/V. Keep `--decode-worker-socket` unset for an
automatically unique temporary path; set it only when launchd supervision or
socket diagnostics require a stable path.

Migration procedure:

1. deploy with the flag absent and confirm the existing in-process smoke;
2. enable `--decode-worker` on one Primary and verify `/healthz`, append,
   streaming generation, cancellation, and one injected child kill;
3. confirm the child PID changes, the current turn resumes from the imported
   allens snapshot plus acknowledged proof checkpoint, and the router RSS does
   not contain a second model;
4. retain flag removal as rollback until the Mac parity, 100-session,
   fault-injection, latency, and four-hour acceptance suite has passed.

Worker mode can import allens portable snapshots but deliberately does not
export a K/V snapshot per token. Locally computed prefill boundaries therefore
are not republished through the router in this release. A cancellation kills
the in-flight child forward; the affected session is still closed by the gRPC
cancellation contract, while other retained session checkpoints are restored
lazily if later used.
