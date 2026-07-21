# Primary decode Mac acceptance

`scripts/bench_agentic/primary_decode_acceptance.py` is the blocking Mac
acceptance runner for the isolated Primary decode worker. It emits JSON
conforming to `schemas/primary_decode_acceptance.schema.json` and JUnit XML.
The full mode includes the four-hour mixed-prompt run; it is only started by
an explicit `--mode all` or `--mode endurance`.

Example full command:

```bash
PYTHONPATH=.:sdks/python python3 \
  scripts/bench_agentic/primary_decode_acceptance.py \
  --mode all \
  --grpc-address 127.0.0.1:50051 \
  --tokenizer-id Qwen/Qwen3-0.6B \
  --worker-control-command "/path/to/decode-worker-acceptance-adapter" \
  --latency-baseline results/platform-tests/bench_mlx_verifier_1779507043.json \
  --output results/platform-tests/primary_decode_acceptance.json \
  --junit-output results/platform-tests/primary_decode_acceptance.xml
```

Start the runtime with its test-only, mode-0600 UDS enabled:

```bash
PYTHONPATH=.:sdks/python python3 scripts/start_grpc_runtime_server.py \
  --backend mlx \
  --verifier-id Qwen/Qwen3-0.6B \
  --bind 127.0.0.1:50051 \
  --capacity 128 --sink 4 --window 64 \
  --decode-worker \
  --decode-worker-timeout-s 110 \
  --decode-worker-acceptance-socket /tmp/kakeya-decode-acceptance.sock
```

The corresponding adapter command is:

```bash
PYTHONPATH=. python3 \
  scripts/bench_agentic/decode_worker_acceptance_adapter.py \
  --socket /tmp/kakeya-decode-acceptance.sock
```

Run `footprint`, `disconnect`, `hang`, `kv-restore`, and `latency` separately
for pre-CI/hardware checks. Reserve `--mode all` for the blocking release run:
it deliberately includes the four-hour endurance workload.

Short development runs can override `--session-count` and
`--endurance-duration-s`, but the gates remain fixed at 100 sessions and
14,400 seconds. A shortened run therefore emits useful diagnostics while
remaining failed and cannot be mistaken for release evidence.

## Decode-worker adapter contract

The runtime provides
`scripts/bench_agentic/decode_worker_acceptance_adapter.py`, passed through
`--worker-control-command`. The harness starts it once per operation, writes
one JSON object to stdin, and expects one JSON object on stdout:

```json
{
  "schema_version": 1,
  "operation": "snapshot",
  "payload": {}
}
```

```json
{
  "schema_version": 1,
  "operation": "snapshot",
  "ok": true,
  "data": {}
}
```

Nonzero exit status, malformed JSON, `ok: false`, or mismatched operation is
reported as a gate error. The adapter must be test-only/local-only and must
not expose fault injection on a network listener.

Required operations:

- `snapshot`: returns integer `runtime_pid`, `worker_pid`,
  `worker_restart_count`, `process_footprint_bytes`, `active_sessions`, and
  `active_generations`. The footprint must cover the runtime plus owned decode
  worker, not only the router process.
- `inject_hang`: accepts payload `phase: "next_forward"` and
  `expected_worker_pid`; atomically arms exactly the next MLX forward in that
  worker and returns `accepted: true`. The injected forward must remain hung
  until the normal watchdog/recycle path kills the worker.
- `kv_restore_parity`: accepts `prompt_token_ids` and performs a normal
  decode, worker recycle, restore from the persisted Allens KV snapshot plus
  proof checkpoint, and repeated decode. It returns
  `baseline_first_token_id`, `restored_first_token_id`,
  `baseline_logits_sha256`, `restored_logits_sha256`, and the literal
  `restore_source: "allens_kv+proof_checkpoint"`. SHA-256 is over a
  canonical, dtype-preserving byte representation of the last-token logits.

The gRPC/router branch must additionally guarantee that closing the client
channel cancels the in-flight Generate and removes its session. The harness
starts timing only after `snapshot.active_generations` becomes nonzero, then
requires both active counters to become zero within five seconds.

## Existing benchmark reuse

Mixed-prompt endurance records are aggregated by
`inference_engine.bench.session_long_run.aggregate_run`. Latency baseline
input accepts either a previous acceptance report's top-level `latency`
summary or the existing `bench_mlx_verifier.py` JSON. For the latter, the
reported MLX generation mean per token is used as the reference for both
p50 and p95 and is identified as such in `baseline_source`; a prior
acceptance report is preferred once available.
