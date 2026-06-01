# Platform test & benchmark archive

This directory is the **historical archive** of platform-test and benchmark
artifacts collected during Kakeya v0.1 → v0.3 development. Reports here are
referenced from ADRs (`docs/adr/`) and from release notes; do **not** edit or
re-run files in place once they have been committed — they are evidence,
not working data.

When a new run is needed, write to a fresh, timestamp-suffixed filename and
add an entry below.

---

## File-naming conventions

| Prefix                        | Meaning                                                                     |
| ----------------------------- | --------------------------------------------------------------------------- |
| `mac-mlx-1{a,b,c}-…`          | Phase-1 MLX backend bring-up tests (v0.1 era).                              |
| `mac-phase-b-…`               | Phase-B sparse-logits proposer tests.                                       |
| `mac-streaming-…`             | E2 server streaming integration tests.                                      |
| `mlx-…`                       | First-pass MLX platform test on Mac M4 24 GB.                               |
| `bench_mac_kvcache_…`         | Sink+window verifier KV peak comparison (vs baseline).                      |
| `bench_mac_m4_…`              | Mac M4 micro-bench (token throughput, single prompt).                       |
| `bench_mlx_speculative_…`     | MLX speculative decoder bench (CPU/CPU vs MLX/MLX).                         |
| `bench_mlx_verifier_…`        | MLX verifier-only forward bench.                                            |
| `bench_param_sweep_…`         | Hyperparameter sweep (block size × num_diffusion_steps × proposer K).       |
| `bench_sparse_vs_dense_…`     | Proposer LM-head sparse vs dense logits comparison.                         |
| `bench_long_session_mac_…`    | Long-session memory-stability run (the v0.3 §2.3.a / §2.3.b evidence).      |
| `…junit.xml` / `…coverage.xml`| Companion test-runner artifacts for the matching `.json`.                   |
| `*.partial.json`              | Live checkpoint written every N turns by `bench_long_session.py`.           |
| `*.aborted.json`              | Annotated abort note when a long run was terminated for triage.             |

---

## v0.3 long-session archive (the ADR 0006 → ADR 0007 → ADR 0008 evidence chain)

These five runs were the empirical chain that drove the architecture from
"OpenAI-compatible HTTP server with stateless turns" toward the
session-bound runtime described in ADR 0008. Each run was originally pushed
on its own `AgentMemory/bench-*-8e7f` branch; consolidated here so ADRs and
release notes can reference stable `main` paths.

### Index (chronological)

| # | UTC date         | File (in this dir)                                       | Wall time        | Successful turns | Errors | Notes                                                          |
| - | ---------------- | -------------------------------------------------------- | ---------------- | ---------------- | -----: | -------------------------------------------------------------- |
| 1 | 2026-05-30 08:42 | `bench_long_session_mac_1780130542.aborted.json`         | 12 412 s aborted | 58               | 0\*    | First 4 h attempt; aborted at ~3.4 h. Triage notes only.       |
|   |                  | `bench_long_session_mac_1780130542.partial.json`         |                  |                  |        | Last live checkpoint from run #1 before abort.                 |
| 2 | 2026-05-30 ~16:* | `bench_long_session_mac_short_1780146230.json`           | 1 800 s (30 min) | 57               | 0      | First clean 30 min after orphan-session fix.                   |
|   |                  | `bench_long_session_mac_short_1780146230.partial.json`   |                  |                  |        |                                                                |
| 3 | 2026-05-31 09:* | `bench_long_session_mac_short2_1780196477.json`           | 1 800 s (30 min) | 58               | 0      | Adds in-flight metrics poller (`metrics_poll_interval_s=0.25`).|
|   |                  | `bench_long_session_mac_short2_1780196477.partial.json`  |                  |                  |        |                                                                |
| 4 | 2026-05-31 13:* | `bench_long_session_mac_short3_1780208693.json`           | 1 800 s (30 min) | 58               | 0      | KV gauge gated to active sessions; KV peak = **7.4 MiB**.      |
|   |                  | `bench_long_session_mac_short3_1780208693.partial.json`  |                  |                  |        |                                                                |
| 5 | 2026-05-31 14:* | `bench_long_session_mac_4h_1780211323.json`               | 14 400 s (4 h)   | 58               | **182**| Memory bounded; throughput collapses to ~0 after turn 58.      |
|   |                  | `bench_long_session_mac_4h_1780211323.partial.json`      |                  |                  |        |                                                                |

\*The `aborted.json` records 0 errors only because every later request was
rejected with HTTP 429 by the scheduler before the bench client even sent
it; the client did not classify those as turn errors. The server log showed
sustained 429s — that is the bug the orphan-session fix addressed.

### What the chain tells us

1. **Memory is bounded.** Runs #2-#5 all hold KV peak ≈ 0 / 0 / 7.4 MiB
   (depending on whether the gauge was wired to the engine yet) with KV
   drift `+0.00 MiB` over 10-min buckets. The 4 h run (#5) holds the same
   bound as the 30-min run (#4). This is the evidence behind
   **ADR 0006 §2.3.a (memory-bounded claim — VERIFIED)**.

2. **Latency is *not* bounded.** Every run shows positive `latency_drift_p50`
   in the +38 s … +41 s range, and per-bucket p50 grows monotonically:

   ```
   bucket 0 (0-10 min):   ~15 s p50
   bucket 1 (10-20 min):  ~38 s p50
   bucket 2 (20-30 min):  ~55 s p50
   ```

   In run #5 (the only run long enough to expose this), p95 keeps rising
   until turns hit the 120 s client timeout and start to error — 182 such
   timeouts in the 3.5 h tail. This is the evidence behind
   **ADR 0006 §2.3.b (latency-bounded claim — NOT achieved in v0.3)**.

3. **The cause is full-history prefill on every turn.** The bench appends
   each prior assistant reply to the next prompt. With sink+window KV
   forced to reset at every request, prefill grows linearly with turn
   count. This is what made cross-request KV reuse a v0.3 hard requirement
   rather than a v0.4 nice-to-have, and is what motivated **ADR 0007**
   (automatic prefix matching). When the Qwen3 chat template was found to
   inject generation-time-only placeholders that break token-id-level
   prefix matching, the design pivoted again, to the explicit
   session-bound protocol described in **ADR 0008**.

### Source branches (audit trail)

| File                                                 | Source branch                                              |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| `bench_long_session_mac_1780130542.{aborted,partial}.json` | `AgentMemory/bench-long-session-mac-results-8e7f` |
| `bench_long_session_mac_short_1780146230.{,partial.}json`  | `AgentMemory/bench-short-test-results-8e7f`       |
| `bench_long_session_mac_short2_1780196477.{,partial.}json` | `AgentMemory/bench-short-test-results-2-8e7f`     |
| `bench_long_session_mac_short3_1780208693.{,partial.}json` | `AgentMemory/bench-short-test-results-3-8e7f`     |
| `bench_long_session_mac_4h_1780211323.{,partial.}json`     | `AgentMemory/bench-long-4h-mac-results-8e7f`      |

These branches remain on `origin/` for the original commit-hash audit
trail; the JSON snapshots are reproduced verbatim here.

---

## Earlier (v0.1 / v0.2) artifacts

The `mlx-…`, `mac-mlx-1{a,b,c}-…`, `mac-phase-b-…`, `mac-streaming-…`,
`bench_mac_*`, `bench_mlx_*`, `bench_param_sweep_…`, and
`bench_sparse_vs_dense_…` files predate the long-session work and were
landed in earlier PRs (Phase-1 bring-up, MLX-1a/1b/1c probes, sparse-logits
A/B, MLX speculative decoder bring-up). They are kept here unchanged for
historical reproducibility and for ADR 0001 / ADR 0002 / ADR 0003 cross-
references.
