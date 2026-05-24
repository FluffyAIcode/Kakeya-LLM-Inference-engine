# ADR 0003 — Verifier ↔ Slab Pool Integration

- **Status**: Accepted
- **Date**: 2026-05-24
- **Decision drivers**: Memory accounting accuracy, multi-session
  serving correctness, engineering risk vs reward at v0.2.0 scope.
- **Depends on**: ADR 0001, ADR 0002.
- **Supersedes**: nothing.

## 1. Context

ADR 0001 §5.3 and `docs/local-inference-engine.md` envisioned a
fixed-slab KV pool replacing the verifier's `transformers.cache_utils.DynamicCache`
entirely. PR #8 shipped the slab pool and admission scheduler; PR
#12 wired HTTP routes through that scheduler; PR #13 added Prometheus
metrics including `scheduler_pool_in_use` and `scheduler_pool_total`
gauges.

There is one residual asymmetry: the slab tensors handed out by
`SlabPool.acquire()` are currently **placeholder bookkeeping bytes**
(1-element bf16 tensors per slab in the default placeholder pool;
~4 bytes total). The verifier's actual KV cache continues to live in
the `DynamicCache` that `transformers` allocates and manages. This
means:

- `scheduler_pool_in_use` reports the count of held slabs honestly,
  but `slab.kv_bytes` and `slab.live_kv_bytes` are misleading: the
  numbers reflect the placeholder tensors, not the real KV memory
  the session is consuming.
- A multi-session deployment with `max_concurrent=N` actually holds
  `N × DynamicCache_bytes` of KV in `transformers`-managed memory.
  None of that shows up in the slab pool's `total_kv_bytes` property.
- The original design vision — *the slab pool's tensors ARE the
  verifier's KV cache* — would close this gap by making the slab
  tensors hold the real K/V data and having the model forward
  consume them directly.

## 2. The full refactor and why we are not doing it now

The full refactor target replaces `DynamicCache` with a custom
`SlabBackedCache` subclass that:

1. Implements every method on `transformers.cache_utils.Cache` that
   the Qwen3 forward uses (`update`, `get_seq_length`,
   `crop_past_key_values`, layer-iteration, etc.).
2. Stores K/V layer tensors as views into the slab's pre-allocated
   `[num_layers, num_heads, capacity, head_dim]` buffers rather
   than allocating fresh per-step tensors.
3. Routes the sink+window trim through `KVSlab.append` /
   `KVSlab.truncate` / the existing window-slide logic.
4. Preserves RoPE correctness: surviving K vectors keep the rotation
   they had at their original positions, and new keys rotate at
   their true global position.
5. Preserves the speculative decoder's bit-equivalence with vanilla
   greedy AR (the existing test contract).

This is a substantial body of work. Two factors push the engineering
risk meaningfully higher than a typical refactor:

- **Correctness fragility.** `transformers` 4.x's `Cache` API has
  documented behaviors but no formal contract. Subtle wrong-output
  bugs from a slightly off `cache_position` or `update()` semantic
  would not show up in our current test suite — we have no
  bit-equivalence harness comparing a `SlabBackedCache` run against
  a `DynamicCache` run on the same prompt. Without that test
  infrastructure, "the tests pass" does not mean "the model is
  generating correctly".
- **Cross-version churn.** Qwen3's modeling code lives inside
  `transformers`; its expectations of `past_key_values` change
  across `transformers` minor versions. A `SlabBackedCache` that
  works on 4.45 may break silently on 4.52. Maintenance load is
  unbounded until we add a CI matrix that exercises both ends of
  our pinned `transformers` range.

The combination of "high probability of subtle wrong-output bugs"
and "no test infrastructure to detect them" makes shipping the full
refactor in v0.2.0 a poor risk/reward trade. We defer it.

## 3. Decision: ship an intermediate step now, full refactor in v0.3

For v0.2.0, we ship the **smallest concrete step that makes the
metrics accurate** without modifying the verifier's model-forward
path:

1. `KVSlab` gains a `live_kv_bytes_override: Optional[int]` attribute
   and the `live_kv_bytes` property returns the override when set.
2. A new `inference_engine/scheduler/pooled_verifier.py` defines
   `PooledVerifier`, a wrapper around any verifier (PyTorch
   `SinkWindowVerifier` or `MLXSinkWindowVerifier`) that:
   - Holds an optional reference to a `SlabPool`.
   - On `prefill()`: acquires a slab (releasing any previously
     held one).
   - On `reset()`: releases the held slab, if any.
   - After every forward (`prefill` / `forward_block` / `append_token`
     / `commit_or_truncate`): writes the verifier's real
     `stats.peak_kv_bytes` snapshot into the slab's
     `live_kv_bytes_override`, so `scheduler_pool_in_use_bytes`
     (a future metric) and `slab.live_kv_bytes` report real numbers.
3. `Scheduler.submit()` continues to acquire / release placeholder
   slabs as today; integrators wiring real verifiers into the
   scheduler use `PooledVerifier(verifier, scheduler.pool)` to bind
   the two.
4. The slab tensors stay as placeholders. The verifier's K/V
   tensors stay in `DynamicCache`. Behavior under model forward is
   bit-identical to v0.1.0.

The intermediate step costs ~150 lines of code + tests. It cannot
introduce wrong-output bugs because it does not touch the model
forward.

## 4. Acceptance criteria for v0.3 (the full refactor)

When the full refactor lands in a future PR, it must:

1. **Pass a bit-equivalence test** comparing N tokens of greedy AR
   output between (a) the old `DynamicCache` path and (b) the new
   `SlabBackedCache` path on real Qwen3-1.7B for at least three
   distinct prompts including one ≥ 256 tokens.
2. **Run on both ends of the supported `transformers` range**
   (currently 4.45.x and 4.52.x; may shift). CI gains a matrix.
3. **Preserve sink+window trim correctness**: a regression test
   exercises a session that exceeds `sink_size + window_size` by
   ≥ 50 % so the slide path runs.
4. **Show measurable memory savings** in the
   `bench_mlx_verifier_quant.py`-style comparison: total resident
   memory at `B=N, S=8192` should be ≤ 1.05× of the analytical
   prediction `N * (sink+window) * num_layers * num_heads * head_dim * 2`.
5. **Be reversible**: a `--legacy-cache` flag on `scripts/serve.py`
   (or a config switch) keeps the `DynamicCache` path available for
   one minor release in case the refactor surfaces a real-world
   issue we miss in CI.

The full refactor has its own ADR (planned 0005) at the time it
ships, which records the test fixtures, the memory measurements,
and the version matrix.

## 5. Alternatives Considered

### 5.1 Ship the full refactor in v0.2.0 (rejected — see §2)

### 5.2 Ship nothing for #3 in v0.2.0; tag v0.2.0 without it (rejected)

The user-visible `scheduler_pool_in_use` gauge is misleading today.
Even a small accuracy improvement is worth shipping. Status-quo
silence on this asymmetry leaves operators unable to size pool
capacity from telemetry alone.

### 5.3 Replace `DynamicCache` only on the MLX backend first (deferred)

MLX's `inference_engine.backends.mlx.cache.SinkWindowKVCache`
already manages slab-like fixed buffers. Unifying it under
`KVSlab` is structurally cleaner than the PyTorch `DynamicCache`
path because we control the entire MLX cache implementation. It is
attractive as a smaller proving ground for the full refactor — but
deferring it to a separate PR alongside the PyTorch refactor lets
both share the bit-equivalence harness rather than each inventing
its own.

## 6. Consequences

### 6.1 Positive

- **Metrics become honest** for v0.2.0 deployments that wire
  `PooledVerifier` into the scheduler. `slab.live_kv_bytes` reports
  real KV memory; `scheduler_pool_in_use` plus a follow-up
  `scheduler_pool_kv_bytes` metric give operators the data to size
  pool capacity.
- **The full refactor's test infrastructure can be specified
  upfront** (§4) rather than retrofitted after a problem is
  observed in production.
- **No correctness risk introduced now**. The model forward path is
  unchanged.

### 6.2 Negative / accepted trade-offs

- The slab pool's `kv_bytes` and `total_kv_bytes` properties remain
  reporting placeholder bytes for v0.2.0 deployments that don't
  wire `PooledVerifier`. They become accurate only via the wrapper.
  This is documented in `inference_engine.memory.pool` docstring.
- Two cache paths coexist in the codebase (DynamicCache via verifier,
  KVSlab via pool) until v0.3. Code reviewers must hold both in
  mind. This is the cost of staging a high-risk refactor.

### 6.3 Implications for code

- `inference_engine/memory/slab.py`: add
  `live_kv_bytes_override: Optional[int]` and modify the
  `live_kv_bytes` property.
- `inference_engine/scheduler/pooled_verifier.py` (new): the
  wrapper class.
- `inference_engine/scheduler/__init__.py`: export `PooledVerifier`.
- README + this ADR cross-referenced from
  `docs/local-inference-engine.md`.
- Tests: pure-CPU unit tests against a `_FakeVerifier` real
  concrete class. No HF cache required for CI.

## 7. Validation

This ADR is considered validated when:

1. The intermediate step (§3) is implemented with 100% line coverage
   on the new code.
2. A walkthrough of `inference_engine.memory` and
   `inference_engine.scheduler` documents which paths are
   "placeholder bookkeeping" and which produce real KV byte counts.
3. The full refactor's acceptance criteria (§4) are restated in
   the future ADR 0005 when that PR opens — this ADR's §4 is
   normative for that future work.

## 8. References

- ADR 0001 — proposer sizing + alignment.
- ADR 0002 — verifier selection + quantization.
- `docs/local-inference-engine.md` — original engine architecture.
- PR #8 (E3 slab pool), PR #9 (E4 scheduler), PR #12 (E2↔E4
  integration), PR #13 (metrics).
- `transformers.cache_utils.Cache` — the contract a future
  `SlabBackedCache` must implement.
