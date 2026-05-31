# ADR 0007 — Cross-request KV cache reuse for long sessions

- **Status**: Proposed (in flight, blocking v0.3.0 GA)
- **Date**: 2026-05-31
- **Decision drivers**: ADR 0006 §2.3.b empirical evidence (4-hour Mac M4
  run); the gap between the "local agent infrastructure" framing and the
  v0.3.0-rc1 implementation; the need to make `bench_long_session.py`
  validate "long-session usability", not just "memory bounded over a
  brief usability window".
- **Depends on**: ADR 0001 (verifier sink+window), ADR 0003 (slab pool),
  ADR 0006 (positioning + §2.3 sub-claims).
- **Supersedes**: nothing. Refines the v0.3.b non-claim from ADR 0006
  into a v0.3 claim.

## 1. Context

The 2026-05-31 4-hour Mac M4 long-session bench produced byte-exact
verification of the §2.3.a memory-bounded claim **and** empirical
exposure of a separate failure mode:

- Successful turns: 58, all in the first 30 minutes
- Errors over the next 3.5 hours: 96 client-side `ReadTimeout` + 86
  HTTP 429 in alternating 60–130 s cycles
- Useful work throughput past the 30-minute mark: zero

Diagnosis (see commit history of `AgentMemory/bench-long-4h-mac-results-8e7f`
and `AgentMemory/adr-0006-section-2.3-revision-8e7f`): per-turn
prefill cost grows linearly with chat history length. With the current
v0.3.0-rc1 implementation, every chat-completions HTTP request resets
the verifier's KV cache and re-prefills the entire history from
scratch. After ~3500 history tokens the prefill alone exceeds the
bench's `timeout_s=120`, and the long-session degrades into a
timeout/recovery loop where the server keeps doing correct compute but
no client gets useful tokens.

ADR 0006 §2.3.b previously framed this as a protocol-level limit
("OpenAI chat-completions is stateless"). That framing was wrong on
the architectural axis: the OpenAI protocol does not *require* the
server to be stateless, only to *accept* stateless requests. The
per-turn `verifier.reset()` at the start of every prefill is a v0.2.0
implementation choice inherited from the chat-speedup framing era. It
was a sensible default in v0.2.0 (no session concept, "default to
safety" for cache isolation), but under the ADR 0006 agentic
infrastructure framing it is a system-level bug — agents are
intrinsically multi-turn, and a long-session-capable agent
infrastructure must keep KV state across turns of the same logical
conversation.

This ADR designs the v0.3 fix: **server-side automatic prefix
matching with sink+window-aware cross-request KV reuse**. No client
protocol changes; existing LangChain, CrewAI, AutoGen, Cursor, and
Open WebUI clients gain long-session usability without modification.

## 2. Decisions

### 2.1 Session-identity strategy: server-side automatic prefix matching

**Decision**: the server identifies a "session continuation" by
**matching the new request's tokenized prompt against the current
verifier cache state**, not by an explicit `session_id` field from
the client.

**Rationale**:

- All target frameworks (LangChain `ChatOpenAI`, CrewAI `LLM`,
  AutoGen `OpenAIChatCompletionClient`, Cursor's custom-endpoint
  bridge, OpenAI Python SDK) speak strict OpenAI chat-completions.
  None of them know about a `session_id` field. Adding client-side
  session identity requires modifying every framework integration —
  exactly the integration cost ADR 0006 §2.2 was designed to avoid.
- Industry precedent: vLLM's automatic prefix caching, llama.cpp's
  `--cache-reuse`, TGI's prefix-cache layer all use server-side
  matching, no client coordination needed.
- For our v0.3 single-tenant scope (`max_concurrent=1`), the
  server holds at most one cache state. The match check is O(N)
  in prompt length, trivial vs the prefill work it saves.

**Consequence**: client code stays untouched. A user running an
existing LangChain agent against a Kakeya endpoint upgraded with this
feature gets cross-request reuse automatically.

### 2.2 Cache state representation: SinkWindowKVCache + parallel token sequence

**Decision**: extend `SinkWindowKVCache` (and its CPU PyTorch peer)
to maintain, in addition to the K/V tensors, a parallel
`logical_token_sequence: list[int]` recording the token id at each
*logical* (post-trim) position in the cache.

**Rationale**:

- Prefix matching needs to compare new request tokens against what
  the cache currently holds. K/V tensors don't expose token ids, so
  we must store them alongside.
- Storage cost: at most `sink+window=68` int32 entries per verifier
  = 272 bytes total (the cache structure is a single list, not
  per-layer). Negligible vs the 7.4 MiB KV.
- The `logical_token_sequence` is updated synchronously with the K/V
  tensors inside `update_and_fetch` and `trim`, preserving the
  invariant `len(logical_token_sequence) == cache.cache_seq_length()`.

**Open question 2.2.a**: Should `logical_token_sequence` also track
the *global* token position (which prefill position each cache slot
came from)? Yes — required for matching against requests where the
cache holds positions [12..79] but the new request's history is
150 tokens long. Stored as an int offset:
`logical_position_start: int` = the global token position of the
first cache slot.

### 2.3 Prefix-matching algorithm

**Decision**: on every chat-completions request, before deciding
whether to reset the verifier:

```python
def find_reusable_prefix(
    new_prompt: list[int],
    cached: list[int],
    cache_position_start: int,
) -> int:
    """Return the number of NEW prompt tokens that are already in the
    cache (i.e. that we can skip during prefill).

    Returns 0 if there is no useful overlap (force a full reset).
    """
    if not cached:
        return 0
    # cached covers global positions [cache_position_start ..
    #     cache_position_start + len(cached)).
    # We can reuse prefix iff new_prompt[0..cache_position_start]
    # exactly equals what was previously prefilled (the dropped
    # sink+window-evicted tokens), AND new_prompt[cache_position_start
    # .. cache_position_start + len(cached)] equals `cached` token
    # by token. The first condition is unverifiable (we threw those
    # tokens away), so we can only reuse if the new prompt starts
    # with the same prefix it had on the previous turn AT and AFTER
    # cache_position_start.
    if len(new_prompt) <= cache_position_start:
        return 0  # new prompt is shorter than the part we evicted
    cache_window_in_new_prompt = new_prompt[
        cache_position_start : cache_position_start + len(cached)
    ]
    # Find longest matching prefix within the cache window
    n_match = 0
    for i, (a, b) in enumerate(zip(cache_window_in_new_prompt, cached)):
        if a != b:
            break
        n_match += 1
    if n_match == 0:
        return 0
    return cache_position_start + n_match
```

The returned value tells `SpeculativeDecoder.generate` how many
prefix tokens to skip during prefill. The remaining tokens are
processed with `forward_block` (which uses sink+window-aware
incremental update) instead of `prefill` (which calls reset).

**Open question 2.3.a**: What if the new request's history is
*shorter* than `cache_position_start + len(cached)`? E.g. the user
edited an earlier turn and the new history is now only [sys, user_1,
asst_1_edited, user_2_edited]. Two paths:

- (Conservative) treat as no-overlap, reset, full prefill.
- (Aggressive) try matching against `cached[: shorter_length]` and
  truncate the cache.

**Decision**: conservative path. If the new prompt cannot extend the
cached state monotonically, reset. Edge case rare enough that the
conservative path is correct.

### 2.4 Reset criteria (when prefix matching fails)

**Decision**: the verifier resets and runs full prefill in any of:

1. The cache is empty (first request after server start).
2. `find_reusable_prefix` returns 0 (no overlap with cache).
3. The matched prefix length is below a threshold (default 4 tokens):
   the savings of skipping 1–3 tokens of prefill don't justify the
   logic complexity, just reset.
4. `cache_position_start > 0` (cache has already evicted earlier
   tokens) AND `len(new_prompt) < cache_position_start`: the new
   prompt is too short to overlap with what the cache currently
   holds.
5. The new request's `system` message differs from the system
   message that was active when the cache state was built (covered
   by automatic mismatch in matching, but called out for clarity).

**Open question 2.4.a**: should we add a `force_reset=True` request-
level escape hatch? Default no — automatic prefix matching is correct
in 100% of cases (mismatch → fall back to reset). An escape hatch
adds protocol surface for marginal benefit. Revisit in v0.4 if a real
user-facing need surfaces.

### 2.5 Cache eviction policy (single-tenant scope)

**Decision** for v0.3: **the cache lives as long as the server
process**. There is exactly one cache state at any time
(`max_concurrent=1`), and it gets replaced via the prefix-match
algorithm on every request. No idle timeout, no LRU, no explicit
eviction.

**Rationale**: with a single cache slot and prefix-match-based reuse,
"eviction" happens implicitly: if a new request doesn't match, the
cache resets. Adding idle timeout machinery for v0.3 is premature
optimization.

**Forward-compatibility**: in v0.4 (multi-tenant), eviction becomes
a real concern (many sessions, finite memory). v0.4's ADR will
specify LRU + idle timeout. The session abstraction we introduce
here is structured so v0.4 can extend it without rewriting the v0.3
core.

### 2.6 Concurrency: explicit single-tenant scope for v0.3

**Decision**: this ADR is **scoped strictly to single-tenant**
(`max_concurrent=1`). Multi-tenant concurrent sessions are deferred
to v0.4 (a separate ADR will address it).

**Rationale**:

- Multi-tenant correctness requires `PooledVerifier` to be wired into
  `serve.py` (it isn't today, per ADR 0003 §3.5). That wiring is
  itself a separate engineering task.
- With multiple verifier instances or one verifier serving multiple
  sessions, the single-cache-state assumption of §2.3 breaks. The
  prefix-match algorithm extends naturally (per-session cache state,
  per-session matching), but the engineering surface grows.
- Coupling cross-request reuse and multi-tenant in one release would
  delay both. v0.3 = single-tenant long sessions; v0.4 = add
  concurrent multi-session.

The v0.3 server config `max_concurrent` parameter remains user-
configurable but the production-quality path is `max_concurrent=1`.
v0.3 release notes must call this out so operators don't accidentally
deploy with `max_concurrent=4` and hit the unfixed multi-tenant
bugs.

### 2.7 Determinism and quality equivalence

**Decision**: cross-request KV reuse must produce **bit-identical
output to the v0.2.x stateless path** for any prompt that does not
trigger reset.

**Rationale**:

- v0.3 already accepts the sink+window approximation (per ADR 0001
  §4); cross-request reuse must not introduce new approximation on
  top.
- Greedy decoding + deterministic K/V computation on the same logical
  positions → same K/V values → same logits → same tokens.
- The only difference between reuse and reset paths is *when* the
  K/V values were computed (this turn vs. a previous turn). The
  values themselves are identical because attention is a pure
  function of input tokens and position.

**Validation requirement**: a paired test that runs the same prompt
sequence through (a) the reuse path and (b) the always-reset path,
and asserts bit-identical output token sequences. This test is
mandatory before v0.3.0 GA.

**Open question 2.7.a**: numerical determinism on Apple Metal —
mlx_lm sometimes produces tiny floating-point differences across
runs even with greedy decoding. If the bit-identical test is too
strict on Mac M4, fall back to "logits agree to within float16
ULPs" and document the relaxation.

### 2.8 Backward compatibility: graceful degradation

**Decision**: cross-request reuse is **transparent and automatic**;
there is no opt-out. If a request comes in that doesn't share a
prefix with the cache (e.g. a totally new conversation, or
multi-tenant traffic in v0.4), the server falls back to reset +
full prefill. Behavior is then identical to v0.3.0-rc1.

**Rationale**: removing the opt-out keeps the protocol surface
small. The fallback path is the v0.3.0-rc1 behavior, which is
already tested and shipped.

### 2.9 Observability

**Decision**: extend Prometheus metrics with:

- `cross_request_kv_reuse_decisions_total{outcome="hit|partial|miss"}`
  — counter of per-request decisions: `hit` (full reuse, prefill
  bypassed), `partial` (some prefix reused), `miss` (full reset).
- `cross_request_kv_reuse_tokens_skipped_total` — counter of cumulative
  prompt tokens that did not need to be prefilled because of prefix
  match.
- `verifier_prefill_duration_seconds` — histogram of prefill wall
  time per request, for observing the win.

These are net additions; existing `scheduler_kv_live_bytes` and
friends keep their semantics.

**Operational use**: in production, an operator should see hit-rate
≥ 95% for a healthy long-session agent. A drop to < 50% means
either (a) prompt-management code on the client side is breaking
the prefix (e.g. inserting timestamps) or (b) different sessions are
multiplexed onto one server (a v0.4 deployment running under v0.3
infrastructure).

## 3. Alternatives Considered

### 3.1 Explicit `session_id` from the client (rejected)

OpenAI extension field or HTTP header carrying a session token.
Server keeps a `dict[session_id → cache_state]`.

**Rejected** because:

- All target framework integrations break (none send session_id
  today). We would need to update each integration's example code
  and convince framework users to adopt our extension. ADR 0006 §2.2
  explicitly chose to consume frameworks "as-is" rather than fork them.
- vLLM, llama.cpp, TGI all chose server-side prefix matching for the
  same reason. We are not the first to face this trade-off.
- Adds protocol surface (server has to manage session lifecycle, IDs,
  authentication of which client owns which session) that automatic
  matching avoids.

### 3.2 TCP connection-level session affinity (rejected)

Use HTTP keep-alive connection identity as the session key.

**Rejected** because:

- Brittle: HTTP/2 multiplexes multiple requests over one connection;
  HTTP/1.1 with keep-alive is intermittent. We would have to special-
  case both.
- Frameworks rotate connections aggressively (LangChain's httpx
  pool, Cursor's per-call invocations). Connection-level identity
  doesn't survive a connection drop, breaking the long-session
  invariant for the very pattern we're trying to support.

### 3.3 Persistent disk-backed cache (deferred)

Persist KV state to disk so sessions survive server restarts.

**Deferred** to ADR 0005 (planned, personal layer). Out of scope for
v0.3.

### 3.4 Switching to a stateful protocol (e.g. WebSocket / gRPC stream) (rejected)

Replace OpenAI chat-completions with a bidirectional streaming
protocol where session state is implicit in the connection.

**Rejected** because: OpenAI compatibility is the cornerstone of ADR
0006 §2.2 (every integration framework speaks OpenAI chat-completions
out of the box). Forking the protocol would invalidate all five
integration guides and force every Kakeya user to write custom client
code. That's a much bigger ask than what we save.

### 3.5 Wait for v0.4 to do this and ship v0.3.0 GA with the §2.3.b caveat (rejected)

Ship v0.3.0 GA with the documented "long-session latency not
bounded" caveat from ADR 0006 §2.3.b. Add cross-request reuse to
v0.4 alongside multi-tenant.

**Rejected** because:

- The §2.3.b caveat reads as a protocol-level limit. It isn't — it's
  an implementation choice. Shipping v0.3.0 GA with this caveat
  perpetuates an inaccurate framing.
- The "local agent infrastructure" positioning of ADR 0006 implicitly
  promises long-session usability. v0.3.0 GA without long-session
  usability undermines the positioning at the precise moment we
  introduce it.
- The 4-hour Mac M4 evidence makes the cost concrete: only 58 turns
  of useful work in 4 hours. That number is too embarrassing to ship
  as a GA release whose framing is "agents on Mac".

## 4. Consequences

### 4.1 Positive

- v0.3.0 GA delivers what the ADR 0006 §2.1 framing promises. ADR 0006
  §2.3.b's caveat can be deleted (replaced by a §2.3.a-style
  measurement evidence section).
- Existing framework integrations (LangChain, CrewAI, AutoGen, Cursor,
  Open WebUI) get cross-request reuse for free with no client-side
  changes.
- Long-running agent applications become viable on Mac M4 / Qwen3-1.7B:
  per-turn prefill cost drops from O(history_length) to
  O(new_user_message). For a typical agent loop with ~30 token
  followups and ~30 token assistant responses, post-warmup per-turn
  latency stabilizes around 1–3 seconds.

### 4.2 Negative / accepted trade-offs

- **v0.3.0 GA delays by ~1-2 weeks.** rc1 → GA path now includes
  this work. The bench infrastructure for verification is already in
  place (PR #24-#28 + ADR #29), so the verification cost is small.
- **Single-tenant only in v0.3.** Concurrent sessions remain a v0.4
  deliverable. Operators using `max_concurrent>1` in v0.3 will hit
  cache-pollution bugs (different sessions sharing the same cache
  state via prefix-match collisions). v0.3 release notes must
  pin `max_concurrent=1` as the production configuration.
- **Quality must be bit-identical.** Cross-request reuse cannot
  introduce new approximation. The §2.7 test gate enforces this, but
  it adds a CI step that compares full outputs.
- **Storage of `logical_token_sequence`** adds 272 bytes per verifier
  state — negligible but worth recording so future readers don't
  wonder where it came from.
- **Edge cases are real.** Edited-previous-turn requests, system-
  prompt changes mid-session, and rare prompt-management bugs that
  break the prefix all degrade gracefully to v0.3.0-rc1's full-reset
  behavior. A bench user observing "performance regressed back to
  the pre-fix level" should check the `cross_request_kv_reuse_decisions_total`
  miss rate first.

### 4.3 Implications for code

New / modified modules:

```
inference_engine/backends/mlx/cache.py
  + SinkWindowKVCache.logical_token_sequence: list[int]
  + SinkWindowKVCache.logical_position_start: int
  + SinkWindowKVCache.update_and_fetch: synchronously update
    the parallel sequence
  + SinkWindowKVCache.trim: synchronously trim the parallel
    sequence

inference_engine/backends/mlx/verifier.py
  + MLXSinkWindowVerifier.find_reusable_prefix(prompt) -> int
  + MLXSinkWindowVerifier.prefill_incremental(new_tokens, skip_n=int)
    — the new path used when a prefix is reusable

kv_cache_proposer/verifier.py
  + same surface as MLX verifier (CPU peer)

kv_cache_proposer/speculative.py
  + SpeculativeDecoder.generate now accepts a hint about reusable
    prefix length (computed by SpeculativeEngine before calling)

inference_engine/server/engine.py
  + SpeculativeEngine.generate consults verifier.find_reusable_prefix
    before invoking decoder.generate

inference_engine/server/metrics.py
  + 3 new metrics (§2.9)

inference_engine/server/app.py
  + /metrics handler emits the new metrics
```

Net surface: ~6 modules touched; estimated ~1500-2000 lines including
tests. PR breakdown in §5 below.

## 5. Implementation plan (concrete PRs)

| # | PR | Scope | Coverage gate |
|---|---|---|---|
| 7-1 | `SinkWindowKVCache` + parallel token sequence (MLX + CPU) | logical_token_sequence + logical_position_start, update/trim invariants, unit tests | 100% on touched modules |
| 7-2 | `Verifier.find_reusable_prefix` + `prefill_incremental` (MLX + CPU) | the prefix-match algorithm + the incremental prefill path; unit tests with synthetic verifier | 100% |
| 7-3 | `SpeculativeDecoder` integration | accept reusable-prefix hint; route between full-prefill and incremental-prefill paths | 100% |
| 7-4 | `SpeculativeEngine` route-handler integration | call find_reusable_prefix before delegating to decoder.generate; emit decision to metrics | 100% |
| 7-5 | Determinism gate test | bit-identical comparison between reuse path and always-reset path on a 30-turn synthetic conversation | mandatory before merge |
| 7-6 | bench_long_session_v2 + 4h Mac re-run | bench observes per-turn cost stable at O(new_message) and §2.3.a still holds | 4h Mac evidence |
| 7-7 | ADR 0006 §2.3.b deletion + §2.3.a expansion | delete the no-longer-valid caveat; expand §2.3.a with v2 bench evidence | doc-only |

Estimated: 7 PRs total. Each independently reviewable. PRs 7-1 →
7-4 are pure code on stack; PR 7-5 is the gate; PRs 7-6 / 7-7 are
validation + docs.

## 6. Validation

This ADR is considered validated when:

1. All 7 implementation PRs land on main.
2. The §2.7 determinism gate passes: bit-identical (or float16-ULP-
   identical on Metal) output between reuse and always-reset paths
   over a 30-turn synthetic test.
3. A 4-hour Mac M4 run with `bench_long_session_v2.py` produces
   ≥ 200 successful turns (vs the 58 turns of v0.3.0-rc1's 4h run)
   with `agg.kv_bounded == True` and `agg.n_errors < 5`.
4. Per-turn p50 latency drift over the 4-hour run is ≤ 5 seconds
   (vs the +39.74 s drift of v0.3.0-rc1).
5. ADR 0006 §2.3.b is deleted and replaced with a paragraph in §2.3.a
   citing this ADR's evidence.
6. `cross_request_kv_reuse_decisions_total{outcome="hit"}` reports
   ≥ 95% hit rate on the 4h bench.

Items 2-4 are GA gates; v0.3.0 cannot promote to GA without them.

## 7. Open questions (require decision before implementation)

These are decisions the ADR author is *not* taking unilaterally and
need explicit approval before PR 7-1 starts:

- **OQ-1**: System-prompt change handling. What if the user's system
  prompt rotates mid-session (an agent's "tool" message inserted
  ahead of the conversation history)? This shifts every position by
  N, breaking the prefix. Default proposal: treat as a full reset.
  Alternative: detect the system-prompt shift specifically and
  handle. **Recommend default until production data shows otherwise.**
- **OQ-2**: Numerical determinism strictness on Mac M4. Strict
  bit-identical or relaxed-to-ULP-equivalent? **Recommend strict
  first, relax only if tests fail.**
- **OQ-3**: Should the `--no-cross-request-reuse` server flag exist
  for debugging? **Recommend no — keep the protocol surface
  minimal. Operators debugging can compare old behavior by checking
  out v0.3.0-rc1.**
- **OQ-4**: Should we also expose the matched-prefix length as a
  per-response field (in the `usage` block)? **Recommend no for
  v0.3** — it leaks server implementation detail. Add only if a
  user need surfaces.
- **OQ-5**: Eviction in v0.3 is "implicit replacement" (next
  non-matching request resets). Is that adequate, or do we need a
  "stale cache idle timeout" even in single-tenant? **Recommend
  no idle timeout for v0.3** — single-tenant servers are usually
  long-running with one client.

Each OQ has a recommended default. If you accept all defaults, this
ADR is ready to move from `Proposed` to `Accepted` and PR 7-1 starts.

## 8. References

- ADR 0001 — Proposer sizing, alignment, verifier decoupling. The
  sink+window choice that this ADR builds on.
- ADR 0003 — Verifier ↔ slab pool integration. The slab abstraction
  this ADR explicitly does NOT use for memory accounting (PR #27
  established that the engine reads KV state directly, not through
  the pool).
- ADR 0006 — Project positioning as local agent infrastructure. The
  framing that demands this fix.
- PR #24-#28 — measurement plumbing that lets the §6 validation
  criteria be checkable.
- PR #29 (in flight) — ADR 0006 §2.3 revision; will need a follow-up
  amendment after this ADR's PR 7-7 lands to delete §2.3.b.
- 4-hour Mac M4 evidence:
  `results/platform-tests/bench_long_session_mac_4h_1780211323.json`.
- Industry references:
  - vLLM Automatic Prefix Caching: <https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html>
  - llama.cpp `--cache-reuse`: server flag, see llama-server docs
  - TGI prefix-cache: see Hugging Face TGI source `router/src/queue.rs`
