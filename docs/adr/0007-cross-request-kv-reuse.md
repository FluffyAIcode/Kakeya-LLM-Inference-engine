# ADR 0007 — Cross-request KV cache reuse for long sessions

- **Status**: Superseded by [ADR 0008](0008-session-bound-runtime-and-grpc-protocol.md) (2026-06-01). Originally Accepted 2026-05-31; superseded after empirical falsification of §2.4 by Qwen3 chat-template re-rendering. Implementation (PR 7-1..7-6, merged via PRs #30-#36 on 2026-05-31, before ADR 0008 was written) is present on `main` and is treated by ADR 0008 as a historical code layer that its Phases A-E will replace incrementally; see ADR 0008 §6.6 for the per-file disposition.
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

### 2.4 Path selection: continuation vs new-session

**Decision**: every request takes exactly one of two deterministic
paths. Both paths are first-class correct actions for their input
class. There is no "fallback" semantic — the project's engineering
principles forbid fallback as a design pattern (alongside no-mock
and no-overfit), and this section's split between continuation and
new-session reflects that prohibition.

#### 2.4.a Continuation path

Triggered when the new request's prompt is a strict monotonic
extension of the cached state. Formally, both must hold:

- `len(new_prompt) >= cache_position_start + len(cached_token_sequence)`
  (the new prompt extends at or past the cached region's logical end), AND
- `new_prompt[cache_position_start : cache_position_start + len(cached_token_sequence)]
  == cached_token_sequence` (every cached position matches the new
  prompt at the same logical position).

Action: skip prefill of the matched logical positions; run
incremental prefill on
`new_prompt[cache_position_start + len(cached_token_sequence):]`.

#### 2.4.b New-session path

Triggered when the request is **not** a continuation, i.e. fails
either of the §2.4.a conditions. Concrete sub-cases:

1. **Cold start**: the cache is empty (first request after server
   boot, or after the previous request's incremental path
   completed and truncated the cache to empty by trim).
2. **Shorter history**: the new prompt is shorter than the cached
   state's logical end. Caused by the client deliberately
   shortening conversation history (e.g., the user opened a new
   chat tab in the agent UI).
3. **Diverging history**: the cached state's tokens disagree with
   the new prompt at one or more cached logical positions. Caused
   by the client switching to a different conversation that may
   share an early prefix (e.g., the same system prompt) but
   diverges before the cache window's end.

Action: reset the verifier and run full prefill on `new_prompt`. The
cache state is replaced with the new session's state.

#### 2.4.c Path semantics

Both paths produce **bit-identical** output for the same input
prompt (per §2.7). The only difference is computational cost: the
continuation path skips already-prefilled tokens.

Selecting the new-session path is **not** a degradation of the
continuation path; it is the **correct** action when the input
does not satisfy the continuation precondition. A new conversation
genuinely requires a fresh prefill — there is no shortcut, and
choosing to fresh-prefill is not a "fall back to a worse path".

The two-path structure exhausts the input space: every valid input
prompt satisfies the continuation precondition or it does not. The
selection function is total. There is no third path. Inputs that
violate the path's preconditions at runtime cannot exist by
construction; if such an input appears, it is an anomaly invariant
violation per §2.9, which is a bug not a fallback.

### 2.5 Cache state lifecycle (single-tenant scope)

**Decision** for v0.3: **the cache holds exactly one state at any
time, lives as long as the server process, and is overwritten via
the path-selection function on every request.** There is no LRU,
no idle timeout, no explicit eviction call.

**Rationale**: with a single cache slot (`max_concurrent=1`), the
state lifecycle is fully described by §2.4's two paths:

- Continuation path: cache state is **extended** with the
  incremental tokens.
- New-session path: cache state is **replaced** with the fresh
  prefill's output.

Both transitions are deterministic correct actions per §2.4.c. No
state is left "stale" — the state at any moment reflects whichever
session was most recently observed. Adding idle-timeout machinery
for v0.3 single-tenant is premature optimization without a
concrete user need.

**Forward-compatibility**: in v0.4 (multi-tenant), the cache space
holds N states (one per concurrent session), and the lifecycle
includes eviction (LRU + idle timeout) because the state count is
bounded but the request stream is not. v0.4's ADR will specify
those policies. The session abstraction this ADR introduces is
structured so v0.4 can extend it without rewriting the v0.3 core.

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
strict on Mac M4, the test gate should be **relaxed once,
explicitly, in this ADR** to "logits agree to within float16 ULPs"
— with the relaxation written down here, not silently applied at
test runtime. A test that adapts its strictness based on whether
the strict path passes is a fallback in disguise.

### 2.8 Backward compatibility: path totality

**Decision**: cross-request reuse is **transparent and automatic**;
there is no opt-out. The path-selection function (§2.4) is total
over all valid inputs. There is no fallback semantic.

**Rationale**:

- A client whose request happens to satisfy the continuation
  precondition (the dominant case for an agent in a multi-turn
  loop) takes the continuation path. The server's behavior on its
  output is bit-identical to v0.3.0-rc1's per-turn-reset behavior.
- A client whose request does not satisfy the continuation
  precondition (cold start, new chat, edited history) takes the
  new-session path. The server's behavior on its output is **also**
  bit-identical to v0.3.0-rc1's per-turn-reset behavior — because
  full prefill is exactly what v0.3.0-rc1 always did.

Therefore the upgrade is observably indistinguishable from
v0.3.0-rc1 on output (same tokens, same `usage` block, same
`/healthz`), with the single observable change being **per-turn
latency**: continuation path turns are O(new tokens), new-session
path turns are O(history length, same as v0.3.0-rc1).

The phrase "graceful degradation" is deliberately not used.
Degradation implies a primary correct path and a less-correct
backup. Both paths here are equally correct for their input
classes, just with different cost profiles. This framing is
required by the project's no-fallback principle (alongside no-mock
and no-overfit).

### 2.9 Anomaly invariants (these are bugs, not states)

The path-selection function (§2.4) is total over the input space,
but the verifier's internal state has invariants that the
implementation is responsible for maintaining. Their violation is
a **bug**, not a path. Violations must surface immediately as
runtime errors; the implementation must not silently recover, retry,
or take an alternate path.

**Required invariants**:

- **INV-1: parallel-sequence consistency.** For every layer's
  `SinkWindowKVCache`,
  `len(cached_token_sequence) == cache.cache_seq_length()` must
  hold after every cache mutation (`update_and_fetch`, `trim`,
  `reset`). The parallel token sequence must never drift from the
  K/V tensor sequence dimension.
- **INV-2: position monotonicity within a session.** During a
  continuation chain (consecutive continuation-path requests for
  the same session), `cache_position_start` is monotonically non-
  decreasing across requests. A continuation that decreases
  `cache_position_start` indicates a cache-management bug.
- **INV-3: continuation-path determinism.** For inputs that satisfy
  the continuation precondition (§2.4.a), the incremental-prefill
  output must be bit-identical (or float-precision-equivalent per
  §2.7) to the full-prefill output for the same input. This is the
  contract that makes §2.4.c's "both paths correct for their
  inputs" claim hold.

**Detection and response**:

- INV-1 is checked at every cache mutation via Python `assert`
  statements (cheap, in-process). Violation raises `AssertionError`
  to the route handler, which surfaces as an HTTP 500 with the
  OpenAI error envelope and a unique error id for log correlation.
- INV-2 is checked when path selection runs; violation raises
  `RuntimeError` with the offending values for the bug report.
- INV-3 is checked offline via the §2.7 determinism gate test
  (mandatory before merge). It is not a runtime check because the
  comparison requires running both paths on the same input, which
  is too expensive in production.

A violation of any of these is a **critical bug**. The
implementation does not retry, does not fall back, does not silently
choose the new-session path to "recover". It raises. Operators
encountering an INV-1 or INV-2 violation should file a bug report
and restart the server. The next request after restart takes the
cold-start sub-case of the new-session path (§2.4.b case 1), which
is correct in its own right — but the assertion that surfaced the
bug must be investigated, never papered over.

The OpenAI error envelope returned for INV-1 and INV-2 violations
follows the convention from PR #13:

```json
{
  "error": {
    "message": "internal cache invariant violation; bug id <UUID>",
    "type": "internal_error",
    "code": "kv_cache_inv_violation"
  }
}
```

The bug id is a UUID logged alongside the assertion stack trace so
the report can be correlated with server logs.

### 2.10 Observability

**Decision**: extend Prometheus metrics with:

- `path_selection_total{path="continuation|new_session"}` —
  counter of per-request path-selection decisions. Both labels are
  first-class outcomes; neither is an "error" or "fallback". The
  ratio `continuation / (continuation + new_session)` over a long
  session indicates how well the upstream client preserves
  prefix-extending history.
- `continuation_tokens_skipped_total` — counter of cumulative
  prompt tokens that the continuation path did not need to
  re-prefill across the lifetime of the server. Concretely
  measures the win.
- `verifier_prefill_duration_seconds{path="continuation|new_session"}`
  — histogram of prefill wall time per request, partitioned by
  path. Continuation-path histogram should center around the
  per-incremental-token cost; new-session-path histogram tracks
  full-prefill cost.
- `cache_invariant_violations_total{kind="inv1|inv2"}` — counter
  of INV-1 / INV-2 anomaly detections (per §2.9). Should always
  be 0. Any non-zero value is a critical alert.

These are net additions; existing `scheduler_kv_live_bytes` and
friends keep their semantics.

**Operational use**:

- **Healthy long-session agent**: continuation rate is high (e.g.
  ≥ 95% of requests for a multi-turn LangChain conversation).
- **Healthy mixed workload**: continuation rate may be lower —
  agents spawning short-lived conversations, multiple parallel
  threads, or system-prompt rotation will all legitimately
  generate new-session-path requests. A "low" continuation rate
  by itself is not a problem; it is a workload characterization.
- **Critical alert**: any non-zero `cache_invariant_violations_total`.
  This is a bug, not a degraded state. Page on this metric.
- **Performance regression alert**: a previously-high continuation
  rate dropping unexpectedly. This indicates an upstream change
  (client started inserting timestamps in history, framework
  upgrade changed message format) that broke the prefix.

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
| 7-1 | `SinkWindowKVCache` + parallel token sequence (MLX + CPU) | `logical_token_sequence` + `logical_position_start`; `update_and_fetch` / `trim` / `reset` paths sync the parallel sequence; **INV-1 assert at every mutation site**; unit tests | 100% on touched modules |
| 7-2 | `Verifier.path_select(prompt) -> ContinuationPlan \| NewSession` + `prefill_incremental(skip_n)` (MLX + CPU) | the path-selection function (§2.4) + the incremental prefill path; **INV-2 assert in path-select**; unit tests with synthetic verifier covering both paths' inputs explicitly | 100% |
| 7-3 | `SpeculativeDecoder` integration | accept the path-selection result; route between full-prefill and incremental-prefill paths | 100% |
| 7-4 | `SpeculativeEngine` route-handler integration | call verifier.path_select before delegating to decoder.generate; emit `path_selection_total` metric; route INV violations to OpenAI error envelope per §2.9 | 100% |
| 7-5 | Determinism gate test (§2.7 + INV-3) | bit-identical comparison between continuation path and always-reset path on a 30-turn synthetic conversation; covers all path-selection branches; mandatory before merge | mandatory before merge |
| 7-6 | bench_long_session_v2 + 4h Mac re-run | bench observes per-turn cost stable at O(new_message); §2.3.a still holds; INV violations counter is 0 over 4h | 4h Mac evidence |
| 7-7 | ADR 0006 §2.3.b deletion + §2.3.a expansion | delete the no-longer-valid caveat; expand §2.3.a with v2 bench evidence | doc-only |

Estimated: 7 PRs total. Each independently reviewable. PRs 7-1 →
7-4 are pure code on stack; PR 7-5 is the gate; PRs 7-6 / 7-7 are
validation + docs.

## 6. Validation

This ADR is considered validated when:

1. All 7 implementation PRs land on main.
2. The §2.7 determinism gate passes: bit-identical (or
   float16-ULP-identical on Metal, per the §2.7 OQ resolution) output
   between continuation path and always-reset path over a 30-turn
   synthetic test. **The relaxation, if any, is recorded in this
   ADR before the gate is changed — never silently relaxed at test
   runtime.**
3. INV-1 and INV-2 (§2.9) assertions never fire during the §6
   validation runs. The `cache_invariant_violations_total` counter
   stays at 0 across the determinism test, the synthetic suite, and
   the 4h Mac M4 run. Any non-zero value is a release blocker.
4. A 4-hour Mac M4 run with `bench_long_session_v2.py` produces
   ≥ 200 successful turns (vs the 58 turns of v0.3.0-rc1's 4h run)
   with `agg.kv_bounded == True` and `agg.n_errors < 5`.
5. Per-turn p50 latency drift over the 4-hour run is ≤ 5 seconds
   (vs the +39.74 s drift of v0.3.0-rc1).
6. ADR 0006 §2.3.b is deleted and replaced with a paragraph in §2.3.a
   citing this ADR's evidence.
7. `path_selection_total{path="continuation"}` reports ≥ 95% of
   total path selections on the 4h bench.

Items 2–5 are GA gates; v0.3.0 cannot promote to GA without all of
them.

## 7. Open questions — resolved 2026-05-31

All five OQs were resolved with the recommended-default answers
on 2026-05-31. The ADR moved from Proposed to Accepted on the
same date. The original questions and resolutions are recorded
below for traceability:

- **OQ-1 (RESOLVED, default accepted)**: System-prompt change
  handling. What if the user's system prompt rotates mid-session
  (an agent's "tool" message inserted ahead of the conversation
  history)? This shifts every position by N, breaking the prefix.
  **Resolution**: take the new-session path uniformly. Do not
  attempt to detect "system prompt only changed" specifically.
  Revisit only if production data shows the simpler behavior is
  costly enough to justify the detection logic.

- **OQ-2 (RESOLVED, default accepted, strengthened)**: Numerical
  determinism strictness on Mac M4. **Resolution**: bit-identical
  by default. If a future test cycle shows the strict gate is
  unreachable on Metal, the relaxation must be **written into this
  ADR explicitly** (as an amendment) before the gate is changed.
  Tests that auto-relax based on whether the strict path passes
  are explicitly forbidden — that pattern is a fallback in
  disguise.

- **OQ-3 (RESOLVED, default accepted)**: Should the
  `--no-cross-request-reuse` server flag exist for debugging?
  **Resolution**: no. Path selection (§2.4) is total; there is
  no failure mode that requires a runtime toggle. Operators who
  want to compare against pre-cross-request-reuse behavior can
  check out the v0.3.0-rc1 tag.

- **OQ-4 (RESOLVED, default accepted)**: Expose matched-prefix
  length in the `usage` block of the response? **Resolution**: no
  for v0.3. `usage` is a public OpenAI-compatible field; adding
  Kakeya-specific keys leaks implementation detail to clients
  that don't need it. The same information is available via
  Prometheus metrics (§2.10) for operators. Revisit only if a
  user-facing need surfaces.

- **OQ-5 (RESOLVED, default accepted)**: Eviction in v0.3 is
  "implicit replacement" (next non-matching request takes the
  new-session path and overwrites the cache). Adequate, or need a
  "stale cache idle timeout" in single-tenant scope?
  **Resolution**: no idle timeout for v0.3. Single-tenant servers
  are typically long-running with one client; stale-cache-on-idle
  has no concrete operational benefit at this scope. v0.4
  multi-tenant ADR will define LRU + idle-timeout policy at that
  scope.

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
