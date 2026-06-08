# ADR 0008 — Session-bound runtime + gRPC protocol; v0.4 GA dLM K/V Restoration architecture

- **Status**: Accepted (2026-06-01) for v0.3 GA scope (§1–§10);
  Accepted (2026-06-08) for v0.4 GA architecture amendment (§11).
- **Date**: 2026-06-01 (original); 2026-06-08 (v0.4 amendment)
- **Decision drivers (original, v0.3)**:
  - Empirical failure of ADR 0007's automatic prefix matching against
    Qwen3's `enable_thinking=False` chat template (local-only smoke
    evidence, 2026-06-01).
  - User-stated strategic constraints recorded 2026-06-01: standalone
    runtime for local agent applications, no framework-coupling
    requirement, no deadline, no sunk-cost reasoning, extreme KV
    efficiency, zero intelligence regression.
  - The need to make the v0.3 GA criterion "long-session usability"
    falsifiable on hardware (Mac M4 24 GB) without depending on the
    accident of any one model's tokenization round-trip stability.
- **Decision drivers (v0.4 amendment, §11)**:
  - 2026-06-06 sink+window quality A/B benchmark
    (`results/platform-tests/sink_window_quality_ab_1780714635.json`)
    measured −83 % mid-context fact recall under v0.3's `sink=4 +
    window=64`. Source of intelligence regression isolated as the
    structurally-bounded verifier KV cache.
  - 2026-06-07 R1c–R1e empirical research (PR #65 / #67 / #68)
    falsified the cross-attention-bridge subspace of ADR 0011
    (frozen-base verifier + frozen-base proposer + trainable
    single-instance bridge). R1e-γ at localization 0.82 with recall
    0.12 is the decisive datum.
  - User-recognised architectural insight 2026-06-08: dLM proposer
    has no KV cache (forward is parallel diffusion, not autoregressive
    sequential), so its K/V tensors at every position are computed
    transiently each forward and discarded. This makes the proposer
    a constant-memory K/V reconstruction source for the verifier.
- **Depends on**: ADR 0001 (proposer sizing + verifier decoupling),
  ADR 0002 (verifier selection), ADR 0003 (slab pool), ADR 0006
  (positioning + §2.3 sub-claims).
- **Supersedes**: ADR 0007 (cross-request KV cache reuse via automatic
  prefix matching). ADR 0007 is retained as a historical record of the
  architecture-discovery process. Its implementation (PR 7-1..7-6,
  merged via PRs #30-#36 on 2026-05-31, before this ADR's strategic
  context was finalized) is present on `main` as of this ADR's merge
  commit and is treated as a historical code layer that Phases A-E
  (§6) will replace incrementally; see §6.6.
- **Rejects (v0.4 amendment)**:
  - **ADR 0010 draft** (full-attention verifier + low-precision INT8/NF4
    KV cache). Rejected because it proposed a generic literature
    baseline (NF4) inferior to the project's existing production-grade
    KV codec, and because its core premise (linear-but-thinner KV
    storage) does not satisfy the constant-memory requirement that v0.4
    targets. The draft's tombstone lives on its branch (PR #66, to be
    closed) and the original 351-line proposal is recoverable from git
    history if anyone needs to read why we did not implement it.
  - **ADR 0011 draft** (cross-attention proposer/verifier coupling).
    Rejected because the §3 frozen-base subspace was empirically
    falsified by R1c–R1e on the toy NIAH task, and because the
    motivating premise — that the verifier's KV cache must be
    structurally bounded and intelligence loss must therefore be
    recovered through a residual-stream rescue mechanism — is itself
    superseded by the §11 architecture in this ADR. ADR 0011's
    research evidence is preserved in `results/research/` and on
    its branch (PR #68, to be closed) as scientific record.
  - These rejections are **part of this ADR's scope, not separate
    documents**. The project does not maintain a sprawl of half-baked
    parallel ADRs; ADR 0008 is the single authoritative architectural
    record and this amendment integrates the v0.4 GA design into it.

---

## Outline

1. Context — where v0.3 stands; why ADR 0007 failed; the strategic frame
2. Decisions — wire protocol, session model, cache binding, determinism, concurrency, observability, deprecated-shim, anomaly invariants
3. SDK delivery — Python and TypeScript SDKs as v0.3 first-class deliverables
4. Alternatives considered (and rejected)
5. Consequences — what is gained, what is given up
6. Implementation plan — phased PR sequence
7. Validation criteria — GA gates for v0.3
8. Open questions — items reserved for resolution during implementation
9. Testing discipline — binding rule for every PR landing under this ADR
10. References

---

## 1. Context

### 1.1 Where v0.3.0-rc1 stands today

Three complementary pieces of evidence, all reproducible from `main`:

- `results/platform-tests/bench_long_session_mac_short3_1780208693.json`
  — 30 min, 58 turns, 0 errors, KV peak `7.4 MiB` flat across all
  10-min buckets. **§2.3.a memory-bounded claim: VERIFIED.**
- `results/platform-tests/bench_long_session_mac_4h_1780211323.json`
  — 4 h, 58 successful turns + **182 client-side timeouts**. KV stayed
  at `7.4 MiB` flat (memory still bounded), per-turn p50 latency
  drifted from `~15 s` (bucket 0) to `~55 s` (bucket 2) before turning
  into 120 s `ReadTimeout` for the remaining 3.5 h.
  **§2.3.b latency-bounded claim: NOT achieved.**
- `results/platform-tests/bench_long_session_mac_1780130542.aborted.json`
  — the 2026-05-30 4 h attempt that surfaced the orphan-session bug,
  retained as the empirical anchor for ADR 0006 §2.3 framing.

Diagnosis (recorded in ADR 0007 §1, still mechanically correct):
each chat-completions request resets the verifier KV cache and
re-prefills the entire chat history. Per-turn prefill cost grows
linearly with history length. After ~3 500 history tokens the prefill
alone exceeds the bench's 120 s timeout. This is not memory
pressure — it is structural. The solution must either (a) avoid
re-prefilling already-cached prefixes, or (b) avoid sending the full
history on each turn.

### 1.2 Why ADR 0007 (automatic server-side prefix matching) failed

ADR 0007's design picked option (a) under a hard constraint: leave
clients untouched (LangChain, CrewAI, Cursor, OpenAI SDK). The plan
was to identify continuation by **token-id-level prefix matching** of
the new request's tokenized prompt against the verifier's cached token
sequence.

Its first real-hardware smoke test (Qwen3-1.7B, Mac M4, 2026-06-01,
`bench_long_session_mac_v2_smoke2_1780238315.json` — local-only,
unsharded by deliberate decision; see §10) reported `continuation_rate
= 0.0` over 5 minutes of multi-turn driving. Diagnosis (also local-
only): Qwen3's chat template, when invoked with
`enable_thinking=False`, **inserts generation-time-only structural
placeholders** (an empty `<think>...</think>` block) into prior
assistant turns when re-rendering the full history. As a result, the
tokenized prompt for turn N+1 is *not* an extension of the tokenized
prompt for turn N — it is a different token-id sequence whose first
~30 tokens diverge from the cache.

This is not a bug in Qwen3; it is the chat template implementing its
own contract. It generalizes:

> **Any tokenizer chat template that re-renders prior history can
> insert, remove, or re-shape tokens that did not appear during the
> generation that produced the cache. Token-id-level prefix matching
> against an LLM-rendered chat history is therefore *not* a stable
> primitive.**

The same hazard applies, with different mechanisms, to Gemma's
`<bos>` / `<start_of_turn>` framing (token-position dependent), to
DeepSeek's tool-call markers (template-version dependent), and to any
future model that grows generation-mode-aware template logic. ADR
0007's design depended on this primitive being stable; the primitive
is not stable; therefore the design fails.

A weaker version of ADR 0007 ("require the user to disable thinking
mode and pin a tokenizer template version") would re-introduce the
"works on Qwen3 with these flags only" coupling that ADR 0006 §2.2
explicitly rejected. We do not pursue it.

### 1.3 Strategic frame (recorded 2026-06-01)

The user reframed scope and constraints, which this ADR commits to:

1. The Kakeya runtime is **not** built to serve LangChain, Cursor, or
   any external chat-completions consumer. It is a **standalone local
   runtime + protocol** for agentic applications that we ourselves
   author or that an SDK user deliberately couples to.
2. v0.3 must support long-running reasoning and local-agent
   workflows. "Long-running" is the load model that the latency-
   bounded claim (§7) is measured under.
3. The single optimization objective is **extreme KV-cache efficiency
   without intelligence loss**. Compatibility surface is a tool, not a
   goal.
4. There is **no deadline**, and we explicitly **reject sunk-cost
   reasoning** when evaluating prior implementation work. (This is
   what enabled the original 2026-06-01 decision C3 = b "close PR
   #30..#36 without merging" — superseded post-hoc by C3-revised = a
   when audit revealed those PRs had already been merged on
   2026-05-31, before this ADR was written; see §6.6 for the
   resulting reconciliation plan.)
5. **No PR may be requested for merge until it carries a passing Mac
   M4 integration-test report on the PR branch.** Unit tests on the
   Linux CI runner are necessary but not sufficient — they have already
   permitted a class of regressions (the Qwen3 template incompatibility,
   PR #34's coverage hole reaching `main`) that integration runs catch.

Items (1) and (3) make a server-side `session_id` strictly
admissible. Items (4) and (5) bind the implementation discipline.

### 1.4 Why this is a v0.3 problem, not a v0.4 problem

v0.3's external framing (ADR 0006) is "production-grade local agent
infrastructure". An infrastructure that holds memory bounded but
cannot deliver useful work past 30 minutes does not satisfy that
framing. Under ADR 0007's analysis we already accepted that
cross-request KV reuse is a **system-level requirement**, not an
optimization — every per-turn full-prefill is structurally wrong for
multi-turn workloads. ADR 0007's *design* was wrong; its *problem
statement* was correct, and remains v0.3-blocking.

### 1.5 Empirical evidence chain (now stable on `main`)

After PR #37, the following on-`main` paths form the audit chain that
this ADR's §1 stands on:

| ID | Path                                                                     | Role                                              |
| -- | ------------------------------------------------------------------------ | ------------------------------------------------- |
| E1 | `results/platform-tests/bench_long_session_mac_1780130542.aborted.json`  | First 4 h attempt; orphan-session bug evidence.   |
| E2 | `results/platform-tests/bench_long_session_mac_short_1780146230.json`    | First clean 30 min after orphan-session fix.      |
| E3 | `results/platform-tests/bench_long_session_mac_short2_1780196477.json`   | 30 min with in-flight metrics poller.             |
| E4 | `results/platform-tests/bench_long_session_mac_short3_1780208693.json`   | 30 min with KV gauge gated; KV peak 7.4 MiB.      |
| E5 | `results/platform-tests/bench_long_session_mac_4h_1780211323.json`       | 4 h: §2.3.a verified, §2.3.b not achieved.        |
| E6 | (local-only) `bench_long_session_mac_v2_smoke_1780236903.json`           | First v2 smoke; partial-cache crash, fixed by hotfix preserved in commit history of `AgentMemory/v030-pr7-2-path-select-and-prefill-incremental-8e7f`. |
| E7 | (local-only) `bench_long_session_mac_v2_smoke2_1780238315.json`          | 5 min smoke after hotfix; `continuation_rate = 0.0`; the empirical falsification of ADR 0007 §2.4. |

E6 and E7 are deliberately not archived to `main` (decision recorded
2026-06-01 as C2 = a). They live on the user's local Mac M4 working
copy and are referenced here only as **annotation**, not evidence the
reader can independently verify. The conclusion drawn from them
(§1.2's chat-template hazard) is independently reproducible from any
Qwen3 + `apply_chat_template(..., enable_thinking=False)` invocation
that re-renders a multi-turn history; it does not require E6/E7 to
stand.

---

## 2. Decisions

### 2.1 Wire protocol: gRPC bidirectional streaming as primary

**Decision**: the primary client ↔ runtime wire protocol is
**gRPC bidirectional streaming** (HTTP/2). The service surface is
defined by a single `.proto` file under `proto/kakeya/v1/runtime.proto`
and is the source of truth from which both Python and TypeScript SDKs
generate.

**Rationale**:

- Schema enforcement: every message has a typed contract, so
  client/server can evolve independently with explicit `proto3`
  optional / repeated semantics. The Qwen3 chat-template incident is
  in part a story of an under-specified contract (free-form
  `messages: list[dict]` re-tokenized at the server) being violated
  by a model upgrade. We do not give a future contributor that same
  rope.
- Bidirectional streaming maps naturally to the runtime semantics we
  need: the **client** streams new tokens (or a single new message) into
  a session and concurrently consumes generated tokens out, in one
  long-lived connection per session. No SSE chunked-encoding workarounds.
- Multimodal-ready: gRPC's binary message frame admits image / audio
  byte payloads without the base64-over-JSON tax; v0.4+ multimodal
  extensions land as new message fields, not new endpoints.
- Production-grade flow control (HTTP/2 stream-level) and cancellation
  (`grpc.Status.CANCELLED` end-to-end) replace the bespoke
  `asyncio.CancelledError` propagation that PR #25 had to manually
  wire through SSE.

**Consequence**: Kakeya is not, after v0.3, an "OpenAI-compatible
HTTP server." The OpenAI surface (HTTP+SSE) remains under §2.7 as a
**deprecated single-shot shim**, intentionally lacking the long-
session features that justify Kakeya's existence; users of LangChain,
Cursor, OpenAI Python SDK, etc. can keep talking to it for chat
acceleration but **cannot** reach the §2.2 session features through
it. This is by design — we will not silently degrade.

### 2.2 Session model: server-issued session id, raw token history, append-only

**Decision**: the runtime exposes the following session lifecycle as
gRPC unary or streaming RPCs. Names are illustrative; the canonical
form lives in the `.proto` file.

| RPC                       | Direction        | Purpose                                                       |
| ------------------------- | ---------------- | ------------------------------------------------------------- |
| `CreateSession`           | unary            | Client requests a session; server returns `session_id`.       |
| `AppendTokens`            | unary            | Client sends *raw token ids* to append; server confirms.      |
| `Generate`                | server-streaming | Client requests N tokens of generation bound to a session_id; server streams generated token ids back as they commit. |
| `CloseSession`            | unary            | Client releases a session; server frees its KV slot.          |
| `GetSessionInfo`          | unary            | Diagnostic: token count, KV bytes, cache invariant counters.  |

The session model has three binding contracts:

1. **The server owns the session id.** Clients cannot fabricate one;
   `CreateSession` is the only way to get one. This blocks accidental
   collision and id-spoofing in trust boundaries (multi-process / multi-
   user is v0.4 scope, but the protocol must already admit it).
2. **The token history is raw token ids.** No `messages: list[dict]`
   field, no role markers, no `apply_chat_template` call inside the
   server. The client (or its SDK) is responsible for tokenization,
   including any chat template the application chooses to use. The
   server treats the history as an opaque sequence of integers in the
   tokenizer vocabulary range.
3. **The token history grows append-only inside a session.** The
   client cannot rewrite prior history mid-session. Forking a
   conversation = `CreateSession` again with the desired starting
   history. This makes byte-exact KV cache binding (§2.4) trivially
   safe.

**Rationale**:

- The chat template is a property of the application, not of the
  runtime. v0.3's local agent applications that we ourselves author
  will, in many cases, not use a chat template at all — agentic
  reasoning often wants a custom serialization (e.g., a tool-call grammar
  or a constrained JSON schema). Building chat-template logic into
  the server forces every application to either accept the template or
  fight it, and exposes the runtime to the §1.2 class of bugs.
- Raw tokens make the server **model-agnostic**: the same gRPC
  service binds equally to a Qwen3, Gemma, or DeepSeek verifier with
  no template plumbing, because no template plumbing exists.
- Append-only is the simplest contract that makes §2.4 byte-exact
  cache binding decidable. It is also a hard architectural choice that
  the SDK must respect — see §3.

**Consequence**: tokenization moves to the client side. The Python
SDK will ship a thin tokenizer-loader helper around `transformers
AutoTokenizer`, and the TypeScript SDK will ship a thin wrapper
around `@huggingface/tokenizers`. Both are explicitly opt-in: an
application that has its own tokenization (e.g., a code agent with a
custom BPE) just constructs token-id arrays directly and never calls
the helper.

### 2.3 KV cache binding: byte-exact contract, sink+window per session (v0.3 GA scope)

> **v0.4 GA scope note**: this section describes the **v0.3 GA**
> verifier KV cache architecture (sink+window only, no reconstruction).
> The 2026-06-06 A/B benchmark measured −83 % mid-context recall under
> this design. **§11 of this ADR (v0.4 GA architecture amendment)
> supersedes the "sink+window only" clause below** by adding a dLM
> proposer-mediated K/V reconstruction layer that restores
> approximately full-attention intelligence at constant memory.
> The session model, byte-exact contract, INV-1/INV-2/INV-3
> determinism guarantees, and SessionStore central abstraction
> below remain unchanged in v0.4.

**Decision (v0.3 GA)**: on every `Generate` call, the verifier's KV cache state
is **byte-exact-bound** to the (session_id, history_token_ids) pair
the server has on file. Specifically:

- The first `Generate` after `CreateSession` (or after a sequence of
  `AppendTokens` that has not yet been generated against) prefills
  the new tokens incrementally on top of the existing cache.
- The verifier maintains one **per-session** sink+window cache slab
  (ADR 0001 + ADR 0003). Slab capacity is the same `sink+window`
  value used by v0.2.
- The cache state is uniquely keyed by `session_id`. Two sessions
  cannot share a slab. There is no implicit cross-session reuse.
- Determinism contract (refines INV-3, ADR 0007 §2.9): for the same
  (session_id, history_token_ids, generation seed) tuple, repeated
  `Generate` calls produce **bit-identical** output token sequences,
  regardless of how the history was *built up* (one `AppendTokens` of
  N tokens vs. N `AppendTokens` of 1 token vs. `CreateSession` with
  initial history vs. `CreateSession` followed by `AppendTokens`).

**Rationale**:

- Per-session slab eliminates §1's linear-prefill problem by
  construction: the second `Generate` only prefills the
  newly-appended tokens, not the full history.
- Byte-exact binding is the strongest invariant we can require, and
  is achievable because the history is raw tokens (§2.2). Without
  template re-rendering, the cache and history are always tip-to-toe
  consistent.
- The determinism clause makes "did I get the same answer because the
  KV cache reuse is correct, or because the model is robust?" a
  decidable question, not an act of faith.

**Consequence**: §6's `SessionStore` becomes the central new
abstraction. It owns slab allocation, lookup by `session_id`, and the
INV-1 / INV-2 / INV-3 enforcement (§2.8) per session.

### 2.4 No chat template at the runtime — ever

**Decision**: the runtime ships **no** code that calls
`tokenizer.apply_chat_template`, `tokenizer.encode_chat`, or any
analog. The runtime ships **no** code that knows about role markers
(`system`, `user`, `assistant`), turn boundaries, EOS-of-turn vs
EOS-of-stream, or thinking-mode flags. All of these are SDK / app
responsibilities.

**Rationale**: ADR 0007 §1.2 is the case study. Embedding template
logic in the runtime (a) couples the runtime to specific tokenizer
versions, (b) injects a re-rendering hazard on every multi-turn
operation, and (c) is the single largest source of model-specific
glue we would otherwise have to write. By moving it out, the runtime
gets one less reason to break when the verifier is swapped.

**Consequence**: every example, integration test, and SDK demo must
build its own tokenization. We provide reference helpers (§3) but no
defaults. This is a deliberate ergonomic cost paid for architectural
clarity.

### 2.5 Concurrency model

**Decision**:

- **v0.3 single-tenant**: the scheduler holds at most `max_concurrent`
  in-flight generations across all sessions; configurable but defaults
  to `1` on a 24 GB Mac M4. A session may have at most one in-flight
  `Generate` call at a time (subsequent `Generate` calls on the same
  session block until the prior one completes).
- **Multiple sessions may exist simultaneously**, each with its own
  slab; the cap on simultaneous *sessions* equals the
  `SessionStore` capacity, not `max_concurrent`. Idle sessions hold
  KV memory until the configured TTL (§2.6) expires.
- The slab pool (ADR 0003) is bound to the `SessionStore` rather than
  to `Scheduler.active_count`. A slab is allocated when a session is
  created and freed when the session is closed or evicted.
- v0.4 multi-tenant scope is explicitly deferred (§4.5).

**Rationale**: this is the simplest concurrency model that makes the
single-Mac long-session story work. Multi-tenant requires
authentication, fairness, and per-tenant quotas — all of which are
worth doing carefully on their own ADR rather than tacked onto v0.3.

### 2.6 Cache state lifecycle: TTL-based eviction, no implicit reset

**Decision**:

- A session's cache is freed if and only if (a) `CloseSession` is
  called, (b) the session has been idle for `session_idle_ttl_s`
  (default 30 min, configurable), or (c) `SessionStore` is at
  capacity and an LRU-by-last-access slab must be evicted to admit a
  new `CreateSession`.
- `Generate` **never** silently resets a session's cache. If the
  cache is in an internally-inconsistent state (an INV-1/2/3
  violation, §2.8), the runtime returns a `FAILED_PRECONDITION`
  status and **closes the session**. The client's SDK is free to
  retry by creating a new session; the runtime does not paper over
  the bug.
- Eviction is observable: when an LRU eviction would occur, the
  runtime emits a `session_evicted_total` counter increment with the
  reason label, and (if the affected client subsequently calls
  `Generate` on the evicted session) returns `NOT_FOUND`. There is
  no implicit re-creation.

**Rationale**: ADR 0007 §2.8 already established the "no graceful
degradation" principle; this section binds it to the session
lifecycle. An invariant violation is a bug, not a state — and bugs
must surface.

### 2.7 HTTP+SSE shim: deprecated, single-shot only

**Decision**: the existing FastAPI surface (`POST /v1/chat/completions`,
`GET /metrics`, etc.) is preserved in v0.3 but is **explicitly
deprecated** and feature-frozen at its v0.3.0-rc1 capability:

- HTTP requests are mapped to **single-shot** sessions: each request
  creates a fresh session, prefills the (chat-template-rendered)
  history end-to-end, generates, and closes the session. There is no
  cross-request reuse. There is no `session_id` exposed.
- Any feature added under this ADR (per-session `Generate` streaming,
  byte-exact cache binding, raw-token history) is reachable via gRPC
  only. The HTTP surface returns the v0.3.0-rc1 behavior plus a
  `Deprecation` and `Sunset` header on every response, pointing to
  this ADR.
- The HTTP surface remains 100%-tested and 100%-covered. No new
  capabilities are added to it. We do not silently fall back to it.

**Rationale**: removing the HTTP surface in v0.3 would block users who
already pinned their workflows to v0.2/v0.3.0-rc1; preserving it as
a frozen single-shot shim gives them a migration window without
diluting the new architecture's claims. v0.4 may delete the shim
entirely (open question OQ-3, §8).

### 2.8 Anomaly invariants (refined for the session model)

The three invariants from ADR 0007 §2.9 are re-stated here, scoped to
"per session":

- **INV-1 (parallel-sequence consistency)**: for every session, the
  in-memory `cached_token_sequence` length equals the K/V tensor
  sequence dimension, *for every* layer of the verifier, after every
  cache mutation. Violation is a bug. Detection raises immediately
  inside the cache mutation; the session is failed
  (`FAILED_PRECONDITION`) and the slab is freed.
- **INV-2 (position monotonicity)**: for every session, the
  `next_global_position` value is non-decreasing across the session's
  lifetime. Violation is a bug.
- **INV-3 (continuation-path determinism, refined)**: for every
  session, the bit-pattern of the generated token stream is identical
  whether the history was built incrementally (one `AppendTokens`
  per turn) or in one shot (single `AppendTokens` covering the whole
  history) before `Generate`. The pre-merge gate
  `tests/integration/test_inv3_session_determinism_gate.py` (Mac M4)
  drives both shapes through the runtime and asserts byte equality.

These are **anomaly invariants**, not steady states. Violation
metrics (`cache_invariant_violations_total{kind=...}`) must read 0
in healthy operation; non-zero values are a paging-grade signal.

### 2.9 Observability

Per-session metrics (Prometheus, exported on the existing `/metrics`
endpoint to keep the operational tooling unified):

| Metric                                               | Type      | Purpose                                                       |
| ---------------------------------------------------- | --------- | ------------------------------------------------------------- |
| `session_active`                                     | Gauge     | Currently allocated sessions.                                 |
| `session_total{outcome="closed|evicted|failed"}`     | Counter   | Lifecycle accounting.                                         |
| `session_kv_live_bytes`                              | Gauge     | Sum of live KV bytes across allocated sessions.               |
| `session_history_tokens`                             | Histogram | Distribution of per-session history length at `Generate` time.|
| `generate_prefill_tokens`                            | Histogram | Tokens prefilled per `Generate` call (v0.2: full history; v0.3: only the appended-since-last-generate delta). |
| `generate_prefill_duration_seconds`                  | Histogram | Wall time of the prefill phase per `Generate`.                |
| `cache_invariant_violations_total{kind="inv1|inv2"}` | Counter   | INV-1/INV-2 violations. Steady-state value MUST be 0.         |
| `session_evicted_total{reason="ttl|lru|close"}`      | Counter   | Eviction accounting; `lru` non-zero is a capacity-pressure signal. |

The deprecated HTTP shim's existing metrics (`scheduler_*`,
`scheduler_kv_live_bytes`, `path_selection_total`) are removed under
this ADR — `path_selection_*` were ADR 0007 metrics with no v0.3
referent. `scheduler_*` are subsumed by `session_*` since the
session is the new coordination unit.

### 2.10 Backward compatibility: explicit "no graceful degradation"

This ADR makes the same explicit rejection of "graceful degradation"
that ADR 0007 §2.8 made: there is no path under which a
runtime-detected anomaly (INV-1 / INV-2 / INV-3 violation, capacity
exhaustion, session not found) is silently masked. Every such case
returns a typed gRPC error to the client and emits a counter. The SDK
surfaces these as typed exceptions, not as warnings or as no-op
returns.

---

## 3. SDK delivery — first-class v0.3 deliverables

### 3.1 Python SDK (`kakeya-py`)

Shipped from the same monorepo, under `sdks/python/`. Public surface:

```python
from kakeya import Client, Session

client = Client("localhost:50051")
session = client.create_session()                  # CreateSession
session.append(token_ids: list[int])               # AppendTokens
for token_id in session.generate(max_tokens=128):  # Generate (server-streaming)
    ...
session.close()                                    # CloseSession
```

The SDK is a thin layer over `grpc.aio` (asyncio) with a synchronous
facade for non-async callers. It does **not** ship a tokenizer; users
who want one bring `transformers AutoTokenizer` and the SDK provides
a `kakeya.tokenization.tokens_for_chat(...)` reference helper that
takes a `messages` list and an `AutoTokenizer` instance and returns
the rendered token ids — explicitly opt-in (§2.4).

100% unit test coverage gate applies. `tests/sdk/python/` runs against
a minimal in-process gRPC server that uses a deterministic engine
test double (the same shape as the existing `DeterministicEngine` in
`tests/inference_engine/server/conftest.py`).

### 3.2 TypeScript SDK (`kakeya-ts`)

Shipped from the same monorepo, under `sdks/typescript/`. Target
runtimes: **Node.js 20+, Electron 30+, Bun 1.1+**. **Not** browser —
gRPC-Web is intentionally out of scope (open question OQ-1, §8).

Public surface mirrors §3.1:

```typescript
import { Client } from "@kakeya/runtime";

const client = new Client("localhost:50051");
const session = await client.createSession();
await session.append(tokenIds);                    // number[]
for await (const tokenId of session.generate({ maxTokens: 128 })) {
    ...
}
await session.close();
```

Transport is `@grpc/grpc-js` (HTTP/2 native client), not `grpc-web`.
Code generation uses `protoc-gen-ts_proto` so the SDK's typing is
auto-generated from the same `.proto` as the Python SDK.

### 3.3 Why both must ship in v0.3

The user-stated runtime use case (§1.3 item 1) is local agent
applications. In the Mac-M4-as-personal-runtime scenario, the most
natural application authoring environments are **Python** (data /
research / personal ML) and **TypeScript on Electron** (desktop apps,
including chat UIs and code-editor integrations). Shipping only one
makes the runtime usable for one of those audiences and unusable for
the other. v0.3's "production-grade local agent infrastructure"
framing (ADR 0006 §2.1) is not honest if the Electron-side audience is
behind an indefinite hand-wave.

This is also why v0.3 ships **two** SDKs but **zero** browser SDK —
a browser SDK requires either a separate gRPC-Web proxy or a WebSocket
shim, both of which add operational complexity that we cannot justify
without a concrete v0.3-era browser application target. v0.4+ may
revisit (§8 OQ-1).

### 3.4 The chat template lives in the SDK examples, not the SDK core

The SDK packages themselves intentionally contain no chat-template
logic. They ship reference example applications under
`sdks/python/examples/` and `sdks/typescript/examples/` that show the
canonical pattern for each model family (Qwen3, Gemma 4, DeepSeek
V4) — including, for Qwen3, an example that demonstrates the
`enable_thinking=False` template behavior that broke ADR 0007, so a
future contributor immediately sees the §1.2 hazard live in
runnable form.

---

## 4. Alternatives considered

### 4.1 ADR 0007 (automatic server-side prefix matching) — superseded

Already discussed (§1.2). The architectural failure mode (chat-
template re-rendering hazard) is not Qwen3-specific and not solvable
by a hardening pass on the prefix-matching algorithm. We do not
re-attempt this.

### 4.2 WebSocket as the primary protocol — rejected

Rejected because:

- No native schema enforcement; we would re-create a `proto`-equivalent
  hand-rolled JSON contract and lose the autogeneration story for
  SDKs.
- No native server-streaming RPC semantics — every WebSocket app
  builds its own framing, message-id correlation, and cancellation,
  which we would inevitably get subtly wrong (cf. SSE
  `should_exit_event` PR #22).
- HTTP/2's stream-level flow control and cancellation, which gRPC
  inherits, has no clean WebSocket analog.

WebSocket *might* re-enter as a v0.4 alternative transport for browser
SDK users (§8 OQ-1), wrapping the same `.proto` schema via a
hand-written framing layer; that would be additive, not replacing.

### 4.3 HTTP+SSE as the primary protocol — rejected

Rejected because the existing v0.3.0-rc1 HTTP+SSE surface is
*precisely* what §1's evidence falsifies. SSE is unidirectional
(server-to-client streaming only) and so cannot carry the bidirectional
"client appends, server streams generation" pattern of §2.2 in one
connection. Building a session model on top of HTTP+SSE requires
either two parallel connections or a hand-rolled multiplexing layer,
neither of which is simpler than just using gRPC.

The deprecated shim (§2.7) preserves HTTP+SSE for the single-shot
case only.

### 4.4 OpenAI Assistants / Responses API as the protocol model — rejected

Rejected because:

- Both bind tightly to the ChatML role taxonomy (`system`, `user`,
  `assistant`, `tool`) — a chat-template assumption at the protocol
  level, exactly the §2.4 hazard we are excluding.
- Responses API's "previous_response_id" mechanism is the right
  *idea* (server-side state continuation), but its server-side state
  includes rendered text, not raw tokens, which puts the
  re-tokenization hazard right back into the protocol.
- We are not optimizing for "drop in replacement for the OpenAI
  client surface"; the deprecated shim (§2.7) is sufficient for that
  audience.

### 4.5 Multi-tenant in v0.3 — deferred to v0.4

Rejected for v0.3 scope. Multi-tenant means at least: per-tenant
authentication; per-tenant fairness in the scheduler; per-tenant
resource quotas; per-tenant audit logging. Each is non-trivial.
v0.3 must satisfy the single-tenant long-session story before adding
load. Recorded as a v0.4 ADR slot.

### 4.6 Ship browser-targeted SDK in v0.3 — rejected

Rejected. See §3.3. Re-evaluated in §8 OQ-1 once a concrete v0.4-era
browser application target exists.

### 4.7 Chat template inside the runtime (with strict version pinning) — rejected

A degenerate version of ADR 0007 would have kept the server-side
template but pinned the tokenizer version to one known to round-trip
cleanly under multi-turn. Rejected because (a) it locks the runtime
to a single tokenizer release that we then cannot update independently
of clients, (b) it does not eliminate the hazard for users on
different tokenizer versions, and (c) it gives away the model-
agnosticism property that §2.4 buys us. We pay the §2.4 ergonomic
cost on purpose.

---

## 5. Consequences

### 5.1 What we gain

- **Latency-bounded long-sessions become possible** under the v0.3
  GA validation gate (§7), eliminating the §1's linear-prefill
  failure mode by construction.
- **Model-agnosticism**: the runtime stops caring which verifier
  family is loaded. Adding a new verifier (Gemma 4, DeepSeek V4) is
  a backend-config change, not a server-side template plumbing
  exercise.
- **A schema-defined contract** that is independently consumable from
  any gRPC-supporting language. The Python and TypeScript SDKs in
  this ADR are the *first* two; nothing in the design prevents a
  future Rust or Go SDK from being added by an external contributor.
- **Determinism is decidable**: the §2.3 byte-exact contract gives
  every PR a falsifiable property to gate against (§7 GA gate G3).

### 5.2 What we explicitly give up

- **OpenAI surface compatibility**: the deprecated shim (§2.7)
  preserves single-shot compatibility, but the long-session features
  this ADR introduces are not reachable from LangChain, Cursor,
  CrewAI, or the OpenAI Python SDK without a coupling we no longer
  pay for. ADR 0006 §7.2's "agent framework integration examples"
  remain valid for the deprecated single-shot mode and should be
  re-scoped accordingly in a follow-up.
- **Zero server-side chat-template magic**: every client must
  tokenize. The Python and TypeScript SDKs ship reference helpers, but
  application authors carry the cognitive load of "what tokens do I
  send for this turn?"
- **Browser-side reach in v0.3**: deferred to a future ADR.

### 5.3 Migration path for v0.2 / v0.3.0-rc1 users

- Existing OpenAI-style HTTP clients keep working against the §2.7
  shim with no code changes; they get the v0.3.0-rc1 single-shot
  behavior, augmented only by `Deprecation`/`Sunset` headers.
- Users who want long-session behavior migrate to the gRPC SDK. The
  SDKs ship a `from_openai_messages` reference helper that takes an
  OpenAI-style messages list + a tokenizer and produces the token-id
  sequence the §2.2 contract expects, so the migration is a one-import
  + one-helper change, not a rewrite.
- The deprecated shim is removed no earlier than v0.5 (open question
  OQ-3, §8).

---

## 6. Implementation plan

Each phase below is one or more PRs. Every PR is independently
reviewable, gated by 100% unit-test coverage on its diff, and (per §9)
gated by a Mac M4 integration-test report on the PR branch before
merge. Phases are sequential where shown; sub-PRs within a phase can
parallelize.

### 6.1 Phase A — Schema and session core (no protocol surface yet)

- **PR-A1**: Land `proto/kakeya/v1/runtime.proto` with the §2.2 RPC
  surface. No code-gen targets yet; the file is documentation in
  `.proto` form. Lints with `buf lint`. CI gains a `buf` step.
- **PR-A2**: Implement `inference_engine/session/store.py` —
  `SessionStore` with create/append/close, in-memory only, no
  scheduler binding yet. INV-1, INV-2 enforced inside. Pure Python,
  no gRPC. 100% unit coverage.
- **PR-A3** *(scope split, recorded 2026-06-01 during implementation
  of PR-A3)*: This phase originally proposed two coupled changes —
  (a) remove the ADR 0007 `path_select` / `prefill_incremental`
  machinery from both verifiers, and (b) refactor slab ownership so
  the KV cache state is constructed and owned by `SessionStore`
  rather than by the scheduler / pool. (a) requires only Linux unit
  tests; (b) crosses into MLX-runtime hardware paths and pulls in
  scheduler / pool API redesign. Atomic merge of (a) is essential
  for coherence: the moment ADR 0007's `path_select` is removed
  from the verifier, every caller has to be removed too, or the
  speculative-decoding loop breaks. (b) does not have that
  coherence pressure — it can land independently after (a).
  Therefore PR-A3 is split:
  - **PR-A3** (this PR): pure removal of ADR 0007 dead code.
    Deletes `kv_cache_proposer/path_plan.py`,
    `tests/core/test_path_plan.py`,
    `tests/core/test_determinism_gate.py` (depends on
    `path_select`); strips `path_select` /
    `prefill_incremental` / `_cached_global_positions` /
    `_prompt_matches_cached_positions` from both verifiers; reverts
    `kv_cache_proposer/speculative.py`'s `generate()` dispatch to a
    single always-prefill path. Inert defaults are retained on
    `SpeculativeRunResult.path_selection` / `tokens_skipped` /
    `prefill_duration_seconds` so the server-side observability
    surface (which §6.6 rows server/* scope to PR-D1) keeps reading
    valid values without code change. 100% Linux unit coverage.
  - **PR-A3b**: the slab-ownership refactor proper. Three deliverables:
      1. **`Session.slab` is real** — typed `Optional[KVSlab]`. When
         the store has a `slab_pool`, `create_session` acquires;
         every removal path (`close_session`, LRU eviction, TTL
         eviction, INV-1 / INV-2 violation) releases. `Session.kv_live_bytes()`
         reads through to `slab.live_kv_bytes` so the §2.9
         `session_kv_live_bytes` gauge becomes real once a pool is
         wired.
      2. **Both verifiers implement `CacheInspector`** — CPU
         `SinkWindowVerifier.k_seq_length` delegates to
         `_cache_seq_length`; MLX `MLXSinkWindowVerifier.k_seq_length`
         delegates to `_cache_buffer_size`. The session argument is
         accepted for protocol conformance but ignored in v0.3
         single-tenant scope (one verifier instance binds to one
         session at a time).
      3. **Existing code paths preserve byte-exact behavior** — the
         deprecated HTTP shim continues using `PooledVerifier` (the
         ADR 0003 wrapper) on its own slab pool; PR-D1 is what
         migrates the HTTP shim to `SessionStore`. `PooledVerifier`
         is left untouched in this PR.
    Reserved for **PR-A3c**: session-scoped binding (so the verifier
    holds per-session cache state when `max_concurrent > 1` is
    enabled in v0.4 multi-tenant scope). v0.3 single-tenant
    intentionally leaves this as a `del session` no-op in
    `k_seq_length`.
    First PR with a mandatory Mac M4 integration test report
    (cf. §9), since the new MLX `k_seq_length` method is reachable
    from MLX-runtime tests in `tests/backends/mlx/test_verifier.py`.

### 6.2 Phase B — gRPC server + Python SDK

- **PR-B1**: Generate Python stubs from the `.proto`. Add
  `inference_engine/server/grpc_app.py` implementing `CreateSession`,
  `CloseSession`, `GetSessionInfo` (no `Generate`/`AppendTokens`
  yet). Server starts on a configurable port, alongside the existing
  FastAPI server.
- **PR-B2**: Implement `AppendTokens` and the §2.3 byte-exact
  prefill-incremental path through `SessionStore` + verifier. Unit
  tests assert INV-3 on the *internal* code path (no SDK yet).
- **PR-B3**: Implement `Generate` server-streaming RPC. Mac M4
  integration test gate becomes mandatory here; this is the first
  PR whose runtime behavior cannot be validated by Linux CI alone.
- **PR-B4**: Ship `sdks/python/` with `kakeya.Client`, `kakeya.Session`,
  100% unit test coverage against an in-process gRPC server +
  deterministic engine.

### 6.3 Phase C — TypeScript SDK

- **PR-C1**: Generate TypeScript stubs from the `.proto` with
  `protoc-gen-ts_proto`. Add `sdks/typescript/` package skeleton.
- **PR-C2**: Implement the Node/Electron/Bun client surface
  (§3.2). Tests run under `vitest` on Node 20 + Bun 1.1 in CI.

### 6.4 Phase D — Deprecated HTTP+SSE shim

*(scope split, recorded 2026-06-01 during implementation of PR-D1.)*

The original PR-D1 entry conflated two coupled changes:

  (a) Remove the ADR 0007 dead code from the server-side surface
      (path_selection metrics, `_emit_path_selection_metric` helper,
      `engine_result` field on the scheduler session, etc.).
  (b) Refactor the HTTP shim's chat-completions handler onto the new
      `SessionStore` so each request becomes a single-shot session
      (prefill → generate → close) instead of being driven by the
      legacy `PooledVerifier`.

(a) is a pure subtraction: the dead code was reachable only from the
ADR 0007 path_select stack that PR-A3 already removed from the
verifier side; the server-side metrics and helpers it left behind
are unreachable at runtime in any healthy completion. (b) is a
larger refactor of feature-frozen code (per §2.7), with a
corresponding test-update tail.

The two are split, same pattern as PR-A3 / PR-A3b:

- **PR-D1** (this PR, dead-code removal): cleans up §6.6 rows for
  `app.py` / `engine.py` / `metrics.py` / `scheduler/session.py` /
  `bench_long_session.py`. The HTTP shim continues to use
  `PooledVerifier` exactly as before; nothing user-observable
  changes except the disappearance of the four ADR 0007 metrics
  from `/metrics` and the `acceptance_rate` field from the OpenAI
  response (the latter was sourced from `engine_result`, which is
  gone). 100% Linux unit coverage.

- **PR-D2** (queued, not in PR-D1's diff): the HTTP-shim refactor
  proper. Each `/v1/chat/completions` request creates a single-shot
  session under `SessionStore`, prefills, generates, and closes;
  `PooledVerifier` is retired. Adds `Deprecation` / `Sunset`
  headers per §2.7. Updates the existing integration suite to
  match. Linux-only path; §9 carve-out continues to apply. PR-D2
  is non-blocking for v0.3 GA — the deprecated shim works on
  `main` post-PR-D1 in its v0.3.0-rc1 shape, just lighter.

### 6.5 Phase E — Mac M4 integration test marker + CI workflow

- **PR-E1**: Add `tests/integration/` with the `pytest.mark.integration`
  marker, the `INV-3 session-determinism gate`, and a Mac M4
  long-session bench that drives the gRPC SDK through the same
  workload as `bench_long_session.py`.
- **PR-E2**: GitHub Actions workflow that runs the integration suite
  on a self-hosted Mac M4 runner, gated on PR labels (only PRs
  labelled `needs-mac-m4` consume the runner). The label is added
  automatically when a PR touches `inference_engine/`, `sdks/`, or
  `proto/`. Failing this job blocks merge.

### 6.6 Phase F — Reconciling pre-existing ADR 0007 implementation on `main`

PRs #30..#36 (the PR 7-1..7-6 stack implementing ADR 0007) merged to
`main` on 2026-05-31, **before** this ADR's strategic context was
finalized. Decision C3-revised = a (recorded 2026-06-01 after audit
revealed the merges) accepts that implementation as a **historical
code layer** that this ADR's Phases A-E will progressively replace.
The revised decision rests on the §1.3-item-4 sunk-cost rejection:
keeping the merged code is not a sunk-cost concession (the code does
correctly implement the rejected design), it is a recognition that
deleting it now and re-implementing it later via Phases A-E in the
same files would be churn for no architectural gain.

Pre-existing ADR 0007 surface on `main` (commits `56e8c5c` PR 7-1
through `0a31ee9` PR 7-6):

| File                                                    | Status   | Disposition                                                                                                  |
| ------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------ |
| `kv_cache_proposer/path_plan.py`                        | added    | Deleted in PR-A3 (verifier session-store rewiring) — `PathPlan` is replaced by `SessionStore` lookup.        |
| `tests/core/test_path_plan.py`                          | added    | Deleted alongside the implementation in PR-A3.                                                               |
| `tests/core/test_determinism_gate.py`                   | added    | Deleted in **PR-A3** (the test depends on `path_select`, which PR-A3 removes; cannot wait for PR-E1). Replaced by `tests/integration/test_inv3_session_determinism_gate.py` in PR-E1, which is created from scratch rather than refactored from the deleted file. |
| `kv_cache_proposer/verifier.py`                         | modified | `path_select` / `prefill_incremental` removed in PR-A3; `cached_token_sequence` retained (still useful for INV-1 inside `SessionStore`). |
| `inference_engine/backends/mlx/verifier.py`             | modified | Same as above.                                                                                               |
| `kv_cache_proposer/speculative.py`                      | modified | `path_select` dispatch removed in **PR-A3** (atomic with the verifier-side removal — leaving the dispatch but removing `verifier.path_select` would break the speculative loop). `generate()` reverts to single always-prefill. PR-B2 is now scoped to "the gRPC `AppendTokens` handler subsumes the *role* that `generate`'s dispatch used to play, this time at the protocol layer not the speculative-decoder layer." |
| `inference_engine/server/app.py`                        | modified | `_emit_path_selection_metric` and `_session_acceptance_rate` paths removed in PR-D1 (deprecated-shim refactor).|
| `inference_engine/server/engine.py`                     | modified | `EngineResult` `path_selection` / `tokens_skipped` / `prefill_duration_seconds` fields removed in PR-D1.     |
| `inference_engine/server/metrics.py`                    | modified | `path_selection_total`, `continuation_tokens_skipped_total`, `verifier_prefill_duration_seconds`, `cache_invariant_violations_total` are removed in PR-D1; replaced by §2.9's `session_*` metrics in PR-B1/B3. |
| `inference_engine/scheduler/session.py`                 | modified | `engine_result` field removed in PR-D1; replaced by gRPC-side response metadata in PR-B3.                    |
| `scripts/bench_agentic/bench_long_session.py`           | modified | Path-selection scrape removed in PR-D1; replaced by `scripts/bench_agentic/bench_session_long_run.py` in PR-E1.|

- **PR-F1**: This ADR's supplement PR (the same PR that lands the §6.6
  rewrite you are reading) is the entire Phase F paperwork. **No code
  PR closes / reverts / deletes anything** — Phases A-E own the
  removals at the points where their own additions make those removals
  natural.

Phases A and the §6.6 supplement PR ran independently; the supplement
PR is doc-only, Phase A involves runtime code. The supplement PR
preceding Phase A guarantees that anyone reading ADR 0008 against
`main` after the supplement merges sees the correct disposition of
the pre-existing surface.

---

## 7. Validation criteria — GA gates for v0.3

A PR cannot land claiming "v0.3 GA-ready" until **all** of the
following are green on `main`:

- **G1 (memory bounded, retained)**: 4 h Mac M4 long-session bench
  reports KV peak drift `<10%` across 10-min buckets with N≥3 buckets
  observed. Inherited from ADR 0006 §2.3.a.
- **G2 (latency bounded, new)**: same 4 h bench reports per-turn
  latency p50 drift `< 50%` from bucket-0 to bucket-N (i.e., a
  long-session at hour 4 must not be more than 1.5× slower than the
  same workload at minute 0). Falsifies §1's linear-prefill failure
  mode.
- **G3 (continuation determinism)**: integration test
  `tests/integration/test_inv3_session_determinism_gate.py` asserts
  bit-identical token streams between (a) one `AppendTokens` carrying
  the full history then `Generate`, vs. (b) N `AppendTokens` of one
  token each then `Generate`, for at least 3 distinct (model,
  history) inputs.
- **G4 (anomaly invariants clean)**: in the 4 h bench,
  `cache_invariant_violations_total{kind="inv1"} = 0` and
  `... {kind="inv2"} = 0`. Non-zero is a paging-grade signal, not a
  steady state.
- **G5 (SDK surface tested end-to-end)**: every `Generate` call in
  the 4 h bench is issued through the Python SDK (not raw gRPC), so
  the bench is also an SDK contract test.
- **G6 (deprecated shim still works)**: the existing 461-test
  integration suite (HTTP+SSE) passes, demonstrating no regression
  for v0.3.0-rc1 OpenAI-shaped users.

The bench script that produces G1 / G2 / G4 / G5 evidence is the
successor to `bench_long_session.py` and lives at
`scripts/bench_agentic/bench_session_long_run.py` (different name to
make the migration explicit; the old script remains for the
deprecated shim under G6).

---

## 8. Open questions

These are reserved for resolution **during** implementation, not
before. Each becomes a small dedicated ADR (or an ADR-supplement PR)
when forced by an implementation decision.

- **OQ-1**: When (if ever) does Kakeya gain a browser-targeted SDK,
  and via what transport (gRPC-Web through a separate proxy, vs. a
  WebSocket framing of the same `.proto`)? **Resolution trigger**:
  the first concrete Kakeya application that targets a browser. Not
  in v0.3.
- **OQ-2**: What is the canonical eviction signal for a session that
  exceeds `sink+window` history length? Two candidates: (a)
  truncation with explicit `History truncated` event in the gRPC
  stream and a counter increment; (b) hard `RESOURCE_EXHAUSTED` with
  forced session close. **Resolution trigger**: first Mac M4 bench
  that drives a session past `sink+window`. Default while unresolved
  is (a), since (b) loses information unrecoverably.
- **OQ-3**: When do we delete the §2.7 deprecated HTTP+SSE shim?
  **Resolution trigger**: explicit user decision after v0.4 lands;
  not before v0.5.
- **OQ-4**: Does `CreateSession` admit a `seed` field for sampling
  determinism, or is the seed a per-`Generate` argument? Both are
  defensible; per-`Generate` is more flexible but breaks the §2.3
  determinism contract if `Generate` is called twice on the same
  session with different seeds expecting the same output.
  **Default while unresolved**: seed is a per-`Generate` argument,
  and the §2.3 determinism contract is parameterized over a fixed
  seed.
- **OQ-5**: Authentication for the gRPC surface in v0.3 (single-
  tenant, but the runtime might still bind to a non-loopback
  interface on a multi-user Mac). The HTTP shim already has API-key
  bearer auth; gRPC needs a parallel mechanism. **Default while
  unresolved**: gRPC binds to `127.0.0.1` only by default and uses
  the same API-key bearer pattern in metadata when bound to
  non-loopback. Multi-tenant auth is v0.4 (§4.5).

---

## 9. Testing discipline (binding rule)

This rule is recorded here as part of the ADR (rather than only in a
CONTRIBUTING.md) so any future ADR that wants to relax it must
supersede this one explicitly:

> **Every PR landing under this ADR must include a Mac M4 integration-
> test report on the PR branch before it is requested for merge.** The
> report is a JSON file under `results/platform-tests/` produced by
> the relevant integration job (PR-E1's `tests/integration/` suite).
> The PR description must link to the JSON file and quote its
> `summary_text` block. Linux CI 100% unit-test coverage is necessary
> but not sufficient.

Rationale (drawn from PR #34 and the §1.2 incident): unit tests on a
Linux runner cannot detect (a) MLX-specific cache-state bugs, (b)
chat-template re-rendering hazards, (c) coverage holes in defensive
branches that are reachable only with real engine timing, or (d) end-
to-end determinism violations that depend on hardware floating-point
ordering. The Mac M4 integration test is the first runtime that
exercises all four.

A PR that cannot produce a Mac M4 report (because it touches only
documentation, or only Linux-runtime code) declares so explicitly in
its description with a one-line justification ("doc-only" /
"Linux-only path"); reviewers may then waive the gate by an explicit
approval comment, recorded in the PR's review history. This ADR
itself is one such PR — it ships a markdown file and adds no
runtime code.

---

## 10. References

### On-`main` evidence (after PR #37)

- `results/platform-tests/README.md` — index for the v0.3 archive.
- `results/platform-tests/bench_long_session_mac_4h_1780211323.json`
  — §2.3.a verified, §2.3.b not achieved (E5 in §1.5).
- `results/platform-tests/bench_long_session_mac_short3_1780208693.json`
  — KV peak 7.4 MiB flat (E4).
- `results/platform-tests/bench_long_session_mac_short2_1780196477.json`
  — in-flight metrics poller introduction (E3).
- `results/platform-tests/bench_long_session_mac_short_1780146230.json`
  — first clean 30 min (E2).
- `results/platform-tests/bench_long_session_mac_1780130542.aborted.json`
  — orphan-session-bug evidence (E1).

### Local-only smoke evidence (annotation, not archived per C2 = a)

- `bench_long_session_mac_v2_smoke_1780236903.json` — first v2 smoke,
  surfaced the partial-cache crash. The hotfix is preserved on `main`
  via PR #32 (commit `f3b3c64`, "PR 7-2 (ADR 0007): path_select +
  prefill_incremental + INV-2"); see §6.6 for the disposition of that
  code under this ADR.
- `bench_long_session_mac_v2_smoke2_1780238315.json` — 5 min smoke
  after hotfix, reported `continuation_rate = 0.0`. The empirical
  falsification of ADR 0007 §2.4.

### Related ADRs

- ADR 0001 — Proposer sizing, alignment, verifier decoupling.
- ADR 0002 — Verifier selection, quantization.
- ADR 0003 — Verifier ↔ slab-pool integration.
- ADR 0004 — Alignment training data preparation policy.
- ADR 0006 — Project positioning as local agent infrastructure.
- ADR 0007 — Cross-request KV cache reuse (superseded by this ADR).

### External references

- [`buf` schema linter](https://buf.build/docs/lint/overview) — used
  in PR-A1's CI step.
- [`@grpc/grpc-js`](https://github.com/grpc/grpc-node/tree/master/packages/grpc-js)
  — Node.js native gRPC client.
- [`protoc-gen-ts_proto`](https://github.com/stephenh/ts-proto) —
  TypeScript stub generator.

---

## 11. v0.4 GA architecture amendment (2026-06-08)

### 11.1 Scope of this amendment

§1–§10 above documented the v0.3 GA design as shipped on `main`. This
§11 is the **v0.4 GA architecture amendment**. It supersedes the
"sink+window only" verifier KV strategy of §2.3 with a constant-memory
dLM-mediated K/V reconstruction architecture, and it explicitly
**rejects the parallel-track ADR 0010 and ADR 0011 drafts** that
emerged on branches but never landed on `main`. Everything else in
§1–§10 (session model, gRPC protocol, byte-exact contract, INV-1 /
INV-2 / INV-3 determinism, SDK delivery, observability, deprecation
of HTTP+SSE shim) carries over to v0.4 unchanged.

### 11.2 Reasoning chain leading to this amendment

The v0.4 architecture has been arrived at by working backward from
two empirical observations on top of v0.3:

1. **v0.3 sink+window=64 destroys mid-context recall**. The 2026-06-06
   A/B benchmark (`results/platform-tests/sink_window_quality_ab_1780714635.json`)
   measured 1/6 (16.7 %) mid-context fact retrieval under the v0.3
   design, vs 6/6 (100 %) under the full-attention oracle. The
   intelligence regression is structural: K/V tensors at evicted
   positions are gone, and no inference-time mechanism within v0.3
   can reconstruct them.
2. **R1c–R1e (PR #65 / #67 / #68) falsified the cross-attention
   bridge subspace** as a means to compensate. The decisive datum is
   R1e-γ at localization 0.82 with cross_attn_recall 0.12: even when
   the bridge attends to the needle 82 % of the time and a full
   pre-norm transformer block sits on the write path, frozen verifier
   layers downstream of the injection point cannot translate the
   located content into the right argmax. ADR 0011 §3 (the specific
   mechanism we tested) is settled-falsified for the frozen-base
   regime; the broader cross-attention training paradigm has untested
   subspaces (see ADR 0011 §11.1 on the R1e branch) but none are
   needed once the §11 design here is adopted.

These two together pushed the design space toward "do not bound the
verifier KV at all". From there, the path that satisfies all five of
the project's hard constraints (§11.4 below) is the one in §11.5.

### 11.3 The dLM proposer's no-cache property is the load-bearing fact

The v0.4 architecture is enabled by a property of the dLM proposer
that v0.3 documented but did not fully exploit:

> **dLM proposer has no KV cache.** Diffusion language models (MDLM,
> ELF, and the variants ADR 0001 / ADR 0002 select) operate by
> parallel denoising over the entire sequence in each iteration. K
> and V tensors at every position are computed transiently inside the
> forward and discarded afterward. There is no persistent cache that
> grows with context length — the proposer's sustained memory
> footprint is its weights and (small, fixed) activation buffers.

This means the dLM proposer can serve as a **constant-memory K/V
reconstruction source** for the verifier's evicted positions: when
the proposer runs its standard forward (which it has to do anyway for
drafting), the K/V tensors it computes at every position — including
positions the verifier has evicted — are available transiently for
the verifier to consume in the same step. After the step they are
freed; nothing accumulates.

This property is the difference between v0.3 (where the proposer was
a black-box "drafts come out, no introspection") and v0.4 (where the
proposer is also the verifier's transient memory).

### 11.4 The five hard constraints v0.4 must satisfy

The user-stated v0.4 strategic frame has five non-negotiable
constraints:

1. **Constant memory in context length** — the sustained memory
   footprint must not grow with prompt size. Rules out KakeyaLattice-
   alone (linear memory at 2.4–2.8× thinner) and full-attention KV
   storage (linear at full bytes).
2. **Zero intelligence regression** — the model must produce the same
   argmax distribution as a full-attention oracle (subject to
   well-bounded reconstruction noise). Rules out v0.3 sink+window-
   only.
3. **Speculative decoding correctness contract preserved** — output
   distribution under speculative decoding must equal the
   distribution of the verifier-as-implemented running standalone
   (Leviathan et al. 2023, Theorem 1). Rules out "trust the proposer"
   shortcuts that break rejection sampling.
4. **No cross-attention bridge** — R1c–R1e empirically settled this
   path is not viable in the cheap-to-explore subspace, and the
   broader paradigm requires training scales (10⁵–10⁶ tokens) and
   joint-training assumptions that violate the project's
   "no-deadline + cheap-validation" research discipline. Rules out
   ADR 0011 §3 family.
5. **Fits Mac mini 24 GB targeting Gemma 4-9B-class verifier** — the
   sustained memory must leave headroom for verifier weights (~5 GB),
   proposer weights (~2 GB), and standard activation/working memory.

§11.5 below is the narrowest design that satisfies all five.

### 11.5 v0.4 GA architecture: dLM K/V Restoration

**Decision**: the v0.4 verifier maintains a minimal sink+window cache
(same machinery as v0.3 §2.3) AND, at every generation step, accepts
transient K/V tensors at evicted positions reconstructed from the
dLM proposer's parallel forward. The verifier performs standard
softmax attention over (sink+window K/V from cache) ⊕ (reconstructed
K/V from proposer transient). After the step, reconstructed K/V are
discarded.

In one diagram:

```
  Per generation step:
  
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  dLM proposer (full attention, NO cache)                        │
  │                                                                 │
  │   Inputs: prompt + drafts_so_far                                │
  │   Forward: 1 parallel denoising pass over all positions         │
  │   Outputs (transient — freed after this step):                  │
  │     • k draft tokens for verification (standard SD)             │
  │     • K, V tensors at every position (the new byproduct)        │
  │                                                                 │
  └────┬─────────────────────────────────────────────┬──────────────┘
       │ drafts                                      │ proposer K/V at all positions
       │                                             │
       ▼                                             ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  Cross-model K/V projection f_θ                                 │
  │   (per-layer, per-head adapter; identity if proposer = verifier │
  │    same checkpoint; learned if cross-model)                     │
  │                                                                 │
  │  Inputs:  proposer's K/V at evicted positions                   │
  │  Outputs: verifier-shape K/V at the same positions              │
  │                                                                 │
  └────────────────────────┬────────────────────────────────────────┘
                           │ reconstructed K/V (transient)
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  AR verifier (per-session sink+window cache + reconstruction)   │
  │                                                                 │
  │  Inputs: drafts to verify                                       │
  │  Cache content for this attention:                              │
  │     • sink + window K/V from session cache (sustained, ~3 MB)   │
  │     • reconstructed K/V at evicted positions (transient)        │
  │  Forward: standard softmax attention over the union             │
  │  Output: per-draft logits, accept/reject via rejection sampling │
  │                                                                 │
  │  After step:                                                    │
  │     • Append accepted drafts' K/V into sink+window cache        │
  │       (with FIFO eviction maintaining (sink, window) shape)     │
  │     • Discard reconstructed K/V (not stored)                    │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

The five properties this design realises:

1. **Constant memory in context length**. Sustained verifier KV
   footprint is `sink + window` (≈ 3 MB for typical settings),
   independent of prompt size. Sustained proposer footprint is its
   weights only (no cache). Transient peak compute memory grows with
   context but is freed each step.
2. **Intelligence approximates full attention**. The verifier's
   attention input at every step is (sink + window + reconstructed),
   which is a superset of the full-attention input modulo
   reconstruction noise. The argmax distribution is `p_v_restored`
   ≈ `p_v_full` to the limit of `f_θ`'s reconstruction fidelity.
3. **Speculative decoding correctness contract preserved**. Rejection
   sampling guarantees the final output token distribution equals the
   distribution of the verifier-as-implemented (i.e., `p_v_restored`).
   This is mathematically the same contract as v0.3 — what changes is
   that `p_v_restored` is much closer to `p_v_full` than v0.3's
   `p_v_bounded` was.
4. **No cross-attention bridge**. The reconstruction injects K/V
   directly into the verifier's standard cache. The verifier's
   frozen attention machinery consumes K/V naturally — no "translate
   cross-attn delta into argmax-flipping signal" step. The R1c–R1e
   failure mode (frozen layers don't decode bridge delta) does not
   apply because the integration point is pre-attention, not
   mid-stack residual.
5. **Fits Mac mini 24 GB**. Sustained: weights (~7 GB) + sink+window
   cache (~3 MB) ≈ 7 GB. Transient peak (Gemma 4-2B proposer
   forward + Gemma 4-9B verifier forward + reconstruction projection)
   stays under ~10 GB even for 100 k-token contexts. Headroom of
   ~14 GB.

### 11.6 Cross-model K/V projection f_θ

The architecture is parameterised by a learned projection
`f_θ: proposer_K/V → verifier_K/V` (one for K, one for V; per-layer;
per-head). Two cases:

1. **Same-model setup (Phase 1 toy / debugging)**: proposer and
   verifier share weights. `f_θ = identity`. K and V tensors at
   any position computed by either model are bit-identical. No
   training needed; this case is for end-to-end pipeline validation
   (does the routing work, does memory stay bounded, does the
   correctness contract hold under round-trip).
2. **Cross-model setup (production)**: proposer is a small dLM (e.g.,
   Gemma 4-2B-class), verifier is a larger AR model (Gemma 4-9B-
   class). Hidden dimensions, head counts, and layer counts differ.
   `f_θ` is a learned per-layer projection that maps a proposer
   `K[L', p, ...]` to a verifier `K[L, p, ...]` (similarly for V),
   trained to minimise `||verifier(reconstructed K/V at evicted) -
   verifier(ground-truth K/V at evicted)||` on logits or a
   downstream-task surrogate. Layer alignment `L' → L` is itself a
   design parameter (uniform, attention-pooled, or learned).

The cross-model projection is structurally easier than the R1c–R1e
cross-attention bridge for three reasons that the R1e-γ (loc=0.82,
recall=0.12) datum makes precise:

| | R1c–R1e bridge | v0.4 K/V projection (this ADR) |
|---|---|---|
| Injection point | Verifier mid-stack hidden state | Verifier KV cache (pre-attention) |
| What downstream frozen verifier does | Must "decode" cross-attn delta | Standard softmax attention |
| Source representation | Proposer post-final-norm hidden (answer space) | Proposer mid-layer K/V (same role as verifier's K/V) |
| Cross-abstraction translation needed | Yes (layer-K hidden → layer-N residual) | No (K/V → K/V, same role) |
| Training objective | Implicit (let bridge learn translation end-to-end) | Explicit (`||K_recon - K_truth||²` per layer per head) |

### 11.7 Implementation phases (v0.4 GA)

Each phase has Linux CI gates plus Mac M4 / vast.ai empirical gates
per ADR 0008 §9. Phase K2 was rescoped on 2026-06-08 to absorb the
former K4 KakeyaLattice composition — see §11.11 for the integration
architecture and motivation. The original K4-as-byte-codec framing
underestimated the compositional value of KL: KL is not a "drop-in
post-hoc compressor" but a load-bearing component that, by enlarging
the verifier's resident local cache, materially reduces the work the
dLM K/V Restoration path must do per decode step. Pulling KL forward
into K2 means the cross-model projection `f_θ` is trained against
the same memory budget the deployed system will run under, which
removes a class of late-stage fitting risk that K3 production
training would otherwise inherit.

| Phase | Scope | Linux CI gate | Empirical gate |
|---|---|---|---|
| **K1** | Same-model toy: proposer and verifier share Gemma 3-1B weights. Implement K/V routing infrastructure (reconstruction hook, cache concatenation, transient memory management). Validate on synthetic NIAH that recall ≈ oracle when projection is identity. | round-trip K/V bit-identical when `f_θ = id`; no leaks across forward steps; INV-3 byte-exact under reconstruction | Mac M4: NIAH small-vocab recall ≥ 95 % at sink+window=4+64 + reconstruction (vs 16 % v0.3) |
| **K2.A** (was K4) | KakeyaLattice integration into the verifier's local sink+window cache. `KVCompressor` interface with `IdentityCompressor` (no-op) + `KakeyaLatticeCompressor` (the in-house codec). Local cache stores compressed K/V; decode-time decompresses lazily into the K/V tensor that feeds attention. **Same model** as K1 (`f_θ = id`); isolates the KL composition risk from the cross-model projection risk. | round-trip identity: `decompress(compress(K, V)) ≈ (K, V)` within published KL fidelity; throughput ≥ K1 oracle baseline | Mac M4: NIAH recall = K1 baseline ± 1pp at 100k context with KL on; throughput improvement ≥ 1.3× over K1 v04 at the same memory budget |
| **K2.B** (was K2) | Cross-model toy: proposer = Gemma 3-1B, verifier = Gemma 3-4B. Train `f_θ` per-layer linear projection with L2 reconstruction loss on long-context corpus. **f_θ trained against KL-quantized cache** so the projection inherits KL's quantization bias and is robust to it. Measure `\|p_v_restored - p_v_full\|`. | reconstruction loss reaches plateau on calibration set; coverage metric for layer alignment; KL-on residual ≤ KL-off residual + 5% | vast H200: NIAH recall ≥ 90 % cross-model at sink+window=4+64 with KL on |
| **K3** | Production scale: proposer = Gemma 4-2B-MDLM, verifier = Gemma 4-9B-class. Full alignment training of `f_θ` on long-context corpus (RULER, NarrativeQA). KL on by default. | training pipeline reproducible; checkpoint integrity manifest | Mac M4: 4 h `bench_session_long_run.py` at 100 k-token context, kv_live_bytes flat, latency p95 stable, INV-3 holds |
| **K4** | _Reserved._ Originally KakeyaLattice composition; absorbed into K2.A on 2026-06-08. Slot kept open for future composition experiments (e.g. tile-wise mixed precision in the proposer's transient K/V — see Q11.4 below). |  |  |
| **K5** | Default flip + docs | feature flag `kv_strategy=dlm_restore` (with `kv_compressor=kakeya_lattice`) becomes default for v0.4; sink+window-only retained as opt-in for memory-constrained edge cases | quickstart updated; v0.3 → v0.4 migration documented |

K1 + K2.A are both doable on Mac M4 alone (no GPU training; KL is
applied at inference time, not learned). K2.B requires vast or
similar single-GPU training; budget ~$5–10. K3 is the production
training and requires real long-context corpus — ~$200–1000 GPU
budget depending on corpus size and number of training tokens.

### 11.8 v0.4 GA validation criteria

A v0.4 release shipping §11 must demonstrate, on reproducible
artifacts in `results/platform-tests/` or `results/research/`:

1. **Quality parity vs full-attention oracle**: NIAH mid-context
   recall ≥ 95 % at 100 k-token context (vs v0.3's 16.7 %).
2. **Constant sustained memory**: `GetSessionInfo.kv_live_bytes`
   does not grow beyond the sink+window slab capacity over a 4 h
   `bench_session_long_run.py` run, regardless of cumulative
   context. Predicted slope = 0.
3. **Determinism preserved**: ADR 0008 §6.5 INV-3 gate passes
   bit-exact between continuation and reset paths under the
   reconstruction layer.
4. **Speculative decoding contract proof**: empirical verification
   that final output distribution under §11 matches running the
   `verifier_with_reconstruction` standalone on the same prompt
   (not the full-attention oracle — that would be a stronger but
   different claim). Sample-based KS test on logit distributions
   at a held-out set of 1k positions.
5. **Cross-platform**: MLX (Apple Silicon) and PyTorch (CUDA)
   backends produce matching argmax across a 50-prompt eval set
   (subject to projection numerics; tolerance < 1 % token disagreement).
6. **Long-session stability**: 4 h benchmark with no errors,
   p95 latency stable, memory flat.
7. **Throughput floor (added 2026-06-08)**: per-config decode
   throughput, in tokens / second, recorded by the K1.E NIAH
   harness (`mean_throughput_tokens_per_sec`, K1.I schema v4),
   must satisfy at the run's context length:

   * v0.4 (with KakeyaLattice on per §11.11) ≥ 1.3× the v0.4
     baseline measured at the same context length without KL.
   * v0.4 (KL on) decode throughput ≥ 0.6× the full-attention
     oracle's throughput at the same context length. The 0.6
     bound reflects the inherent extra work per step (proposer
     forward + reconstruction + verifier forward) and is a
     consequence of the architecture, not a regression. K2.A
     closes much of the gap by enlarging the resident local
     cache (compressed) so reconstruction fires less often.

   Throughput is measured per-sample, then aggregated as
   mean / median / min / max. The metric is reported alongside
   recall, peak memory, and effective attention window so all
   four release-gating axes are visible in the same JSON
   evidence.

### 11.9 Open questions for v0.4 GA

- **Q11.1**: Optimal layer alignment for cross-model `f_θ`. Linear
  in proposer-layer / verifier-layer ratio? Attention-pooled across
  proposer layers? Per-verifier-layer learned routing? Initial
  recommendation: linear ratio with learned residual; revisit in
  Phase K2 ablation.
- **Q11.2**: How aggressive can sink+window be once reconstruction
  is on? `sink=0 + window=0` (no cache, all reconstruction) vs
  `sink=4 + window=64` (anchor + locality)? Probably want `sink>0`
  to anchor instruction, but `window` could shrink to ~16. Validate
  in K1.
- **Q11.3**: When the proposer's draft is rejected, the
  reconstruction step's K/V are still computed (proposer ran its
  forward). Is there a way to amortize? Probably no within one
  step, but across steps with KV-cache-style proposer (which
  contradicts the dLM no-cache property) — defer.
- **Q11.4** (partially resolved 2026-06-08; see §11.11): The main
  KakeyaLattice composition target — compressing the verifier's
  resident sink+window cache — has been pulled forward from K4
  into K2.A and is now a load-bearing component, not an optional
  post-hoc codec. The remaining unresolved fragment is whether to
  ALSO compress the proposer's transient K/V before the
  reconstruction projection `f_θ` consumes it. This is now a
  separate question: the proposer K/V lives one forward, so
  compression there only pays off if the (compress, decompress)
  round-trip is cheaper than transmitting the uncompressed tensor
  through `f_θ`. Empirically unproven; deferred to a future
  composition experiment, with the K4 phase slot reserved for it.
- **Q11.5**: Multi-tenant scheduling. With the proposer running a
  full forward per generation step, throughput is bottlenecked by
  proposer cost, not memory. Compute-throughput vs memory-density
  trade-off shifts vs v0.3 — schedule design needs revisiting in
  v0.4.

### 11.10 Why ADR 0010 and ADR 0011 drafts were specifically wrong

To document the lesson for future contributors:

- **ADR 0010 (NF4 KV quant)** was wrong because it proposed a
  generic literature baseline that was already inferior to the
  project's existing in-house KV codec
  (`github.com/FluffyAIcode/LLM-KV--Cache-compress`, KakeyaLattice
  v1.4 D4 / v1.5 E8, beats Google TurboQuant 12/12 on H200) AND
  because its target compression shape (linear-but-thinner) does
  not satisfy the constant-memory requirement of v0.4. The agent
  drafting ADR 0010 was unaware of the in-house codec; the lesson
  is that future ADRs touching KV memory must explicitly survey
  the project owner's existing repos before recommending external
  baselines.
- **ADR 0011 (cross-attention bridge)** was wrong in its problem
  framing, not its mechanism. It tried to solve "how do we recover
  the intelligence loss from sink+window?" — but sink+window=64
  itself was a v0.3 budget compromise (memory budget on Mac mini
  24 GB without KV compression), not a design decision that needed
  defending. The real fix was always to remove the structural
  bound on the verifier's effective attention range, not to
  build a residual-stream rescue mechanism for an artificially-
  bounded view. The agent drafting ADR 0011 (and the R1c–R1e
  research that followed) treated sink+window as a fixed constraint
  rather than as a contingent v0.3 implementation choice. The
  lesson is that future ADRs proposing "rescue mechanisms" must
  first argue why the underlying constraint cannot be removed.

Both lessons are deposited in this ADR (rather than as separate
"rejected ADR" tombstones) so future readers see them in the
context of the architecture that replaces them.

### 11.11 KakeyaLattice integration into the verifier's K/V path (K2 amendment, 2026-06-08)

This section was added 2026-06-08 in response to the user directive
"在 k2 阶段，把 kakeyalattice 的 kv cache 压缩集成进 verifier 的 kv
过程里". It pulls the in-house **KakeyaLattice** KV codec
(`github.com/FluffyAIcode/LLM-KV--Cache-compress`, v1.4 D4 / v1.5
E8 lattice, 2.4–2.8× compression at < 1 % perplexity loss on H200,
beats Google TurboQuant 12/12) forward from the original K4 slot
into K2.A. The motivation is not "smaller cache" — at sink=4,
window=64 the local cache is ~17 KB at bf16 for Gemma 3-1B,
already negligible — but a **composition effect** with the dLM
K/V Restoration architecture of §11.5 that materially changes the
quality / cost trade-off.

#### 11.11.1 Why integrate KakeyaLattice and dLM K/V Restoration as one design point

§11.5 chose a small `sink+window` deliberately so total KV memory
is bounded constant in T. But "small" was set at the v0.3 budget
without compression. With KL on, the same memory budget supports
a 2.4–2.8× larger resident window — at fixed memory, sink+window
can grow from 4+64 = 68 to ~190 effective positions; at fixed
window, memory drops 2.4–2.8×. The two design points have
different empirical signatures:

* **Memory-fixed, window-expanded** (the K2.A primary path):
  identical sustained memory as K1 v04, but the resident window
  covers ~3× more positions. Eviction (and therefore dLM K/V
  Restoration) only fires past position ~190 instead of ~68. For
  short-to-medium context (T ≤ 2k tokens) the verifier runs
  almost entirely from its compressed local cache and the
  proposer reconstruction path is a no-op — collapsing to a
  faster, oracle-equivalent decode. For long context (T = 100k)
  the eviction rate is roughly (T - 190) / T ≈ 99.8 %, marginally
  worse than (T - 68) / T ≈ 99.9 % — so the long-tail behaviour
  is unchanged, but the short-context end is dramatically
  faster.
* **Window-fixed, memory-reduced** (the K2.A optional path):
  identical resident window as K1 v04, but sustained memory is
  ~2.5× lower. Useful for multi-tenant batching where verifier
  memory is scaled out across N concurrent sessions and per-session
  footprint matters more than per-session quality.

The primary path is the first; throughput is the goal, not memory
shrinkage. K1 v04 already achieved constant memory at the K1
budget — that constraint is satisfied. The K2.A win is throughput.

#### 11.11.2 Where KL plugs in (and where it does NOT)

The v0.4 verifier holds K/V tensors in three structurally
different forms during one decode step:

| K/V kind | Lifetime | Compresses? | Why |
|---|---|---|---|
| **resident local cache** (sink + window K/V) | persists across decode steps; sustained working set | **YES — K2.A target** | persistent working set is the only K/V whose memory cost compounds; KL's compression amortises across steps |
| **dLM-reconstructed K/V** at evicted positions | one decode step | NO | reconstructed K/V is computed fresh from the proposer's transient state every step and consumed by the verifier's attention in the same forward; compressing it does not save sustained memory and adds round-trip cost per step |
| **proposer's transient K/V** during its forward | one decode step | possibly (Q11.4 fragment) | Same lifetime argument as above; only worth it if the (compress, transmit-to-`f_θ`, decompress) path is cheaper than passing the raw tensor. Empirically unproven; reserved for K4. |

**KL applies to the resident local cache only.** This is the
load-bearing claim of K2.A. Compressing the other two K/V kinds
is either a no-op (no sustained memory saved) or a net cost
(per-step round-trip), so the K2.A integration deliberately does
not touch them.

#### 11.11.3 Compositional model: how dLM K/V Restoration and KL stack

The verifier's attention at decode position q ∈ [0, T) must see
some approximation of K/V at every preceding position k ∈ [0, q].
With both K2.A components on, the K/V at position k is sourced
from one of two paths:

```
                     k ∈ resident local positions  ┐
                                                    │ → KL.decompress() → K/V  ┐
                       (sink ∪ window slots)        ┘                          │
                                                                               ├→ verifier attention
                     k ∈ evicted positions          ┐                          │
                                                    │ → dLM forward → f_θ      ┘
                       (T \ local_positions)        ┘
```

The two paths produce K/V tensors of identical shape; they
differ only in source. The verifier's attention does not know
which is which, which preserves the speculative-decoding contract
of §11.5 (the verifier's output distribution is a function only
of (current logits | K/V at all preceding positions), independent
of how that K/V was produced).

Critically, the **set membership of the local positions is the
design lever**. K2.A enlarges that set under fixed memory, which
shifts more of `[0, q]` into the KL path and less into the
reconstruction path. Per decode step, that means:

* **Throughput** ↑ (reconstruction path costs more than KL
  decompression — the proposer's forward is the dominant cost,
  and avoiding it on more positions means fewer proposer
  forwards per step in batched implementations, eventually
  converging to "one proposer forward per N decode steps" once
  the local cache covers most of T).
* **Quality** ↑ slightly (KL has bounded perplexity loss; `f_θ`
  has unbounded learned-projection loss when the cross-model
  projection is imperfect — see K2.B). At the K2.A same-model
  identity-projection point, both paths are exact, so quality
  parity is automatic.
* **Memory** = (the budget is fixed; KL's compression is spent
  on enlarging the local cache, not on reducing memory).

#### 11.11.4 Implementation contract: `KVCompressor` interface

K2.A introduces a narrow, codec-agnostic interface so the
KakeyaLattice dependency lives behind one boundary:

```python
class KVCompressor(Protocol):
    """One verifier-cache compressor instance per (layer, head_kv).

    Stateful: holds the compressed representation; decode step
    sequence is compress → ... → decompress on the same instance.
    """
    def compress(self, k: Tensor, v: Tensor, positions: Tensor) -> None:
        """Add (k, v) at given positions to the compressed store.
        Idempotent on repeated positions (overwrites)."""

    def decompress(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Return (k, v) at the given resident positions. Shape
        matches a slice of the original K/V tensor."""

    def evict(self, positions: Tensor) -> None:
        """Drop the given positions from the compressed store."""

    def memory_bytes(self) -> int:
        """Sustained byte size of the compressed store. Used by
        K1.G memory accounting to verify constant memory."""
```

Two implementations ship in K2.A:

* **`IdentityCompressor`** — stores `(k, v)` uncompressed, identity
  on `compress`/`decompress`. Default for K1 and as the v0.4 GA
  fallback when KL is unavailable.
* **`KakeyaLatticeCompressor`** — wraps the in-house codec.
  `compress` runs D4/E8 lattice quantisation; `decompress` runs
  the inverse. The codec's published fidelity guarantees become
  the K2.A round-trip-identity gate.

`DLMRestoredVerifier` (K1.D) is parameterised on a `KVCompressor`
instance (default `IdentityCompressor`); the K1 NIAH evidence
already validates the architecture on the identity compressor, so
K2.A is purely an interface-and-codec swap, not a re-architecting.

#### 11.11.5 K2.A acceptance gates

K2.A merge requires evidence of all three:

1. **Round-trip identity**: per-tensor numerical: `‖decompress(compress(K, V)) - (K, V)‖ / ‖(K, V)‖` within KakeyaLattice's published fidelity envelope (per layer per head). Linux unit-test gate; deterministic on synthetic K/V.
2. **No quality regression**: K1.E NIAH harness, same Gemma 3-1B
   identity-projection setup, KL on vs KL off:
   * recall(KL on) ≥ recall(KL off) − 1pp at every context rung
     in §11.12 ladder (1.4k, 5.6k, 22k, 64k, 100k).
   * `effective_attention_fraction` from K1.H schema: identical
     between KL on and KL off (KL is structurally invisible to
     the attention-mask path).
3. **Throughput improvement**: K1.I throughput metric (schema v4):
   * `mean_throughput_tokens_per_sec(KL on) / mean_throughput_tokens_per_sec(KL off) ≥ 1.3` at the 22k+ rungs of the §11.12 ladder.
   * The 1.3× floor is conservative; theoretical upper bound is
     the inverse of the KL-on eviction rate, which approaches the
     full-attention oracle's throughput as the local cache grows
     to cover most of T.

#### 11.11.6 K2.B (was K2): cross-model `f_θ` trained against KL-on cache

The K2.B phase trains the cross-model linear projection `f_θ`
(Gemma 3-1B proposer → Gemma 3-4B verifier) against a verifier
running with KL on, not against an idealised uncompressed
verifier. This is the "fit the projection to the deployed
runtime" discipline: if K3 production training is also done
against KL-on, the K3-trained `f_θ` inherits robustness to
KL's quantisation bias for free.

The empirical risk this addresses: if K2.B trained `f_θ` on a
KL-off verifier and K3 deployed it against a KL-on verifier,
the small KL quantisation error compounds with the larger `f_θ`
projection error in unpredictable ways. Training the projection
in the same compression regime it will be served in is the
standard discipline for compressed-deployment models; we make
it explicit here because the project's earlier ADR 0010 / 0011
sequence showed how easy it is to mis-frame the runtime
constraints during training-phase design.

#### 11.11.7 What K2.A is NOT

To prevent scope creep:

* **Not a new compression algorithm.** KakeyaLattice already exists
  at `github.com/FluffyAIcode/LLM-KV--Cache-compress` as v1.4 D4 /
  v1.5 E8. K2.A is integration plumbing, not algorithmic R&D.
* **Not a memory-budget change.** Constant-memory is already
  satisfied by K1; K2.A spends KL's compression headroom on
  throughput, not on cutting memory further.
* **Not a quality-recovery mechanism.** K1.E already showed v0.4
  recall = oracle at the same-model identity-projection point;
  K2.A's quality gate is "no regression vs K1", not "improve
  recall". Quality improvement comes in K2.B from the cross-model
  projection.
* **Not coupled to the cross-model projection.** K2.A ships KL on
  the same-model setup first (`f_θ = id`); K2.B introduces the
  cross-model projection. This staging matches the K1 → K2 risk
  isolation discipline of §11.7.

#### 11.11.8 Why this is the right phase boundary

Considered alternatives, rejected:

* **Land KL in v0.4 GA but as an optional flag (was K4).** Rejected:
  if KL is optional, the throughput criterion in §11.8 has to be
  written twice (KL-on path and KL-off path), each with different
  acceptance numbers, and v0.4 release evidence has to cover both
  matrices. Throughput is the v0.4 user-visible win; making it
  flag-gated dilutes the release.
* **Land KL after K3 production training (was implicit).** Rejected:
  K3-trained `f_θ` against KL-off, then deployed with KL on, is the
  exact "training-deployment compression mismatch" failure mode
  documented in §11.11.6. Better to take the integration risk
  during K2 (small models, fast iteration) than during K3
  (production models, expensive iteration).
* **Land KL during K1 (i.e. before this amendment was even
  written).** Rejected: K1's job was validating the dLM K/V
  Restoration architecture itself; adding KL in K1 would have
  conflated two independent risks (architecture validity ×
  codec composition) into one phase.

K2.A is therefore the narrowest phase in which KL composition is
both necessary (throughput is unmeasurable without it under the
fixed-memory budget) and sufficient (the architecture has just
been proven in K1, so KL can be added without simultaneously
defending the architecture).

#### 11.11.9 Mac M4 portability for K2.A (added 2026-06-08, post-K1.H)

User directive 2026-06-08: "k2 的 Mac mini 版本的也要支持。所以
要把 kakeyalattice 适配到 Mac mini 上." K2.A must run on Apple
Silicon (Mac M4 24 GB) on PyTorch's MPS backend, not just on
NVIDIA H200 / H100 (which are KakeyaLattice's published
benchmark hardware). This subsection documents how that's
achievable with no changes to the codec library and what
empirical evidence is required.

**Why portability is the default state, not a separate engineering
project.**  KakeyaLattice's hot-path source
(`kakeyalattice/python/kakeyalattice/lattice_codebooks.py`,
inspected 2026-06-08) is **pure PyTorch**:

* Sylvester–Hadamard rotation: `torch.cat`, `torch.tensor`
  initialisation, matmul.
* Per-vector qmax: `.abs().max()`, `.clamp(min=eps)`, division.
* Conway–Sloane closest-lattice-point (D4 / E8): `torch.round`,
  `argmax`, `gather`, `scatter_`, `where`, `sum`.
* Dtype handling: `to(torch.float16)` for storage; the codec
  internally up-casts to fp32 for the lattice math.

None of these ops require CUDA-specific kernels. The "GPU" in the
class name `V14KakeyaZamirLatticeGPU` is a project naming
convention ("strict GPU — no numpy, no CPU detour" per the
module docstring), not a platform restriction. The constructor
accepts `device: str` and forwards it verbatim to PyTorch tensor
creation calls.

**Implementation plumbing in this repo (PR-K2.A.0).**  The K2.A
integration scaffold (`inference_engine/v04/kv_compressor.py`)
forwards the verifier's active device through the
`KakeyaLatticeCompressor` constructor unchanged:

```python
KakeyaLatticeCompressor(
    head_dim=256,
    device=torch.device("mps"),   # the K2.A Mac M4 dispatch
    lattice="D4",
    q_range=38,
)
```

The adapter coerces the device to a string (`str(device) == "mps"`)
because KakeyaLattice's published API takes a `str`. This is the
**load-bearing line** for Mac M4 portability — without it, the
codec would silently materialise tensors on CPU even though the
verifier is on MPS, the device-mismatch overhead per decode step
would dwarf the K2.A throughput win. A unit test
(`test_mps_device_forwarded_as_string` in
`tests/inference_engine/v04/test_kv_compressor.py`) pins this
behaviour against future regression.

**`kakeyalattice` as an optional dependency.** The K2.A integration
treats `kakeyalattice` as optional (`pip install kakeyalattice`
not in the runtime's `install_requires` until K2.A integration
PR ships). When the package is missing, `KakeyaLatticeCompressor`
construction raises `KakeyaLatticeUnavailable` with an actionable
install hint, and `make_default_compressor(prefer_kakeya=True)`
catches that error and falls back to `IdentityCompressor` with a
warning. The runtime continues to operate in K1-equivalent mode
on hosts where KL isn't installed yet. This is deliberate: it
lets the K2.A scaffold land before the production deployment
story for `kakeyalattice` distribution is settled.

**Mac M4 acceptance gate (separate from the K2.A integration
gates of §11.11.5).**  Empirical evidence required from the Mac
M4 reviewer aid `scripts/review_pr_k2a_kl_smoke_on_mac.sh`,
running `scripts/research/k2a_kl_mac_smoke.py`:

1. **Direct codec round-trip on MPS.** `V14KakeyaZamirLatticeGPU
   (D=256, q_range=38, device='mps').roundtrip(K)` produces a
   reconstruction with relative MSE ≤ 5e-4 — the bound is 10×
   the published CUDA fidelity envelope (which is ~3e-5 for D4
   Q=38 on typical K/V) to absorb MPS bf16 reduction-order
   numerics. If MPS produces materially worse reconstruction
   than CUDA at the same Q, that's a finding to escalate to
   the KakeyaLattice repository, not a blocker for K2.A in
   this repo.
2. **Adapter-level round-trip.**
   `KakeyaLatticeCompressor.compress / decompress` on synthetic
   `[num_kv_heads=1, n_positions=256, head_dim=256]` K/V tensors
   on MPS produces `K, V` reconstructions whose RMS error matches
   the direct-codec result within numerical noise (≤ 1.05× the
   direct rmse). Validates the adapter's reshape / clone
   logic.
3. **Eviction state machine on MPS.** After `compress(positions
   [0..255])` followed by `evict(positions[128..255])`,
   `decompress(positions[0..127])` succeeds and
   `decompress(positions[128..255])` raises `KeyError`. Validates
   that the per-position dictionary state machine works on MPS
   (it doesn't depend on tensor device, but pinning the
   behaviour catches future tensor-device-bookkeeping regressions).
4. **Factory dispatch on MPS.**
   `make_default_compressor(device=torch.device('mps'),
   prefer_kakeya=True)` returns an instance of
   `KakeyaLatticeCompressor` (not the Identity fallback) AND
   the codec name reflects the requested lattice / Q. Validates
   that the dispatch correctly recognises `kakeyalattice` is
   available on the active device.

The Mac smoke script emits a JSON report at
`results/research/k2a_kl_mac_smoke_<stamp>.json` with
`summary.status == "pass"` and `summary.mps_active == true` when
all four checks pass. That file is the K2.A Mac M4 portability
evidence, committed alongside the K2.A integration PR.

**What `kakeyalattice` install on Mac M4 looks like.**  PyPI
release: `pip install kakeyalattice`. Source install (recommended
during early K2.A iteration so changes to the codec are local):
clone `github.com/FluffyAIcode/LLM-KV--Cache-compress`, then
`pip install -e <clone>/kakeyalattice/python`. The package's
`pyproject.toml` declares only PyTorch as a hard dependency, so
the install is fast (~10 s) on Mac M4. The `vllm_backend/` plugin
of the upstream repo is **not** installed on Mac (vLLM is a
CUDA-only stack); only the pure-Python codec layer is needed for
K2.A.

**Which lattice on Mac M4.** D4 (v1.4) is the K2.A default for
Mac because it has lower per-block compute (4-D blocks vs 8-D
for E8) and the per-decode-step latency budget on Mac M4 is
tighter than on H100 / H200. E8 (v1.5) gives +0.29 dB shaping
gain over D4 at matched bit budget but takes ~25–30 % more
compute per block; on Mac M4 this trade-off favours D4 unless
empirical Mac latency under E8 is materially better than D4
(possible if MPS dispatches D4's parity-flip branch poorly), to
be measured in the K2.A throughput rung at 22k+ context.

**What we do not commit to in this amendment.**

* MLX (Apple's native framework) backend for KakeyaLattice. MLX
  is typically 1.5–3× faster than PyTorch MPS on Apple Silicon
  for matmul-heavy workloads, so an MLX backend for KL would
  improve Mac M4 K2.A throughput further. That's a separate
  workstream, not a Mac portability requirement: PyTorch MPS is
  sufficient for the K2.A acceptance gates above, and porting
  KakeyaLattice's codec to MLX requires upstream coordination
  with the `kakeyalattice` repository. Track as a future K4-slot
  optimisation.
* Bit-exact parity between the Mac M4 KL output and the H200 KL
  output. bf16 reduction order differs across backends; we accept
  the 10× rmse slack in gate (1) above to absorb this. If the
  difference grows beyond 10× during K2.B cross-model training,
  the discipline note of §11.11.6 ("train `f_θ` against KL-on")
  applies — train against the **deployed** backend's output, not
  CUDA's.
