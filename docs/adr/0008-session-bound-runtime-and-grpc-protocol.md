# ADR 0008 — Session-bound runtime + gRPC protocol

- **Status**: Accepted (2026-06-01)
- **Date**: 2026-06-01
- **Decision drivers**:
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
- **Depends on**: ADR 0001 (proposer sizing + verifier decoupling),
  ADR 0002 (verifier selection), ADR 0003 (slab pool), ADR 0006
  (positioning + §2.3 sub-claims).
- **Supersedes**: ADR 0007 (cross-request KV cache reuse via automatic
  prefix matching). ADR 0007 is retained as a historical record of the
  architecture-discovery process; none of its implementation reaches
  `main`.

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
   reasoning** when evaluating prior implementation work (this is
   what enables the C3 = "close PR #30..#36 without merging" decision
   recorded for v0.3).
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

### 2.3 KV cache binding: byte-exact contract, sink+window per session

**Decision**: on every `Generate` call, the verifier's KV cache state
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
- **PR-A3**: Refactor the verifier (CPU + MLX) so its KV cache state
  is constructed and owned by `SessionStore` rather than by the
  scheduler / pool. Slab-pool integration (ADR 0003) becomes "slab
  per session" instead of "slab per scheduler slot". Internal-only
  refactor; the existing HTTP shim still works against the new
  internal shape. 100% unit coverage; no behavior change to the
  HTTP surface.

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

- **PR-D1**: Update `inference_engine/server/app.py` so each
  `/v1/chat/completions` request creates a single-shot session under
  the new `SessionStore`, prefills, generates, and closes. Removes
  any path-selection / cross-request logic (none of which exists on
  `main` after C3). Adds `Deprecation` / `Sunset` headers. Updates
  the existing 461-test integration suite to match.

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

### 6.6 Phase F — Old PRs cleanup

- **PR-F1**: Close PRs #30..#36 (recorded as decision C3=b,
  2026-06-01) without merging. Their commit history remains
  reachable via the `AgentMemory/v030-pr7-*` branches for archival
  reading; their content is superseded by Phases A–E.

Phases A and F can run in parallel; F is a paperwork PR with no code.

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
  surfaced the partial-cache crash. The hotfix is preserved in the
  commit history of `AgentMemory/v030-pr7-2-path-select-and-prefill-
  incremental-8e7f` even though that branch is not merged (per
  C3 = b).
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
