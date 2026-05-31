# ADR 0006 — Project Positioning as Local Agent Infrastructure

- **Status**: Accepted
- **Date**: 2026-05-30
- **Decision drivers**: External positioning, v0.3+ release framing,
  alignment between the project's design choices and its actual best-fit
  application domain.
- **Depends on**: nothing — this is a positioning decision, not a
  technical decision. Builds context from ADR 0001 (proposer/alignment),
  ADR 0002 (verifier selection), ADR 0003 (slab pool), ADR 0004
  (alignment data prep), and the (planned) ADR 0005 (personal layer).

## 1. Context

Through v0.1.0 and v0.2.0, the project's external framing has
implicitly been "yet another local LLM inference engine that adds
DLM-based speculative decoding for chat acceleration." The README,
release notes, and benchmark scripts have all been organized around
single-prompt chat throughput vs vanilla AR.

A series of comparison analyses against existing engines surfaced a
sharper truth:

- vs **vLLM**: not a peer (we don't target data-center GPUs).
- vs **llama.cpp**: structurally a research extension layer, not a
  peer (llama.cpp dominates the consumer-hardware general-AR-LLM
  niche; we cannot and should not compete).
- vs **mlx_lm**: we depend on it for ~70% of basic infrastructure
  and add ~30% of unique algorithmic value (DLM speculative,
  cross-model alignment, production server features).
- vs **NVIDIA Nemotron self-speculation**: structurally weaker at
  the same-architecture single-model scale, but in a different
  deployment quadrant.

What the comparisons consistently revealed is that Kakeya's
**design choices** map cleanly to the **requirements of local
agentic applications**, but were never explicitly justified in
that frame. Specifically:

| Local-agent requirement                  | Pre-existing Kakeya design choice                |
| ---------------------------------------- | ------------------------------------------------ |
| Multiple concurrent agents per machine    | Scheduler + admission control (ADR 0003 + PR #9) |
| Long sessions (hours, ≥ 32k tokens)       | Sink+window KV cache (ADR 0001 §4)               |
| Cross-session memory ("remember our project") | Personal data store (planned ADR 0005)           |
| Per-user codebase / domain personalization | Repr-Align personal LoRA (ADR 0001 + ADR 0004)   |
| Tool-call JSON output reliability          | Greedy bit-deterministic decoding (ADR 0001 §2.2) |
| Mid-stream cancellation                   | `Scheduler.cancel_session` + lifespan (PR #9, #12) |
| Production monitoring                     | Prometheus `/metrics` (PR #13)                   |
| Multi-tenant access control                | API-key auth (PR #13)                            |

This is not retroactive marketing. Every row is a decision recorded
in an existing ADR or PR. The pattern was visible in the design tree
all along; it just was not surfaced as the project's primary frame.

For comparison, the same requirements against `mlx_lm.server`
(today, late May 2026):

| Requirement                                | mlx_lm.server  | Kakeya v0.2.0      | Kakeya v0.4 (planned) |
| ------------------------------------------ | -------------- | ------------------ | --------------------- |
| Multiple concurrent agents                  | Single-tenant  | Multi-tenant       | Multi-tenant          |
| Long-session **memory** stability             | KV grows linearly | sink+window bounded | sink+window bounded     |
| Long-session **latency** stability            | grows linearly with history | grows linearly with history (stateless API) | bounded via cross-request KV reuse |
| Cross-session memory                        | None           | Schema in ADR 0005 | Personal data store   |
| Per-user personalization                   | None           | Stage 1 surgery    | Personal LoRA layer   |
| Production metrics                          | None           | Prometheus         | Prometheus            |
| API key auth                                | None           | Bearer token       | Bearer token          |
| Mid-stream cancel                           | Basic          | Full lifecycle     | Full lifecycle        |

The "memory stability" and "latency stability" rows are deliberately
separated. Sink+window bounds **memory** at the verifier level, but
the OpenAI chat-completions protocol is stateless — every turn the
client re-sends the full chat history and the server re-prefills it
end-to-end. v0.3 makes no claim that per-turn latency stays bounded
across a long session; cross-request KV reuse is a v0.4 feature.

For single-user, single-agent, short-session use, `mlx_lm.server`
is a better fit (simpler, more model selection). For
**multi-agent / long-session / personalized** use, Kakeya's
designed-in features have no peer in the local-Mac ecosystem.

This ADR makes the agentic positioning explicit so future
release framing, integration documentation, benchmarking, and
prioritization decisions all flow from a coherent product story
rather than from accumulated technical decisions.

## 2. Decisions

### 2.1 Reframe v0.3+ release notes from "chat acceleration" to "local agent infrastructure"

**Old framing** (used in v0.2.0 release notes and benchmark
scripts):

> v0.3 ships representation-alignment training to lift speculative
> acceptance rate from 0.10 to 0.40, producing 2.5–3× speedup over
> vanilla AR.

**New framing** (mandatory for v0.3+ release):

> Kakeya v0.3 is a production-grade **local agent infrastructure
> for Mac**. It runs multiple concurrent agents on a single
> machine with **per-session KV memory bounded by sink+window**
> (verified to the byte against the theoretical limit on Mac M4 —
> see §2.3.a), learns per-user codebase and workflow patterns
> through on-device alignment training, retains conversation
> history across sessions, and exposes Prometheus metrics +
> API-key auth for long-running deployment.
>
> Per-turn latency is not bounded across long sessions in v0.3
> because the OpenAI chat-completions protocol is stateless;
> cross-request KV reuse is a v0.4 feature (see §2.3.b).

The technical detail (acceptance rate, alignment training,
speculative speedup) becomes implementation evidence, not the
headline claim. The headline claim is **what the engine enables
the user to build**.

This reframing applies to:

- README top-of-file paragraph (currently chat-acceleration
  framing).
- v0.3 release notes (annotated tag message).
- HuggingFace model card text if we publish one.
- Any external talks / blog posts / paper abstract.
- The bench scripts' top-level docstrings.

The technical-internals framing stays in module docstrings and ADRs
where it belongs, but is no longer the surface story.

### 2.2 Ship integration examples as first-class deliverables

Most local-agent users today consume an inference engine through
**a framework**, not by writing OpenAI-API client code directly.
The dominant frameworks at the time of writing:

- LangChain (`langchain-openai` / `ChatOpenAI`)
- CrewAI (`Agent`, `Crew` with OpenAI-compatible LLM)
- Microsoft AutoGen (`AssistantAgent` + OpenAI config)
- Cursor (uses an OpenAI-compatible custom endpoint)
- Open WebUI / LM Studio (drop-in via OpenAI URL)

We commit to shipping **`docs/integrations/`** with one short
markdown page per framework, each ~50 lines containing:

1. Required Kakeya server config (CLI flags, env vars).
2. Framework client config (5–10 lines of code) pointing at our
   `/v1/chat/completions`.
3. A worked example demonstrating multi-agent concurrent execution
   (the discriminator vs `mlx_lm.server`).
4. Notes on what works / what does not (e.g., `temperature`
   accepted but ignored — ADR 0001 §2.2).

Concrete file list:

```
docs/integrations/
  README.md           # Index + matrix of supported frameworks
  langchain.md        # ChatOpenAI(base_url=..., api_key=...)
  crewai.md           # Crew with multiple agents
  autogen.md          # AssistantAgent multi-agent
  cursor-bridge.md    # Custom Cursor endpoint config
  openwebui.md        # Drop-in URL config
```

These ship with v0.3.0 (not later — they make the v0.3 release
useful from day one).

### 2.3 Add agentic benchmarks alongside chat benchmarks

Current bench scripts (`bench_mlx_speculative.py`,
`bench_mlx_verifier_quant.py`, etc.) measure **single-prompt
short-session** characteristics. These still belong in the suite —
they validate the algorithm. But they do **not** validate the
positioning.

We add a new bench category that tests the actual workload shape:

```
scripts/bench_agentic/
  bench_long_session.py        # ≥ 4-hour session, growing context
  bench_multi_agent.py         # 3 concurrent agents, mixed workloads
  bench_tool_call_reliability.py  # 1000 tool calls, JSON parse-rate
  bench_cancellation.py        # Mid-stream cancellation latency
  bench_persistent_memory.py   # Cross-session recall with personal layer
```

Each script emits a JSON report with the same shape conventions
as existing bench scripts (per
`scripts/bench_mlx_verifier_quant.py`'s pattern). Each script
has a corresponding `mlx_lm` equivalent run so the comparison is
explicit.

The headline numbers from these benchmarks become the v0.3.0
release evidence. Specifically, the v0.3.0 release notes claim
must be backed by:

- "3 concurrent agents on M4 24GB" → measured by `bench_multi_agent.py`
- "Per-session KV bounded across long sessions" → measured by
  `bench_long_session.py` (the §2.3.a sub-claim below)
- "100% tool-call JSON validity" → measured by `bench_tool_call_reliability.py`

If any benchmark fails to back its claim, the release notes
adjust the claim, not the benchmark.

The long-session benchmark validates **two distinct sub-claims**;
v0.3 makes only the first:

#### 2.3.a Memory bounded across long sessions (v0.3 claim)

Per-session KV cache stays bounded by the configured sink+window
regardless of session duration or generated-token count. This is
the §2.3 headline `bench_long_session.py` exists to measure.

**Status (v0.3.0-rc1)**: VERIFIED on two independent runs:

| Run | Wall time | Successful turns | KV peak per turn | Spread |
|---|---|---|---|---|
| 30-min short test #3 | 1,800 s | 58 | 7,798,784 bytes (×58) | 0.00% |
| 4-hour run | 14,400 s | 58 (first 30 min) | 7,798,784 bytes (×58) | 0.00% |

Both runs were on Mac M4 with Qwen3-1.7B and `sink_size=4
window_size=64`. Both recorded a per-turn KV peak of exactly
**7,798,784 bytes** — drift 0.00 MiB, observed/expected = 100.0000%
to the byte. The 4-hour run additionally confirmed the
orphan-session fix invariant (PR #25) over 4 hours: `idle
pool_in_use` stayed at 0 throughout, even while the run was
processing 182 timeout/429 cycles in the §2.3.b regime.

The observed value matches the theoretical sink+window bound:

```
68 tokens × (28 layers × 2 (K+V) × 8 KV-heads × 128 head_dim × 2 bytes) = 7,798,784
```

Evidence files:
- `results/platform-tests/bench_long_session_mac_short3_1780208693.json`
- `results/platform-tests/bench_long_session_mac_4h_1780211323.json`

#### 2.3.b Latency bounded across long sessions (NOT a v0.3 claim)

Per-turn latency does **not** stay bounded as chat history grows.
The OpenAI chat-completions protocol is stateless: every turn the
server re-tokenizes and re-prefills the full conversation history
end-to-end. Sink+window only bounds the generation-phase KV
footprint, not prefill compute.

**Status (v0.3.0-rc1)**: NOT achieved by sink+window alone, and
empirically observed:

- Short test #3 (30 min): p50 latency drifted 15.5 s (0–10 min)
  → 38.6 s (10–20 min) → 55.3 s (20–30 min) as the chat history
  grew from ~50 to ~3,700 tokens.
- 4-hour run: completed only 58 turns of useful work (matching
  the 30-min run exactly), then accumulated 182 errors over the
  remaining 3.5 hours. Error breakdown: 96 client-side
  ReadTimeouts (per-turn latency exceeded the bench's
  `timeout_s=120`) interleaved with 86 HTTP 429s (the timed-out
  request's slab still held while the server worker thread
  finished its prefill). This is exactly the §2.3.b pattern:
  prefill cost grew past the bench's fixed timeout, and the
  long-session degraded into a timeout/recovery loop.

**Practical envelope on Mac M4 / Qwen3-1.7B with `--max-tokens 64
--turn-spacing-s 5 --timeout-s 120`**: useful work for ~30 min /
~60 multi-turn turns of a single continuous session. Past that,
client-side prompt management is required.

**Mitigation in v0.3**: client-side prompt management (summarization,
sliding windows, history truncation) is the user-side fix.
Acceptable for short-turn tool-use agents; insufficient for
hours-long single-session workloads.

**v0.4 plan**: cross-request KV reuse via session affinity — a
follow-up ADR will design the protocol extension and the engine
support for it. See ADR 0006 §2.5 for prioritization context.

The v0.3 release framing must therefore say "long-session
**memory** stability", not "long-session stability". Bench
reports always carry the latency-drift series alongside the
KV-bounded series so the trade-off is transparent to operators.

### 2.4 Establish a "what we are not" stance

To keep the agentic positioning sharp, we explicitly **decline**
the following positioning ambitions, even when they look
adjacent:

- **Not a llama.cpp replacement.** llama.cpp is the right tool for
  general-purpose local AR LLM serving. We don't compete on model
  coverage, hardware portability, or single-binary distribution.
- **Not a vLLM replacement.** Data-center GPU serving is not our
  deployment target; we do not optimize for DGX / multi-node /
  GPU-cluster scenarios.
- **Not a complete agent framework.** We provide the inference
  substrate that agents run on. We do not provide planning loops,
  tool registries, agent personas, or memory-graph orchestration.
  Those belong to LangChain / CrewAI / AutoGen / Cursor / etc.
- **Not a chat product.** Despite the OpenAI-compatible API, our
  optimization target is agentic workloads, not interactive chat.
  Single-user-single-prompt benchmarks may show parity or slight
  loss vs `mlx_lm.server`; we accept that.
- **Not a multi-model gateway.** We support Qwen3 family and the
  dllm-hub MDLM proposer paired with it. Model coverage is not a
  product axis we compete on.

These exclusions free us from feature pressure that does not
serve the agentic-infrastructure thesis.

### 2.5 Re-prioritize the v0.3 / v0.4 path under this lens

The technical roadmap from prior ADRs does not change, but its
**ordering and presentation** do:

| Existing work item                              | Prior framing                  | Re-framed under ADR 0006                                      |
| ----------------------------------------------- | ------------------------------ | ------------------------------------------------------------- |
| Stage 2/3/4 alignment training (ADR 0001 §4, ADR 0004) | "Boost acceptance to 0.40"     | "Make the engine learn the user's codebase / domain"           |
| Personal layer (ADR 0005, planned)              | "Personal LoRA on history"     | "Cross-session memory + per-user agent specialization"        |
| `bench_*` benchmarks                            | "Acceptance rate vs vanilla"   | "Number of agents per Mac, hours-of-stable-session, JSON parse rate" |
| `/metrics` endpoint (ADR 0006 §2.4 below)       | "Prometheus instrumentation"   | "Long-running agent service observability"                    |
| `verifier-id` parameterization (PR #6)          | "ADR 0002 v2 ship enabler"     | "Let agents pick the right model for their task class"        |

No technical decisions reverse. What changes is which features are
*marketed* and *prioritized*. For instance, an integration
example for Cursor (deliverable from §2.2) jumps in priority over,
say, adding a fifth quantization backend — because the former
directly serves the agentic story while the latter does not.

## 3. Alternatives Considered

### 3.1 "Generic local LLM engine" positioning (rejected)

Continue marketing as a generic chat-acceleration engine.
**Rejected** because:

- llama.cpp dominates this niche (ADR 0006 §1 comparison). Direct
  competition on model coverage / hardware portability / single
  binary distribution is unwinnable.
- Single-user-single-prompt benchmarks underplay our actual
  strengths (multi-tenant, long-session, personalized).
- Confuses users about when to choose us vs `mlx_lm.server`.

### 3.2 "Research engine for DLM speculative decoding" positioning (rejected)

Position purely as a research artifact. **Rejected** because:

- We have shipped a production-grade HTTP API + scheduler +
  metrics + auth (PR #7, #9, #12, #13). Calling it "research"
  understates the engineering investment.
- Research-engine positioning sets the wrong expectations for
  contributors and integrators (e.g., "is this stable enough to
  ship in my product?").
- Forces us to keep the algorithmic contribution as the headline,
  which constrains what end users can build with the engine.

### 3.3 "Chat speedup competing with `mlx_lm` speculative decoding" positioning (rejected)

Compete head-to-head with `mlx_lm`'s own speculative decoding
support (`--draft-model` flag). **Rejected** because:

- On the chat-speedup axis, `mlx_lm` will likely match us within
  ~20% via continued integration of speculative decoding into
  their core. We do not have a sustainable structural advantage
  on this axis.
- Forces ongoing point-by-point benchmark comparisons that will
  swing as `mlx_lm` improves, instead of letting our actual
  designed-in agentic strengths carry the story.

### 3.4 "Agent framework + inference engine" combined product (rejected)

Build agent framework features (planning loops, tool registries,
memory graphs) into the engine. **Rejected** because:

- LangChain / CrewAI / AutoGen / Cursor all have far more
  invested. We cannot match them.
- Forces us to take a position on agent design that is largely
  orthogonal to the inference-engine improvements we actually
  contribute.
- "Substrate, not framework" is a more sustainable role and
  matches the unix-philosophy precedent.

### 3.5 Wait until v0.4 to reposition (rejected)

Defer the framing change until the personal layer (ADR 0005) ships.
**Rejected** because:

- v0.3 release framing is set within weeks; making it correctly
  the first time is cheap.
- Mid-release reframing creates a marketing inconsistency.
- The agentic positioning is already justified by v0.2.0
  capabilities (Scheduler, sink+window, metrics, auth) — it does
  not depend on personal layer work landing.

## 4. Consequences

### 4.1 Positive

- **Coherent external story** that ties existing technical
  decisions (sink+window, scheduler, metrics, auth) into a single
  product thesis a non-technical reader can grasp in 30 seconds.
- **Sharper feature prioritization**: every proposed feature is
  evaluated against the question "does this serve the local
  agent infrastructure thesis?"
- **Realistic competitive positioning**: stops trying to win
  comparisons we cannot win (general chat vs llama.cpp, GPU
  serving vs vLLM) and starts highlighting comparisons we can
  win (multi-agent on Mac vs `mlx_lm.server`).
- **Lower marketing-engineering mismatch risk**: the engine ships
  what the marketing claims, because the claims describe what the
  engine was designed for all along.

### 4.2 Negative / accepted trade-offs

- **Cedes the chat-speedup market**. Single-user-single-prompt
  users get directed to `mlx_lm.server` or `llama.cpp`. Some
  current Kakeya users may find this disorienting.
- **Requires shipping integration examples**, which is real work
  (~5 short docs + maintenance as frameworks evolve). The
  alternative (no integrations) is worse — Kakeya looks isolated.
- **Locks the project into Mac-first**. CUDA / ROCm support
  becomes lower-priority because the agentic-on-Mac story is
  what we lead with. CUDA contribution still welcome but moves to
  community-driven rather than core-team-driven.
- **Some bench scripts may show parity-or-loss vs `mlx_lm.server`
  on chat workloads**. Under this ADR, that is fine — it does
  not invalidate the agentic positioning. But it requires
  discipline to not over-react and start optimizing for the wrong
  benchmark.

### 4.3 Implications for code

- **README**: top-paragraph rewrite to lead with agentic
  infrastructure framing.
- **`docs/integrations/`** directory created (§2.2 deliverable
  list).
- **`scripts/bench_agentic/`** directory created (§2.3
  deliverable list).
- **v0.3.0 release notes draft** uses agentic framing throughout.
- **v0.2.x branch (current main)** does not retroactively update
  framing; this ADR governs v0.3.0 onward.

## 5. Validation

This ADR is considered validated when:

1. v0.3.0 README and release notes lead with the agentic-
   infrastructure framing (§2.1).
2. At least three of the five integration examples in `docs/integrations/`
   ship with v0.3.0 (§2.2).
3. The agentic benchmark suite (§2.3) has at least
   `bench_multi_agent.py` and `bench_long_session.py` shipped
   with v0.3.0, with comparison numbers vs `mlx_lm.server`.
4. **The §2.3.a memory-bounded sub-claim is validated by an
   on-device measurement that matches the theoretical sink+window
   bound to within 1% across at least 30 minutes of continuous
   single-session traffic.** Validated by two independent
   v0.3.0-rc1 runs on Mac M4 with Qwen3-1.7B:
   - 30-min short test #3 — 58/58 turns at exactly 7,798,784 bytes
     (`bench_long_session_mac_short3_1780208693.json`)
   - 4-hour run — same 58 turns at the same 7,798,784 bytes,
     plus orphan-session invariant verified for 4 hours of
     continuous server uptime
     (`bench_long_session_mac_4h_1780211323.json`)
   Both runs: observed/expected = 100.0000% to the byte.
5. **The §2.3.b latency-not-bounded caveat is documented in v0.3.0
   release notes**, README, and bench script docstrings — readers
   must not infer "long-session stability" without seeing the
   memory-vs-latency split.
6. Any post-v0.3.0 release positioning that contradicts §2.4
   (the "what we are not" stance) requires a follow-up ADR
   superseding this one — not a unilateral marketing decision.

If item 1 is satisfied but items 2-3 are not, the v0.3.0 release
notes acknowledge the gap and identify which integrations /
benchmarks are deferred to v0.3.1. Items 4 and 5 are GA gates —
v0.3.0 cannot promote from rc to GA without the §2.3.a validation
landing in `results/` and the §2.3.b caveat appearing in release
notes.

## 6. References

- ADR 0001 — Proposer sizing, alignment, verifier decoupling
  (technical foundation that the agentic story builds on).
- ADR 0002 — Verifier selection, quantization (model-class scope).
- ADR 0003 — Verifier ↔ slab pool integration (memory-stability
  foundation for long sessions).
- ADR 0004 — Alignment training data preparation policy
  (per-user / per-domain alignment foundation).
- ADR 0005 (planned) — Personal layer / personal data store
  (cross-session memory foundation).
- PR #7 (HTTP API), #9 (scheduler), #12 (HTTP↔scheduler integration),
  #13 (metrics + auth) — deployment-side production capabilities
  this ADR's positioning relies on.
- `docs/local-inference-engine.md` — original engine architecture
  document; predates the agentic positioning but is consistent with
  it.
