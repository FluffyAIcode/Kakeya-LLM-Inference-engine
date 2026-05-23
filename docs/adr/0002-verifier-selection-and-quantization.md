# ADR 0002 — Verifier Selection, Quantization, and the Open-vs-Closed-Weight Constraint

- **Status**: Accepted
- **Date**: 2026-05-23
- **Decision drivers**: Memory fit on consumer hardware, perceived
  latency, alignment-training feasibility, future-proofing across the
  Qwen / Gemma / DeepSeek roadmap.
- **Depends on**: [ADR 0001](0001-proposer-sizing-and-alignment.md)
  (proposer is a constant 0.25–1 B regardless of verifier choice).

## 1. Context

ADR 0001 fixed the proposer in a 0.25–1 B band and established that
verifier swaps are *data-and-fine-tune* operations rather than
re-architecture operations. That decision deliberately deferred the
question of *which* verifier to ship against. This ADR resolves that
question for the project's first two ship targets and lays out the rule
for future swaps.

The decision space is constrained by four hard, independent factors:

1. **Memory budget.** Primary deployment target is consumer hardware:
   Mac M-series (16–64 GB unified memory) and consumer Nvidia GPUs
   (RTX 4090: 24 GB, RTX 3090: 24 GB). The 24 GB Mac M4 is the
   project's reference machine and is the lower bound below which we
   refuse to design.
2. **Perceived latency.** The chat REPL (`scripts/chat.py`) has a
   project-internal target of ≤ 30 s for a 200-token Chinese response on
   the reference Mac. Beyond that, the streaming UX is no longer
   acceptable.
3. **Alignment availability.** Per ADR 0001, every verifier we ship
   against requires its own representation-alignment-trained proposer.
   A verifier we cannot align against is a verifier we cannot ship.
4. **Open weights for hidden states.** EAGLE-3 alignment requires
   read access to the verifier's embedding, last-layer hidden state, and
   LM head. Closed-weight (API-only) verifiers cannot be aligned with
   the project's primary recipe; they fall back to a degraded path that
   tops out at lower acceptance (see section 5).

The current measured baseline (commit `e207aed`, MLX backend on M4):

| Metric                        | Value                  |
| ----------------------------- | ---------------------- |
| Verifier                      | `Qwen/Qwen3-1.7B`, bf16 |
| Resident memory               | ~5.5 GB                |
| Wall-time (zh KV-cache prompt, 150 tokens) | 12.07 s   |
| Acceptance (no alignment)     | 0.06–0.12              |

## 2. Decision

### 2.1 Ship sequence

The project ships against verifiers in this order:

| Ship | Verifier            | Backend         | Quant   | Triggered when                                       |
| ---- | ------------------- | --------------- | ------- | ---------------------------------------------------- |
| v1   | `Qwen/Qwen3-1.7B`   | MLX / CUDA      | bf16    | ADR 0001 Validation #2 passes (α ≥ 0.40 at K=2)      |
| v2   | `Qwen/Qwen3-8B`     | MLX / CUDA      | **4-bit (AWQ-style)** | v1 in production + alignment retrained for 8B verifier |
| v3+  | larger / MoE        | TBD             | TBD     | Recorded in a future ADR (0004+)                     |

v1 reuses the verifier we have today. v2 is the planned upgrade. The
2-step sequence exists deliberately: v1 proves that the alignment
pipeline (ADR 0001) actually works on a verifier we can run end-to-end
on a 24 GB Mac without quantization risk; v2 then exercises the
verifier-decoupling claim of ADR 0001 §2.3 — same proposer architecture,
new alignment artifacts, new quantized verifier weights.

### 2.2 Quantization rule

**bf16 below 4 B parameters; 4-bit MLX (or AWQ on CUDA) at 4 B and above.**

Concretely:

- Verifier ≤ 2 B params: bf16 unconditionally. Memory headroom is
  sufficient on the reference 24 GB Mac; 4-bit gains nothing
  meaningful and adds quantization noise that hurts acceptance.
- 2 B < Verifier < 8 B: bf16 if it fits in ≤ 60 % of available unified
  memory, else 4-bit. Decision is made per target machine in the engine
  config, not statically per verifier.
- Verifier ≥ 8 B: 4-bit by default. bf16 is reserved for non-consumer
  GPU paths (A100 / H100 / MI300) recorded in a separate engine
  deployment ADR.

The 60 % threshold leaves ~10 GB headroom on a 24 GB machine for the
proposer (≤ 1 GB), per-forward activations (~1–2 GB), the OS, and other
applications the user is running.

### 2.3 Open-weight requirement

**The project's primary alignment recipe (ADR 0001) requires
open-weight verifiers.** Closed-weight verifiers (e.g. GPT-4-class
APIs, Claude-class APIs, Gemini-API-only models) cannot be aligned with
EAGLE-3 because they expose neither embedding weights nor hidden
states. Section 5 documents the degraded fallback path; it is *not* a
ship target for v1 or v2.

### 2.4 Latency budget enforcement

For each verifier candidate, before training the alignment proposer
against it, run a *no-proposer-baseline* benchmark on the reference
hardware:

```
verifier.generate(reference_prompt, max_new_tokens=200)
```

If this baseline exceeds 50 s on the reference Mac, the verifier is
rejected at the planning stage — speculative decoding cannot recover a
verifier that is fundamentally too slow.

For Qwen3-8B 4-bit on M4: estimated baseline 35–45 s for 200 tokens,
which leaves margin. For Qwen3-32B-class models on M4: estimated
baseline > 90 s, which rejects them at this gate. They become
candidates only on cloud / data-center deployment, recorded in a
separate ADR.

## 3. Alternatives Considered

### 3.1 Skip v1, go straight to Qwen3-8B (rejected)

Rationale for considering: 1.7 B is "too small to matter" as a
production verifier; the project's value proposition is at 7 B+ scale.

Why rejected:

- Two unproven things at once (alignment pipeline correctness *and*
  quantized 8B fit/latency) compound risk. If something goes wrong, we
  cannot tell which factor caused it.
- ADR 0001's validation explicitly requires acceptance ≥ 0.40 on a
  verifier we have measurements on. Switching that verifier to one we
  don't yet have measurements on changes what "validation" means.
- Cost: training the 8B-verifier alignment requires ~30–50 GB of
  on-policy hidden-state cache. Burning that compute before validating
  the recipe on the smaller verifier is wasteful.

### 3.2 Skip Qwen3-8B, go directly to Qwen3-32B or DeepSeek-V2.5 (rejected)

Why rejected:

- 32 B 4-bit ~ 16 GB; with proposer + activations + OS, total resident
  is ~21 GB on a 24 GB Mac — same memory cliff that rejected bf16 for
  8B. Pushing harder without first hardening the engine layer is bad
  engineering sequencing.
- Latency budget: 32 B at 4-bit on M4 is estimated > 90 s for a
  200-token reply, exceeding the perceived-latency target by 3×.
- These verifiers are appropriate for the cloud-deployed verifier
  pattern (proposer local, verifier remote) sketched in
  `docs/local-inference-engine.md`. That deployment mode is recorded in
  a future ADR; this ADR's scope is local-only.

### 3.3 Use a non-Qwen verifier for v1 / v2 (rejected for now)

Candidates: Gemma 4, DeepSeek V3/V4 distill, Llama 3.x.

Why rejected for v1 / v2:

- Project commitment in the original product brief is Qwen / Gemma /
  DeepSeek as parallel targets. Sequencing them serially (Qwen first)
  is a planning choice, not an architectural one.
- Tokenizer continuity: the current proposer (`dllm-hub Qwen3-0.6B-mdlm`)
  shares Qwen3 tokenizer. Switching verifier families forces a
  proposer family switch too, which compounds with v1's alignment
  validation in the same way as 3.1.
- Multi-family support is a v3+ concern recorded in a future ADR.

### 3.4 8-bit instead of 4-bit at the 8 B boundary (rejected)

Why rejected:

- 8-bit Qwen3-8B ≈ 8.5 GB resident. With proposer + activations + OS,
  total ~13–14 GB. Fits but eats most of the headroom needed for
  serving multiple sessions or running other apps concurrently.
- 4-bit MLX (group-wise quantization, group_size=64) measures ~1 % of
  perplexity degradation on Qwen3 family, well below the noise floor of
  speculative decoding's accept/reject decisions.
- 4-bit is also the format with mature MLX community releases
  (`mlx-community/Qwen3-8B-4bit`), which removes a conversion step
  from the engineering path.

### 3.5 Mix: bf16 verifier on CUDA, 4-bit on MLX (deferred)

Tempting because RTX 4090 has the same 24 GB as M4 but with faster
memory bandwidth, so bf16 8B (16 GB) would fit. Deferred because:

- It bifurcates the alignment training: bf16 verifier and 4-bit
  verifier produce slightly different hidden states, which in
  principle requires two alignment runs.
- Empirically the difference is small enough (per Qwen3 4-bit
  literature) that one alignment run usually transfers, but we have no
  in-house measurement yet.
- Deferred to a follow-up ADR after v2 ships and we measure the
  quantization-transfer gap directly.

## 4. Consequences

### 4.1 Positive

- **v1 ships on a verifier we can already run.** No new model
  acquisition, no quantization conversion, no memory cliff. The only
  unknown in v1 is whether ADR 0001's alignment recipe actually works.
- **v2 has a clear, bounded scope.** When v1 ships, v2 is mechanical:
  add `--verifier-id Qwen/Qwen3-8B-4bit` flag, regenerate hidden-state
  cache, retrain proposer adapters, ship.
- **The 60 % memory rule generalizes.** Future verifier candidates can
  be evaluated with a one-line calculation; we are not redesigning
  memory budgets per model.
- **Alignment pipeline reuse.** The `training/repr_align/` package
  built for v1 will run unchanged for v2. The verifier-decoupling claim
  of ADR 0001 §2.3 gets exercised in production rather than just on
  paper.

### 4.2 Negative / accepted trade-offs

- **v1 is a "stepping stone" verifier.** 1.7 B is below the size where
  the project's KV-cache-savings story becomes economically interesting
  (KV/token at 1.7 B is small enough that the proposer's weight
  amortization breakeven sits at uncomfortably large B × S). This is
  acknowledged: v1's purpose is recipe validation, not user-facing
  value.
- **Quantization noise interacts with alignment.** When v2's verifier
  is 4-bit, the alignment recipe trains against quantized hidden
  states, which are very slightly different from bf16 hidden states.
  Acceptance may be 1–3 percentage points lower than equivalent bf16
  alignment. We accept this; the absolute target (α ≥ 0.50 for v2 at
  K=2) is set with that haircut already factored in.
- **Closed-weight models are out of scope.** GPT-4 / Claude / Gemini
  cannot be served by this engine in its primary mode. Section 5
  explains why and what the lossy fallback would look like, but the
  fallback is not a v1/v2/v3 commitment.

### 4.3 Implications for current and future code

- **`scripts/setup_*.sh`**: `download_models` becomes parameterized
  over a model list rather than hard-coding `Qwen3-1.7B`. v1's list
  remains `[Qwen3-0.6B-mdlm, Qwen3-1.7B]`; v2's list adds
  `mlx-community/Qwen3-8B-4bit`.
- **`inference_engine/backends/mlx/verifier.py`**: gains a `--verifier-id`
  CLI flag and propagates it through `MLXSinkWindowVerifier(config)`.
- **`scripts/run_platform_tests.sh`**: HF cache pre-flight check
  becomes verifier-id-aware (currently hard-coded to `Qwen3-1.7B`).
- **`training/repr_align/`** (introduced for ADR 0001): its
  hidden-state cache directory is keyed by verifier id; v1 and v2
  produce non-overlapping caches that can coexist on disk.
- **Future ADR 0003** records per-verifier K values and tree-spec
  configuration, which depend on measured acceptance from this ADR's
  v1 / v2 deliveries.
- **Future ADR 0004** records remote/cloud-deployed verifier pattern
  (proposer local, verifier in the data center). That is where
  Qwen3-32B / DeepSeek-V2.5 / GPT-OSS-120B class models become
  in-scope.

## 5. Closed-Weight Verifiers — Why and What If

A recurring question is whether EAGLE-3-style alignment can be applied
to commercial API-only models (GPT-4, Claude, Gemini, Qwen-Max). The
honest answer has three parts.

### 5.1 What EAGLE-3 demands and which APIs supply it

EAGLE-3 alignment uses three classes of verifier signal:

| Signal                         | Required for                | Available from API? |
| ------------------------------ | --------------------------- | ------------------- |
| Embedding weights              | Shared `embed_tokens` in proposer | **No (any API)** |
| LM head weights                | Shared `lm_head` in proposer | **No (any API)** |
| Last-layer hidden state per token | Hidden-state alignment loss   | **No (any API)** |
| Per-token top-K log-probs      | Logits-distill auxiliary loss | OpenAI (top-20), Anthropic (no), Gemini (top-5 in some endpoints), Qwen-Max (no) |
| Sampled token sequence         | On-policy token-level supervision | **Yes (all APIs)** |

The first three rows are the core of EAGLE-3 and are uniformly
unavailable from closed APIs. This is not an oversight by API providers:
exposing hidden states would leak the model's internal representation
in a way that damages competitive moats and aids extraction attacks. It
is highly unlikely to change for frontier models.

### 5.2 What's still possible — degraded paths

Three increasingly weak alternatives, all worse than EAGLE-3:

1. **Logits-only distillation** (when API exposes top-K log-probs,
   e.g. OpenAI). Loss reduces to KL between proposer's full
   distribution and the API's top-K. Empirically observed acceptance
   ceiling: ~0.45–0.55 (vs 0.70–0.80 for full EAGLE-3). This is the
   regime that early speculative-decoding papers (the original DeepMind
   work) operated in before EAGLE introduced hidden-state alignment.
2. **Sequence-level behavioral cloning** (when API exposes only
   sampled tokens). Loss is standard next-token cross-entropy on
   verifier-generated sequences. Empirically observed acceptance
   ceiling: ~0.30–0.40. This is essentially "train a small model to
   imitate the API's output style"; it does not exploit the verifier's
   probability distribution at all.
3. **Hybrid with a local proxy** — train alignment against a
   *similar-but-open* verifier (e.g. align against Qwen3-72B-Instruct
   weights, deploy with Qwen-Max API). Produces a proposer aligned to
   the wrong target; transfer quality depends on how close the open
   proxy is to the closed model. Empirically: 5–15 percentage points
   below pure EAGLE-3 against the actual proxy.

### 5.3 Why none of these are v1/v2 ship targets

- The project's primary value proposition (KV-cache replacement, local
  memory savings) requires running the verifier locally. A closed API
  verifier is run remotely; the "memory savings" become "the user pays
  per token". Different product, different ADR.
- Acceptance ceilings of 0.30–0.55 collapse the speculative speedup to
  1.3–1.8×, which does not justify the engineering complexity.
- Closed APIs charge per token of *both* prompt and completion. The
  proposer's per-step verifier call contains ~K candidate tokens that
  may be rejected; rejected tokens still cost money. The economic
  break-even moves against speculative decoding in this setting.

### 5.4 What we *will* support if the closed-API mode becomes a goal

If the project later decides to address closed APIs (recorded in a
future ADR), the path is:

- A `RemoteVerifier` adapter exposing the same interface as
  `MLXSinkWindowVerifier` / `SinkWindowVerifier`.
- A degraded `training/logit_distill_remote/` package implementing
  alternative #1 above when the target API exposes top-K log-probs.
- A separate evaluation harness that reports acceptance, throughput,
  *and* token cost per generated user-visible token, because the
  third axis is what dominates the closed-API economics.

This is a non-trivial body of work and explicitly out of scope for v1
and v2. Pursuing it without first finishing v2 would distract from
proving the core alignment recipe works.

## 6. Validation

This ADR is considered validated when:

1. **v1 validation**: ADR 0001 §6 conditions are met against
   `Qwen/Qwen3-1.7B` (α ≥ 0.40 at K=2), confirming the bf16 path
   functions end-to-end.
2. **v2 validation**: with no changes to proposer architecture,
   training scripts, or serving code, the alignment pipeline produces
   a proposer for `mlx-community/Qwen3-8B-4bit` that achieves α ≥ 0.50
   at K=2 on the held-out evaluation set, and the engine runs the
   reference 200-token Chinese prompt within the 30 s perceived-latency
   target on the M4 reference machine.
3. **Memory-rule validation**: the 60 % memory threshold from §2.2
   correctly predicts fit/no-fit on at least three independent target
   machines (24 GB Mac, 32 GB Mac, 24 GB RTX-class GPU) without
   per-machine tuning.

If item 2 fails on perceived latency but passes on acceptance, the ADR
is partially superseded by an engine-side optimization ADR. If item 2
fails on acceptance, ADR 0001's recipe is what needs revision, and
this ADR's v2 commitment is paused until that revision lands.

## 7. References

- ADR 0001 — Proposer sizing, alignment, verifier decoupling
  (this ADR is the verifier counterpart of that proposer-side decision).
- `docs/local-inference-engine.md` — describes the serving stack that
  consumes the verifiers selected here.
- MLX community: `mlx-community/Qwen3-8B-4bit` for v2.
- Qwen3 4-bit perplexity studies (community-published) supporting the
  4-bit-at-8B decision.
- Original DeepMind speculative decoding paper for the logits-only
  alignment regime referenced in §5.2.
