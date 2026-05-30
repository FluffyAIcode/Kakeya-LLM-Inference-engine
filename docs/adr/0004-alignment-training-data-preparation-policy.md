# ADR 0004 — Alignment Training Data Preparation Policy (Nemotron-informed)

- **Status**: Accepted
- **Date**: 2026-05-25
- **Decision drivers**: Acceptance-rate quality, deployment-time
  robustness across domains, training cost discipline,
  reproducibility of the v0.3 alignment training pipeline.
- **Depends on**: ADR 0001 (proposer/verifier alignment recipe),
  ADR 0002 (verifier selection + quantization), ADR 0003 (verifier
  ↔ slab pool integration).
- **Informed by**: NVIDIA Nemotron-Labs-Diffusion-14B technical
  report (Fu et al. 2026) and the underlying Efficient-DLM paper
  (arXiv:2512.14067).

## 1. Context

ADR 0001 §4 specified a minimum-viable EAGLE-3-style alignment
recipe with high-level shape (50 k prompts, on-policy data,
LoRA rank 32 on QKV, smooth-L1 + KL distill + mask recovery
loss). The recipe was deliberately under-specified pending real
data on what works at our scale.

Two pieces of new information now warrant tightening the
specification before v0.3 implementation begins:

1. **NVIDIA's Nemotron-Labs-Diffusion technical report (May 2026)**
   reports a LoRA configuration that diverges sharply from EAGLE-3
   defaults — they target only `o_proj` with rank 128, alpha 512,
   and report TPF (tokens-per-forward) gains of +14.4 % / +32.5 %
   / +27.6 % at 3 B / 8 B / 14 B model scale on their
   self-speculation drafter. The result has implications for our
   own LoRA configuration choice even though our architectural
   setting (two independent models) differs from theirs (one
   model, two modes).

2. **Production deployment scenarios** revealed in our v0.2.0
   integration testing show that domain coverage in the training
   data is the single largest determinant of deployment-time
   acceptance. A single-domain training set produces
   embarrassingly low cross-domain acceptance numbers regardless
   of how aggressive the LoRA tuning is.

This ADR locks in the data preparation policy that will govern
the v0.3 alignment work — what prompts to use, how to capture
verifier behavior, what hyperparameters to pin, what acceptance
criteria gate the v1 ship — so the training pipeline implementation
can proceed against a fixed contract.

## 2. Decisions

### 2.1 Prompt pool composition (50 k prompts)

The training prompt pool is **deliberately multi-domain** with
the following composition:

| Domain                   | Source                          | Share | Count       |
| ------------------------ | ------------------------------- | ----- | ----------- |
| Chat (English)           | ShareGPT (cleaned + dedup)      | 30 %  | 15 k        |
| Chat (Chinese)           | WildChat (zh subset)            | 20 %  | 10 k        |
| Code generation          | HumanEval-X + MBPP + CodeAlpaca | 15 %  | 7.5 k       |
| Math reasoning           | GSM8K + MATH (subset)           | 10 %  | 5 k         |
| Long context (≥ 2k tok)  | LongBench (subset)              | 10 %  | 5 k         |
| Multi-turn conversations | MT-Bench-Conv + WildChat 多轮     | 10 %  | 5 k         |
| Tool calls / JSON output | ToolBench (subset)              | 5 %   | 2.5 k       |

**Quality filtering (mandatory before sampling)**:

- Length filter: 5 ≤ prompt_len ≤ 4096 tokens.
- Language ID: fasttext or langdetect; language tagged on each row.
- MinHash dedup: Jaccard similarity > 0.85 → drop.
- Manual blacklist: known toxic / NSFW prompts.

**Adversarial / OOD slice (separate from training pool)**: a held-
back 1 k-prompt set covering deliberately weird inputs (typo-heavy,
super-short, super-long with repetition, prompt-injection
attempts). This set is for evaluation only and must produce a
non-zero acceptance number — but is not gated.

### 2.2 On-policy verifier rollout (Stage 2 capture)

The rollout configuration **must mirror deployment exactly** so
training-test attention patterns match:

```yaml
verifier_id: "Qwen/Qwen3-1.7B"                          # v1
# verifier_id: "mlx-community/Qwen3-8B-4bit"             # v2 (separate run)
decoding: greedy                                          # ADR 0001 §2.2
max_new_tokens: 512
sink_size: 4
window_size: 64
system_prompts:                                           # rotate over multiple
  - "You are a helpful, concise assistant."
  - "You are an expert programmer."
  - "You are a careful mathematician..."
  # 5–10 distinct system prompts mixed evenly so the
  # proposer learns to be system-prompt-robust
chat_template: "from verifier tokenizer (apply_chat_template)"
```

**Capture per generated token**:

- Last-layer hidden state (bf16, dim = `verifier.hidden_size`)
- Top-20 logits with associated probabilities
- Committed token id
- Position id (global, post sink+window trim)
- `cache_logical_size` at emission time
- Block-aligned views: also persist hidden states grouped into
  blocks of size K = 4 (the deployment block size), so the
  alignment loss can supervise at block boundaries (Nemotron-
  informed; see §3.2).

**Post-rollout filtering**:

- Drop tokens where verifier top-1 probability < 0.30 (low-
  confidence regions destabilize LoRA gradients).
- Drop sequences with token-level repetition > 30 % of generated
  span (degenerate completions).
- Drop sequences hitting `max_new_tokens` without EOS (often
  indicates failed completion patterns).

### 2.3 LoRA configuration (Nemotron-informed)

**Default for v0.3 production training**:

| Hyperparameter   | Value      | Source / rationale                                        |
| ---------------- | ---------- | --------------------------------------------------------- |
| Target modules   | `o_proj`   | Nemotron technical report (preserves attention behavior)  |
| Rank `r`         | 128        | Nemotron technical report                                 |
| Alpha `α`        | 512        | Nemotron technical report (scale = α/r = 4)               |
| Dropout          | 0.05       | Standard PEFT default                                     |
| Bias             | `none`     | Standard PEFT default                                     |
| Trainable params | ~16 M      | Smaller than EAGLE-3 default (50 M); concentrated on o_proj |

**Rationale for diverging from EAGLE-3 default (QKV, rank 32)**:
Nemotron's report shows that LoRA on `o_proj` only — without
touching QKV — preserves the backbone's attention behavior more
faithfully and produces +14–32 % TPF gains. The mechanism: QKV
LoRA changes *how the model attends*; `o_proj` LoRA changes *how
attention output flows into the FFN*. The latter is a more
conservative perturbation that better preserves the backbone's
pretrained behavior, which matters for alignment quality.

**Required A/B validation before locking in for production**: the
v0.3 implementation must run an A/B/C experiment on a 5 k-prompt
subset (cost: ~$30, ~6 H200-hours) comparing:

- A: EAGLE-3 default (QKV, r = 32, α = 16)
- B: Nemotron default (o_proj, r = 128, α = 512)
- C: Hybrid (QKV + o_proj, r = 64, α = 128)

The variant with highest held-out acceptance becomes the
production configuration. Default expectation: B wins, but A/B
results are normative for the actual training run.

### 2.4 Loss formulation

```
L_total = 1.0 · L_repr_alignment    (smooth_L1 of W·h_q vs h_v)
        + 0.5 · L_logit_distill     (KL with T = 2, top-20)
        + 0.1 · L_mask_recovery     (cross-entropy on masked tokens)
```

**Position-dependent masking probability** (Nemotron-informed,
from Efficient-DLM §2.2):

```
p_mask(position_in_block) = 0.3 + 0.4 · (position_in_block / block_size)
```

Block-end positions get higher mask probability (~0.7), block-start
positions lower (~0.3). This mirrors test-time confidence-based
decoding order, where high-confidence tokens at the start of a
block are committed before low-confidence tokens at the end.

### 2.5 Verifier-specific data isolation

**Each verifier requires its own independent training data.**
A LoRA trained on Qwen3-1.7B hidden states is not transferable to
Qwen3-8B; a LoRA trained on bf16 hidden states is not transferable
to 4-bit (per ADR 0002 §3.5).

**Data versioning convention**:

- Path layout: `data/alignment/<verifier_id>/<verifier_dtype>/<schema_version>/`
- Per-row tag: every Parquet row carries `verifier_id`,
  `verifier_dtype`, `system_prompt_hash`, `block_size`, and
  `schema_version` so the trainer can refuse to load mismatched
  data.

### 2.6 Greedy-only training assumption

The recipe assumes **temperature = 0 (greedy decoding)** at both
training time and deployment time. Per ADR 0001 §2.2, the
HTTP API accepts `temperature` and `top_p` in requests but does
not honor them; the same constraint binds the alignment training.

If the project later commits to non-greedy deployment, a separate
ADR specifies the temperature-aware alignment training recipe.
Until that ADR exists, deployment with `temperature > 0` produces
acceptance numbers below the gates documented here, and that
degradation is the user's responsibility to measure.

### 2.7 Acceptance gates (v1 ship criteria)

| Slice                | Gate (acceptance @ K=2) | Gate (TPF @ K=4) |
| -------------------- | ----------------------- | ---------------- |
| Aggregate            | ≥ 0.40                  | ≥ 2.0            |
| Chat (en)            | ≥ 0.45                  | ≥ 2.2            |
| Chat (zh)            | ≥ 0.40                  | ≥ 2.0            |
| Code                 | ≥ 0.25                  | ≥ 1.5            |
| Math                 | ≥ 0.30                  | ≥ 1.7            |
| Long context         | ≥ 0.30                  | ≥ 1.7            |
| Multi-turn           | ≥ 0.35                  | ≥ 1.9            |
| Tool calls           | ≥ 0.40                  | ≥ 2.0            |
| Adversarial / OOD    | ≥ 0.10 (no hard gate)   | ≥ 1.2            |

The v1 ship is gated on the aggregate row plus *all* domain rows
except adversarial. Failure on any gated row blocks the ship.

### 2.8 Evaluation metrics (Nemotron-comparable)

The evaluation harness reports — per slice and aggregate:

- `acceptance_rate` at K ∈ {1, 2, 4}
- `tokens_per_forward` (TPF) at K ∈ {2, 4} — the headline
  Nemotron metric: `(1 + accepted_tokens) / (proposer_forwards + verifier_forwards)`
- `mean_acceptance_length` — mean number of consecutively accepted
  tokens before first rejection
- `speedup_vs_vanilla_AR` — wall-time speedup against the same
  verifier running greedy AR with no proposer; this is the
  user-visible quantity

Reporting absolute throughput in tokens/sec is **not** required
because our deployment hardware (Mac M-series, consumer GPU) is
incomparable to Nemotron's GB200 numbers. Relative speedup is
the apples-to-apples metric.

## 3. Where Nemotron's findings apply (and where they don't)

### 3.1 Applies — adopt directly

- **LoRA `o_proj`-only target** with rank 128, alpha 512. The
  rationale (preserve backbone attention behavior, change only
  the output projection) is architecture-agnostic; it applies as
  much to our cross-model alignment as to their same-model
  self-speculation.
- **Position-dependent masking schedule**. The "dLMs retain a
  left-to-right tendency at inference, so training-time masking
  should mirror that" insight is a property of masked diffusion
  models in general, not Nemotron-specific.
- **Block-aligned hidden state capture** during data collection.
  The training-test attention pattern matching argument from
  Efficient-DLM §2.2 directly transfers.
- **TPF and acceptance length as headline metrics**. Reporting
  these alongside acceptance rate makes our numbers Nemotron-
  comparable for external readers.

### 3.2 Does not apply — do not adopt

- **Single-model self-speculation architecture**. Structurally
  incompatible with our two-model design (ADR 0001 §1.2). We
  cannot share KV between drafter and verifier when they are
  different models.
- **Joint AR + diffusion pretraining objective**. Our proposer
  (`dllm-hub/Qwen3-0.6B-mdlm`) is already a pre-trained DLM; we
  fine-tune via LoRA, not continued pretraining. The 10 B / 100 B
  token training scales from Efficient-DLM apply to AR-to-dLM
  conversion, not to the LoRA fine-tune we're doing.
- **Custom CUDA kernels**. Nemotron's 1015 tok/sec on GB200
  comes from kernel-level optimization, not the alignment
  recipe. Our v0.3 work is alignment training; kernel
  optimization is v0.4+ and gets its own ADR.
- **14B model scale**. Their LoRA effects are measured at
  3B/8B/14B; our proposer is 0.6B and our verifier is 1.7B/8B.
  The qualitative direction (o_proj LoRA helps) transfers, but
  the quantitative numbers (+14 / +32 / +27 % TPF) do not. The
  A/B/C experiment in §2.3 is what tells us our actual numbers.

## 4. Alternatives Considered

### 4.1 Single-domain training (50 k all ShareGPT) — rejected

Cheapest data acquisition; predictably collapses on every other
domain. Production users with code/math/tool workloads would see
acceptance < 0.10 on those workloads despite the chat number
being good. The asymmetry is unacceptable for a public-facing
release.

### 4.2 Use EAGLE-3 default LoRA (QKV, r = 32) without A/B test — rejected

Saves ~6 H200-hours and ~$30. But Nemotron's data is strong
enough that ignoring it is willful. Running the A/B/C is cheap
insurance; if Nemotron's config doesn't translate to our setting,
we fall back to QKV with confidence; if it does, we save 0.05–0.15
acceptance points on the production run.

### 4.3 Skip block-aligned capture, just record per-token states — deferred

Cheaper Stage 2 implementation. The training-test attention
pattern mismatch this introduces costs an estimated 0.05–0.10
acceptance points (Efficient-DLM §2.2 + our own architectural
asymmetry). For a v0.3 release where every acceptance point
matters, paying the implementation cost (~50 lines in
`rollout_worker.py`) is cheap. Revisit if implementation reveals
the cost is much higher than estimated.

### 4.4 Train one universal LoRA across multiple verifiers — rejected

Tempting because it would let one alignment cover Qwen3-1.7B and
Qwen3-8B. But:

- Hidden-state distributions across verifiers are too different
  for a single LoRA to track both well — empirical results from
  EAGLE-3 multi-verifier experiments show 10–15 percentage point
  acceptance drops vs verifier-specific training.
- The data-isolation policy (§2.5) is consistent with ADR 0001
  §2.3's verifier-decoupling design: same proposer architecture,
  per-verifier alignment artifacts.

### 4.5 Wait until v0.3 starts to write this ADR — rejected

The data-prep choices materially constrain the trainer
implementation. Writing the ADR after starting the trainer means
either retrofitting decisions (waste) or churning the ADR mid-
implementation (worse). The cost of writing this ADR pre-v0.3 is
low (~3 hours of writing); it makes the v0.3 PR sequence cleaner.

## 5. Consequences

### 5.1 Positive

- **Predictable v0.3 implementation**: trainer + data-collection
  PRs land with their inputs / outputs already specified, no
  re-litigation of "what counts as good training data" mid-PR.
- **External reproducibility**: external readers can compare our
  v0.3 numbers to Nemotron's directly via TPF + acceptance length
  metrics.
- **Domain transparency**: per-slice acceptance gates surface
  domain weaknesses in the release; users with code-heavy
  workloads can see that gate explicitly rather than discovering
  it in production.
- **Lower training-test gap**: block-aligned capture +
  position-dependent masking are both attention-pattern-matching
  improvements that should additively lift acceptance.

### 5.2 Negative / accepted trade-offs

- **Higher initial data-collection complexity**: we cannot just
  `verifier.generate()` and call it done; we need 7-domain
  prompt-pool curation, multi-system-prompt rotation,
  block-aligned hidden-state views, post-rollout filtering. The
  Stage 2 implementation grows from a ~300-line MVP to a
  ~1500-line production-grade pipeline. Acceptable cost given
  the alternative is a release that fails on every non-chat
  workload.
- **Per-verifier data duplication**: Qwen3-1.7B and Qwen3-8B
  alignment runs cannot share data; storage doubles from ~85 GB
  to ~210 GB. Acceptable: storage is cheap compared to GPU time.
- **Mandatory A/B test before production**: adds ~6 H200-hours /
  $30 to the v0.3 budget. Acceptable: high information value.
- **Greedy-only is a real product limitation**: clients that
  require non-greedy sampling get degraded acceptance with no
  in-recipe remedy. Documented in §2.6; revisited in a future
  ADR if and when the product needs it.

### 5.3 Implications for code

- **`training/repr_align/data_collection/`** new module (~1500
  lines): `prompt_pool.py`, `rollout_worker.py`,
  `parquet_writer.py`, `post_filter.py`, plus 7 per-domain
  config YAMLs.
- **`training/repr_align/trainer.py`** consumes the schema
  defined here; specifically reads `block_aligned_views` and
  applies the position-dependent mask schedule from §2.4.
- **`training/repr_align/eval.py`** reports the metrics from §2.8
  per slice from §2.7.
- **`tests/training/repr_align/`** adds unit tests against fake
  verifier / fake hidden states, covering filter logic, schema
  versioning, A/B config switching, and per-slice aggregation.
  Real-weight runs are platform tests, not unit tests.

## 6. Validation

This ADR is considered validated when:

1. The v0.3 data-collection implementation produces a 50 k-prompt
   Parquet shard whose schema matches §2.2 exactly.
2. The A/B/C experiment from §2.3 is run, documented, and
   the production LoRA configuration is selected from its
   results (recorded in a follow-up ADR addendum or in the v0.3
   PR description).
3. The v1 ship gate from §2.7 is met or explicitly waived (with
   the waiver documented per the ADR convention in
   `docs/adr/README.md`).
4. The eval harness reports all metrics from §2.8 in the v0.3 PR.

If §2.7 gates are not met after exhausting reasonable iteration
(remediation paths from ADR 0001 §4 fallback section), the v1
ship is paused and a follow-up ADR analyzes which gate failed
and what to change.

## 7. References

- ADR 0001 — Proposer sizing, alignment, verifier decoupling.
- ADR 0002 — Verifier selection, quantization.
- ADR 0003 — Verifier ↔ slab pool integration.
- Fu et al. 2026, "Nemotron-Labs-Diffusion: A Tri-Mode Language
  Model Unifying Autoregressive, Diffusion, and Self-Speculation
  Decoding" (NVIDIA technical report).
- arXiv:2512.14067 — "Efficient-DLM: From Autoregressive to
  Diffusion Language Models, and Beyond in Speed" (Fu et al.,
  Dec 2025).
- HuggingFace model card:
  https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-14B
  (LoRA configuration `subfolder="linear_spec_lora"`).
