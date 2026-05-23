# ADR 0001 — Proposer Sizing, Alignment Strategy, and Verifier Decoupling

- **Status**: Accepted
- **Date**: 2026-05-23
- **Decision drivers**: Maximizing acceptance rate, controlling training cost,
  enabling verifier swaps without re-architecting the proposer.

## 1. Context

The project's core proposition is to replace the verifier's KV cache cost
with a small DLM proposer plus speculative decoding, while preserving exact
verifier-equivalent output. The economic value of this architecture depends
almost entirely on a single empirical quantity: the **token acceptance
rate** $\alpha$ produced by rejection sampling between proposer and
verifier. At $\alpha < 0.3$, speculative decoding produces ~1.1× speedup
and the proposer's weight memory dominates net byte accounting. At
$\alpha \geq 0.5$, speedup crosses 2× and the architecture becomes
genuinely useful. At $\alpha \geq 0.7$, it becomes the SOTA serving
pattern.

As of this ADR, our measured operating point is:

| Component        | Configuration                                           | Params | Acceptance |
| ---------------- | ------------------------------------------------------- | ------ | ---------- |
| Proposer         | `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` (off-the-shelf) | 0.6 B  | —          |
| Verifier         | `Qwen/Qwen3-1.7B` (bf16, MLX backend)                   | 1.7 B  | —          |
| Speculative pair | (no alignment training)                                  | —      | **0.06–0.12** |

This sits well below the "useful" threshold and far below the "SOTA"
threshold, which forces a decision about where to invest:

- **Larger proposer?** (commonly assumed lever, but possibly wrong)
- **Smaller proposer + better training?** (less obvious, but supported by literature)
- **Different verifier?** (independent question, addressed by ADR 0002 once written)

We need a defensible baseline answer to all three, framed in a way that
remains valid as we move from `Qwen3-1.7B` to `Qwen3-8B` and beyond.

### 1.1 What "alignment" means here

Throughout this ADR, **alignment** refers to the *output-distribution*
similarity between proposer and verifier — formally, the importance ratio
$\mathbb{E}_{x \sim p_q}[p_v(x)/p_q(x)] \to 1$ that drives the
rejection-sampling acceptance probability
$\min(1, p_v/p_q)$. It does **not** refer to parameter-space similarity
($\|\theta_v - \theta_q\|$), which is neither necessary nor sufficient
for high acceptance. Two models can have identical parameters yet produce
divergent token distributions (different attention masks, different
training objectives), and two models with completely different parameter
shapes can produce near-identical token distributions if trained to do so.

### 1.2 What the published evidence shows

EAGLE-1/2/3, HASS, and Medusa report consistent empirical regularities
across LLaMA2/3, Vicuna, Mixtral, and DeepSeek-V2.5 verifiers:

| Verifier            | Verifier params | Proposer params | Ratio   | Acceptance | Speedup |
| ------------------- | --------------- | --------------- | ------- | ---------- | ------- |
| Vicuna-7B           | 7 B             | ~0.24 B         | 1 : 29  | 0.78       | 3.0×    |
| LLaMA2-13B-Chat     | 13 B            | ~0.34 B         | 1 : 38  | 0.78       | 3.1×    |
| LLaMA2-70B-Chat     | 70 B            | ~0.55 B         | 1 : 127 | 0.74       | 3.4×    |
| Mixtral-8×7B        | 47 B (13 B active) | ~0.45 B      | 1 : 104 | 0.71       | 3.2×    |
| LLaMA3-70B-Instruct | 70 B            | ~0.5 B          | 1 : 140 | 0.75       | 3.5×    |
| DeepSeek-V2.5       | 236 B (21 B active) | ~1.0 B      | 1 : 236 | 0.65       | 3.0×    |

Two non-obvious patterns:

1. **The proposer's absolute size is nearly verifier-independent.** Going
   from a 7 B verifier to a 70 B verifier requires roughly 2× more
   proposer parameters, not 10×. The proposer is not a "miniature
   verifier"; it learns the conditional behaviour of the verifier given
   the verifier's own hidden state, a task whose intrinsic difficulty
   does not scale linearly with verifier capacity.

2. **Acceptance is governed by alignment quality, not parameter ratio.**
   1 : 30 and 1 : 240 ratios both produce $\alpha \in [0.65, 0.78]$ when
   the proposer is trained with EAGLE-3-style representation alignment.
   The alignment training, not the size choice, is the load-bearing piece.

## 2. Decision

### 2.1 Proposer sizing

**Adopt a fixed proposer-size band of 0.25 B to 1 B parameters,
independent of verifier size, for all verifiers up to 200 B active
parameters.**

- **Default working size**: 0.5 B. Sufficient headroom for K-step deep
  drafting (HASS-style multi-step training), comfortable margin above the
  ~0.25 B vocabulary/depth floor, fits in 1 GB at bf16 / 0.3 GB at 4-bit.
- **Hard floor**: 0.25 B. Below this, alignment training fails to
  converge — hidden-state capacity is insufficient to absorb verifier
  representation; vocabulary projection collapses on long-tail tokens.
- **Hard ceiling**: 1.0 B. Above this, speedup plateaus while training
  cost and weight-amortization break-even seq-length both grow. Reserved
  for very large verifiers (≥ 200 B active) where the additional capacity
  measurably helps long-tail acceptance.

Our current `dllm-hub/Qwen3-0.6B-mdlm` proposer (0.6 B) sits comfortably
inside this band and **does not need to be replaced**.

### 2.2 Alignment strategy

**Adopt EAGLE-3-style representation alignment as the canonical training
recipe**, with a project-specific minimal-viable variant detailed in
section 4.

The recipe has four load-bearing components, none of which are optional:

1. **Shared embedding and LM head.** Proposer embedding and output
   projection are *copied verbatim* from the verifier and *frozen*. This
   is the single highest-leverage decision in the entire training
   pipeline — it grants the proposer full access to the verifier's
   ~1–2 B parameters of vocabulary representation for free.
2. **Hidden-state alignment loss.** MSE or smooth-L1 loss between
   proposer's last-layer hidden state and verifier's last-layer (or
   weighted multi-layer, EAGLE-3 style) hidden state, with a learned
   linear projection to handle dimension mismatch.
3. **On-policy data collection.** Training prompts are completed by the
   verifier itself; proposer is supervised on the verifier's actual
   trajectories, not on ground-truth corpus text. Off-policy training
   produces a trained-vs-deployed distribution shift that EAGLE measured
   at ~15 percentage points of acceptance.
4. **Logits distillation as auxiliary.** Standard temperature-scaled KL
   between top-K logits, $T \in [2, 4]$. Auxiliary, not primary —
   representation alignment dominates the gradient signal in well-tuned
   recipes.

### 2.3 Verifier decoupling

**Treat proposer training and verifier choice as separable concerns,
sharing 90 % of the training pipeline across verifier swaps.**

Concretely:

- The proposer architecture (dllm-hub MDLM 0.6 B backbone) is fixed
  across verifiers.
- For each new verifier we train a separate set of *small* artifacts:
  the projection matrix $W$ (a few MB), the LoRA adapters on the
  proposer backbone (~50 M params), and a fresh on-policy hidden-state
  cache (the only large item, ~30–50 GB on disk per dataset, regenerable).
- The proposer's frozen body and the training scripts are unchanged.

This means future verifier swaps (1.7B → 8B → larger) are
*data-and-fine-tune* operations, not *re-architecture* operations.

## 3. Alternatives Considered

### 3.1 Scale proposer with verifier (rejected)

The intuitive lever: 1.7 B verifier → 0.6 B proposer; 8 B verifier →
~3 B proposer; 70 B verifier → ~25 B proposer. Rejected because:

- Empirically wrong: EAGLE-3's 0.5 B proposer serves a 70 B verifier at
  $\alpha = 0.75$. The "scale together" heuristic predicts $\alpha < 0.3$
  at this ratio, which contradicts published results.
- Defeats the project goal: a 25 B proposer for a 70 B verifier would
  consume more memory than it saves on the verifier's KV cache.
- Confounds verifier swaps: every new verifier would require choosing,
  obtaining, and training a new proposer at a new scale.

### 3.2 Pure logits distillation, no representation alignment (rejected)

Train proposer to match verifier's top-K logits only, no hidden-state
loss. Simpler. Rejected because:

- Acceptance plateau is consistently 10–20 points lower than
  representation-aligned recipes (EAGLE-1 ablation, Medusa-2 comparison).
- Long-tail tokens are systematically mispredicted because logits-only
  loss provides weak signal in low-probability regions.
- Convergence speed is ~3× slower in the same compute budget.

### 3.3 Medusa-style multi-head heuristic on the verifier (rejected)

Bolt parallel LM heads onto the verifier itself, no separate proposer.
Rejected because:

- Acceptance ceiling is ~0.5 (independence assumption between heads
  hurts long-context coherence).
- Does not help our project's primary goal: KV cache replacement.
  Medusa heads run on the verifier's KV-cached state and therefore
  carry the full KV cost.
- Architecturally orthogonal to our DLM proposer thesis.

### 3.4 Joint training of verifier and proposer (rejected)

Train both models together with a shared loss. Theoretically optimal.
Rejected because:

- Requires training-from-scratch access to the verifier, which we do
  not have for any commercial Qwen / Gemma / DeepSeek model.
- Conflates two roles that we want to keep separable for operational
  reasons (verifier swaps, multi-tenant serving with different
  verifiers behind the same proposer).

### 3.5 Train proposer larger than 1 B (deferred)

For verifiers above ~200 B active, the literature has insufficient data
points to set a confident sizing. Deferred to a future ADR (likely
0003+) when we actually have such a verifier in scope.

## 4. Project-Specific Minimum Viable Recipe

Distilling the decision into a concrete training plan that can start
without further architectural debate:

```
Stage 1 — Frozen artifacts (one-time, ~minutes)
  • Copy Qwen3-1.7B embedding → proposer.embed_tokens (frozen)
  • Copy Qwen3-1.7B lm_head    → proposer.lm_head     (frozen)
  • Initialize linear projection W : R^(d_q) → R^(d_v)

Stage 2 — On-policy data collection (one-time, ~hours, single A100/H100)
  • Prompt pool: ShareGPT (en) + WildChat (zh) ≈ 50 k prompts
  • For each prompt: verifier.generate(max_new_tokens=512), capture
      (a) generated token sequence
      (b) last-layer hidden state per token (~ float16, ~3 KB/token)
      (c) top-20 logits per token (~ 60 B/token)
  • Persist as Parquet shards, ~30–50 GB total

Stage 3 — Proposer fine-tune (single A100/H100, ~6–12 hours)
  • Backbone: dllm-hub MDLM 0.6 B + LoRA rank-32 adapters on attention QKV
  • Trainable params: ~50 M (LoRA) + ~2 M (W)
  • Loss = 1.0 · smooth_L1(W h^q, h^v)              # primary
        + 0.5 · KL(softmax(z_v/T) ‖ softmax(z_q/T))  # T=2, auxiliary
        + 0.1 · mask_recovery_loss                    # MDLM regularizer
  • AdamW, lr=2e-4 on LoRA, lr=1e-3 on W, cosine schedule
  • Effective batch 64, ~10 k steps

Stage 4 — Acceptance evaluation
  • Held-out 1 k prompts, measure α at K ∈ {1, 2, 4}
  • Acceptance gate: α ≥ 0.40 at K=2 → ship as v1
                     α ≥ 0.55 at K=4 → ship as v2 with tree spec
```

Expected outcome: $\alpha$ moves from current 0.06–0.12 to 0.40–0.55 on
1.7 B verifier, unlocking 2–3× speculative speedup as a baseline; an
EAGLE-3 multi-layer-fusion follow-up training is expected to reach
0.55–0.70.

## 5. Consequences

### 5.1 Positive

- **Verifier swaps become tractable.** Moving from 1.7 B to 8 B verifier
  requires re-running Stages 1–3 with the new verifier and re-training
  W + LoRA; the proposer architecture, training scripts, and serving
  code are unchanged. Estimated incremental cost per swap: a few GPU-days.
- **Memory accounting becomes predictable.** Proposer weight cost is a
  bounded constant (≤ 1 GB at bf16, ≤ 0.3 GB at 4-bit) regardless of
  verifier scale, which makes Net Bytes per Token forecasts stable
  across product configurations.
- **Training cost is bounded and reproducible.** All training is single-GPU,
  single-day. No multi-node coordination, no large-cluster booking,
  no stalled experiments waiting for capacity.
- **The proposer becomes a reusable asset.** A single
  alignment-trained proposer can serve a fleet of verifiers, with
  per-verifier specialization happening only in cheap projection +
  LoRA artifacts.

### 5.2 Negative / accepted trade-offs

- **The off-the-shelf proposer is not directly usable.** The current
  `dllm-hub/Qwen3-0.6B-mdlm` weights are not aligned with `Qwen3-1.7B`;
  using them as-is yields the observed 0.06–0.12 acceptance.
  Alignment training is mandatory; there is no zero-cost path to high
  acceptance.
- **On-policy data is verifier-specific and non-trivial in size.**
  Each verifier swap regenerates a 30–50 GB hidden-state cache. Storage
  is cheap; the operational discipline of versioning, garbage-collecting,
  and reproducing these caches is a real cost.
- **We are committing to EAGLE-3 lineage assumptions.** If a fundamentally
  different paradigm (e.g. continuous latent draft models, joint pretrained
  speculative pairs) becomes SOTA, this ADR will need to be superseded.
  The risk is mitigated by EAGLE-3's broad empirical validation across
  model families and the fact that the recipe's individual ingredients
  (shared LM head, on-policy data, hidden-state alignment) each have
  independent literature support.

### 5.3 Implications for current and future code

- **`kv_cache_proposer/proposer.py`** keeps its current API. A new
  `training/repr_align/` package will load proposer weights, swap
  embedding / lm_head with verifier's, attach LoRA, and run the loss
  defined in Stage 3.
- **`inference_engine/backends/mlx/proposer.py`** does not need
  architectural changes; it loads whatever proposer weights it is
  pointed at. Post-alignment proposer weights are a drop-in
  replacement.
- **A future ADR 0002** should record the verifier-selection decision
  (1.7 B vs 8 B vs larger) including memory budgets and quantization
  choice. This ADR explicitly does not foreclose that decision.
- **A future ADR 0003** should record the K-value and tree-spec strategy
  once we have measured post-alignment acceptance.

## 6. Validation

This ADR is considered validated when **all** of the following are true:

1. Stages 1–3 of the minimum viable recipe are implemented in
   `training/repr_align/` with unit-test coverage on the deterministic
   pieces (loss computation, projection initialization, LoRA insertion).
2. A run of Stage 4 on `Qwen3-1.7B` verifier achieves $\alpha \geq 0.40$
   at K = 2 on the held-out evaluation set.
3. The same training scripts run unchanged when pointed at a
   `Qwen3-8B` verifier (modulo the data-source flag), demonstrating the
   verifier-decoupling claim of section 2.3.

If item 2 fails (acceptance plateaus below 0.40), this ADR is
superseded by a follow-up that either revises the recipe or revisits
proposer sizing.

## 7. References

- EAGLE-1/2/3 papers and official implementations
  (representation alignment, multi-layer fusion, dynamic tree spec).
- HASS: multi-step training for autoregressive draft models.
- Medusa-1/2: multi-head speculative decoding (rejected alternative).
- DeepMind speculative decoding paper: rejection sampling formalism
  underlying $\alpha$ and acceptance probability.
- Project document `docs/local-inference-engine.md`: where the trained
  proposer is consumed by the serving stack.
