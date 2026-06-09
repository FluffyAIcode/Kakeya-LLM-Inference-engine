# K3 `f_θ` training pipeline — skeleton

**Status**: design draft (2026-06-09).
**Implementation PR**: not yet opened.
**Companion contract**: [k3-cross-model-dlmrestored-verifier-contract.md](k3-cross-model-dlmrestored-verifier-contract.md)
**Authoritative architectural framing**: ADR 0008 §11.5 / §11.6 / §11.11.6 / §11.13.6 / §11.15

This document specifies the training pipeline for the `f_θ`
projection that maps the DFlash drafter's K/V into Gemma 4 26B-A4B's
K/V shape, enabling K3 cross-model dLM K/V Restoration. It is a
**skeleton** — exact hyperparameters, dataset shards, and convergence
criteria are determined empirically during the K3 implementation PR.

## 1. What `f_θ` is and why it must be trained

Per ADR 0008 §11.5, `f_θ: drafter_K/V → verifier_K/V` is a learned
per-(verifier_layer, verifier_kv_head) linear projection that adapts
the DFlash drafter's transient K/V (computed during the drafter's
own forward pass) into K/V tensors in the verifier's shape so they
can be plugged into the verifier's attention at evicted positions.

In K1 / K2.A, proposer and verifier share weights — `f_θ` is
identity, no training needed. In K3 with DFlash drafter (0.4B,
block-diffusion, ~6-12 layers TBD) and Gemma 4 26B-A4B-it verifier
(26B MoE, 30 layers, head_dim per Google spec), the K/V shapes
differ across:

- **Layer count**: drafter has fewer layers than verifier; layer
  alignment strategy from §3 of the contract doc.
- **`head_dim`**: typically same within Gemma family, but DFlash's
  block-diffusion architecture may use different. Verify on load.
- **`num_kv_heads`**: Gemma 4 26B-A4B uses GQA (per Gemma family
  convention); DFlash drafter's KV heads TBD. Different head
  counts require expanding/contracting via `f_θ`.

`f_θ` is a parametric model trained to make
`verifier(restored K/V at evicted positions) ≈ verifier(ground-
truth K/V at evicted positions)` for the long-context retrieval
distribution we care about.

## 2. Training data schema

Per training example: a (prompt_text, target_continuation) pair
from a long-context retrieval distribution (RULER, NarrativeQA,
or similar). Per ADR §11.7 K3 row, the corpus selection is
"long-context" — at least 8k context, ideally up to 100k+ to
match the §11.12 evidence ladder.

For each example, the training pipeline computes:

1. **Drafter K/V capture** (the input):
   ```
   drafter_K[layer ∈ drafter_layers, position ∈ [0..T), kv_head, head_dim]
   drafter_V[same shape]
   ```
   Captured via the same hook mechanism as K1.A
   (`capture_proposer_kv` from
   `inference_engine.v04.kv_capture`), running the DFlash drafter
   forward on `prompt_text`.

2. **Verifier ground-truth K/V** (the target):
   ```
   verifier_K[layer ∈ verifier_layers, position ∈ [0..T), kv_head, head_dim]
   verifier_V[same shape]
   ```
   Captured via the same hook mechanism, running the **Gemma 4
   26B-A4B-it verifier itself** forward on `prompt_text`. Note:
   this is the EXPENSIVE step; the verifier's forward at long
   context is what K3 is trying to make fast at inference time,
   but during training we run it once per example to get
   ground-truth labels.

The training set is therefore a stream of `(drafter_K/V,
verifier_K/V)` paired tensors. Both tensors have the same
position dimension `T` but different layer / head_dim / kv_head
dimensions.

### Storage estimate

For a single 100k-token example:

- Drafter K/V: ~8 layers × 100k × 4 kv_heads × 256 head_dim × 2
  bytes (bf16) × 2 (K + V) ≈ 1.6 GB per example
- Verifier K/V: ~30 layers × 100k × 16 kv_heads × 256 head_dim × 2
  bytes × 2 ≈ 12 GB per example
- Total: ~14 GB per example

For 1000 long-context examples (modest training set), total
storage = ~14 TB. **This is significant**. Mitigations:

- Train at shorter context first (8k → 32k) to reduce per-example
  storage by ~10x; long-context generalization tested at validation
  time only.
- Cache training data on shared blob storage (e.g. S3 / HF Datasets);
  stream batches at training time rather than holding full data.
- Filter to only "evicted-position" positions (positions outside
  sink+window of the T being processed). For sink=4 + window=64 at
  T=100k, the evicted set is ~99.93% of positions, so this doesn't
  help much at training time but does for sampling diverse retrieval
  patterns at validation.

## 3. Loss form

Per ADR §11.6:

> `f_θ` is a learned per-layer projection that maps a proposer
> `K[L', p, ...]` to a verifier `K[L, p, ...]` (similarly for V),
> trained to minimise `||verifier(reconstructed K/V at evicted) -
> verifier(ground-truth K/V at evicted)||` on logits or a
> downstream-task surrogate.

Concretely, two loss formulations to try (in order of cheapness):

### 3.1 K/V-level reconstruction loss (cheapest, recommended start)

```
L_recon = Σ_(layer, position, head) || f_θ(drafter_K)[layer, p] - verifier_K_truth[layer, p] ||²
        + (same for V)
```

Compute time: just `f_θ` forward — milliseconds per batch.
Storage: drafter K/V + verifier K/V ground-truth (the ~14 GB/example
estimate above).

**Pros**: cheap to compute, easy to differentiate, parameter-
gradient analysis is interpretable per-layer.
**Cons**: minimising K/V L2 doesn't directly minimise the downstream
attention output difference. Two K/V tensors with small L2
distance can produce attention outputs with large logit differences
in some directions.

### 3.2 Logit-level loss (more expensive, more architecturally aligned)

```
L_logit = Σ_(position, vocab) || verifier_logits_with_f_θ_K/V(p) - verifier_logits_with_truth_K/V(p) ||²
```

Compute time: per batch = drafter forward + f_θ forward + verifier
forward TWICE (once with restored K/V, once with truth K/V) =
expensive at 26B verifier scale.
Storage: same as 3.1 plus running activations through verifier.

**Pros**: trains f_θ for the actual quantity that matters
(downstream argmax matching).
**Cons**: 5-10x more compute per training step. Practical for
fine-tuning f_θ after L_recon converges, but slow as a from-scratch
objective.

### 3.3 Recommendation

Two-stage training:

1. **Stage 1 (~70-90% of training compute)**: L_recon. Get f_θ to
   reconstruct K/V tensor-level within bounded relative error.
2. **Stage 2 (~10-30%)**: L_logit fine-tune. Tighten f_θ on the
   downstream argmax distribution.

Stop criterion at each stage: validation NIAH recall plateau (see §5).

## 4. Hyperparameters (skeleton)

Initial guesses to anchor the training PR. Tune empirically.

| param | initial value | notes |
|---|---|---|
| optimizer | AdamW | standard for f_θ-style projections |
| LR (Stage 1) | 1e-4 | per-(layer, head) linear is small; not too sensitive |
| LR (Stage 2) | 1e-5 | logit loss is more sensitive; lower LR |
| batch size | 4 sequences × 8k tokens (effective 32k tokens) | trade off between batch noise and per-batch memory |
| total tokens (Stage 1) | ~1B tokens of drafter+verifier K/V | informed by typical projection-head training size |
| total tokens (Stage 2) | ~100M tokens | shorter due to logit loss expense |
| weight decay | 0.01 | standard |
| gradient clip | 1.0 | standard |
| precision | bf16 mixed | matches inference precision |
| f_θ init | identity-on-shared-subspace per ADR §11.11.6 | not random |

GPU budget estimate per ADR §11.7: ~$200-1000 for K3 production
training. With H100 at ~$2/hr and ~1B Stage 1 + 100M Stage 2
tokens, this is plausible — tighter estimate after first
implementation iteration.

## 5. Validation harness

Reuse the existing K1.E NIAH validation harness
(`scripts/research/k1e_niah_validation.py`) with cross-model setup
once K3 implementation lands. Acceptance per ADR §11.8 1a:

- `recall(v0.4 K3 with f_θ) ≥ recall(oracle Gemma 4 26B-A4B-it) − 5pp`
  at every §11.12 ladder rung.

Validation cadence: every 50M training tokens (Stage 1) / every
10M (Stage 2). Track:

- L_recon per layer (Stage 1) — convergence detection
- L_logit (Stage 2)
- NIAH recall at the 5.6k rung (cheap canary; full ladder once
  per major checkpoint)
- Per-position attention output cosine similarity (interpretive
  diagnostic; not a gate)

## 6. Checkpoint format

`f_θ` checkpoint is a single state_dict with:

```python
{
    "version": "k3.fθ.v1",
    "drafter_id": "z-lab/gemma-4-26B-A4B-it-DFlash",
    "drafter_kv_shape": (drafter_layers, drafter_kv_heads, drafter_head_dim),
    "verifier_id": "google/gemma-4-26B-A4B-it",
    "verifier_kv_shape": (verifier_layers, verifier_kv_heads, verifier_head_dim),
    "layer_alignment_strategy": "uniform" | "pooled" | "learned",
    "layer_alignment": list[int] | None,  # if static; None if learned
    "weights": {  # the trainable W_K, W_V, b_K, b_V tensors
        "W_K": Tensor[verifier_layers, verifier_kv_heads, verifier_head_dim, drafter_kv_heads, drafter_head_dim],
        "W_V": Tensor[same shape],
        "b_K": Tensor[verifier_layers, verifier_kv_heads, verifier_head_dim],
        "b_V": Tensor[same shape],
    },
    "training_metadata": {
        "stage": "stage1" | "stage2",
        "total_training_tokens": int,
        "validation_niah_recall_5_6k": float,
        "validation_niah_recall_100k": float,
        "config_snapshot": dict,  # frozen hyperparams
    },
}
```

Save format: `safetensors` with separate JSON sidecar for the
metadata block (so the metadata is human-readable and grep-able
without loading tensor weights).

## 7. Cost-aware staging

Per the user directive 2026-06-09 ("K3 first; K2 Qwen as
backport"), the order is:

1. **K3 dry-run**: implement the cross-model `DLMRestoredVerifier`
   with `LinearLayerProjection` and **identity-initialized
   weights** (no training yet). Run K3 NIAH at the §11.12 ladder.
   Expected: massive recall regression (because untrained f_θ
   doesn't actually project meaningfully). This is the **baseline
   for what untrained K3 looks like** — establishes the gap to
   close.
2. **K3 training Stage 1**: L_recon training. Validate every
   50M tokens. Stop when validation NIAH recall plateaus.
3. **K3 training Stage 2**: L_logit fine-tune. Validate every
   10M tokens. Stop when ADR §11.8 1a gate (Δ ≤ 5pp at every
   §11.12 rung) passes OR when 100M tokens spent without further
   improvement (then the gap is structural — escalate to a
   richer `f_θ` architecture per the §11.11.4 escalation paths).
4. **K2.B (Qwen backport)**: re-run the K3 pipeline on the
   smaller research-scale pair (Qwen3.5-4B + DFlash 0.4B
   drafter, scale ratio 10:1). Useful as: (a) reproducibility
   check that the K3 pipeline isn't K3-specific, (b) cheaper
   research vehicle for tuning hyperparameters between K3
   iterations, (c) Mac-fit alternative for development /
   smoke testing without the 4-bit-quantized verifier path.

## 8. Failure modes & escalation

If K3 training Stage 1 doesn't converge to L_recon < bounded
threshold:

- **`f_θ` capacity insufficient**: per-(layer, head) linear is
  too restrictive for the 65:1 scale ratio. Escalate to:
  - per-(layer, head) two-layer MLP with non-linearity
  - per-layer matrix factorisation (rank-r approx)
  - learnable layer alignment (drafter→verifier soft mixture)
- **Layer alignment wrong**: try `pooled` or `learned` instead
  of `uniform`.
- **K1.A hook mechanism doesn't apply to DFlash drafter**:
  drafter uses `trust_remote_code=True` and may have a
  non-standard attention class hierarchy. Inspect drafter
  modeling code; adapt hook pattern.

If Stage 2 doesn't tighten Δ to within 5pp of oracle even at
plateau:

- Per ADR §11.13.6 staleness analysis, K3 cached resident K/V
  at K2.A.2 stateful caching may have suffix-drift staleness
  from the dLM proposer's full-attention behaviour. The Δ that
  Stage 2 cannot close may be the staleness gap, not f_θ
  capacity. Test by running K3 with K1.D-style stateless
  caching (no §11.13.6.1 staleness) and seeing if Δ closes.
  If it does, escalate to §11.13.6.4 freshness designs for
  K2.A.2-style stateful caching.

## 9. What this skeleton does NOT specify

- Exact corpus shards. Need to verify RULER and NarrativeQA
  licensing for our use, and possibly add proprietary
  long-context corpora.
- Multi-GPU training topology. Single H100 may suffice for
  Stage 1 with batched K/V capture; Stage 2 requires careful
  parallelism due to verifier-twice-per-step.
- Validation NIAH dataset — should it be the same as K1.E's
  needle-in-haystack synthetic, or a real RULER subset?
  K1 used synthetic; K3 may want real for stronger transfer.
- Distillation alternative: instead of training `f_θ` to match
  the verifier's K/V, train the **drafter** to produce K/V that
  match what the verifier expects. Out of scope for K3 (the
  drafter is z-lab's published checkpoint), but worth noting.
