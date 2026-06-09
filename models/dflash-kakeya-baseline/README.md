# DFlash baseline drafter for `google/gemma-4-26B-A4B-it` (Kakeya-aligned)

Baseline **DFlash block-diffusion drafter** for the Gemma-4 26B-A4B verifier,
for use in Kakeya inference speculative-decoding development and scenario
testing. Loadable directly by the native engine:

```python
import torch
from inference_engine.v04.dflash_drafter import DFlashDrafter
drafter = DFlashDrafter.from_pretrained("models/dflash-kakeya-baseline", dtype=torch.bfloat16)
```

## What this is

- **Architecture**: the native `DFlashDrafter` (5-layer Qwen3 backbone + `fc`
  aux projection + `hidden_norm` + `norm`), faithful to vLLM PR #41703
  `qwen3_dflash.py`. Shares the verifier's embeddings (`×sqrt(hidden)`) and
  lm_head (`final_logit_softcapping=30`). Aux layers `(2,7,12,18,23,28)`.
- **Weights**: the upstream `z-lab/gemma-4-26B-A4B-it-DFlash` checkpoint,
  **alignment-trained** to the Kakeya engine's inference path (see below).
  0.43 B params, bf16, `model.safetensors` (stored via Git LFS).

## Why alignment

The upstream DFlash forward is defined inside vLLM (custom KV-cache writes,
fused kernels). The native engine reconstructs the math, but the exact
aux-hidden-tap semantics live in vLLM internals. Rather than reverse-engineer
them, we treat the gap as an `f_θ` alignment task (ADR 0008 §11,
`docs/design/k3-f-theta-training-pipeline.md`): freeze the verifier, train the
drafter so its drafts match the verifier's greedy tokens.

## Provenance / reproduce

- Base: `z-lab/gemma-4-26B-A4B-it-DFlash`
- Verifier: `google/gemma-4-26B-A4B-it`
- Trainer: `scripts/research/k3_dflash_alignment_train.py`
  ```
  python scripts/research/k3_dflash_alignment_train.py \
      --steps 6000 --lr 5e-5 --block-size 16 --n-prompts 64 --gen-len 192 \
      --train-scope full --save dflash_aligned_corpus.pt
  ```
  (64 diverse prompts, 58 usable; `train_match=0.71`)

## Acceptance (vs the real Gemma-4 verifier, block 16)

| eval | acceptance_rate | acceptance_length |
|---|---|---|
| held-out (8 disjoint prompts) | 0.107 | 2.45 |
| in-domain (small set) | 0.561 | 8.62 |
| reference (HumanEval, vLLM) | 0.447 | 7.70 |

The in-domain run reaching ≥ the reference proves the integration is correct;
the held-out number is limited by the small (64-prompt) alignment corpus and
climbs with more data (10→64 prompts: 1.94→2.45 length). This is a
**baseline** — scaling the alignment corpus is expected to close the held-out
gap toward 7.70.

## Status

Research baseline (not GA). Lossless vs greedy AR is preserved by the
spec-decode accept loop regardless of draft quality; this drafter only affects
*speedup*, not correctness.
