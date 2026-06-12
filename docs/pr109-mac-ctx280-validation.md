# PR109 Mac ctx280 Validation

This note records the review-driven rerun for PR #109 after fixing the
measurement issues called out in review.

## Review Corrections

- Fair timing: cross and oracle now report the same `e2e_prefill_plus_decode`
  scope, plus per-sample `prefill_s`, `decode_s`, and `e2e_s`.
- Chunked prefill: MLX prompt prefill now uses `--prefill-chunk-size` to avoid
  the long-context one-shot forward path that can OOM.
- Adaptive native path: Step 2 adaptive S5 native skips `build_restoration`,
  f_theta restoration, and aux capture.
- Gemma4 stop tokens: `<turn|>` is treated as a generation stop token alongside
  `<eos>`.
- Gate scale: validation was rerun with `n=5`, `max_new_tokens=32`, and
  haystack lines `238..322`, producing prompt lengths `4406..5810`.

## Command

```bash
PYTHONPATH=.:sdks/python python scripts/research/k3_integrated_niah_eval_mac.py \
  --verifier-path /Users/fluffy314/Documents/Kakeya-LLM-Inference-engine-pr94-resolve/models/gemma-4-26B-A4B-it-mlx-4bit \
  --drafter-id z-lab/gemma-4-26B-A4B-it-DFlash \
  --f-theta-dir results/research/f_theta_v5_s5_sliding \
  --s5-exact-full-attn --fused-specdecode --block-size 4 \
  --n-samples 5 --haystack-min-lines 238 --haystack-max-lines 322 \
  --max-new-tokens 32 --prefill-chunk-size 512 --decode-warmup-tokens 1 \
  --output results/research/k3_mlx_fused_fair_ctx280_n5_gen32_20260612_105807.json
```

## Result

- Recall: cross `5/5 = 1.0`, oracle `5/5 = 1.0`, delta `0pp`.
- Prompt lengths: `4406..5810` tokens.
- Timing scope: `e2e_prefill_plus_decode` for both cross and oracle.
- Cross Step 2 throughput: `0.2217 tok/s` (`39 tok / 175.893s`).
- Oracle AR throughput: `0.0858 tok/s` (`39 tok / 454.484s`).
- Speedup vs oracle AR: `2.584x`.
- KV memory: S5 `132.92 MB`, naive full KV `1308.88 MB`, savings `89.8%`.

## Interpretation

This validation supports Step 2 adaptive S5 native under the corrected e2e
measurement scope at ctx280 scale on the tested Mac setup. It does not claim
that Step 1 incremental is fixed; earlier evidence still shows Step 1 remains
slow and should be treated as a separate optimization target.
