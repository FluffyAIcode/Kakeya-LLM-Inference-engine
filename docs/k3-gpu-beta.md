# K3 GPU beta — Kakeya inference (f_θ + S5 K/V-Restoration)

Status: beta, GPU-validated on NVIDIA H200 with `google/gemma-4-26B-A4B-it`
(verifier) + `z-lab/gemma-4-26B-A4B-it-DFlash` (drafter) + the trained f_θ v5
checkpoint (`results/research/f_theta_v5_s5_sliding/`). Recall 1.0 throughout.

## What it is

The verifier keeps only a **sink+window** local KV cache; at every *evicted*
position its attention reads **reconstructed** K/V, so it attends over the full
context while holding `O(sink+window)` resident KV (ADR 0008 §11).

  verifier (Gemma 4 26B-A4B):  sink+window resident KV
      ├─ sliding layers  → evicted K/V restored via f_θ(drafter K/V)
      └─ full-attn layers (S5: [5,11,17,23,29]) → verifier's OWN exact K/V
                            (recall-critical; f_θ cannot reconstruct these —
                             proven by the α-sweep, eval rel_mse floor ~1.4)

  drafter (DFlash 0.4B): no KV cache; constant-memory K/V reconstruction
      source (its K/V are projected into verifier space by f_θ).

## Components (this branch)

| piece | file |
|---|---|
| DFlash drafter (block diffusion, faithful to z-lab `qwen3_dflash`) | `inference_engine/v04/dflash_drafter.py` |
| f_θ projection (drafter K/V → verifier K/V) | `inference_engine/v04/f_theta.py` |
| Cross-model restored verifier (CUDA) + S5 | `inference_engine/v04/cross_model_dlm_verifier.py` |
| Cross-model restored verifier (MLX / Apple Silicon) | `inference_engine/backends/mlx/cross_model_dlm_verifier.py` |
| Incremental restored verifier (`SinkWindowVerifier` API) | `inference_engine/v04/restored_sink_window_verifier.py` |
| Served-path factories + gRPC `--backend restored` | `inference_engine/v04/build_restored.py`, `scripts/start_grpc_runtime_server.py` |

## Three engines (decode modes)

* **Re-forward** (`incremental=False`) — memory-optimal, eval-grade; recomputes
  restoration each step (O(T)/step). Bit-equivalent reference for the gate.
* **Gap-A incremental** (`incremental=True`) — capture restored K/V into a
  `DynamicCache` at prefill, decode natively (O(L)/block). **= AR decode speed**,
  KV 16.9×–43.9× smaller, recall 1.0.
* **Fused spec-decode** (`restored_specdecode_fused`) — DFlash block draft +
  incremental verify, with three prefill-built, incrementally-extended caches:
  (A) verifier aux hidden captured from the verify forward, (B) drafter context
  K/V cache, (C) Gap-A restored KV. Per-block O(L). **> AR** (see below).

## Validated results (H200, ctx 1238, gemma-4-26B-A4B)

| path | decode tok/s | vs AR | recall |
|---|---|---|---|
| standalone AR | 21.1 | 1.0× | 1.0 |
| Gap-A incremental restored | 21.7 | 1.03× | 1.0 |
| fused DFlash spec-decode (aggregate) | 26.8 | **1.27×** | 1.0 |

KV memory: restored resident KV constant **16.71 MB** vs AR 282 MB @1238 tok →
733 MB @3238 tok (**16.9× → 43.9×**, grows with context). DFlash acceptance on
HumanEval ≈ official gemma-4-26B parity (length ~3.9 ≈ official 3.3× speedup).

## Run

```bash
# Incremental restored decode vs AR (memory + tok/s + recall)
PYTHONPATH=.:sdks/python python scripts/research/k3_e2e_gpu_bench.py \
  --verifier-id google/gemma-4-26B-A4B-it \
  --drafter-id z-lab/gemma-4-26B-A4B-it-DFlash \
  --f-theta-dir results/research/f_theta_v5_s5_sliding \
  --incremental --haystack-lines 60,160

# Fused DFlash spec-decode vs AR
PYTHONPATH=.:sdks/python python scripts/research/k3_specdecode_gpu_bench.py \
  --drafter-id z-lab/gemma-4-26B-A4B-it-DFlash --skip-unfused

# gRPC server with the restored backend
PYTHONPATH=.:sdks/python python scripts/start_grpc_runtime_server.py \
  --backend restored --device cuda \
  --verifier-id google/gemma-4-26B-A4B-it \
  --drafter-id z-lab/gemma-4-26B-A4B-it-DFlash \
  --f-theta-dir results/research/f_theta_v5_s5_sliding --sink 4 --window 64
```

## Canonical proposer

The proposer/drafter is **`z-lab/gemma-4-26B-A4B-it-DFlash`** (the official
checkpoint, with the Gap-B embed-scale fix) — used uniformly for both drafting
and as the f_θ restoration K/V source across all entry points. The earlier
`models/dflash-kakeya-baseline` was alignment-trained against a buggy
(`×sqrt(hidden)`-scaled) embed pipeline and is not the beta drafter.

f_θ v5 was trained against the kakeya-baseline drafter, so its **sliding-layer**
restoration is technically off for z-lab K/V — but this is **harmless for
recall**: recall is carried by the S5 exact full-attention layers, and the
sliding-layer restored K/V are window-masked during decode. Both incremental
decode and fused spec-decode measure **recall 1.0** with z-lab. (If pure
sliding-layer restoration is ever needed, retrain f_θ on z-lab K/V.)

All **inference/eval** entry points default to z-lab (`k3_e2e_gpu_bench`,
`k3_specdecode_gpu_bench`, `k3_integrated_niah_eval`(+`_mac`),
`k3_dflash_specdecode_eval`(+`_mac`); the gRPC server takes an explicit
`--drafter-id`). The **f_θ training** script (`k3_f_theta_train.py`) and its
orchestration `.sh` keep `models/dflash-kakeya-baseline` because that is how the
shipped v5 checkpoint was historically trained.

## Notes / scope

* Drafting conditions on the restored verifier hidden for committed decode tokens
  (clean aux for the prompt) — resolves the bounded-KV vs clean-aux tension
  natively; no SGLang/vLLM dependency.
* Stable decode requires loading the verifier without `device_map` (no accelerate
  per-forward hooks; the 26B-A4B fits on one H200) + a full-length warmup.
* f_θ v5 restores the sliding layers; recall is carried by the S5 exact
  full-attention layers, so f_θ fidelity is not the recall bottleneck.
