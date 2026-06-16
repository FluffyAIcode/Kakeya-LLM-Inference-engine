# Kakeya restored-S5 vs vLLM — multi-tenant parallel decode (same H200, gemma-4-26B-A4B)

Apples-to-apples comparison of Kakeya's recall-preserving **restored-S5** batched
multi-tenant decode against **vLLM** (PagedAttention, the production baseline),
run back-to-back on the **same H200** so there is zero cross-host variance.

## Methodology

| Axis | Setting (identical for both engines) |
| --- | --- |
| GPU | 1× NVIDIA H200 (~140 GB), `vastgpu4` |
| Model | `google/gemma-4-26B-A4B-it`, **bf16** (the precision the Kakeya bench loads) |
| Workload | NIAH, `make_niah_dataset(haystack_lines=60, seed=0)` → modal prompt **1238 tokens** (= the quoted ctx≈1238) |
| Prompts | the modal-length bucket tiled to N, fed as **identical token-ids** to both engines |
| Concurrency | N = 1, 2, 4, 8 sessions decoded in parallel |
| Decode | greedy (temperature 0), **gen=128** (primary; stable steady-state rate) and gen=24 (matches the original table) |
| Recall | answer needle substring in the decoded text |
| Kakeya | `scripts/research/k3_cuda_multitenant_parallel_bench.py` — batched restored-S5 (per-session KV row), eager HF transformers decode loop |
| vLLM | `scripts/research/vllm_multitenant_parallel_bench.py` — `vllm==0.23.0`, continuous batching; decode tok/s from per-request metrics (prefill excluded), matching the Kakeya decode-loop timing |

## Results (gen=128, ctx 1238, recall 1.0 throughout)

| N | Kakeya restored-S5 tok/s | (×N=1) | vLLM tok/s | (×N=1) | **vLLM / Kakeya** | Kakeya recall | vLLM recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.2 | 1.00× | 224.8 | 1.00× | **13.9×** | 1.0 | 1.0 |
| 2 | 32.0 | 1.98× | 364.8 | 1.62× | **11.4×** | 1.0 | 1.0 |
| 4 | 65.9 | 4.08× | 579.9 | 2.58× | **8.8×** | 1.0 | 1.0 |
| 8 | 131.7 | **8.15×** | 1048.0 | 4.66× | **8.0×** | 1.0 | 1.0 |

(gen=24 reproduces the same picture: Kakeya 17.9/35.6/71.0/110.7 tok/s, peaks
57.7/60.5/66.2/77.5 GB — see below; vLLM's gen=24 decode rate is noisy because
24 tokens is too short to measure a steady decode window, hence gen=128 is the
primary comparison.)

## Memory

| N | Kakeya restored-S5 peak GPU | vLLM |
| --- | --- | --- |
| 1 | 57.7 GB | model ~52 GB + **static KV pool** |
| 2 | 60.6 GB | reserves `gpu_memory_utilization=0.9` ≈ **126.6 GB** |
| 4 | 66.2 GB | (KV pool = **71.1 GiB → 337,590 tokens**, 82× concurrency for 4096-len) |
| 8 | 77.6 GB | independent of N (pool reserved up front) |

- **Kakeya restored-S5** allocates per batch and grows modestly with N
  (57.7 → 77.6 GB). Per-session KV is **bounded**: sink+window on the sliding
  layers + 5 exact full-attention layers (S5). This reproduces the quoted table
  near-exactly (57.6/60.4/66.0/77.3 GB), confirming identical model/precision/ctx.
- **vLLM** is full-KV PagedAttention: it reserves a **static KV pool** (default
  90% util ≈ 126.6 GB measured peak) sized for 337k tokens, and per-token KV
  covers **all layers/positions** — so its KV footprint grows with total tokens
  and is **unbounded in context length**, the opposite of Kakeya's bounded S5.

## Findings (honest)

1. **vLLM is ~8–14× faster in absolute decode throughput.** This is a
   **kernel/implementation** gap — vLLM ships optimized paged-attention, CUDA
   graphs, and fused-MoE kernels; the Kakeya restored bench runs an **eager HF
   transformers decode loop** (`attn_implementation="eager"`, per-step Python
   `model()` calls). It is **not** an algorithmic property of restored-S5.
2. **Kakeya scales more linearly** (8.15× at N=8 vs vLLM's 4.66×). The metric
   favors Kakeya only because its per-request rate is low (lots of batching
   headroom), while vLLM's per-request rate is already high (less headroom) — so
   parallel **speedup** and absolute **throughput** point in opposite
   directions, and absolute throughput is what ships.
3. **Both preserve recall 1.0** at ctx 1238.
4. **Different value axes.** vLLM wins raw throughput; Kakeya restored-S5's claim
   is **bounded per-session KV + recall restoration**, whose payoff is memory at
   long context / many tenants, not raw speed in this eager implementation.
   Closing the absolute-throughput gap would require porting restored-S5 onto
   optimized kernels — the bounded-KV + recall contribution is orthogonal to
   kernel optimization.

## Cross-host note

This H200 is slower (and likely more contended) than the host that produced the
originally-quoted table: Kakeya N=1 here is 16.2 tok/s vs the quoted 27.4, while
**peak memory reproduces the quoted table almost exactly**. Because both engines
were measured on the **same** host, the transferable result is the **ratio**
(vLLM ≈ 8–14× the Kakeya restored-S5 decode throughput at equal recall), not the
absolute tok/s.

## Evidence

- `results/research/k3_cuda_multitenant_parallel_h200nvl_ctx1238_gen128.json`
- `results/research/vllm_multitenant_parallel_h200nvl_ctx1238_gen128.json`
- `results/research/k3_cuda_multitenant_parallel_h200nvl_ctx1238.json` (gen=24)
- `results/research/vllm_multitenant_parallel_h200nvl_ctx1238.json` (gen=24)
