# Local Inference Engine — Architecture (Mac + Ubuntu)

This document is the design for a local inference engine that wraps the
DLM-proposer + AR-verifier framework (already implemented under
`kv_cache_proposer/`) into a production-grade serving stack with two
explicit goals:

1. **Extreme memory savings** — run Qwen3-1.7B verifier + Qwen3-0.6B-MDLM
   proposer at S=128k context in **< 2 GB total memory** on consumer
   hardware.
2. **Extreme token throughput** — ≥ 150 tok/s single-request and ≥ 500
   tok/s aggregate on an M3 Max; ≥ 400 / ≥ 1500 on an RTX 4090.

## 0. Why we are *not* using PagedAttention

PagedAttention solves three problems that arise from "KV cache is an
unbounded, growing object of unpredictable size":

| PagedAttention capability | Problem it solves |
| ------------------------- | ----------------- |
| Fixed 16-token pages, on-demand allocation | KV cache **fragmentation** across many sessions |
| Copy-on-write page sharing                 | Multiple sessions can **reuse a shared prefix** |
| Block-table indirect addressing            | KV may live **non-contiguously** in physical memory |

In our framework, the sink+window invariant means **every session's KV
cache is a constant-size object** (`(sink + window) × per-token-bytes`),
e.g., 14.8 MB / session at NF4 quantization. None of the three problems
above remain:

* No fragmentation, because every slab is the same size.
* No prefix sharing, because the prefix beyond the sink (4 tokens) is
  *evicted by design*.
* No non-contiguity, because surviving KV is two contiguous segments
  (sink + window), which standard FlashAttention handles natively.

So instead of paged KV we use a **fixed-size slab pool**:

```python
class SinkWindowKVPool:
    def __init__(self, n_max_sessions: int, sink: int, window: int, model_cfg):
        slab = torch.empty(
            n_max_sessions, sink + window, ...,  # one big preallocation
            dtype=torch.uint8,                    # NF4-packed
        )
        self.buffer = slab
        self.free_list = list(range(n_max_sessions))
    def acquire(self) -> int:    return self.free_list.pop()      # O(1)
    def release(self, sid: int) -> None: self.free_list.append(sid)  # O(1)
    def view(self, sid: int):    return self.buffer[sid]          # zero-copy
```

This is ~30 lines, has zero metadata overhead, and lets attention kernels
operate on contiguous memory (no block-table gather → ~5–15% faster than
PagedAttention's indirect path on the same hardware).

## 1. Layered architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  L7 Public API           OpenAI-compat HTTP / gRPC / CLI / SDK    │
├──────────────────────────────────────────────────────────────────┤
│  L6 Scheduler            Continuous batching + priority queue +  │
│                          cancellation, mid-stream tool injection │
├──────────────────────────────────────────────────────────────────┤
│  L5 Speculative engine   Proposer worker  ┐                      │
│                          Verifier worker  │ async pipelined      │
│                          Acceptance loop  ┘ (tree-spec capable)  │
├──────────────────────────────────────────────────────────────────┤
│  L4 Memory subsystem     Fixed-slab KV pool (sink+window)         │
│                          Quantized weight loader (4-bit AWQ/Q4)   │
│                          NF4 KV quantizer                         │
│                          Activation recycler                      │
├──────────────────────────────────────────────────────────────────┤
│  L3 Backend abstraction  Mac:   MLX                               │
│                          Linux: PyTorch + CUDA (Flash-Attn 3,     │
│                                 Marlin 4-bit GEMM)                │
├──────────────────────────────────────────────────────────────────┤
│  L2 Op library           matmul / RoPE / GQA-attn / RMSNorm /     │
│                          softmax / sampling                       │
├──────────────────────────────────────────────────────────────────┤
│  L1 Tensor runtime       MLX (Mac) | torch+cuda (Linux) |         │
│                          torch+cpu (last-resort)                  │
└──────────────────────────────────────────────────────────────────┘
```

Layers L0–L3 split per platform; L4–L7 are shared code.

## 2. Backend choice per platform

### Mac (Apple Silicon, M1–M4)

**Primary backend: MLX**. Reasons:

* **Unified memory** (up to 192 GB on M2/M3 Ultra) eliminates host↔device
  copies — the single biggest performance win on Apple Silicon.
* **Lazy graph compilation + kernel fusion** out of the box; ≈ 2–3× faster
  than PyTorch MPS for our workload.
* **Native 4-bit quantization** (`mlx.nn.quantize`) with fused Q4 GEMM
  kernels, no manual marlin-equivalent needed.

Fallback: `llama.cpp` Metal backend for compatibility (gguf format), at
~70% of MLX speed and harder to integrate custom speculative kernels.

### Ubuntu / Linux

**Primary backend: PyTorch + CUDA 13.x**, borrowing — but not depending
on the engine of — vLLM/SGLang component kernels:

* **Flash-Attention 3** for attention.
* **Marlin** kernel for 4-bit AWQ GEMM.
* **NVIDIA cuda-python** bindings for stream / graph control.

We use these as algorithm components, not as the engine, because their
schedulers don't accommodate a DLM proposer in the speculative loop.

### CPU (last-resort)

Keep the existing `torch+cpu` path for environments without GPU. Same
correctness, lower throughput. The current demo runs here.

## 3. L4 memory subsystem (the "extreme savings" core)

### 3.1 Memory budget on a 16 GB Mac M-series / 24 GB RTX

For Qwen3-1.7B verifier + Qwen3-0.6B-MDLM proposer, S=128k context,
single concurrent session:

| Component                              | Naive bf16 | Optimized                    |
| -------------------------------------- | ---------- | ---------------------------- |
| Verifier weights                       | 3.40 GB    | **0.85 GB** (NF4 4-bit)      |
| Proposer weights                       | 1.50 GB    | **0.38 GB** (NF4 4-bit)      |
| Verifier KV cache (full)               | 14.7 GB    | **14.8 MB** (sink+W=512, NF4)|
| Proposer persistent KV                 | n/a        | 0 (recomputed per block)     |
| Activation peak (sparse-logits)        | n/a        | 5 MB                         |
| Runtime buffers (attn scratch, sample) | ~200 MB    | ~200 MB                      |
| **Total**                              | **~19.6 GB** | **~1.43 GB**               |

≈ 13× total compression. Runs comfortably on 8 GB systems.

### 3.2 Required techniques (in priority order)

1. **Fixed-slab KV pool** (above). 30 LoC. No paging.
2. **Sink+window invariant** (already implemented in `verifier.py`). Per
   session, KV is bounded to `sink + window` slots forever.
3. **NF4 KV quantization** at the slab level. Each slot stores K,V as 4-bit
   with per-block scale (group=64 elements). Read-time dequant inside
   FlashAttention via custom kernel; ~5% slower attention vs bf16, 4× memory.
4. **AWQ 4-bit weight quantization** for both proposer and verifier. Use
   AutoAWQ on Linux, `mlx_lm.convert -q --bits 4` on Mac.
5. **Embedding tying** (already in Qwen3). Saves ~600 MB for the 152k vocab.
6. **Sparse-logits proposer optimization** (TODO). Compute logits only at
   masked positions during diffusion → bounds activation at `L_block × V`
   regardless of context length. Required for S>32k.
7. **Optional: weight streaming** for very large verifiers (Qwen3-32B+) on
   memory-constrained Macs. Stream layer weights from disk, hot layers
   pinned in unified memory.

## 4. L5 / L6 throughput subsystem

### 4.1 Pipeline diagram

```
                Async Pipeline (one tick per block)
   ┌────────────────────────────────────────────────────────────┐
   │ tick 1     tick 2      tick 3      tick 4                   │
   │                                                              │
   │ Proposer:  block_1     block_2     block_3                  │
   │            (K=16 dif)  (K=16 dif)  (K=16 dif)               │
   │                  │            │            │                 │
   │ Verifier:        └─verify─┐   └─verify─┐   └─verify─┐        │
   │                           │            │            │        │
   │                  +append  │   +append  │   +append  │        │
   │                           │            │            │        │
   │ Stream:        tokens_1   │   tokens_2 │   tokens_3 │  …     │
   └────────────────────────────────────────────────────────────┘
```

K=16 diffusion forwards on a 0.6B proposer ≈ 1 forward of a 1.7B verifier.
With async dispatch the two halves overlap ~70% of the time.

### 4.2 Required techniques

| Technique                  | Mechanism                                              | Win                              |
| -------------------------- | ------------------------------------------------------ | -------------------------------- |
| Continuous batching        | Dynamically add/remove requests in the active batch    | aggregate throughput +3–5×       |
| Tree speculative decoding  | Proposer emits top-k tree, verifier checks in one pass | acceptance +30–50%               |
| Async proposer/verifier    | Two CUDA streams (or MLX command queues) with a queue  | wall-clock −20–35%               |
| CUDA Graph / MLX compile   | Static graph capture eliminates kernel launch overhead | small-batch +30–50%              |
| Flash-Attention 3 / MLX-FA | Fused attention compute                                | +20–40%                          |
| Speculative streaming      | Stream optimistic tokens before verifier confirms      | TTFT −60% (UI may show fix-ups)  |
| Chunked prefill            | Long prompt prefill chunked into 4k pieces             | TTFT −50% on long prompts        |

## 5. Speculative engine integration details (L5)

What changes from the current `speculative.py` to make it production-grade:

1. **Two stream / queue model.** Proposer and verifier run on separate
   compute streams. A lock-free queue holds proposed blocks; verifier
   pulls and checks; result pushed to a streaming output queue.
2. **Optimistic KV state.** Maintain `(committed_kv, optimistic_kv)`. The
   proposer keeps generating block N+1 against `optimistic_kv` while the
   verifier finalizes block N. On rejection, drop the optimistic side and
   restart block N+1.
3. **Tree-spec extension.** DLM already produces per-position softmax; turn
   the linear chain `d[0..L-1]` into a top-k tree. Verifier-side mask is a
   path-compatible causal mask; the longest accepted root-to-leaf path is
   committed.
4. **Per-session sink+window slab.** From the L4 pool. Proposer and verifier
   address the same slab via the slab id.
5. **Session resume / persistence.** Slab can be `to('cpu')` / serialized
   to disk for multi-day conversations.

## 6. Recommended technology stack

### Mac
```
Public API     custom OpenAI-compat (Swift app or Python aiohttp + uvloop)
Scheduler      Python asyncio (Mac doesn't need extreme QPS)
Spec engine    kv_cache_proposer/ ported to MLX
Memory         MLX 4-bit weights + custom NF4 KV quantizer + fixed-slab pool
Backend        mlx >= 0.18 + bits of mlx_lm (model definitions, AWQ converter)
Quant tools    mlx_lm.convert -q --bits 4
Distribution   Signed .dmg via PyInstaller, or native Swift Mac app bundle
```

### Linux
```
Public API     FastAPI + uvicorn (or Rust axum if we want zero-overhead server)
Scheduler      Custom continuous batcher
Spec engine    kv_cache_proposer/ + tree-spec + async pipeline
Memory         AutoAWQ weights + NF4 KV + fixed-slab pool
Backend        PyTorch 2.8+cuda + Flash-Attention 3 + Marlin GEMM
Quant tools    AutoAWQ for one-shot weight quant
Distribution   Docker image + apt deb
```

### Shared code (≈70% of the codebase)
```
core/
├── speculative.py      already exists, cross-platform
├── proposer.py         already exists
├── verifier.py         already exists, needs slab-pool integration
├── tree_spec.py        new: tree speculative decoding
├── memory/
│   ├── slab_pool.py    new, ~30 LoC
│   ├── nf4_kv.py       new: NF4 KV quant/dequant
│   └── awq_loader.py   new
├── scheduler/          platform-neutral
└── server/             OpenAI-compat
backend/
├── mlx_backend/        Mac-specific kernels + model wiring
└── cuda_backend/       Linux-specific
```

## 7. Phased build plan

### P0 — single-session walking skeleton (~2–3 weeks, 1 engineer)

* MLX backend port of `kv_cache_proposer/`
* AWQ 4-bit weight loading (verifier)
* Fixed-slab KV pool with NF4 quantization
* Sparse-logits proposer optimization
* Smoke + equivalence tests pass on Mac M-series

**Acceptance**: M2 Mac runs single-request Qwen3-1.7B verifier under 2 GB
total memory, output bit-equal to baseline.

### P1 — multi-session + high throughput (~2 months)

* Continuous-batching scheduler
* Async proposer/verifier pipeline (two streams + lock-free queue)
* Tree speculative decoding
* CUDA Graph / MLX compile capture
* Tree-mask FlashAttention integration

**Acceptance**: M3 Max ≥ 150 tok/s single, ≥ 500 tok/s aggregate; RTX
4090 ≥ 400 / ≥ 1500.

### P2 — productization (~1 month)

* OpenAI-compat HTTP API + streaming SSE
* Session persistence (slab → CPU/disk between turns)
* Configuration + observability dashboard
* Signed .dmg / .deb packages

### P3 — polish (~ongoing)

* Speculative streaming (mid-block UI updates)
* Cross-CPU/GPU offload for huge verifiers
* Online RL fine-tuning of proposer (acceptance 0.3 → 0.7)
* Per-session dynamic block-size controller

## 8. Quantitative success criteria

| Metric                          | M3 Max (96 GB unified) | RTX 4090 (24 GB) | Pi 5 (8 GB) |
| ------------------------------- | ---------------------- | ---------------- | ----------- |
| Resident memory (1.7B + 0.6B + 128k ctx) | < 2 GB                 | < 2 GB           | < 1.5 GB (more aggressive quant) |
| Single-request tok/s            | ≥ 150                  | ≥ 400            | ≥ 8         |
| Aggregate tok/s (acc=0.6 spec)  | ≥ 500                  | ≥ 1500           | ≥ 20        |
| TTFT (S=2k prompt)              | < 200 ms               | < 80 ms          | < 2 s       |
| Acceptance rate (Repr-Align proposer) | 0.6–0.85               | 0.6–0.85         | 0.6–0.85    |

## 9. Out of scope for this engine

* Multi-target verifier routing (Qwen / Gemma / DeepSeek). Same-engine
  binary supports only one verifier family at a time; multi-target is a
  cluster-level orchestration concern.
* Federated / OTA continual learning of proposer. The local engine consumes
  proposer checkpoints; producing them is a separate training pipeline.
* Session-affinity scheduling across machines. Local engine is single
  node; affinity is meaningful only in cloud deployments.
