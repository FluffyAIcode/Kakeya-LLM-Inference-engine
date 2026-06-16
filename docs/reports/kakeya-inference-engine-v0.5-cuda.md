# Kakeya Inference Engine v0.5 for CUDA — release scorecard

**What v0.5-cuda ships:** Kakeya Attention's bounded-window (S5) KV management
running **on the vLLM runtime** — so the three runtime components the engine
needs are inherited unchanged (all Apache-2.0):

| component | owner in v0.5-cuda | role |
| --- | --- | --- |
| **Fused MoE Triton kernel** | **vLLM** (Apache-2.0) | grouped-GEMM expert kernel — the dominant ~90 % of the gemma-4-26B-A4B decode forward |
| **CUDA graphs** | **vLLM** (Apache-2.0) | fixed-shape decode capture (`enforce_eager=False`) — removes per-token launch overhead |
| **Continuous-batching scheduler** | **vLLM** (Apache-2.0) | request scheduler + paged KV-manager — drives multi-tenant throughput |
| **Kakeya Attention (bounded window / KV)** | **Kakeya** | bounds the resident sliding-layer KV to `sink + window` (S5 = 68) |

Entrypoint: `inference_engine.engine.KakeyaVLLM` (see *Usage* below). This is the
**KIE-v2** strategy from [ADR 0015](../adr/0015-kakeya-attention-and-engine-substrate.md)
and the [feasibility note](../design/kakeya-vllm-backend-feasibility.md): rather
than rebuild vLLM's fused-MoE + graphs + scheduler (a multi-component kernel
project that was shown blocked in KIE-v1.1.z), Kakeya Attention runs *on* vLLM
and contributes the bounded-KV attention layer.

> **Honest scope.** v0.5-cuda is the **gemma-4 bounded-window** instantiation:
> gemma-4's hybrid (5 full + 25 sliding) needs **no per-token restoration** (the
> S5 "free lunch"), so the bounded window is delivered via vLLM `hf_overrides`.
> The **restoration backend** (f_θ + dLLM-proposer at prefill) for **full-attention**
> models (Qwen/Llama) — the large ~6× memory differentiator — is the **v0.6**
> roadmap item, not in this release.

---

## 1. Token throughput — decode tok/s ≥ vLLM (H200, gemma-4-26B-A4B, recall 1.0)

Kakeya bounded window (`sliding_window=68`) **on vLLM** vs vLLM default
(`sliding_window=1024`), same H200, ctx 16k, gen 128, recall **1.0** throughout:

| N (concurrent) | **Kakeya-on-vLLM** decode tok/s | vLLM default | ratio |
| --- | --- | --- | --- |
| 1 | **195.6** | 159.3 | **1.23×** |
| 4 | **231.9** | 198.6 | 1.17× |
| 8 | **539.0** | 467.5 | 1.15× |
| 70 | **1079.0** | 894.9 | **1.21×** |

**Decode throughput exceeds vLLM by ~1.15–1.23×** at recall 1.0 — it inherits
vLLM's fused-MoE + CUDA-graph + scheduler, and Kakeya's tighter sliding window
makes the sliding-layer attention cheaper than vLLM's default. This is the axis
the eager research engine (KIE-v1.1.y/z, ~25–31 tok/s aggregate) could not reach;
on vLLM it is 195–1079 tok/s.

**Long context (ctx 62k, N=70) — the edge shrinks on gemma-4:**

| metric (62k, N=70) | Kakeya-on-vLLM sw=68 | vLLM default sw=1024 | ratio |
| --- | --- | --- | --- |
| decode tok/s | 20.38 | 19.03 | 1.07× |
| vLLM max concurrency (66k req) | 16.15× | 15.51× | 1.04× |

The edge falls from ~1.2× (16k) to ~1.07× (62k) because gemma-4's **5
full-attention layers hold full-ctx KV in both configs** and dominate the
footprint — shrinking the sliding window is only a ~4–7 % saving. This is the
gemma-4 caveat (see §3); the large win needs a full-attention model.

## 2. Parallel inference (concurrency)

- **On vLLM (v0.5-cuda path):** the bounded window is measured to **N=70**
  concurrent sessions at ctx 16k, recall 1.0, while *increasing* decode tok/s vs
  vLLM default (§1) — vLLM's continuous-batching scheduler + the smaller resident
  window scale cleanly.
- **Research engine ceiling (KIE-v1.1.y/z, eager, demonstrator):** the bounded-KV
  + int8 + quantized-attention path reached **N=75 @ 62k, recall 1.0** (≈4.8×
  vLLM's 15.5 concurrency ceiling) — proving the *memory/concurrency* axis even
  before moving to the vLLM runtime. Decode speed on that eager path was the weak
  axis (~31 tok/s aggregate), which is exactly what running on vLLM fixes.

So v0.5-cuda gets **both** axes: vLLM's decode speed **and** the bounded-window
concurrency, with recall 1.0.

## 3. Memory-saving efficiency (honest, model-dependent)

The bounded-KV win scales with the model's **full-attention fraction**:

| model class | full-attn layers | Kakeya resident-KV edge vs vLLM | shipped in |
| --- | --- | --- | --- |
| **gemma-4-26B-A4B** (hybrid 5 full / 25 sliding) | 5 / 30 | **~7 % @ 62k** (vLLM already hybrid-bounds 25/30; the 5 full layers dominate both) | **v0.5-cuda** |
| **full-attention** (Qwen/Llama, all layers full) | all | **~6×** (vLLM keeps all full; Kakeya keeps exact + window + restoration) | **v0.6** (restoration backend) |

- The bounded-KV cost model (`inference_engine.engine.admission`, 9 unit tests):
  per-session resident **2.56 GB @ 62k** vs full-KV **15.2 GB** — a **~6×** edge —
  but that edge is only realized end-to-end on a **full-attention** model. On
  gemma-4 the native hybrid means vLLM already bounds the sliding layers, so the
  *measured* long-context saving over vLLM is the honest **~7 %**.
- **Takeaway:** v0.5-cuda's memory win on gemma-4 is modest-but-real; the engine's
  large memory differentiation is a **full-attention-model** property delivered by
  the v0.6 restoration backend. We report this rather than overclaim gemma-4.

## 4. Usage

```python
from inference_engine.engine import KakeyaVLLM
from vllm import SamplingParams

# Kakeya Attention (S5 bounded window) on vLLM's fused-MoE + CUDA-graph + scheduler.
engine = KakeyaVLLM(
    "google/gemma-4-26b-a4b-it",
    sink=4, window=64,        # Kakeya S5 window (total resident = 68)
    max_model_len=16384,
)
out = engine.generate(prompts, SamplingParams(temperature=0.0, max_tokens=128))
```

`KakeyaVLLM` builds a `vllm.LLM` with `hf_overrides={"sliding_window": 68,
"text_config": {"sliding_window": 68}}` and `enforce_eager=False` (CUDA graphs +
fused-MoE on). The pure config layer (`kakeya_hf_overrides`, `KakeyaVLLMConfig`)
is torch/vllm-free and unit-tested (`tests/inference_engine/engine/test_kakeya_vllm.py`).

## 5. Verification status

- ✅ **Throughput / concurrency / recall numbers** (gemma-4-26B) above were measured
  on H200 (Vast.ai) and committed in the KIE-v2 integration
  (`scripts/research/vllm_multitenant_parallel_bench.py --sliding-window 68`;
  commits `7ec3a03`, `48ded1e`, `e2cf137`).
- ✅ **`KakeyaVLLM` entrypoint validated end-to-end on H200** (vLLM 0.23.0): it builds
  the vLLM engine with **CUDA graphs captured** (PIECEWISE + FULL), the Kakeya window
  (68) **reaches vLLM's model config** (`hf_config.sliding_window == 68`), and
  generation is coherent (Paris / 2+2=4 / story) at **777 tok/s** (batch 3). The
  validation used **Qwen/Qwen3-4B** — the 26B model does not fit this box's 4 GB free
  disk, so the wrapper *mechanism* is validated here and the 26B *performance* is the
  committed measured path above. Evidence:
  `kakeya_vllm_v05_h200_validation.log`.
- ✅ The wrapper auto-detects `text_config` nesting (multimodal gemma-4 → nested;
  text-only Qwen/Llama → flat), fixing a crash where unconditional `text_config`
  injection broke text-only models. Config layer unit-tested (13 tests).

## 6. What's next (v0.6)

The **restoration backend** for full-attention models (Qwen/Llama): inject f_θ +
dLLM-proposer restoration at vLLM prefill and a graph-capturable quantized-exact
attention kernel, to realize the **~6×** resident-KV edge end-to-end on vLLM. That
is the genuine custom-backend work (ADR 0015 §KIE-v2 caveats); v0.5-cuda proves
the throughput axis and ships the gemma-4 bounded-window engine.

## Evidence

- Throughput tables: [`docs/design/kakeya-vllm-backend-feasibility.md`](../design/kakeya-vllm-backend-feasibility.md) §5b
- Concurrency / memory journey: [`docs/reports/kakeya-engine-vs-vllm-h200.md`](kakeya-engine-vs-vllm-h200.md)
- Architecture / milestones: [`docs/adr/0015-kakeya-attention-and-engine-substrate.md`](../adr/0015-kakeya-attention-and-engine-substrate.md)
- Entrypoint: `inference_engine/engine/kakeya_vllm.py`; tests: `tests/inference_engine/engine/test_kakeya_vllm.py`
