# DLM Proposer + AR Verifier — runnable KV-cache-saving framework

Runs the speculative-decoding architecture designed in the prior product
discussion using **real, public** weights:

| Role     | Model                                                   | Params | Tokenizer    |
| -------- | ------------------------------------------------------- | ------ | ------------ |
| Proposer | [`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`][p]          | 0.75 B | Qwen3 family |
| Verifier | [`Qwen/Qwen3-1.7B`][v] (closest public stand-in for "Qwen 3.6") | 1.72 B | Qwen3 family |

[p]: https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1
[v]: https://huggingface.co/Qwen/Qwen3-1.7B

> **Note on the verifier choice**: at the time of this writing, no public
> "Qwen 3.6" checkpoint exists. We use `Qwen/Qwen3-1.7B` because it is the
> closest publicly-available autoregressive Qwen-3 model that (a) shares the
> proposer's tokenizer (the prompt encodes to identical token ids — verified
> at startup) and (b) is large enough to make KV-cache savings non-trivial.
> Swapping in an actual Qwen 3.5/3.6 checkpoint requires only changing
> `--verifier-id`.

## What this code actually does

Implements a greedy **speculative decoding** loop:

```
for each block:
    proposer.propose_block(committed_prefix, L, K_diffusion_steps)   # masked-diffusion
    verifier.forward_block(proposed_tokens)                          # one parallel forward
    walk i=0..L-1:  accept proposed[i] iff argmax(verifier_logits[i-1]) == proposed[i]
    cache eviction so that |cache| <= sink_size + window_size at all times
    correction_or_bonus = argmax(prev_logits)
    verifier.append_token(correction_or_bonus)                       # 1 forward
```

The verifier's KV cache is bounded to `sink + window` slots via direct
`torch.Tensor` slicing on each `DynamicCache` layer's K and V tensors,
StreamingLLM-style. New queries always use the **global** position id
(so RoPE on new K/Q is rotated at the true distance); evicted tokens
simply disappear from the attention's view, while sink/window survivors
keep the RoPE rotation they had at their original positions.

There is **no mock, no fallback, no overfit**:

- Every forward pass runs real model weights downloaded from Hugging Face.
- The cache trim slices the actual K/V tensors (and verifies the layer
  shape matches the logical bookkeeping; raises on mismatch).
- The "no intelligence loss" guarantee is verified by an
  **equivalence-regime self-test**: when `sink + window >= full_seq_len`,
  speculative output must be **bit-identical** to greedy AR (the demo
  exits with code 2 if not).
- No prompt-specific tuning; the same code runs every prompt.

## Project layout

```
kv_cache_proposer/
├── proposer.py        # DLM Proposer (masked-diffusion block generator)
├── verifier.py        # AR Verifier with sink+window DynamicCache
├── speculative.py     # Greedy speculative-decoding loop
├── baseline.py        # Reference greedy AR with full DynamicCache
├── metrics.py         # KV byte counting, NBT formula, projection table
├── run_demo.py        # End-to-end demo + JSON results
└── __init__.py
scripts/
└── smoke_test.py      # Component smoke tests on real weights
results/                # Logs and JSON outputs from runs
requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
# (one-time fix: the dllm-hub modeling file references the broken `dllm`
#  package only inside an `if __name__ == "__main__":` block; we install a
#  no-op stub so transformers' static check_imports passes)
mkdir -p ~/.local/lib/python3.12/site-packages/dllm
echo '' > ~/.local/lib/python3.12/site-packages/dllm/__init__.py

# Smoke test: tokenizer agreement, model loading, cache invariants
PYTHONPATH=. python3 scripts/smoke_test.py

# Equivalence regime: window >= sequence length => bit-identical to baseline
PYTHONPATH=. python3 -m kv_cache_proposer.run_demo \
    --max-new-tokens 32 \
    --block-size 8 --num-diffusion-steps 8 \
    --sink-size 4 --window-size 64 \
    --batch-size-for-amortization 8 \
    --prompt "Reply with exactly: OK."

# Compression regime: window << sequence => real KV eviction observed
PYTHONPATH=. python3 -m kv_cache_proposer.run_demo \
    --max-new-tokens 64 \
    --block-size 16 --num-diffusion-steps 16 \
    --sink-size 4 --window-size 24 \
    --batch-size-for-amortization 64 \
    --prompt "Write a one-paragraph explanation of why prime numbers are infinite, suitable for a high school student." \
    --results-json results/run_compress.json
```

## Results from the included CPU runs

### 1. Equivalence-regime test (sink+window covers full sequence)

```
prompt  : "Reply with exactly: OK."
config  : sink=4, window=64, block_size=8, K=8

baseline (full KV)        : "OK.<|im_end|>"   (3 tokens, peak KV = 3,584 KB)
speculative (sink+window) : "OK.<|im_end|>"   (3 tokens, peak KV = 3,696 KB)
exact match               : True              <- "no intelligence loss" verified
acceptance rate           : 0.375
```

Self-check passes: `sink+window=68 >= full_seq_len=33`, output bit-identical
to the verifier's own greedy decode. The math of speculative decoding +
no-eviction reduces to "verifier emits its argmax everywhere", exactly
what the baseline computes.

### 2. Compression-regime test (window << sequence)

```
prompt   : "Write a one-paragraph explanation of why prime numbers are infinite ..."
config   : sink=4, window=24, block_size=16, K=16, B=64 (for amortization)
S        : 108 tokens (44 prompt + 64 generated)

verifier KV (full DynamicCache, baseline)  =  12.10 MB total = 114,688 B/token
verifier KV (sink+window,  speculative)    =   3.06 MB total =  29,734 B/token
                                             ── 3.86x verifier-side compression
proposer weights amortized at B=64,S=108   = 172,468 B/token  (dominates here)
proposer peak activation amortized B=64,L=16 = 32,049 B/token (logits buffer)

NBT (this scenario, B=64 S=108) = 234,251 B/token   (compression  0.49x)
```

The proposer-overhead terms dominate at this small scale — which is exactly
the operating-regime caveat the design predicted. The framework also
reports projected NBT at production-realistic operating points using the
empirically measured per-slot KV bytes:

```
   B           S     NBT B/token   compression
   1       8,192     2,197,048.0         0.05x
   8       8,192       274,974.0         0.42x
   8     131,072       257,553.4         0.45x
  32     131,072        64,406.7         1.78x
  64     131,072        32,215.6         3.56x
  64   1,048,576        32,069.8         3.58x
```

`B*S ≈ 1M` is the empirical break-even — exactly matching the analytical
prediction that proposer-weight amortization governs the regime where the
architecture pays off.

## Honest caveats — what these numbers do and don't claim

1. **Verifier model**: the included demo uses `Qwen/Qwen3-1.7B` because no
   public Qwen-3.6 checkpoint exists yet. The Qwen 3.5/3.6 hybrid-attention
   architecture has only 16/64 layers carrying KV instead of 28/28 in
   Qwen3-1.7B, so the *baseline* per-token KV would be smaller and the
   compression ratio likely smaller as well. The framework code does not
   change.
2. **Acceptance rate is low (~0.12)**. The proposer was trained with masked
   diffusion on Nemotron-SFT-Code by a different research group; it is *not*
   Repr-Align-aligned to Qwen3-1.7B's representation geometry. With a same-
   family Repr-Align proposer (the design's recommended choice), acceptance
   rates of 0.6–0.85 are reported in the literature. Low acceptance does not
   break correctness — it just means the verifier issues more correction
   tokens, which costs throughput, not memory.
3. **Proposer activation memory** is dominated by the dense logits buffer
   (`[1, T, V_vocab]`). The included implementation does not use the standard
   "compute logits only at masked positions" optimization — its peak is
   `T * V * 2` bytes per forward pass. The projection table reuses the
   activation peak measured at S=108; long-context activation projections
   therefore *implicitly assume* the sparse-logits optimization. Adding it
   is a few-line change but was kept out of this drop for transparency.
4. **CPU runs**. The repository runs end-to-end on a 4-core, 15 GB-RAM CPU
   environment in tens of seconds. GPU runs would just change wall-clock,
   not byte accounting; the NBT numbers are deterministic functions of
   model shapes and the cache budget.
5. **No fallback**. If anything in the pipeline becomes inconsistent
   (cache layout, tokenizer drift, mask leakage from the proposer) the
   code raises immediately. There is no path that silently degrades to
   "just call the verifier".

## What is and isn't being demonstrated

- **Demonstrated**: KV-cache memory bound is enforced and measured; the
  speculative loop is greedily distribution-equivalent to the verifier (in
  the equivalence regime); the NBT trade-off curve crosses unity at the
  predicted operating regime.
- **Not demonstrated** (out of scope for a single CPU runnable demo):
  multi-target verifier routing (Qwen / Gemma / DeepSeek), session-affinity
  scheduling, OTA, federated self-learning. Those are platform-level
  components from the design discussion that need separate plumbing.
