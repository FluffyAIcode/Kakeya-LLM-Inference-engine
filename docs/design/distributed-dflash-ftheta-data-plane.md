# Distributed DFlash + f_θ data plane (ADR 0009 §4 "F3")

Status: **machinery landed + unit-tested; real-model engine is the next phase.**
PR: #158 (stacked on #157).

## Goal

Run the **production Kakeya config** across two hosts so the real engine — not
the n-gram toy — earns a true distributed RTT:

- **Host A (verifier):** gemma-4-26B-A4B-it-mlx-4bit on a Mac mini (MLX), with
  sink+window restored KV.
- **Host B (proposer):** the DFlash drafter + f_θ K/V projection on a GPU.

Correctness containment is **structural and unchanged**: every emitted token is
decided by host A's local greedy verify, so the output is byte-identical to
local greedy regardless of what host B drafts.

## Protocol (gRPC `DFlashProposerService`, stateful per decode session)

### Per turn (prefill / restoration)

1. **Restore** (A→B `prompt_ids`; B→A restored K/V): host B embeds the prompt
   with the verifier embedding, runs the DFlash drafter to get its K/V, and maps
   them through f_θ into verifier K/V space. Under S5 (`s5_exact_full_attn`) the
   full-attention layers are omitted — the verifier's native cache owns them (on
   gemma-4 this is the "free lunch": f_θ-projected sliding-layer K/V are
   recall-irrelevant, so Restore can even return empty); with `--force-f-theta`
   semantics the projected sliding-layer banks are shipped and injected.
2. Host A `verifier.prefill(prompt_ids, restored, evicted_positions)`.
3. **SeedContext** (A→B `aux`): host A's verifier aux-layer hidden over the
   prompt (`capture_aux_hidden`, `num_aux × [1,T,hidden]`) seeds host B's drafter
   context K/V (`make_context_kv`).

### Per decode block

4. **DraftBlock** (A→B `bonus,context_len,L-1`; B→A drafts): host B
   `draft_block_cached(ctx_kv, bonus, embed_fn, lm_head_fn, block_size=L-1,
   context_len)`.
5. Host A `verify_block([bonus]+drafts)` → greedy accept count; `commit` (drop
   rejected KV, append correction on partial accept).
6. **ExtendContext** (A→B committed `aux` + positions, O(block_size)): host B
   `extend_context_kv(ctx_kv, make_context_kv(new_aux, new_positions))`.

### Wire payloads (per [tensor_codec](../../inference_engine/distributed/tensor_codec.py))

| Message | Direction | Payload | Size class |
|---|---|---|---|
| Restore | A→B / B→A | prompt ids / f_θ K/V banks (sliding layers) | O(T) one-time (empty under S5 free-lunch) |
| SeedContext | A→B | `num_aux × [1,T,hidden]` aux | O(T) one-time |
| DraftBlock | A→B / B→A | scalars / `L-1` ids | O(block) |
| ExtendContext | A→B | `num_aux × [1,k,hidden]` aux, k≈accept+1 | O(block) (~152 KB/block at L=16) |

## Landed in this PR (fully unit-tested, framework-agnostic)

| Component | File | Tests |
|---|---|---|
| `Tensor`/`LayerKV` + `DFlashProposerService` proto | `proto/kakeya/v1/distributed.proto` | proto-drift CI |
| `WireTensor` codec (numpy + torch/mlx bridges) | `inference_engine/distributed/tensor_codec.py` | `test_tensor_codec.py` (17) |
| `RestorationDraftEngine` contract + servicer + `RemoteDFlashProposer` | `inference_engine/distributed/dflash_service.py` | `test_dflash_service.py` (7) |
| `DistributedFusedDecoder` + `RestoringVerifier` contract | `inference_engine/distributed/fused_decode.py` | `test_fused_decode.py` (10, byte-identical for perfect AND wrong drafts) |

## Next phase — real-model engine (construction recipe)

Two concrete classes, placed in `inference_engine/backends/mlx/` (not
coverage-gated; they import mlx/torch), wired from the proven helpers in
`scripts/research/k3_integrated_niah_eval_mac.py` and
`inference_engine/backends/mlx/fused_specdecode.py`:

1. **`MLXRestorationDraftEngine`** (host B, implements `RestorationDraftEngine`):
   - load: `DFlashDrafter.from_pretrained(drafter_id)` (torch) or
     `MLXDFlashDrafter.from_pretrained(drafter_id)`, `FThetaProjection
     .from_pretrained(f_theta_dir)`, and a verifier-embedding source for
     `embed_fn`/`lm_head_fn` (`make_native_embed_lm_head` / `make_bridge_embed_lm_head`).
   - `restore`: replicate `capture_drafter_kv` (embed prompt → drafter forward,
     hook `k_proj`/`v_proj`) + `f_theta.forward_kv_pack`; return projected
     sliding-layer K/V as `WireTensor` (empty under S5 free-lunch).
   - `seed_context`/`extend_context`: `make_context_kv` / `extend_context_kv`,
     keyed by `session_id`.
   - `draft_block`: `draft_block_cached(ctx_kv, bonus, embed_fn, lm_head_fn, ...)`.

2. **`MLXRestoringVerifierAdapter`** (host A, implements `RestoringVerifier`):
   wraps `MLXRestoredIncrementalVerifier` — `prefill`, `next_token_logits`
   argmax, `forward_block` (with `_capture_aux=True`), the greedy accept loop,
   `commit_or_truncate`/`append_token`, `last_aux_torch_slice` → `WireTensor`,
   `aux_over_prompt` = `capture_aux_hidden`.

### Validation plan

- **In-process real-model E2E** (single gemma-4 load, avoids 2×26B OOM on one
  Mac): drive `DistributedFusedDecoder` with an in-process proposer calling the
  engine directly, compare to `fused_specdecode_generate` → assert byte-identical.
- **True cross-host RTT**: gemma-4 verifier on the Mac mini ↔ DFlash+f_θ engine on
  the GPU over gRPC; measure per-block `DraftBlock`+`ExtendContext` RTT and
  end-to-end tok/s, vs the single-host fused baseline (4.72 tok/s).

## Real-model validation (landed)

`inference_engine/backends/mlx/dflash_distributed.py` implements the two model-bound
contracts (`MLXRestorationDraftEngine`, `MLXRestoringVerifierAdapter`) + an
`InProcessDFlashProposer`; `scripts/research/k3_distributed_dflash_e2e_mac.py` runs
the real engine (gemma-4-26B-A4B-it-mlx-4bit + torch DFlash + f_θ, loaded once)
and asserts byte-identical-to-greedy. Bridge presets `mlx-distributed-dflash-e2e-{inproc,grpc}`.

On the Mac mini (DFlash drafter on CPU), 28-tok prompt:

| Run | output | acceptance | greedy | distributed |
|---|---|---|---|---|
| In-process | ✅ byte-identical | 0.892 (33/37) | 11.81 tok/s | 6.57 tok/s |
| Loopback gRPC | ✅ byte-identical | 0.863 (44/51) | 19.60 tok/s | 8.78 tok/s |

**Per-RPC RTT + payload (loopback gRPC, block=4, 64 tok):**

| RPC | n | p50 | mean | payload |
|---|---|---|---|---|
| Restore | 1 | 162 ms | 162 ms | 11.47 MB (f_θ sliding-layer K/V, one-time) |
| SeedContext | 1 | 11.7 ms | 11.7 ms | 1.89 MB (prompt aux) |
| **DraftBlock** | 17 | **232 ms** | 272 ms | O(1) (scalars + L-1 ids) |
| ExtendContext | 17 | 11.8 ms | 19.2 ms | 4.33 MB total (~0.25 MB/block aux) |

### Cross-host motivation (what the numbers show)
`DraftBlock`'s ~232 ms p50 is **the DFlash drafter's forward on the Mac's CPU**, not
network — it is the single dominant per-block cost. This is exactly the work the
GPU topology offloads: on an H200 the DFlash forward is single-digit ms, so the
cross-host per-block cost becomes **GPU draft (~ms) + network RTT (~52 ms p50,
measured VM↔H200) + ExtendContext aux (~0.25 MB)** — i.e. moving the proposer to
the GPU is projected to cut `DraftBlock` from ~232 ms to well under network RTT.
The one-time `Restore` (11.5 MB) + `SeedContext` (1.9 MB) amortize over the turn.

## Live cross-host RTT (landed)

Deployed the torch engine on an H200: `inference_engine/v04/dflash_distributed_engine
.TorchRestorationDraftEngine` (torch gemma-4-26B-A4B-it for the embed + DFlash +
f_θ) served by `scripts/research/k3_dflash_proposer_server.py`; a verifier host
connects with `RemoteDFlashProposer`. The MLX verifier adapter filters restored
layers to the verifier's KV-source layers (gemma-4 cross-layer sharing).

Measured VM→H200 over an SSH `-L` tunnel (real GPU compute; true data-plane payloads):

| RPC | p50 | payload | note |
|---|---|---|---|
| Restore | 2310 ms | 11.47 MB | one-time; f_θ-projected sliding-layer K/V (25 layers) |
| SeedContext | 947 ms | 1.89 MB | one-time; prompt aux |
| **DraftBlock** | **108 ms** | O(1) | H200 DFlash forward + net RTT — **vs 232 ms on the Mac CPU (loopback)**: the GPU offload cuts draft compute |
| ExtendContext | 140 ms | 0.27 MB/block | committed aux — bandwidth-dominated cross-host |

Per-block (draft+extend) p50 ≈ **248 ms** over the SSH tunnel. Caveats: the SSH
single-stream inflates transfer-bound RPCs vs a direct gRPC link; VM↔H200 base RTT
≈ 52 ms; byte-identical correctness is proven on the Mac loopback (same engine code).
The Mac↔H200 byte-identical run uses the same path via `mlx-distributed-dflash-e2e-
crosshost` with `ssh -p 43350 root@107.206.71.138 -L 50070:localhost:50070` active.

### (historical) Remaining for the LIVE Mac↔GPU number
The GPU (CUDA) cannot run MLX, so the GPU-side engine needs a **torch embedding**
source for `embed_fn`/`lm_head_fn` (gemma-4 tied embed). Two options:
1. one-time ship of the verifier embedding weights Mac→GPU at session setup
   (~1.5 GB), then a pure-torch `TorchRestorationDraftEngine`; or
2. embed/lm_head RPC back to host A per block (no weight ship, +2 hops/block).
Output stays byte-identical either way (greedy verify is authoritative); only the
drafter's numerics (and thus acceptance) may shift slightly vs the MLX 4-bit embed.

### Open considerations

- **embed/lm_head on host B**: DFlash needs the verifier's tied embedding for the
  query block; host B either replicates the gemma-4 embedding weights (~1.5 GB
  torch) or RPCs `query_ids → embeddings/logits` back to host A.
- **MLX↔torch on the wire**: handled by `tensor_codec` (bf16 via uint16 bits).
- **RTT economics** (from ADR 0014 fused crosshost sims): break-even ≈ 100 ms/block;
  same-rack deployment keeps DraftBlock+ExtendContext RTT sub-ms–single-digit-ms.
