# Skill: Build a distributed speculative-decode inference engine (remote DFlash + f_θ proposer)

**Reusable across agents (Claude / Codex / Cursor).** This is the SOP for taking a
single-host fused spec-decode engine (an AR verifier + an EAGLE-style drafter +
f_θ KV restoration) and splitting it across hosts — **verifier on host A, drafter
+ f_θ proposer on host B** — over a real gRPC data plane (ADR 0009 §4 "F3"). The
concrete example is Kakeya's gemma-4 verifier (MLX, Mac) ↔ DFlash+f_θ (torch, GPU),
but the pattern is general.

The non-negotiable invariant that makes this safe: **correctness containment** —
the verifier's local greedy verify decides every token, so the output is
**byte-identical to local greedy regardless of what the remote proposer drafts**.
A wrong/stale/garbage draft can only lower the acceptance rate, never change a token.

---

## 1. When to use this skill

- You have a working **single-host** fused spec-decode loop and want to offload the
  drafter (+ f_θ) to another machine (GPU fleet utilization, memory split, etc.).
- The drafter is **EAGLE-style** (needs the verifier's aux-layer hidden states +
  the verifier's tied embedding), so it is NOT a token-ids-only proposer.
- You need a real cross-host **RTT / throughput / bounded-memory** measurement of
  the production config, not a toy proposer.

If your proposer is **model-free / token-ids-only** (e.g. an n-gram prompt-lookup),
you do NOT need this — use the simpler `ProposerService` / `RemoteProposer`
(ADR 0009 control plane). This skill is specifically for the **bulk-tensor data
plane** (aux hidden + f_θ-projected K/V crossing the wire).

---

## 2. Architecture: two layers

Keep the **transport/protocol** strictly separate from the **model math** so the
former is unit-testable without GPUs/models and the latter is swappable per
framework.

### Layer 1 — framework-agnostic machinery (pure-python, 100%-unit-tested)
- `tensor_codec` — a self-describing `WireTensor` ↔ proto `Tensor` (dtype string +
  int64 shape + raw little-endian bytes). bf16 has no numpy scalar → carry it as
  `uint16` bits under the logical name `"bfloat16"`; rebuild via thin torch/mlx
  bridges. **No torch/mlx import in the codec** (mlx bridges are `# pragma: no cover`).
- `dflash_service` — a `RestorationDraftEngine` Protocol (WireTensor in/out), an
  async gRPC servicer, and a sync `RemoteDFlashProposer` client. Engine `KeyError`
  → `NOT_FOUND`, `ValueError` → `INVALID_ARGUMENT`.
- `fused_decode` — `DistributedFusedDecoder` (mirrors the in-process fused loop)
  driving a `RestoringVerifier` Protocol. Aux/K-V cross the verifier↔decoder
  boundary as `WireTensor`, so the loop is framework-agnostic and fully fakeable.

### Layer 2 — real-model engines (mlx/torch, validated on-device, NOT coverage-gated)
- **Host A (verifier):** a `RestoringVerifier` adapter wrapping your restored
  incremental verifier (Kakeya: `MLXRestoringVerifierAdapter` over
  `MLXRestoredIncrementalVerifier`).
- **Host B (proposer):** a `RestorationDraftEngine` impl holding the drafter + f_θ
  + the verifier's tied embedding (Kakeya: `MLXRestorationDraftEngine` for an
  all-Mac loopback, `TorchRestorationDraftEngine` for a CUDA host).

### Wire protocol (stateful session)
Per turn: **Restore** (prompt → host B captures drafter K/V → f_θ → verifier K/V
banks; host A prefills) → **SeedContext** (host A's verifier aux hidden over the
prompt → host B's drafter context K/V). Per block: **DraftBlock** (bonus +
context_len → exactly `block_size` drafts) → host A verifies/commits →
**ExtendContext** (committed tokens' aux, O(block) → grow host B's context).
**CloseSession** frees host-B state.

| Message | Dir | Size class |
|---|---|---|
| Restore | A→B ids / B→A K/V banks | O(T) one-time (empty under S5 free-lunch) |
| SeedContext | A→B aux | O(T) one-time |
| DraftBlock | A↔B | O(1) / O(block) |
| ExtendContext | A→B committed aux | O(block) (the per-block bandwidth term) |

---

## 3. SOP — build order

1. **Ground the dataflow first.** Read the EXACT single-host fused loop and write
   down, per block, every tensor that crosses the drafter↔verifier boundary
   (shapes, dtype, which model produces it). Decide what stays local (drafter
   context K/V, verifier KV cache, full logits — send only the bonus int) vs what
   crosses (aux hidden O(block), draft ids, restored K/V once).
2. **Build Layer 1 + unit tests FIRST.** Codec roundtrip + dtype/byte-count
   validation; servicer over a real `grpc.aio` server with a fake engine (status
   mapping, dead-address wrap, draft-count refusal); decoder with a fake verifier
   that models a fixed greedy continuation + fake remotes returning **perfect AND
   wrong** drafts — assert **byte-identical to greedy in both cases**. This proves
   containment before any model is involved.
3. **Build the real engines (Layer 2)** by REUSING the in-process fused helpers
   (capture-drafter-KV, f_θ projection, `make_context_kv`/`draft_block_cached`/
   `extend_context_kv`, the restored verifier). Don't reimplement the math.
4. **Climb the validation ladder** (each rung adds one risk, all assert
   byte-identical):
   - **in-process** (single model load, no gRPC) — validates engine+adapter+loop;
   - **loopback gRPC** (real wire + codec, same host) — validates serialization;
   - **cross-host** (real network) — validates deployment + measures RTT.
   Use **block_size=1 as the greedy baseline** (the same decoder at block=1 is pure
   greedy) so baseline and distributed share one code path.
5. **Deploy** with the scripts in §5 and **measure** throughput / bounded-memory / RTT.

---

## 4. Gotchas / lessons (the expensive ones)

- **MLX is Apple-only.** A CUDA host B cannot run the MLX verifier's embedding;
  give host B a **torch** embedding (load the base verifier, or ship just the
  ~1.5 GB tied-embed weight). Output stays byte-identical (greedy verify is
  authoritative); only the drafter numerics / acceptance shift.
- **transformers version.** gemma-4 (torch) needs `transformers>=5.0`; older
  custom modeling that depends on `decoder_layer.attention_type` breaks under 5.x
  (see `requirements.txt`). Also: 5.x `apply_chat_template` returns a dict — pass
  `tokenize=True, return_dict=False`.
- **Cross-layer KV sharing.** gemma-4 shares K/V across layers. Ship every
  non-exact f_θ layer from host B, but on host A **filter restored layers to the
  verifier's `kv_source_layer_map` source layers** — the verifier only injects
  those. Keep that filter on the host-A (MLX) side where the layout lives.
- **f_θ is prefill-only** under S5; on gemma-4 the projected sliding-layer K/V are
  recall-irrelevant ("free lunch") so `Restore` can be empty — force f_θ (ship the
  banks) only when you want it load-bearing / to exercise the path.
- **gRPC max message size.** Restored K/V (~11 MB) and per-block aux exceed gRPC's
  4 MiB default — set `grpc.max_{send,receive}_message_length` high on both ends.
- **Don't sync-RPC on the server's event loop in tests.** A sync client `close()`
  that issues an RPC will deadlock an in-process `grpc.aio` server sharing the
  thread; drive it via `asyncio.to_thread`. (In production the server is remote —
  no constraint.)
- **vast / cloud port mapping.** Portal ports (Caddy) return HTTP 401 to gRPC, and
  some mapped ports silently drop. Use a **plain high port** (e.g. 50070) reached
  over an **SSH `-L` tunnel** — do not rely on the externally-mapped portal ports.
- **Big model cache.** The base verifier may exceed the root disk; cache it in a
  RAM-disk (`/dev/shm`).
- **Verify, don't trust comments.** Every "should be byte-identical" claim must be
  asserted by an actual run on each rung of the ladder.

---

## 5. Deployment + startup scripts

| Host | Script | What it does |
|---|---|---|
| B (GPU) | `scripts/deploy/dflash_proposer_server_gpu.sh` | ensure transformers 5.x, fetch gemma-4 (embed) + DFlash into `/dev/shm` HF cache, serve `DFlashProposerService` on a non-portal port |
| A (verifier) | `scripts/deploy/dflash_verifier_client.sh` | (optionally) open the SSH `-L` tunnel, probe it, run the byte-identical + RTT E2E against `localhost:<port>` |
| both | `scripts/research/k3_dflash_proposer_server.py` / `k3_distributed_dflash_e2e_mac.py` | the underlying server + harness (in-process / `--grpc` / `--remote-addr`) |

Typical run:
```bash
# Host B (GPU):
bash scripts/deploy/dflash_proposer_server_gpu.sh --port 50070
# Host A (Mac): open the tunnel with YOUR creds, then:
ssh -p <ssh_port> root@<gpu_host> -L 50070:localhost:50070   # in another shell
bash scripts/deploy/dflash_verifier_client.sh \
    --verifier-path /path/to/gemma-4-26B-A4B-it-mlx-4bit --port 50070
```
On a self-hosted Mac runner, the same E2E runs via the bridge preset
`mlx-distributed-dflash-e2e-crosshost` (it expects the tunnel open on the runner).

---

## 6. What "good" looks like (Kakeya gemma-4 ↔ H200, measured)

- **Correctness:** PASS byte-identical-to-greedy on all three rungs (in-process,
  loopback gRPC, real Mac↔H200), DFlash acceptance ≈ **0.86–0.89** (vs n-gram 0.10).
- **Bounded memory:** verifier-side invariant unchanged by the split — ~235 MB
  resident KV, constant over a 1241-token generation (S5: 25 sliding layers bound
  to sink+window, 5 exact layers full-context).
- **RTT (Mac↔H200 over SSH tunnel):** Restore ~3.2 s / 11.5 MB (one-time),
  SeedContext ~0.4 s, DraftBlock ~268 ms, ExtendContext ~316 ms / 0.27 MB-per-block,
  per-block ~584 ms; throughput 3.7 tok/s (block=4) vs 1.0 (block=1). The DFlash
  forward is offloaded to the GPU (a VM→H200 probe shows DraftBlock 108 ms is
  mostly net-RTT vs the 232 ms Mac-CPU compute); cross-host cost is then network
  RTT + per-block aux bandwidth bound. **GA levers:** aux quantization/compression,
  same-rack placement.

---

## 7. Reference (Kakeya impl)

- Machinery: `inference_engine/distributed/{tensor_codec,dflash_service,fused_decode}.py`
  + tests under `tests/inference_engine/distributed/`.
- Engines: `inference_engine/backends/mlx/dflash_distributed.py` (host A + Mac host B),
  `inference_engine/v04/dflash_distributed_engine.py` (CUDA host B).
- Proto: `proto/kakeya/v1/distributed.proto` (`DFlashProposerService`).
- Design + measured report: `docs/design/distributed-dflash-ftheta-data-plane.md`.
