"""K3 integrated NIAH eval — **Mac (MLX) path**.

Apple-Silicon counterpart of ``scripts/research/k3_integrated_niah_eval.py``
(the validated CUDA K3 product gate). Wires:

  * verifier = Gemma 4 26B-A4B  (MLX 4-bit, ``mlx_lm.load``)
  * drafter  = DFlash 0.4B       (PyTorch ``DFlashDrafter``, MPS/CPU)
  * f_θ      = trained K/V proj  (PyTorch ``FThetaProjection``)
  * S5       = full-attention layers kept exact (``--s5-exact-full-attn``)

Each generated token runs the **restored** verifier forward (sink+window
local cache + evicted-position K/V restoration), exactly mirroring the CUDA
``CrossModelDLMRestoredVerifier`` semantics. Cross-runtime tensors are
bridged via numpy (see ``scripts/research/k3_dflash_mlx_bridge.py``).

Run on the Mac mini (Apple Silicon, ~24 GB):

  HF_TOKEN unnecessary for the local MLX 4-bit verifier. From repo root:

    PYTHONPATH=.:sdks/python python3 scripts/research/k3_integrated_niah_eval_mac.py \\
        --verifier-path models/gemma-4-26B-A4B-it-mlx-4bit \\
        --drafter-id    z-lab/gemma-4-26B-A4B-it-DFlash \\
        --f-theta-dir   results/research/f_theta_v5_s5_sliding \\
        --s5-exact-full-attn \\
        --n-samples 10 --haystack-min-lines 238 --haystack-max-lines 322 \\
        --sink-size 4 --window-size 64 --max-new-tokens 24 \\
        --output results/research/k3_s5_niah_ctx280_mac.json

Quick sanity (smaller / faster):

    --n-samples 4 --haystack-min-lines 60 --haystack-max-lines 81 \\
    --max-new-tokens 16

Diagnostics:
  --identity-restore   restore ALL evicted K/V with the verifier's own true
                       K/V (should match oracle — validates the MLX
                       restoration machinery independent of f_θ / drafter).

Output JSON mirrors the CUDA harness gate schema (recall_cross_model,
recall_oracle, recall_delta_vs_oracle_pp, architectural_correctness).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-path", default="models/gemma-4-26B-A4B-it-mlx-4bit")
    ap.add_argument("--drafter-id", default="z-lab/gemma-4-26B-A4B-it-DFlash")
    ap.add_argument("--f-theta-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--haystack-min-lines", type=int, default=238)
    ap.add_argument("--haystack-max-lines", type=int, default=322)
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--incremental", action="store_true",
                    help="Use the INCREMENTAL restored decode (MLX Gap-A): "
                         "prefill captures restored K/V into a persistent cache, "
                         "decode via mlx_lm generate_step (O(L)/token). Fixes the "
                         "per-token re-forward throughput collapse. Free-gen only.")
    ap.add_argument("--fused-specdecode", action="store_true",
                    help="Use the FUSED DFlash spec-decode engine (MLX port of "
                         "#107 A+B+C): drafter context K/V cache + aux capture from "
                         "the verify forward + incremental restored verify with "
                         "trim_prompt_cache accept/reject. Free-gen only.")
    ap.add_argument("--force-fused-specdecode", action="store_true",
                    help="Deprecated alias: --fused-specdecode now ALWAYS runs "
                         "the fused engine (evidence-gate constraint; the "
                         "silent greedy fallback that produced blocks=0 "
                         "reports labelled fused is no longer reachable).")
    ap.add_argument("--native-baseline-bypass", action="store_true",
                    help="Run the verifier on its NATIVE cache (no restoration, "
                         "no drafter/f_theta) and label the run as "
                         "system_under_test=native_ar_baseline. This is the "
                         "only way to run the former 'adaptive S5 native' "
                         "path; it can no longer occupy the cross-model slot "
                         "or claim recall/speedup (k3_report_gate rules "
                         "BASELINE_AS_SUT / SPEEDUP_SELF_COMPARISON).")
    ap.add_argument("--direct-answer-prompt", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="For NIAH free generation, add a strict instruction to "
                         "answer with only the secret code. This keeps short "
                         "smokes from spending the token budget on Gemma4's "
                         "thought/channel preamble. Use --no-direct-answer-prompt "
                         "to reproduce the legacy prompt exactly.")
    ap.add_argument("--chat-template-prompt", action="store_true",
                    help="Deprecated compatibility flag; chat-template prompting "
                         "is the default for Gemma4 NIAH.")
    ap.add_argument("--raw-completion-prompt", action="store_true",
                    help="Diagnostic only: bypass chat template and encode the "
                         "NIAH prompt as a raw completion prompt.")
    ap.add_argument("--content-channel-prefill",
                    action=argparse.BooleanOptionalAction,
                    default=True,
                    help="With chat-template direct-answer prompts, append "
                         "Gemma4's content channel marker before generation so "
                         "short smokes do not spend tokens on the thought channel.")
    ap.add_argument("--all-mlx-drafter", action="store_true",
                    help="Step-2 rescue: run the DFlash drafter natively in "
                         "MLX (inference_engine.backends.mlx.dflash_drafter) "
                         "instead of PyTorch — zero mx<->torch bridge "
                         "crossings per block. Requires --s5-exact-full-attn "
                         "(the all-MLX path uses native-S5 injection; the "
                         "f_theta sliding restoration path stays torch).")
    ap.add_argument("--single-fused", action="store_true",
                    help="PROBE: with --cuda-trim, fuse drafter+verifier into ONE "
                         "graph (skip the two-phase eval) to classify the Metal "
                         "instability (fundamental command-buffer vs fixable SDPA "
                         "fallback). Reports per-block eval times.")
    ap.add_argument("--cuda-trim", action="store_true",
                    help="All-MLX fused with the CUDA-parity rollback: all-KVCache "
                         "verifier layout + native trim_prompt_cache (keep accepted "
                         "K/V, drop only rejected) instead of the v3 carry "
                         "re-forward. Requires --all-mlx-drafter --fused-specdecode.")
    ap.add_argument("--code-prompts", action="store_true",
                    help="Replace the NIAH dataset with code-completion prompts "
                         "(naturally-long, predictable generation = the spec-decode "
                         "sweet spot). Recall metric is N/A; measures honest "
                         "decode-only throughput + acceptance on a real workload.")
    ap.add_argument("--ignore-turn-stop", action="store_true",
                    help="Do not include Gemma4 <turn|> as a stop token. "
                         "Useful for throughput evidence runs that require "
                         "decode median >= 32 tokens.")
    ap.add_argument("--block-size", type=int, default=4,
                    help="Spec-decode block size (drafted tokens per block).")
    ap.add_argument("--teacher-forced", action="store_true",
                    help="DIAGNOSTIC ONLY (under-measures retrieval): single "
                         "restored forward per sample, check argmax at the "
                         "needle-code span. Note this misses the model's "
                         "preamble so it reads ~0 even for the oracle — use "
                         "the default free-generation for a real recall "
                         "number. Free-gen oracle uses mlx's fast native "
                         "incremental cache; the restored cross path does a "
                         "full forward per token (slow on M4 — see notes).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drafter-device", default="cpu",
                    help="torch device for the DFlash drafter + f_θ (mps|cpu)")
    ap.add_argument("--torch-cpu-threads", type=int, default=0,
                    help="Override torch CPU intra-op threads for drafter/f_theta "
                         "(0 keeps torch default).")
    ap.add_argument("--s5-exact-full-attn", action="store_true",
                    help="Keep full-attention layers' K/V exact (S5).")
    ap.add_argument("--identity-restore", action="store_true",
                    help="Restore ALL evicted K/V with the verifier's own "
                         "true K/V (machinery check; should match oracle).")
    ap.add_argument("--compress-full-attn", action="store_true",
                    help="KakeyaLattice-compress the exact full-attention "
                         "layers' K/V (lossy round-trip) to shrink the O(T) "
                         "linear term. Reports the compression ratio + recall "
                         "under compression.")
    ap.add_argument("--kl-lattice", default="D4", choices=["D4", "E8"])
    ap.add_argument("--kl-q-range", type=int, default=38)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--decode-warmup-tokens", type=int, default=1,
                    help="Run a tiny untimed native decode warmup before "
                         "cross/oracle measurements so MLX/Metal compilation "
                         "cost does not fall only on the first measured path.")
    ap.add_argument("--prefill-chunk-size", type=int, default=512,
                    help="Chunk MLX prompt prefill/forward calls to avoid the "
                         "long-context one-shot forward OOM path. Set <=0 to "
                         "use a single full prompt forward.")
    ap.add_argument("--output", default=None)
    # ---- interactive chat on the FULL fused engine (verifier+proposer+f_θ+S5) ----
    ap.add_argument("--chat", action="store_true",
                    help="Interactive REPL on the FULL fused spec-decode engine "
                         "(verifier + DFlash proposer + f_θ + S5 bounded KV) — "
                         "NOT verifier-only. Requires the fused flags "
                         "(--fused-specdecode --force-fused-specdecode "
                         "--all-mlx-drafter --s5-exact-full-attn --cuda-trim).")
    ap.add_argument("--chat-scripted", default=None,
                    help="Non-interactive chat: '||'-separated user turns "
                         "(for Mac-bridge verification); writes a transcript.")
    ap.add_argument("--force-f-theta", action="store_true",
                    help="Run f_θ restoration even under --s5-exact-full-attn "
                         "(bypass the S5 native-prefill short-circuit). On gemma-4 "
                         "the restored sliding-layer K/V are recall-irrelevant "
                         "(the exact layers carry recall), but f_θ EXECUTES and "
                         "its output is injected — exercising the full verifier/"
                         "proposer/f_θ pipeline. Requires the torch drafter+f_θ "
                         "(do NOT combine with --all-mlx-drafter).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    import mlx.core as mx  # type: ignore
    import mlx_lm  # type: ignore
    import torch

    from inference_engine.v04 import (
        DFlashDrafter, FThetaProjection, NIAHSample,
        aggregate_recall, make_niah_dataset, recall_predicate,
    )
    from inference_engine.v04.kv_merge import compute_evicted_positions
    from inference_engine.backends.mlx.cross_model_dlm_verifier import (
        resolve_mlx_text_model, mlx_full_attention_layer_indices,
        kv_source_layer_map, capture_own_kv, restored_logits,
        per_layer_kv_geometry, kv_memory_report,
        restored_prefill_cache, restored_incremental_generate,
    )
    from inference_engine.backends.mlx.fused_specdecode import (
        MLXRestoredIncrementalVerifier, capture_aux_hidden,
        make_bridge_embed_lm_head, fused_specdecode_generate,
        fused_specdecode_generate_mlx, fused_specdecode_generate_mlx_trim,
    )
    from inference_engine.v04.kv_compressor import make_default_compressor
    from inference_engine.bench.k3_report_gate import (
        CLAIM_ORACLE_DECODE_LOOP, MIN_MEDIAN_DECODE_TOKENS, MIN_PERF_SAMPLES,
        MAX_PREFILL_SPREAD, NATIVE_BASELINE_LABEL,
        decode_only_block, prefill_spread, summarize_violations,
        validate_report,
    )
    from scripts.research.k3_dflash_mlx_bridge import (
        mx_to_torch, torch_to_mx,
    )

    torch.manual_seed(args.seed)
    if int(args.torch_cpu_threads or 0) > 0:
        torch.set_num_threads(int(args.torch_cpu_threads))
        print(f"[mac] torch CPU threads={torch.get_num_threads()}",
              file=sys.stderr, flush=True)
    dev = torch.device(args.drafter_device if (
        args.drafter_device == "cpu" or torch.backends.mps.is_available()
    ) else "cpu")

    # ---------- Load verifier (MLX) ----------
    print(f"[mac] loading MLX verifier {args.verifier_path}", file=sys.stderr, flush=True)
    mlx_model, tokenizer = mlx_lm.load(args.verifier_path)
    text_model = resolve_mlx_text_model(mlx_model)
    embed_scale = float(getattr(text_model, "embed_scale", 1.0))
    n_layers = len(text_model.layers)
    full_attn_idx = mlx_full_attention_layer_indices(text_model)
    src_map = kv_source_layer_map(text_model)
    print(f"[mac] verifier layers={n_layers} full_attn={full_attn_idx}", file=sys.stderr)

    # Evidence-gate path resolution (k3_report_gate):
    #   * --fused-specdecode ALWAYS executes the fused engine. The former
    #     implicit "adaptive S5 native" bypass silently replaced the system
    #     under test with the native baseline while keeping the fused label
    #     (committed reports showed blocks=0 on every sample).
    #   * The native baseline is still runnable — but only explicitly, and
    #     it is labelled as a baseline in the report.
    if args.native_baseline_bypass and args.force_fused_specdecode:
        raise SystemExit(
            "--native-baseline-bypass and --force-fused-specdecode are "
            "mutually exclusive: a run is either the native baseline or "
            "the fused system under test.")
    if args.native_baseline_bypass:
        args.fused_specdecode = True  # route through the cache-based loop
    elif args.fused_specdecode or args.force_fused_specdecode:
        args.fused_specdecode = True
        args.force_fused_specdecode = True
    adaptive_s5_native = args.native_baseline_bypass
    # Interactive chat runs the FULL verifier/proposer/f_θ pipeline by DEFAULT:
    # f_θ executes each turn (torch drafter + f_θ) unless the fast all-MLX path
    # (--all-mlx-drafter, f_θ bypassed) or the native baseline is explicitly chosen.
    if args.chat and not args.all_mlx_drafter and not args.native_baseline_bypass:
        if not args.force_f_theta:
            print("[chat] f_θ default-ON for interactive chat (torch drafter + f_θ); "
                  "pass --all-mlx-drafter for the fast f_θ-bypassed path.",
                  file=sys.stderr, flush=True)
        args.force_f_theta = True
    if args.all_mlx_drafter and not args.s5_exact_full_attn:
        raise SystemExit(
            "--all-mlx-drafter requires --s5-exact-full-attn: the all-MLX "
            "path uses native-S5 prefill injection; the f_theta sliding "
            "restoration path is torch-only.")
    drafter = None
    mlx_drafter = None
    f_theta = None
    fcfg = None
    if adaptive_s5_native:
        print("[mac] native baseline bypass: skipping drafter/f_theta load; "
              "report will be labelled system_under_test=native_ar_baseline",
              file=sys.stderr, flush=True)
    elif args.all_mlx_drafter:
        # ---------- Step-2 rescue: drafter native in MLX ----------
        from inference_engine.backends.mlx.dflash_drafter import (
            MLXDFlashDrafter,
        )
        print(f"[mac] loading ALL-MLX drafter {args.drafter_id}",
              file=sys.stderr, flush=True)
        mlx_drafter = MLXDFlashDrafter.from_pretrained(args.drafter_id)
        # No torch drafter / f_theta: S5-native injection covers prefill
        # restoration and the drafter never leaves the Metal stream.
    else:
        # ---------- Load drafter + f_θ (PyTorch) ----------
        print(f"[mac] loading drafter {args.drafter_id} on {dev}", file=sys.stderr, flush=True)
        drafter = DFlashDrafter.from_pretrained(args.drafter_id, dtype=torch.float32)
        drafter = drafter.to(dev).eval()
        for p in drafter.parameters():
            p.requires_grad_(False)
        f_theta = FThetaProjection.from_pretrained(
            args.f_theta_dir, dtype=torch.float32, device=dev,
        )
        fcfg = f_theta.config

    # ---------- Optional KakeyaLattice compression of full-attn layers ----------
    geom = per_layer_kv_geometry(text_model)
    compressors: Dict[int, Any] = {}
    kl_bits_per_head: Optional[float] = None
    if args.compress_full_attn:
        for li in full_attn_idx:
            n_kv, hd, _ = geom[li]
            comp = make_default_compressor(
                head_dim=hd, device=torch.device("cpu"),
                prefer_kakeya=True, lattice=args.kl_lattice, q_range=args.kl_q_range,
            )
            compressors[li] = comp
            codec = getattr(comp, "_codec", None)
            if codec is not None and kl_bits_per_head is None:
                kl_bits_per_head = float(getattr(codec, "bits_per_token_per_head", 0)) or None
        print(f"[mac] KakeyaLattice compression ON for full-attn layers "
              f"({args.kl_lattice} Q{args.kl_q_range}); "
              f"bits/token/head={kl_bits_per_head}", file=sys.stderr)

    def _compress_roundtrip(li: int, k_mx: Any, v_mx: Any):
        """Lossy KakeyaLattice round-trip of a full-attn layer's pre-norm K/V.
        mx [B,T,n_kv,hd] → torch [B,n_kv,T,hd] (positions=-2) → codec → back."""
        comp = compressors[li]
        kt = mx_to_torch(k_mx, dtype=torch.float32, device="cpu").transpose(1, 2).contiguous()
        vt = mx_to_torch(v_mx, dtype=torch.float32, device="cpu").transpose(1, 2).contiguous()
        T = kt.shape[-2]
        pos = torch.arange(T)
        comp.compress(kt, vt, pos)
        kh, vh = comp.decompress(pos)
        comp.evict(pos)  # keep state bounded between tokens
        kh = kh.transpose(1, 2).contiguous()   # [B,T,n_kv,hd]
        vh = vh.transpose(1, 2).contiguous()
        return torch_to_mx(kh), torch_to_mx(vh)

    # ---------- Drafter K/V capture (Mac): MLX embed → torch → drafter layers ----------
    def capture_drafter_kv(ids: List[int]):
        ids_mx = mx.array([ids])
        emb_mx = text_model.embed_tokens(ids_mx)
        embedded = mx_to_torch(emb_mx, dtype=torch.float32, device=dev)  # [1,T,H]
        layers = list(drafter.layers)
        k_cap: List[Optional[torch.Tensor]] = [None] * len(layers)
        v_cap: List[Optional[torch.Tensor]] = [None] * len(layers)
        handles = []
        for i, layer in enumerate(layers):
            a = layer.self_attn
            handles.append(a.k_proj.register_forward_hook(
                lambda m, inp, out, i=i: k_cap.__setitem__(i, out.detach())))
            handles.append(a.v_proj.register_forward_hook(
                lambda m, inp, out, i=i: v_cap.__setitem__(i, out.detach())))
        try:
            with torch.no_grad():
                T = embedded.size(1)
                qpos = torch.arange(T, device=dev)
                h = embedded
                for layer in layers:
                    h = layer(h, qpos, ctx_k=None, ctx_v=None)
        finally:
            for hh in handles:
                hh.remove()
        dh, ddim = fcfg.drafter_num_kv_heads, fcfg.drafter_head_dim
        d_k = [k_cap[i].view(1, -1, dh, ddim) for i in range(len(layers))]
        d_v = [v_cap[i].view(1, -1, dh, ddim) for i in range(len(layers))]
        return d_k, d_v

    # ---------- Per-sample restoration (amortized: captured ONCE over the
    # prompt, reused for all decode steps). The evicted positions are the
    # fixed prompt mid-context; with <= window generated tokens nothing else
    # is evicted, so the prompt's restored K/V cover every injected slot.
    exact_set = set(range(n_layers)) if args.identity_restore else set(full_attn_idx)

    def build_restoration(prompt_ids: List[int], *, prefill_native_s5: bool = False):
        """Build restored K/V banks.

        For incremental/fused S5 decode, full-attention exact K/V should come
        from the same MLX prefill that populates the native cache. Supplying no
        restored bank for those layers lets mlx_lm store their own post-RoPE
        cache directly and avoids the extra clean verifier forward.
        """
        if (prefill_native_s5 and args.s5_exact_full_attn
                and not args.identity_restore and not args.force_f_theta):
            return {}, {}, len(prompt_ids)
        if drafter is None or f_theta is None or fcfg is None:
            raise RuntimeError("drafter/f_theta are required for this restoration mode")
        d_k, d_v = capture_drafter_kv(prompt_ids)
        with torch.no_grad():
            vk, vv = f_theta.forward_kv_pack(d_k, d_v)
        own = None
        if exact_set and not prefill_native_s5:
            own = capture_own_kv(mlx_model, prompt_ids)
        rk: Dict[int, Any] = {}
        rv: Dict[int, Any] = {}
        for li in range(n_layers):
            if src_map[li] != li:
                continue
            if prefill_native_s5 and li in exact_set:
                continue
            if li in exact_set and own is not None and li in own:
                k_mx, v_mx = own[li]
                if li in compressors:
                    k_mx, v_mx = _compress_roundtrip(li, k_mx, v_mx)
                rk[li], rv[li] = k_mx, v_mx
            else:
                rk[li], rv[li] = torch_to_mx(vk[li]), torch_to_mx(vv[li])
        return rk, rv, len(prompt_ids)

    def _pad(rdict, t_src, t_dst):
        if t_dst <= t_src:
            return rdict
        out = {}
        for li, a in rdict.items():
            pad = mx.zeros((a.shape[0], t_dst - t_src, a.shape[2], a.shape[3]), dtype=a.dtype)
            out[li] = mx.concatenate([a, pad], axis=1)
        return out

    def restored_forward(ids: List[int], rk, rv, t_src, *, return_all: bool):
        T = len(ids)
        evicted = compute_evicted_positions(T, args.sink_size, args.window_size)
        if not evicted:
            out = mlx_model(mx.array([ids])); mx.eval(out)
            return out[0] if return_all else out[0, -1]
        return restored_logits(
            mlx_model, ids,
            restored_k_per_layer=_pad(rk, t_src, T),
            restored_v_per_layer=_pad(rv, t_src, T),
            evicted_positions=evicted, return_all=return_all,
        )

    # ---------- Dataset ----------
    if args.code_prompts:
        _CODE = [
            "Write a complete Python implementation of a binary search tree class "
            "with insert, search, and in-order traversal methods. Include type "
            "hints and docstrings.",
            "Implement a Python LRU cache class with get and put methods using an "
            "OrderedDict. Include type hints and docstrings.",
            "Write a Python function that parses a CSV string into a list of dicts, "
            "correctly handling quoted fields and embedded commas. Add error handling.",
            "Implement quicksort in Python with an in-place partition helper. "
            "Include docstrings and a small example in a __main__ block.",
            "Write a Python class for a fixed-capacity ring buffer with push, pop, "
            "and is_full methods, raising on overflow. Include type hints.",
            "Implement a recursive descent parser in Python for arithmetic "
            "expressions with + - * / and parentheses. Return the evaluated value.",
            "Write a Python decorator `retry` that retries a function up to n times "
            "with exponential backoff on exception. Include type hints and docstring.",
            "Implement a thread-safe counter class in Python using threading.Lock, "
            "with increment, decrement, and value methods.",
        ]
        n = min(args.n_samples, len(_CODE))
        samples: List[NIAHSample] = [
            NIAHSample(prompt_text=p, answer_text="", needle_line_index=0,
                       needle_text="")
            for p in _CODE[:n]
        ]
        print(f"[mac] CODE-PROMPTS workload: {n} prompts (recall N/A; "
              f"measuring decode throughput + acceptance)", file=sys.stderr)
    else:
        samples = make_niah_dataset(
            n_samples=args.n_samples,
            haystack_min_lines=args.haystack_min_lines,
            haystack_max_lines=args.haystack_max_lines,
            seed=args.seed,
        )

    def encode(prompt_text: str) -> List[int]:
        if args.direct_answer_prompt:
            # The legacy padding repeats "answer" on every line; Gemma4 can
            # latch onto that distractor in very short completion smokes.
            # Keep the needle/question unchanged but make filler semantically
            # neutral so recall measures retrieval of the secret code.
            prompt_text = prompt_text.replace(
                "and does not contain the answer.",
                "and is unrelated filler.",
            )
            prompt_text = (
                prompt_text
                + "\n\nReturn only the secret code in PREFIX-NNNN format. "
                  "Do not explain, reason, or add any other text."
            )
        if args.direct_answer_prompt and args.raw_completion_prompt:
            try:
                ids = tokenizer.encode(prompt_text, add_special_tokens=True)
            except TypeError:
                ids = tokenizer.encode(prompt_text)
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return list(ids)
        msgs = [{"role": "user", "content": prompt_text}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = list(ids)
        if args.direct_answer_prompt and args.content_channel_prefill:
            try:
                marker = tokenizer.encode(
                    "<|channel>content\n<channel|>", add_special_tokens=False)
            except TypeError:
                marker = tokenizer.encode("<|channel>content\n<channel|>")
            if hasattr(marker, "tolist"):
                marker = marker.tolist()
            ids.extend(list(marker))
        return ids

    def encode_answer(answer_text: str) -> List[int]:
        try:
            aid = tokenizer.encode(answer_text, add_special_tokens=False)
        except TypeError:
            aid = list(tokenizer.encode(answer_text))
            bos = getattr(tokenizer, "bos_token_id", None)
            if aid and bos is not None and aid[0] == bos:
                aid = aid[1:]
        return list(aid)

    sample_ids = [encode(s.prompt_text) for s in samples]
    answer_ids = [encode_answer(s.answer_text) for s in samples]
    seq_lens = [len(t) for t in sample_ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    end_ids = set()
    if eos_id is not None:
        end_ids.add(int(eos_id))
    try:
        eot_ids = tokenizer.encode("<turn|>", add_special_tokens=False)
    except TypeError:
        eot_ids = tokenizer.encode("<turn|>")
    if hasattr(eot_ids, "tolist"):
        eot_ids = eot_ids.tolist()
    if (not args.ignore_turn_stop) and len(eot_ids) == 1:
        end_ids.add(int(eot_ids[0]))
    print(f"[mac] {len(samples)} samples, prompt len "
          f"min={min(seq_lens)} max={max(seq_lens)}", file=sys.stderr)

    def native_prefill(input_ids: List[int]):
        """Prefill the verifier cache on MLX's native path, optionally chunked."""
        cache = (getattr(mlx_model, "make_cache", lambda: None)())
        chunk = int(args.prefill_chunk_size)
        if chunk <= 0 or len(input_ids) <= chunk:
            out = mlx_model(mx.array([input_ids]), cache=cache)
            mx.eval(out)
            return cache, out[0, -1]

        last = None
        for start in range(0, len(input_ids), chunk):
            part = input_ids[start:start + chunk]
            if not part:
                continue
            last = mlx_model(mx.array([part]), cache=cache)
            mx.eval(last)
        if last is None:
            last = mlx_model(mx.array([input_ids]), cache=cache)
            mx.eval(last)
        return cache, last[0, -1]

    def warmup_decode() -> None:
        """Warm MLX/Metal decode kernels before comparing cross vs oracle."""
        if args.decode_warmup_tokens <= 0 or not sample_ids:
            return
        cache, logits = native_prefill(sample_ids[0])
        for _ in range(args.decode_warmup_tokens):
            tok = int(mx.argmax(logits).item())
            out = mlx_model(mx.array([[tok]]), cache=cache)
            mx.eval(out)
            logits = out[0, -1]
            if tok in end_ids:
                break

    def eval_teacher_forced(logits_all_fn) -> Tuple[List[str], List[float], List[int]]:
        """One restored forward per sample over [prompt + needle-code]; check
        the argmax at the answer span reproduces the code (substring predicate
        — same as CUDA). O(T) per sample, no autoregressive loop."""
        decoded, lats, toks = [], [], []
        for i, pid in enumerate(sample_ids):
            aid = answer_ids[i] or [eos_id or 0]
            full = pid + aid
            t0 = time.perf_counter()
            logits_all = logits_all_fn(pid, full)  # [T_full, V]
            Tp = len(pid)
            preds = [int(mx.argmax(logits_all[Tp - 1 + j]).item())
                     for j in range(len(aid))]
            lats.append(time.perf_counter() - t0)
            decoded.append(tokenizer.decode(preds))
            toks.append(len(aid))
            print(f"[mac]   sample {i}: T={seq_lens[i]} pred[:48]={decoded[-1][:48]!r}",
                  file=sys.stderr)
        return decoded, lats, toks

    def eval_free_gen_cross() -> Tuple[List[str], List[float], List[int]]:
        """Restored free generation: 1 restored full forward per token
        (amortized restoration). Correct recall metric; slow on M4."""
        decoded, lats, toks = [], [], []
        rows = []
        for i, pid in enumerate(sample_ids):
            e2e_t0 = time.perf_counter()
            build_t0 = time.perf_counter()
            rk, rv, tsrc = build_restoration(pid)
            build_s = time.perf_counter() - build_t0
            cur = list(pid); gen: List[int] = []
            t0 = time.perf_counter()
            for _ in range(args.max_new_tokens):
                last = restored_forward(cur, rk, rv, tsrc, return_all=False)
                nxt = int(mx.argmax(last).item()); gen.append(nxt)
                if nxt in end_ids:
                    break
                cur.append(nxt)
            decode_s = time.perf_counter() - t0
            e2e_s = time.perf_counter() - e2e_t0
            lats.append(e2e_s)
            decoded.append(tokenizer.decode(gen)); toks.append(len(gen))
            rows.append({
                "sample": i,
                "build_restoration_s": round(build_s, 3),
                "decode_s": round(decode_s, 3),
                "e2e_s": round(e2e_s, 3),
                "restoration_active": True,
                "decode_loop": "full_reforward_per_token",
            })
            print(f"[mac]   sample {i}: T={seq_lens[i]} -> {decoded[-1][:48]!r}",
                  file=sys.stderr)
        eval_free_gen_cross.stage_rows = rows
        return decoded, lats, toks

    def eval_free_gen_cross_incremental() -> Tuple[List[str], List[float], List[int]]:
        """INCREMENTAL restored free generation (MLX port of CUDA Gap-A):
        prefill ONCE capturing restored K/V into a persistent cache, then
        decode with mlx_lm's native incremental step (O(L)/token). Fixes the
        per-token re-forward throughput collapse. Recall via S5 full-attn."""
        decoded, lats, toks = [], [], []
        rows = []
        for i, pid in enumerate(sample_ids):
            e2e_t0 = time.perf_counter()
            build_t0 = time.perf_counter()
            rk, rv, tsrc = build_restoration(pid, prefill_native_s5=True)
            build_s = time.perf_counter() - build_t0
            T = len(pid)
            evicted = compute_evicted_positions(T, args.sink_size, args.window_size)
            prefill_t0 = time.perf_counter()
            if not evicted:
                cache, first = native_prefill(pid)
            else:
                cache, first = restored_prefill_cache(
                    mlx_model, pid,
                    restored_k_per_layer=_pad(rk, tsrc, T),
                    restored_v_per_layer=_pad(rv, tsrc, T),
                    evicted_positions=evicted,
                    prefill_chunk_size=args.prefill_chunk_size)
            prefill_s = time.perf_counter() - prefill_t0
            decode_t0 = time.perf_counter()
            gen = restored_incremental_generate(
                mlx_model, cache, first,
                max_tokens=args.max_new_tokens,
                eos_ids=end_ids)
            decode_s = time.perf_counter() - decode_t0
            e2e_s = time.perf_counter() - e2e_t0
            lats.append(e2e_s)
            decoded.append(tokenizer.decode(gen)); toks.append(len(gen))
            rows.append({
                "sample": i,
                "build_restoration_s": round(build_s, 3),
                "prefill_s": round(prefill_s, 3),
                "decode_s": round(decode_s, 3),
                "e2e_s": round(e2e_s, 3),
                "restoration_active": True,
                "decode_loop": "generate_step",
            })
            print(f"[mac]   incr {i}: T={seq_lens[i]} "
                  f"prefill={prefill_s:.1f}s decode={decode_s:.1f}s "
                  f"-> {decoded[-1][:48]!r}", file=sys.stderr)
        eval_free_gen_cross_incremental.stage_rows = rows
        return decoded, lats, toks

    def eval_fused_specdecode() -> Tuple[List[str], List[float], List[int]]:
        """FUSED DFlash spec-decode (MLX port of #107 A+B+C): drafter context
        K/V cache + aux captured from the verify forward + incremental restored
        verify with trim_prompt_cache accept/reject. Target: tok/s > AR."""
        argmax_fn = lambda row: int(mx.argmax(row).item())
        active_drafter = mlx_drafter if mlx_drafter is not None else drafter
        aux_layer_ids = (tuple(active_drafter.cfg.aux_layer_ids)
                         if active_drafter is not None else ())
        softcap = None
        for obj in (getattr(mlx_model, "language_model", None), mlx_model):
            cap = getattr(obj, "final_logit_softcapping", None) if obj is not None else None
            if cap:
                softcap = float(cap); break
        if mlx_drafter is not None:
            # All-MLX path (Step-2 rescue): drafter, embed/lm_head, aux
            # slices, positions, and concat all stay on the Metal stream —
            # zero mx<->torch crossings per block.
            from inference_engine.backends.mlx.dflash_drafter import (
                make_native_embed_lm_head,
            )
            bridge = None  # identity: aux slices stay mx
            embed_fn, lm_head_fn = make_native_embed_lm_head(
                text_model, softcap=softcap)
            arange_fn = lambda s, e: mx.arange(int(s), int(e))
            cat_aux_fn = lambda parts: (
                parts[0][None] if len(parts) == 1
                else mx.concatenate(list(parts), axis=0)[None])
        elif args.force_fused_specdecode:
            if drafter is None:
                raise RuntimeError("--force-fused-specdecode requires drafter/f_theta")
            bridge = lambda a: mx_to_torch(a, dtype=torch.float32, device=dev)
            embed_fn, lm_head_fn = make_bridge_embed_lm_head(
                text_model, mx_to_torch=mx_to_torch, torch_to_mx=torch_to_mx,
                device=dev, torch_dtype=torch.float32, softcap=softcap)
            arange_fn = lambda s, e: torch.arange(int(s), int(e), device=dev)
            cat_aux_fn = lambda parts: torch.cat(list(parts), dim=0).unsqueeze(0)
        else:
            bridge = lambda a: mx_to_torch(a, dtype=torch.float32, device=dev)
            embed_fn = lm_head_fn = arange_fn = cat_aux_fn = None
        adapter = MLXRestoredIncrementalVerifier(
            mlx_model, embed_scale=embed_scale, aux_layer_ids=aux_layer_ids,
            bridge_to_torch=bridge)

        def _run_fused_chat() -> Tuple[List[str], List[float], List[int]]:
            """Interactive/scripted chat on the FULL fused engine — reuses the
            EXACT per-turn sequence of the eval loop below (build_restoration →
            S5 prefill → aux capture → fused_specdecode_generate_mlx_trim), so the
            gemma-4 verifier + DFlash proposer + S5 bounded KV are all live. NOT
            verifier-only: each turn the proposer drafts blocks the verifier
            accepts (reports blocks / mean_accept_len). On gemma-4 f_θ restoration
            is bypassed via S5 native exact-layer prefill (the free lunch);
            f_θ is load-bearing on full-attention models."""
            _allmlx_ok = mlx_drafter is not None and args.cuda_trim
            _torch_ftheta_ok = drafter is not None and f_theta is not None
            if not (args.force_fused_specdecode and (_allmlx_ok or _torch_ftheta_ok)):
                raise SystemExit(
                    "--chat needs the FULL fused engine. Either:\n"
                    "  (a) --fused-specdecode --all-mlx-drafter --s5-exact-full-attn "
                    "--cuda-trim  (verifier + proposer; f_θ bypassed on gemma-4 "
                    "via S5), or\n"
                    "  (b) --fused-specdecode --force-f-theta  (torch DFlash drafter "
                    "+ f_θ that ACTUALLY RUNS; do NOT pass --all-mlx-drafter).")
            # Stop at gemma's natural turn end: <end_of_turn> + eos
            # (convert_tokens_to_ids is the reliable special-token lookup).
            chat_eos = set(end_ids)
            unk = getattr(tokenizer, "unk_token_id", None)
            native = getattr(tokenizer, "eos_token_ids", None)
            if native:
                chat_eos |= {int(x) for x in native}
            for m in ("<end_of_turn>", "<eos>"):
                try:
                    tid = tokenizer.convert_tokens_to_ids(m)
                except Exception:
                    tid = None
                if isinstance(tid, int) and tid >= 0 and tid != unk:
                    chat_eos.add(int(tid))

            def _encode_chat(history: List[Dict[str, str]]) -> List[int]:
                try:
                    cids = tokenizer.apply_chat_template(
                        history, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    cids = tokenizer.apply_chat_template(
                        history, add_generation_prompt=True)
                return list(cids.tolist() if hasattr(cids, "tolist") else cids)

            def _gen_turn(pid: List[int]) -> Dict[str, Any]:
                rk, rv, tsrc = build_restoration(pid, prefill_native_s5=True)
                # f_θ ran iff build_restoration produced restored banks via the
                # torch drafter+f_θ (under --force-f-theta the S5 short-circuit is
                # bypassed → rk holds f_θ-projected sliding-layer K/V).
                f_theta_ran = bool(rk) and (drafter is not None and f_theta is not None)
                T = len(pid)
                evicted = compute_evicted_positions(
                    T, args.sink_size, args.window_size)
                aux_prompt_mx = capture_aux_hidden(
                    mlx_model, pid, aux_layer_ids, embed_scale=embed_scale)
                aux_prompt = (aux_prompt_mx if bridge is None
                              else [bridge(a) for a in aux_prompt_mx])
                adapter.prefill(
                    pid, restored_k_per_layer=_pad(rk, tsrc, T),
                    restored_v_per_layer=_pad(rv, tsrc, T),
                    evicted_positions=evicted,
                    prefill_chunk_size=args.prefill_chunk_size, full_kv=args.cuda_trim)
                t0 = time.perf_counter()
                if mlx_drafter is not None and args.cuda_trim:
                    res = fused_specdecode_generate_mlx_trim(
                        adapter, active_drafter, aux_prompt=aux_prompt,
                        embed_fn=embed_fn, lm_head_fn=lm_head_fn,
                        gen_tokens=args.max_new_tokens, block_size=args.block_size,
                        eos_ids=chat_eos, single_fused=args.single_fused)
                elif mlx_drafter is not None:
                    res = fused_specdecode_generate_mlx(
                        adapter, active_drafter, aux_prompt=aux_prompt,
                        embed_fn=embed_fn, lm_head_fn=lm_head_fn,
                        gen_tokens=args.max_new_tokens, block_size=args.block_size,
                        eos_ids=chat_eos)
                else:
                    res = fused_specdecode_generate(
                        adapter, active_drafter, aux_prompt=aux_prompt,
                        embed_fn=embed_fn, lm_head_fn=lm_head_fn,
                        gen_tokens=args.max_new_tokens, block_size=args.block_size,
                        eos_ids=chat_eos, argmax_fn=argmax_fn, arange_fn=arange_fn,
                        cat_aux_fn=cat_aux_fn, allow_greedy_fallback=False)
                res["decode_s"] = round(time.perf_counter() - t0, 3)
                res["f_theta_ran"] = f_theta_ran
                res["f_theta_layers"] = sorted(rk.keys()) if rk else []
                try:
                    txt = tokenizer.decode(res["tokens"], skip_special_tokens=True)
                except TypeError:
                    txt = tokenizer.decode(res["tokens"])
                for marker in ("<turn|>", "<end_of_turn>", "<eos>"):
                    txt = txt.replace(marker, "")
                # gemma-4 sometimes bleeds its reasoning channel after the answer
                # (e.g. a trailing "\nthought ...") — cut at the first channel
                # marker so the chat shows only the natural-language answer.
                for cut in ("<|channel", "<channel", "\nthought", "\nthink"):
                    idx = txt.find(cut)
                    if idx > 0:
                        txt = txt[:idx]
                res["text"] = txt.strip()
                res["resident_kv_bytes"] = int(
                    sum(int(getattr(c, "nbytes", 0)) for c in (adapter._cache or [])))
                return res

            print(f"[chat] FULL fused engine: verifier={args.verifier_path} "
                  f"drafter={args.drafter_id} f_theta={args.f_theta_dir} "
                  f"S5 sink={args.sink_size} window={args.window_size} "
                  f"block={args.block_size} | chat_eos={sorted(chat_eos)}",
                  file=sys.stderr, flush=True)

            history: List[Dict[str, str]] = []
            if args.chat_scripted is not None:
                turns = [t for t in args.chat_scripted.split("||") if t.strip()]
                transcript = []
                for u in turns:
                    history.append({"role": "user", "content": u})
                    res = _gen_turn(_encode_chat(history))
                    history.append({"role": "assistant", "content": res["text"]})
                    tps = (res["decode_tokens"] / res["decode_s"]
                           if res["decode_s"] > 0 else 0.0)
                    transcript.append({
                        "user": u, "text": res["text"],
                        "tokens": res["decode_tokens"], "blocks": res["blocks"],
                        "mean_accept_len": res["mean_accept_len"],
                        "f_theta_ran": res["f_theta_ran"],
                        "f_theta_layers": res["f_theta_layers"],
                        "decode_s": res["decode_s"], "decode_tps": round(tps, 2),
                        "resident_kv_bytes": res["resident_kv_bytes"]})
                    print(f"[chat] USER {u!r}", file=sys.stderr, flush=True)
                    print(f"[chat] GEMMA-4 {res['text'][:200]!r} (blocks="
                          f"{res['blocks']}, accept_len={res['mean_accept_len']}, "
                          f"f_theta_ran={res['f_theta_ran']} "
                          f"layers={res['f_theta_layers']}, "
                          f"{round(tps,2)} tok/s, kv={res['resident_kv_bytes']/1e6:.1f}MB)",
                          file=sys.stderr, flush=True)
                report = {
                    "kind": "mac_gemma4_kakeya_fused_chat", "schema_version": 1,
                    "engine": ("Kakeya-for-Mac fused spec-decode (gemma-4 verifier "
                               "+ DFlash proposer + S5 bounded KV; f_θ restoration "
                               "bypassed on gemma-4 via S5 native exact-layer "
                               "prefill — the free lunch — and load-bearing on "
                               "full-attention models)"),
                    "model_path": args.verifier_path, "drafter_id": args.drafter_id,
                    "f_theta_dir": args.f_theta_dir, "sink": args.sink_size,
                    "window": args.window_size, "block_size": args.block_size,
                    "exact_layers": full_attn_idx, "chat_eos": sorted(chat_eos),
                    # §4 liveness contract: f_θ is INTENDED on the torch path
                    # (not --all-mlx-drafter); the evidence gate asserts
                    # f_theta_ran on every turn when this is true.
                    "f_theta_intended": mlx_drafter is None,
                    "fallbacks_taken": [],
                    "turns": transcript}
                if args.output:
                    op = Path(args.output)
                    op.parent.mkdir(parents=True, exist_ok=True)
                    op.write_text(json.dumps(report, indent=2), encoding="utf-8")
                    print(f"[chat] wrote transcript -> {op}", file=sys.stderr)
                else:
                    print(json.dumps(report, indent=2))
                raise SystemExit(0)

            print("[chat] ready. Type a message; blank line / Ctrl-D quits.",
                  file=sys.stderr, flush=True)
            while True:
                if sys.stdin.isatty():
                    sys.stderr.write("\nyou> "); sys.stderr.flush()
                line = sys.stdin.readline()
                if not line:
                    break
                u = line.strip()
                if not u:
                    break
                history.append({"role": "user", "content": u})
                res = _gen_turn(_encode_chat(history))
                history.append({"role": "assistant", "content": res["text"]})
                tps = (res["decode_tokens"] / res["decode_s"]
                       if res["decode_s"] > 0 else 0.0)
                sys.stdout.write("gemma-4> " + res["text"] + "\n")
                sys.stdout.flush()
                print(f"[chat] blocks={res['blocks']} accept_len="
                      f"{res['mean_accept_len']} {round(tps,2)} tok/s "
                      f"bounded-KV {res['resident_kv_bytes']/1e6:.1f}MB",
                      file=sys.stderr, flush=True)
            raise SystemExit(0)

        if args.chat:
            _run_fused_chat()

        decoded, lats, toks = [], [], []
        rows = []
        for i, pid in enumerate(sample_ids):
            e2e_t0 = time.perf_counter()
            build_t0 = time.perf_counter()
            if adaptive_s5_native:
                rk, rv, tsrc = {}, {}, len(pid)
            else:
                rk, rv, tsrc = build_restoration(pid, prefill_native_s5=True)
            build_s = time.perf_counter() - build_t0
            T = len(pid)
            evicted = compute_evicted_positions(T, args.sink_size, args.window_size)
            aux_t0 = time.perf_counter()
            if args.force_fused_specdecode:
                aux_prompt_mx = capture_aux_hidden(
                    mlx_model, pid, aux_layer_ids, embed_scale=embed_scale)
                if bridge is None:
                    aux_prompt = aux_prompt_mx  # all-MLX: stay on Metal
                else:
                    aux_prompt = [bridge(a) for a in aux_prompt_mx]  # [1,C,H] torch
            else:
                aux_prompt = []
            aux_s = time.perf_counter() - aux_t0
            prefill_t0 = time.perf_counter()
            if adaptive_s5_native:
                cache, first = native_prefill(pid)
                adapter._cache = cache
                adapter.next_token_logits = first
                adapter._past_len = len(pid)
            else:
                adapter.prefill(
                    pid,
                    restored_k_per_layer=_pad(rk, tsrc, T),
                    restored_v_per_layer=_pad(rv, tsrc, T),
                    evicted_positions=evicted,
                    prefill_chunk_size=args.prefill_chunk_size,
                    full_kv=args.cuda_trim)
            prefill_s = time.perf_counter() - prefill_t0
            t0 = time.perf_counter()
            if args.force_fused_specdecode:
                if mlx_drafter is not None and args.cuda_trim:
                    # CUDA-parity: keep accepted K/V, trim only rejected.
                    res = fused_specdecode_generate_mlx_trim(
                        adapter, active_drafter, aux_prompt=aux_prompt,
                        embed_fn=embed_fn, lm_head_fn=lm_head_fn,
                        gen_tokens=args.max_new_tokens,
                        block_size=args.block_size, eos_ids=end_ids,
                        single_fused=args.single_fused)
                elif mlx_drafter is not None:
                    # Single-sync all-MLX loop (levers ①②③) + v3 carry rollback.
                    res = fused_specdecode_generate_mlx(
                        adapter, active_drafter, aux_prompt=aux_prompt,
                        embed_fn=embed_fn, lm_head_fn=lm_head_fn,
                        gen_tokens=args.max_new_tokens,
                        block_size=args.block_size, eos_ids=end_ids)
                else:
                    res = fused_specdecode_generate(
                        adapter, active_drafter, aux_prompt=aux_prompt,
                        embed_fn=embed_fn, lm_head_fn=lm_head_fn,
                        gen_tokens=args.max_new_tokens, block_size=args.block_size,
                        eos_ids=end_ids,
                        argmax_fn=argmax_fn, arange_fn=arange_fn, cat_aux_fn=cat_aux_fn,
                        allow_greedy_fallback=False)
                res["drafter_runtime"] = "mlx" if mlx_drafter is not None else "torch"
            else:
                t_greedy = time.perf_counter()
                adapter._capture_aux = False
                gen = []
                logits_row = adapter.next_token_logits
                while len(gen) < args.max_new_tokens:
                    tok = int(argmax_fn(logits_row))
                    gen.append(tok)
                    out = mlx_model(mx.array([[tok]]), cache=adapter._cache)
                    mx.eval(out)
                    logits_row = out[0, -1]
                    if tok in end_ids:
                        break
                adapter.next_token_logits = logits_row
                res = {
                    "tokens": gen,
                    "blocks": 0,
                    "mean_accept_len": 0.0,
                    "decode_tokens": len(gen),
                    "adaptive_mode": "native_ar_baseline",
                    "time_breakdown_s": {
                        "greedy_decode_s": round(time.perf_counter() - t_greedy, 3)
                    },
                }
            decode_s = time.perf_counter() - t0
            e2e_s = time.perf_counter() - e2e_t0
            lats.append(e2e_s)
            gen = res["tokens"]
            decoded.append(tokenizer.decode(gen)); toks.append(len(gen))
            rows.append({
                "sample": i,
                "build_restoration_s": round(build_s, 3),
                "aux_prompt_capture_s": round(aux_s, 3),
                "prefill_s": round(prefill_s, 3),
                "decode_s": round(decode_s, 3),
                "e2e_s": round(e2e_s, 3),
                "restoration_active": not adaptive_s5_native,
                "decode_loop": ("fused_specdecode" if args.force_fused_specdecode
                                else "per_token_eval"),
                "fused": res,
            })
            print(f"[mac]   fused {i}: T={seq_lens[i]} acc_len={res['mean_accept_len']} "
                  f"-> {decoded[-1][:48]!r}", file=sys.stderr)
        eval_fused_specdecode.stage_rows = rows
        return decoded, lats, toks

    def eval_free_gen_oracle() -> Tuple[List[str], List[float], List[int]]:
        """Oracle free generation using mlx's NATIVE incremental KV cache.

        Decodes via ``restored_incremental_generate`` (mlx_lm
        ``generate_step``: chunked + async-pipelined) — the SAME decode
        primitive as the cross incremental path. The previous hand-rolled
        per-token ``mx.eval`` loop is the documented MLX anti-pattern
        (docs/mlx-port-lessons.md) and depressed the baseline; the gate
        rule SPEEDUP_ORACLE_LOOP rejects headline speedups measured
        against it.
        """
        decoded, lats, toks = [], [], []
        rows = []
        for i, pid in enumerate(sample_ids):
            e2e_t0 = time.perf_counter()
            prefill_t0 = time.perf_counter()
            cache, logits = native_prefill(pid)
            prefill_s = time.perf_counter() - prefill_t0
            decode_t0 = time.perf_counter()
            gen = restored_incremental_generate(
                mlx_model, cache, logits,
                max_tokens=args.max_new_tokens,
                eos_ids=end_ids)
            decode_s = time.perf_counter() - decode_t0
            e2e_s = time.perf_counter() - e2e_t0
            lats.append(e2e_s)
            decoded.append(tokenizer.decode(gen)); toks.append(len(gen))
            rows.append({
                "sample": i,
                "prefill_s": round(prefill_s, 3),
                "decode_s": round(decode_s, 3),
                "e2e_s": round(e2e_s, 3),
                "decode_loop": "generate_step",
            })
            print(f"[mac]   oracle {i}: T={seq_lens[i]} -> {decoded[-1][:48]!r}",
                  file=sys.stderr)
        eval_free_gen_oracle.stage_rows = rows
        return decoded, lats, toks

    def cross_logits_all(prompt_ids, full_ids):
        rk, rv, tsrc = build_restoration(prompt_ids)
        return restored_forward(full_ids, rk, rv, tsrc, return_all=True)

    def oracle_logits_all(prompt_ids, full_ids):
        out = mlx_model(mx.array([full_ids])); mx.eval(out); return out[0]

    label = "identity" if args.identity_restore else (
        "s5" if args.s5_exact_full_attn else "f_theta_all")
    eval_mode = ("teacher_forced" if args.teacher_forced
                 else "native_ar_baseline" if adaptive_s5_native
                 else "free_gen_fused_specdecode" if args.fused_specdecode
                 else "free_gen_incremental" if args.incremental else "free_gen")
    warmup_decode()
    print(f"[mac] running restored cross-model verifier ({label}, {eval_mode})",
          file=sys.stderr, flush=True)
    if args.teacher_forced:
        cross_dec, cross_lat, cross_tok = eval_teacher_forced(cross_logits_all)
        # Diagnostic mode: one restored forward per sample; synthesize the
        # per-sample path-identity rows the evidence gate requires.
        cross_rows = [
            {"sample": i, "restoration_active": True,
             "decode_loop": "teacher_forced_single_forward"}
            for i in range(len(sample_ids))
        ]
    elif args.fused_specdecode:
        cross_dec, cross_lat, cross_tok = eval_fused_specdecode()
        cross_rows = getattr(eval_fused_specdecode, "stage_rows", [])
    elif args.incremental:
        cross_dec, cross_lat, cross_tok = eval_free_gen_cross_incremental()
        cross_rows = getattr(eval_free_gen_cross_incremental, "stage_rows", [])
    else:
        cross_dec, cross_lat, cross_tok = eval_free_gen_cross()
        cross_rows = getattr(eval_free_gen_cross, "stage_rows", [])
    sut_label = (NATIVE_BASELINE_LABEL if adaptive_s5_native
                 else "restored_cross_model")
    cross_name = ("native_ar_baseline_mac" if adaptive_s5_native
                  else "k3_cross_model_mac")
    cross_res = aggregate_recall(cross_name, samples, cross_dec, cross_lat, cross_tok)
    print(f"[mac] {sut_label} recall = {cross_res.recall:.3f} "
          f"({cross_res.samples_correct}/{cross_res.samples_total})", file=sys.stderr)

    oracle_res = None
    o_lat: List[float] = []
    o_tok: List[int] = []
    if not args.skip_oracle:
        print("[mac] running oracle", file=sys.stderr, flush=True)
        if args.teacher_forced:
            o_dec, o_lat, o_tok = eval_teacher_forced(oracle_logits_all)
        else:
            o_dec, o_lat, o_tok = eval_free_gen_oracle()  # fast native incremental
        oracle_res = aggregate_recall("oracle_mac", samples, o_dec, o_lat, o_tok)
        print(f"[mac] oracle recall = {oracle_res.recall:.3f}", file=sys.stderr)

    # ---------- KV-memory accounting (bounded S5 engine) ----------
    T_max = max(seq_lens)
    exact_for_mem = full_attn_idx  # S5: full-attn layers kept exact / compressed
    mem_s5 = kv_memory_report(
        text_model, sink_size=args.sink_size, window_size=args.window_size,
        seq_len=T_max, exact_layer_indices=exact_for_mem,
        compress_full_bits_per_token_per_head=(
            kl_bits_per_head if args.compress_full_attn else None),
    )
    # Baselines for the savings story:
    mem_naive = kv_memory_report(   # all layers O(T), no bound, no compress
        text_model, sink_size=T_max, window_size=0, seq_len=T_max,
        exact_layer_indices=list(range(n_layers)))
    print(f"[mac] KV resident @T={T_max}: S5={mem_s5['total_resident_mb']} MB "
          f"(growth {mem_s5['per_token_growth_kb']} KB/tok); "
          f"naive-full={mem_naive['total_resident_mb']} MB", file=sys.stderr)

    # ---------- Throughput ----------
    def _tps(lats, toks):
        tot_t = sum(lats)
        tot_n = sum(toks)
        return {
            "tokens": tot_n, "wall_seconds": round(tot_t, 3),
            "tokens_per_second": round(tot_n / tot_t, 4) if tot_t > 0 else None,
            "mean_latency_per_sample_s": round(tot_t / max(len(lats), 1), 3),
        }
    cross_tps = _tps(cross_lat, cross_tok)
    cross_tps["timing_scope"] = "e2e_prefill_plus_decode"
    cross_tps["eval_mode"] = eval_mode
    cross_tps["restored_forwards_per_sample"] = (
        1 if args.teacher_forced else args.max_new_tokens)
    cross_tps["stage_timings"] = cross_rows
    oracle_rows = getattr(eval_free_gen_oracle, "stage_rows", [])
    oracle_tps = _tps(o_lat, o_tok) if (o_lat and o_tok) else None
    if oracle_tps:
        oracle_tps["timing_scope"] = "e2e_prefill_plus_decode"
        oracle_tps["stage_timings"] = oracle_rows
        if oracle_rows and not args.teacher_forced:
            oracle_tps["decode_loop"] = "generate_step"

    # ---------- Headline speedup: emitted ONLY when admissible ----------
    # (k3_report_gate SPEEDUP_* rules; the harness withholds the number
    # rather than publish a claim its own gate would reject.)
    import statistics as _stats
    decode_only = decode_only_block(cross_rows, cross_tok, oracle_rows, o_tok)
    speedup_withheld: List[str] = []
    if adaptive_s5_native:
        speedup_withheld.append(
            "native_baseline_self_comparison: cross arm IS the oracle computation")
    if oracle_tps is None:
        speedup_withheld.append("no_oracle_arm")
    else:
        if len(cross_rows) < MIN_PERF_SAMPLES or len(oracle_rows) < MIN_PERF_SAMPLES:
            speedup_withheld.append(
                f"n_samples<{MIN_PERF_SAMPLES} (cross={len(cross_rows)}, "
                f"oracle={len(oracle_rows)})")
        tok_medians = [
            _stats.median(t) for t in (cross_tok, o_tok) if t
        ]
        if len(tok_medians) < 2 or min(tok_medians) < MIN_MEDIAN_DECODE_TOKENS:
            speedup_withheld.append(
                f"median_decode_tokens<{MIN_MEDIAN_DECODE_TOKENS} "
                f"(prefill-dominated wall time)")
        if decode_only is None:
            speedup_withheld.append("decode_only_medians_unavailable")
        if oracle_tps.get("decode_loop") != CLAIM_ORACLE_DECODE_LOOP:
            speedup_withheld.append(
                f"oracle_decode_loop!={CLAIM_ORACLE_DECODE_LOOP}")
        for arm_name, arm_rows in (("cross", cross_rows), ("oracle", oracle_rows)):
            spread = prefill_spread(arm_rows)
            if spread is not None and spread > MAX_PREFILL_SPREAD:
                speedup_withheld.append(
                    f"{arm_name}_prefill_spread {spread:.2f}x > "
                    f"{MAX_PREFILL_SPREAD}x (e2e ratio would be noise)")
    speedup_vs_oracle = None
    if not speedup_withheld and oracle_tps and oracle_tps["tokens_per_second"] \
            and cross_tps["tokens_per_second"]:
        speedup_vs_oracle = round(
            cross_tps["tokens_per_second"] / oracle_tps["tokens_per_second"], 3)
    if speedup_withheld:
        print("[mac] speedup WITHHELD (evidence gate): "
              + "; ".join(speedup_withheld), file=sys.stderr)
    print(f"[mac] cross-model throughput ({eval_mode}): "
          f"{cross_tps['tokens_per_second']} tok/s "
          f"({cross_tps['tokens']} tok / {cross_tps['wall_seconds']} s, "
          f"{cross_tps['mean_latency_per_sample_s']} s/sample)", file=sys.stderr)

    # ---------- Measured (not analytical) accelerator memory ----------
    def _mx_peak_mb() -> Optional[float]:
        for holder in (mx, getattr(mx, "metal", None)):
            fn = getattr(holder, "get_peak_memory", None) if holder is not None else None
            if callable(fn):
                try:
                    return round(float(fn()) / 1e6, 1)
                except Exception:
                    return None
        return None

    # The analytical sink+window table only describes runs where every
    # cross sample actually executed restoration with S5/identity exact
    # layers (k3_report_gate MEMORY_CLAIM_MISMATCH).
    restoration_all_active = bool(cross_rows) and all(
        bool(r.get("restoration_active")) for r in cross_rows)
    formula_matches_run = bool(
        restoration_all_active
        and (args.s5_exact_full_attn or args.identity_restore))

    delta = (abs(cross_res.recall - oracle_res.recall)
             if (oracle_res and not adaptive_s5_native) else None)
    report = {
        "schema_version": 2,
        "kind": "k3_integrated_niah_acceptance_mac",
        "config": {
            "native_baseline_bypass": bool(args.native_baseline_bypass),
            "all_mlx_drafter": bool(args.all_mlx_drafter),
            "block_size": args.block_size,
            "verifier_path": args.verifier_path,
            "drafter_id": args.drafter_id,
            "f_theta_dir": args.f_theta_dir,
            "n_samples": args.n_samples,
            "sink_size": args.sink_size,
            "window_size": args.window_size,
            "haystack_min_lines": args.haystack_min_lines,
            "haystack_max_lines": args.haystack_max_lines,
            "max_new_tokens": args.max_new_tokens,
            "prefill_chunk_size": args.prefill_chunk_size,
            "decode_warmup_tokens": args.decode_warmup_tokens,
            "direct_answer_prompt": bool(args.direct_answer_prompt),
            "content_channel_prefill": bool(args.content_channel_prefill),
            "seed": args.seed,
            "eval_mode": eval_mode,
            "teacher_forced": bool(args.teacher_forced),
            "s5_exact_full_attn": bool(args.s5_exact_full_attn),
            "identity_restore": bool(args.identity_restore),
            "compress_full_attn": bool(args.compress_full_attn),
            "kl_lattice": args.kl_lattice if args.compress_full_attn else None,
            "kl_q_range": args.kl_q_range if args.compress_full_attn else None,
            "kl_bits_per_token_per_head": kl_bits_per_head,
            "full_attention_layers": full_attn_idx,
            "prompt_token_lens": seq_lens,
        },
        "results": {
            "k3_cross_model": {
                **dataclasses.asdict(cross_res),
                "system_under_test": sut_label,
            },
            **({"oracle": dataclasses.asdict(oracle_res)} if oracle_res else {}),
        },
        "gate": {
            # A native-baseline run may not claim cross-model recall
            # (k3_report_gate BASELINE_RECALL_CLAIM): its recall is the
            # oracle's recall by construction.
            "recall_cross_model": (None if adaptive_s5_native else cross_res.recall),
            "recall_native_baseline": (cross_res.recall if adaptive_s5_native else None),
            "recall_oracle": oracle_res.recall if oracle_res else None,
            "recall_delta_vs_oracle_pp": (delta * 100 if delta is not None else None),
            "recall_delta_within_5pp": (delta is not None and delta <= 0.05),
        },
        "memory": {
            "s5": {
                **mem_s5,
                "scope": "analytical_formula",
                "formula_matches_run": formula_matches_run,
            },
            "naive_full_kv": {
                "total_resident_mb": mem_naive["total_resident_mb"],
                "per_token_growth_kb": mem_naive["per_token_growth_kb"],
            },
            # Savings only claimable when the formula describes the run.
            "savings_vs_naive_pct": (round(
                100 * (1 - mem_s5["total_resident_bytes"]
                       / max(mem_naive["total_resident_bytes"], 1)), 1)
                if formula_matches_run else None),
            "measured_peak_mb": _mx_peak_mb(),
        },
        "throughput": {
            "k3_cross_model": cross_tps,
            **({"oracle_native_ar": oracle_tps} if oracle_tps else {}),
            "decode_only": decode_only,
            "cross_model_speedup_vs_oracle_ar": speedup_vs_oracle,
            "speedup_withheld_reasons": speedup_withheld or None,
        },
    }

    # ---------- Evidence gate: the harness validates its own output ----------
    violations = validate_report(report)
    report["gate"]["evidence_violations"] = [
        dataclasses.asdict(v) for v in violations
    ]
    out_path = Path(args.output) if args.output else Path(
        f"results/research/k3_integrated_niah_mac_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[mac] DONE. {sut_label}={cross_res.recall:.3f} "
          f"oracle={oracle_res.recall if oracle_res else 'skipped'} "
          f"-> {out_path}", file=sys.stderr)
    if violations and args.code_prompts:
        print("[mac] code-prompts throughput probe: recall is N/A by design; "
              "evidence gate informational only (not aborting):\n"
              + summarize_violations(violations), file=sys.stderr)
    elif violations:
        print("[mac] EVIDENCE GATE FAILED — this report is NOT admissible "
              "as evidence:\n" + summarize_violations(violations),
              file=sys.stderr)
        return 2
    else:
        print("[mac] evidence gate: PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
