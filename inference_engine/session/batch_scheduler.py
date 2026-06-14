"""Batched decode scheduler — PR-A3c throughput step.

§3.6 made the served path *correct* multi-tenant (per-session binding), but
execution was RPC-serialized: N concurrent ``Generate`` calls each ran their own
verifier forward, one after another. This scheduler **fuses** the decode step of
a cohort of sessions into **one batched forward** — the served-path realisation
of the parallel throughput validated at the engine level in ADR 0014 §3.5.

Scope: a **fixed-cohort** batched decoder (the common multi-tenant burst: N
sessions admitted together, decoded in lockstep). It sources its KV from the
per-session adapters created by :class:`PerSessionVerifierRegistry`, stacks
their restored caches along the batch dim, and runs one ``verifier_model``
forward per step. Sessions that hit EOS/max drop out of the batch; the remainder
keep batching. Dynamic mid-flight arrival + ragged-length continuous batching is
a follow-up (this covers the synchronized cohort that dominates burst load).

Recall-preserving only (the per-session adapters are restored S5).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import torch


class BatchedDecodeScheduler:
    """Fuse a cohort of per-session restored adapters into batched decode.

    Parameters
    ----------
    verifier_model
        The shared HF verifier ``nn.Module`` (the same weights every adapter
        wraps). One batched forward serves the whole cohort.
    device
        Torch device for the batched tensors.
    """

    def __init__(self, verifier_model: Any, device: Any) -> None:
        self._model = verifier_model
        self._device = torch.device(device)

    @staticmethod
    def _stack_caches(adapters: Sequence[Any]):
        """Concatenate the per-session DynamicCaches into one batched cache.

        Every adapter must have an incremental ``_past`` (DynamicCache) at the
        SAME sequence length (synchronized cohort). Returns a new batched
        DynamicCache ``[K, heads, T, dim]`` per layer.
        """
        from transformers.cache_utils import DynamicCache

        pasts = [a._past for a in adapters]
        if any(p is None for p in pasts):
            raise ValueError("all adapters must be prefilled (incremental _past)")
        lengths = {int(a._past_len) for a in adapters}
        if len(lengths) != 1:
            raise ValueError(f"cohort must share one cache length; got {lengths}")
        n_layers = len(pasts[0].layers)
        batched = DynamicCache()
        for li in range(n_layers):
            k = torch.cat([p.layers[li].keys for p in pasts], dim=0)
            v = torch.cat([p.layers[li].values for p in pasts], dim=0)
            batched.update(k, v, li)
        return batched

    @torch.no_grad()
    def run_cohort(
        self,
        adapters: List[Any],
        *,
        max_tokens: int,
        eos_ids: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Decode ``adapters`` in lockstep via batched forwards.

        Each adapter is a prefilled restored verifier (own KV) at the same
        cache length. Returns per-session generated tokens + timing.
        """
        eos_ids = eos_ids or set()
        K = len(adapters)
        if K == 0:
            return {"tokens": [], "decode_s": 0.0, "decode_tokens_per_s": 0.0}
        T = int(adapters[0]._past_len)
        cache = self._stack_caches(adapters)
        # batched next-token logits from each adapter's prefill
        logits = torch.cat([a.next_token_logits.view(1, -1) for a in adapters], dim=0)
        gen: List[List[int]] = [[] for _ in range(K)]
        active = list(range(K))            # rows still generating
        # map current batch row -> original session index
        row_to_sess = list(range(K))
        torch.cuda.synchronize(self._device) if self._device.type == "cuda" else None
        t0 = time.perf_counter()
        step = 0
        while active and step < max_tokens:
            nxt = logits.argmax(-1)                         # [B]
            B = nxt.size(0)
            keep_rows = []
            for r in range(B):
                sidx = row_to_sess[r]
                tok = int(nxt[r].item())
                gen[sidx].append(tok)
                if tok not in eos_ids:
                    keep_rows.append(r)
            step += 1
            if step >= max_tokens or not keep_rows:
                break
            cur = nxt.view(B, 1)
            pos = torch.full((B, 1), T + step - 1, device=self._device, dtype=torch.long)
            cpos = torch.tensor([T + step - 1], device=self._device)
            out = self._model(input_ids=cur, position_ids=pos, cache_position=cpos,
                              past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
            if len(keep_rows) != B:
                # Drop finished rows from the batch (shrink) — keeps the
                # forward dense over only-active sessions.
                idx = torch.tensor(keep_rows, device=self._device)
                logits = logits.index_select(0, idx)
                for layer in cache.layers:
                    layer.keys = layer.keys.index_select(0, idx).contiguous()
                    layer.values = layer.values.index_select(0, idx).contiguous()
                row_to_sess = [row_to_sess[r] for r in keep_rows]
            active = keep_rows
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        dt = time.perf_counter() - t0
        total = sum(len(g) for g in gen)
        return {
            "tokens": gen,
            "decode_s": dt,
            "decode_tokens_per_s": round(total / dt, 3) if dt > 0 else 0.0,
            "sessions": K,
            "total_tokens": total,
        }
