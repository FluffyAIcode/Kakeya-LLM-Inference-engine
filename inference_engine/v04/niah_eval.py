"""Needle-in-haystack evaluation harness for v0.4 K/V Restoration.

ADR 0008 §11.8 gate (a): NIAH mid-context recall ≥ 95 % at 100k-token
context (vs v0.3's 16.7 % from the 2026-06-06 A/B benchmark in
`results/platform-tests/sink_window_quality_ab_1780714635.json`). This
module provides the harness; the empirical run lives on Mac M4 via
``scripts/review_pr_k1e_on_mac.sh``.

Three verifier configurations are evaluated on the same NIAH samples:

1. **Full-attention oracle** — standard ``model.forward``. Upper
   bound on what the architecture can possibly recall. Target ≈ 1.0.
2. **v0.3 sink+window** — Gemma3 forward with a 4D sink+window
   attention mask (sink=4, window=64 by default). Confirms the
   regression v0.4 is designed to fix; target ≈ 0.17 per the
   2026-06-06 A/B benchmark.
3. **v0.4 DLMRestoredVerifier** — the new architecture. ADR 0008
   §11.8 gate (a) target: ≥ 0.95 at 100k context.

Decoding for all three is greedy with ``max_new_tokens=24``; recall
is the fraction of samples whose decoded continuation contains the
exact needle code substring (e.g. ``"ALPHA-1234"``).

The NIAH dataset constructor and recall computation are tested on
Linux without HF dependency. The actual model evaluations are
covered by the Mac M4 reviewer aid (real Gemma 3-1B-it).

Why this lives in `inference_engine/v04/` (not `scripts/research/`)
-------------------------------------------------------------------

* The harness is a stable, reusable evaluation tool — every K-phase
  PR plus the v0.4 GA acceptance criteria reference it.
* It depends on ``DLMRestoredVerifier`` and is naturally co-located
  with the architecture it validates.
* Linux unit tests live in ``tests/inference_engine/v04/`` next to
  the rest of the v04 test suite.
"""

from __future__ import annotations

import dataclasses
import math
import random
import time
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# NIAH dataset
# ---------------------------------------------------------------------------


# Closed needle vocabulary — same shape as cross_attn_toy_prototype.py used
# in R1c-e research, kept here so the v0.3 vs v0.4 comparison is on
# bit-comparable test stimuli.
DEFAULT_NEEDLE_PREFIXES: Tuple[str, ...] = (
    "ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA",
    "ETA", "THETA", "IOTA", "KAPPA", "ORCHID", "PINE",
    "MAPLE", "OAK", "BIRCH",
)
DEFAULT_NEEDLE_CODE_MIN: int = 1000
DEFAULT_NEEDLE_CODE_MAX: int = 9999


@dataclasses.dataclass
class NIAHSample:
    """One needle-in-haystack evaluation example.

    Attributes
    ----------
    prompt_text
        The full haystack + question, as a plain string. The reviewer
        is responsible for applying the chat template (per ADR 0008
        §2.4 / R1b Bug C: the runtime is template-free).
    answer_text
        The exact needle code that the model should recall, e.g.
        ``"ALPHA-1234"``. ``answer_text in greedy_decode_text`` is
        the recall predicate.
    needle_line_index
        Position in the haystack's padding-line list where the needle
        was inserted (for diagnostics / per-position recall analysis).
    needle_text
        The exact needle line as inserted, e.g.
        ``"\\nIMPORTANT: the secret code is ALPHA-1234.\\n"``. Useful
        for downstream attention-localization analysis (cf. R1d-β
        instrumentation).
    """

    prompt_text: str
    answer_text: str
    needle_line_index: int
    needle_text: str


def make_niah_dataset(
    *,
    n_samples: int = 30,
    haystack_min_lines: int = 60,
    haystack_max_lines: int = 80,
    seed: int = 42,
    needle_prefixes: Sequence[str] = DEFAULT_NEEDLE_PREFIXES,
    needle_code_min: int = DEFAULT_NEEDLE_CODE_MIN,
    needle_code_max: int = DEFAULT_NEEDLE_CODE_MAX,
) -> List[NIAHSample]:
    """Build a list of NIAH samples with random needle codes inserted
    at random middle positions in a synthetic haystack.

    Each sample is structured as::

        Note 0000: this paragraph is unrelated padding...
        Note 0001: ...
        ...
        IMPORTANT: the secret code is ALPHA-1234.    ← needle (random pos)
        ...
        Note NNNN: ...
        Question: what is the secret code? Answer:

    The needle is always inserted **outside the first 4 and last 4
    padding lines** so that any sink+window with sink=4 + window
    covering ~4 lines worth of tokens at the tail still misses the
    needle — i.e. the v0.3 baseline genuinely fails by construction.

    Parameters
    ----------
    n_samples
        How many samples to generate.
    haystack_min_lines, haystack_max_lines
        Range of padding-line counts. Each padding line is roughly
        12-15 tokens once tokenized, so ``haystack_max_lines = 80``
        gives prompts of ~1-2 k tokens. Scale up to test longer
        contexts.
    seed
        RNG seed; reproducible across runs.
    needle_prefixes, needle_code_min, needle_code_max
        Closed vocabulary the needle code is drawn from. Default
        matches R1c-e cross_attn_toy_prototype's full vocab.

    Returns
    -------
    A list of ``NIAHSample`` of length ``n_samples``. Determined
    purely by the seed; two calls with the same seed produce the
    same dataset bit-for-bit.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive; got {n_samples}")
    if haystack_min_lines > haystack_max_lines:
        raise ValueError(
            f"haystack_min_lines ({haystack_min_lines}) > "
            f"haystack_max_lines ({haystack_max_lines})"
        )
    if haystack_min_lines < 10:
        raise ValueError(
            f"haystack_min_lines must be >= 10 to leave room for "
            f"sink-anchored and window-anchored regions; got {haystack_min_lines}"
        )
    if not needle_prefixes:
        raise ValueError("needle_prefixes must be non-empty")
    if needle_code_min > needle_code_max:
        raise ValueError(
            f"needle_code_min ({needle_code_min}) > needle_code_max "
            f"({needle_code_max})"
        )

    rng = random.Random(seed)
    samples: List[NIAHSample] = []
    for _ in range(n_samples):
        n_lines = rng.randint(haystack_min_lines, haystack_max_lines)
        prefix = rng.choice(needle_prefixes)
        code_num = rng.randint(needle_code_min, needle_code_max)
        code = f"{prefix}-{code_num}"
        needle = f"\nIMPORTANT: the secret code is {code}.\n"

        padding = [
            f"Note {i:04d}: this paragraph is unrelated padding "
            "and does not contain the answer."
            for i in range(n_lines)
        ]
        # Insert outside first/last 4 lines so neither sink (4 lines
        # worth at start) nor window (small tail) can plausibly catch
        # the needle just from positional luck.
        insert_at = rng.randint(4, n_lines - 4)
        padding.insert(insert_at, needle)

        prompt = "\n".join(padding) + (
            "\nQuestion: what is the secret code? Answer:"
        )
        samples.append(
            NIAHSample(
                prompt_text=prompt,
                answer_text=code,
                needle_line_index=insert_at,
                needle_text=needle,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Recall scoring
# ---------------------------------------------------------------------------


def recall_predicate(decoded_text: str, sample: NIAHSample) -> bool:
    """Return True iff ``decoded_text`` contains the sample's exact
    needle code substring.

    The check is case-sensitive and substring-based — matching the
    convention used in R1b/R1c evaluation. Any leading/trailing
    whitespace in the model's generation is fine; what matters is
    that the exact code appears somewhere.
    """
    return sample.answer_text in decoded_text


@dataclasses.dataclass
class NIAHEvalResult:
    """Aggregate result for one (verifier, dataset) evaluation.

    Attributes
    ----------
    name
        Human-readable label, e.g. ``"v04_restored"``.
    samples_total
        Total NIAH samples evaluated.
    samples_correct
        Count whose greedy decode contained the needle code.
    recall
        ``samples_correct / samples_total``.
    mean_latency_s, median_latency_s
        Per-sample wall-clock generation latency.
    per_sample_decoded
        The raw decoded text for each sample. Useful for post-hoc
        diagnostics ("what did the model say when it was wrong?").
    per_sample_correct
        Boolean correctness flag per sample.
    """

    name: str
    samples_total: int
    samples_correct: int
    recall: float
    mean_latency_s: float
    median_latency_s: float
    per_sample_decoded: List[str]
    per_sample_correct: List[bool]


def aggregate_recall(
    name: str,
    samples: Sequence[NIAHSample],
    decoded_texts: Sequence[str],
    latencies_s: Sequence[float],
) -> NIAHEvalResult:
    """Combine per-sample decoded outputs + latencies into an
    aggregate :class:`NIAHEvalResult`.

    ``len(decoded_texts)`` must equal ``len(samples)``; same for
    ``latencies_s``. Mismatch raises ``ValueError``.
    """
    n = len(samples)
    if len(decoded_texts) != n:
        raise ValueError(
            f"decoded_texts ({len(decoded_texts)}) != samples ({n})"
        )
    if len(latencies_s) != n:
        raise ValueError(
            f"latencies_s ({len(latencies_s)}) != samples ({n})"
        )
    if n == 0:
        raise ValueError("samples must be non-empty")

    correct_flags = [
        recall_predicate(text, sample)
        for text, sample in zip(decoded_texts, samples)
    ]
    n_correct = sum(correct_flags)
    sorted_lat = sorted(latencies_s)
    median = sorted_lat[n // 2] if n % 2 == 1 else (
        (sorted_lat[n // 2 - 1] + sorted_lat[n // 2]) / 2
    )
    return NIAHEvalResult(
        name=name,
        samples_total=n,
        samples_correct=n_correct,
        recall=n_correct / n,
        mean_latency_s=sum(latencies_s) / n,
        median_latency_s=median,
        per_sample_decoded=list(decoded_texts),
        per_sample_correct=correct_flags,
    )


# ---------------------------------------------------------------------------
# Sink+window 4D attention mask — the v0.3 baseline forward
# ---------------------------------------------------------------------------


def make_sink_window_4d_mask(
    seq_len: int,
    sink: int,
    window: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a ``[1, 1, seq_len, seq_len]`` additive attention mask
    that masks out positions outside the per-query (sink ∪ window)
    region.

    For each query position ``q``, the allowed key positions are::

        {0, 1, ..., sink-1}                  (sink)
        ∪ {max(sink, q-window+1), ..., q}    (window, capped at q)

    All other positions are masked with the dtype's ``finfo.min``
    (not ``-inf`` — bf16/fp16 attention kernels NaN-propagate from
    ``-inf + 0`` in some implementations). Used to drive the v0.3
    baseline forward in the K1.E reviewer.

    Returns
    -------
    A 4D tensor of shape ``[1, 1, seq_len, seq_len]`` ready to pass
    as ``attention_mask`` (or wrapped in the Gemma3
    ``{full_attention, sliding_attention}`` dict) to the model.
    """
    if seq_len < 0 or sink < 0 or window < 0:
        raise ValueError(
            f"seq_len={seq_len}, sink={sink}, window={window} must all "
            "be non-negative"
        )
    neg_inf = (
        torch.finfo(dtype).min if dtype.is_floating_point
        else float("-inf")
    )
    mask = torch.full(
        (seq_len, seq_len), neg_inf, device=device, dtype=dtype,
    )
    for q in range(seq_len):
        # sink range
        sink_end = min(sink, q + 1)
        mask[q, :sink_end] = 0.0
        # window range
        window_start = max(sink, q - window + 1)
        mask[q, window_start : q + 1] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Greedy decoders for the three verifier configurations
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_decode_oracle(
    *,
    model: nn.Module,
    prompt_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 24,
) -> str:
    """Greedy decode under the model's standard forward (full-attention
    oracle). Returns the decoded continuation text only (excluding
    the prompt)."""
    cur = prompt_ids
    for _ in range(max_new_tokens):
        out = model(input_ids=cur, use_cache=False)
        next_token = int(torch.argmax(out.logits[:, -1, :]).item())
        cur = torch.cat(
            [cur, torch.tensor([[next_token]], device=cur.device)], dim=1,
        )
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(
        cur[0, prompt_ids.size(1):], skip_special_tokens=True,
    )


@torch.no_grad()
def greedy_decode_sink_window(
    *,
    model: nn.Module,
    prompt_ids: torch.Tensor,
    tokenizer,
    sink_size: int,
    window_size: int,
    is_gemma3: bool = True,
    max_new_tokens: int = 24,
) -> str:
    """Greedy decode under a v0.3-style sink+window 4D attention mask.

    For Gemma3-class models, the mask is wrapped in the ``{full_attention,
    sliding_attention}`` dict convention so both layer types are
    bounded identically (per HF Gemma3 mask dispatch).

    The mask is rebuilt each iteration to match the growing sequence
    length — production v0.3 deployments keep an incremental cache,
    but for this evaluation harness the simpler "rebuild each step"
    approach is sufficient and avoids cache state entanglement across
    samples.
    """
    cur = prompt_ids
    base_dtype = next(model.parameters()).dtype
    for _ in range(max_new_tokens):
        seq_len = cur.size(1)
        mask_4d = make_sink_window_4d_mask(
            seq_len, sink_size, window_size,
            device=cur.device, dtype=base_dtype,
        )
        if is_gemma3:
            attention_mask = {
                "full_attention": mask_4d,
                "sliding_attention": mask_4d,
            }
        else:
            attention_mask = mask_4d

        out = model(
            input_ids=cur,
            attention_mask=attention_mask,
            use_cache=False,
        )
        next_token = int(torch.argmax(out.logits[:, -1, :]).item())
        cur = torch.cat(
            [cur, torch.tensor([[next_token]], device=cur.device)], dim=1,
        )
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(
        cur[0, prompt_ids.size(1):], skip_special_tokens=True,
    )


@torch.no_grad()
def greedy_decode_v04(
    *,
    verifier,  # DLMRestoredVerifier
    prompt_ids: torch.Tensor,
    tokenizer,
    apply_rotary_pos_emb: Callable,
    eager_attention_forward: Callable,
    all_attention_functions=None,
    max_new_tokens: int = 24,
) -> str:
    """Greedy decode under the v0.4 :class:`DLMRestoredVerifier`.

    Each generation step calls ``verifier.forward(...)``, which
    internally runs the proposer-role capture, computes evicted
    positions, installs the patched Gemma3Attention.forward on every
    layer, runs the verifier-role forward, removes the patches, and
    returns logits.
    """
    cur = prompt_ids
    for _ in range(max_new_tokens):
        logits = verifier.forward(
            cur,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
            eager_attention_forward=eager_attention_forward,
            all_attention_functions=all_attention_functions,
        )
        next_token = int(torch.argmax(logits[:, -1, :]).item())
        cur = torch.cat(
            [cur, torch.tensor([[next_token]], device=cur.device)], dim=1,
        )
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(
        cur[0, prompt_ids.size(1):], skip_special_tokens=True,
    )


# ---------------------------------------------------------------------------
# High-level eval orchestration
# ---------------------------------------------------------------------------


def evaluate(
    name: str,
    samples: Sequence[NIAHSample],
    decode_fn: Callable[[NIAHSample], Tuple[str, float]],
) -> NIAHEvalResult:
    """Run ``decode_fn`` on each sample, time it, and aggregate into
    a :class:`NIAHEvalResult`.

    ``decode_fn(sample) -> (decoded_text, latency_s)``. The caller is
    responsible for chat-template encoding the prompt, picking the
    device, and managing tokenizer state — this function is purely
    the eval-loop scaffold so the same harness covers all three
    verifier configurations.
    """
    decoded_texts: List[str] = []
    latencies_s: List[float] = []
    for sample in samples:
        text, latency = decode_fn(sample)
        decoded_texts.append(text)
        latencies_s.append(latency)
    return aggregate_recall(name, samples, decoded_texts, latencies_s)
