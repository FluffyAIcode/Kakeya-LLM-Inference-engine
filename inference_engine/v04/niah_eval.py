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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


# ---------------------------------------------------------------------------
# Memory measurement helpers
# ---------------------------------------------------------------------------
#
# ADR 0008 §11.5 §"Five properties" item 1 — "constant memory in
# context length" — is a measurable claim, not a presumption. The
# helpers below let runners record per-config peak / current memory
# on the active device and emit it into the run's JSON evidence so
# the constant-memory claim becomes empirically verifiable rather
# than rhetorical.
#
# CUDA: torch.cuda.max_memory_allocated tracks the high-water mark
# since the last reset. Reset before each config evaluation, sample
# after, and the peak is the config's memory cost.
#
# MPS: torch.mps does not expose a peak counter as of torch 2.x, so
# we record current_allocated and driver_allocated as point-in-time
# samples. Mac runs cannot demonstrate the sustained-memory claim
# with the same precision as CUDA runs but they can still show
# rough magnitudes.
#
# CPU: optional dependency on psutil. If present, RSS is recorded;
# if absent, memory fields are None and the run continues. Tests
# pass psutil-less to verify graceful degradation.


def reset_memory_peak(device: torch.device) -> None:
    """Reset the device's peak-memory counter so a subsequent
    :func:`record_memory` capture reflects only the period after
    this call.

    Idempotent. Safe to call on devices that don't track peaks
    (MPS, CPU); the call is a no-op there.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        # No-op: torch.mps does not expose reset_peak_memory_stats
        # in the current torch line. Documented limitation; the
        # MPS branch reports point-in-time allocations only.
        pass
    # CPU path: nothing to reset; RSS is process-level and we
    # baseline against a "before" snapshot in record_memory if
    # the caller wants per-config delta.


def record_memory(device: torch.device) -> Dict[str, Any]:
    """Capture a memory snapshot on the given device.

    Returns a dict whose shape depends on the device kind:

    * **cuda**: ``{
        "device_kind": "cuda",
        "current_allocated_bytes": int,
        "current_reserved_bytes": int,
        "peak_allocated_bytes": int,        # since last reset
        "peak_reserved_bytes": int,         # since last reset
        "device_total_bytes": int,
      }``
    * **mps**: ``{
        "device_kind": "mps",
        "current_allocated_bytes": int,
        "driver_allocated_bytes": int,
        "peak_allocated_bytes": None,       # not exposed on MPS
        "peak_reserved_bytes": None,
        "device_total_bytes": None,
      }``
    * **cpu**: ``{
        "device_kind": "cpu",
        "current_allocated_bytes": int|None,  # process RSS via psutil
        "peak_allocated_bytes": None,
        ...
      }``

    All bytes fields are ``int`` when measurable, ``None`` when the
    device kind doesn't expose that metric. JSON-serialisable.

    Synchronizes the CUDA stream before sampling so async kernels
    have committed; MPS path doesn't currently expose a sync API for
    memory accounting (kernels are typically already complete when
    the eval loop is between samples).
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        props = torch.cuda.get_device_properties(device)
        return {
            "device_kind": "cuda",
            "device_name": props.name,
            "device_total_bytes": int(props.total_memory),
            "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    if device.type == "mps":
        # torch.mps.current_allocated_memory and
        # torch.mps.driver_allocated_memory are stable since torch 2.0.
        try:
            current = int(torch.mps.current_allocated_memory())
        except Exception:
            current = None
        try:
            driver = int(torch.mps.driver_allocated_memory())
        except Exception:
            driver = None
        return {
            "device_kind": "mps",
            "device_name": "Apple MPS",
            "device_total_bytes": None,
            "current_allocated_bytes": current,
            "driver_allocated_bytes": driver,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    # CPU or other: try psutil for process RSS.
    rss: Optional[int] = None
    try:
        import psutil  # type: ignore
        rss = int(psutil.Process().memory_info().rss)
    except Exception:
        rss = None
    return {
        "device_kind": device.type,
        "device_name": str(device),
        "device_total_bytes": None,
        "current_allocated_bytes": rss,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }


# ---------------------------------------------------------------------------
# Effective attention-window metric
# ---------------------------------------------------------------------------
#
# ADR 0008 §11.5 §"Five properties" item 2 — "approximates full
# attention intelligence" — turns on a structural property: how
# many of the prompt's preceding key positions can the verifier's
# last query *actually* attend to? In v0.3 sink+window, that
# number is bounded at ``sink + window`` (≈ 68 for sink=4 +
# window=64) regardless of context length T, so the verifier sees
# ~5 % of context at T=1.4k and ~0.07 % at T=100k — a direct
# intelligence cap. v0.4's dLM K/V Restoration design fills the
# evicted positions with reconstructed K/V, so the structural
# attention range is the full preceding context T regardless of
# the verifier's local cache size.
#
# This metric is *structural*, not behavioural. Behavioural
# attention-mass measurement (count keys whose post-softmax weight
# exceeds ε) requires materialising the [B, H, T, T] attention
# matrix and is incompatible with the SDPA path K1.F enabled for
# long-context runs (SDPA fuses softmax inside the kernel and
# does not return weights). The structural metric is sufficient
# to answer the user-facing question "did the inference engine
# reduce the verifier's intelligence by capping its attention
# range?" — and it composes cleanly with the recall metric: if
# v0.4 restores recall to oracle parity *and* preserves full
# structural attention range, then the architecture really does
# satisfy the "no intelligence loss" claim of ADR 0008 §11.5.
#
# Knowing the metric is derived from the configuration alone (no
# instrumentation required, no SDPA incompatibility) lets it be
# computed at any context length, including the canonical 100k
# rung that K1.F unlocked.


def compute_effective_attention_window(
    config_name: str,
    *,
    seq_len: int,
    sink_size: int,
    window_size: int,
) -> Dict[str, Any]:
    """Compute the structural effective attention window for one
    sample under one verifier configuration.

    Parameters
    ----------
    config_name
        One of ``"oracle_full_attention"``, ``"v03_sink_window"``, or
        ``"v04_dlm_restored"``. Other values raise ``ValueError``.
    seq_len
        Prompt token length T (i.e. the number of preceding keys
        available to the last query at the first decode step).
        The metric for later decode steps differs by at most
        ``max_new_tokens``, which is negligible for the long-context
        runs this metric targets.
    sink_size, window_size
        v0.3 cache shape. Required for ``v03_sink_window``; ignored
        for ``oracle_full_attention`` and ``v04_dlm_restored`` (kept
        in the dict for self-describing JSON evidence).

    Returns
    -------
    dict with keys

    * ``config``: echoed ``config_name``.
    * ``seq_len``: echoed ``seq_len``.
    * ``effective_keys_at_last_query``: number of preceding key
      positions the last query can structurally attend to. Equals
      ``seq_len`` for oracle and v0.4; equals
      ``min(sink + window, seq_len)`` for v0.3.
    * ``effective_attention_fraction``: that count divided by
      ``seq_len`` — a unit-free intelligence-coverage metric. ≈ 1.0
      for oracle and v0.4; bounded < 1 for v0.3 once
      ``seq_len > sink + window``.
    * ``structural_constraint``: human-readable description of the
      constraint (used by the run summary).

    The ``v04_dlm_restored`` entry assumes the architecture's claim
    holds — that ``prepare_restored_attention_kv`` fills evicted
    positions with proposer K/V so the verifier's attention can
    reach all preceding tokens. If that contract ever regresses,
    the recall metric (in the same JSON) will diverge from oracle,
    making the failure visible. The two metrics together form a
    cross-check.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative; got {seq_len}")
    if sink_size < 0 or window_size < 0:
        raise ValueError(
            f"sink_size={sink_size}, window_size={window_size} must be "
            "non-negative"
        )
    if config_name == "oracle_full_attention":
        accessible = seq_len
        constraint = "causal"
    elif config_name == "v03_sink_window":
        accessible = min(sink_size + window_size, seq_len)
        constraint = f"sink={sink_size}+window={window_size}"
    elif config_name == "v04_dlm_restored":
        accessible = seq_len
        constraint = (
            f"causal_with_dlm_reconstruction (local_cache="
            f"sink={sink_size}+window={window_size})"
        )
    else:
        raise ValueError(
            f"unknown config_name {config_name!r}; expected one of "
            "'oracle_full_attention', 'v03_sink_window', 'v04_dlm_restored'"
        )
    fraction = (accessible / seq_len) if seq_len > 0 else 0.0
    return {
        "config": config_name,
        "seq_len": seq_len,
        "effective_keys_at_last_query": int(accessible),
        "effective_attention_fraction": float(fraction),
        "structural_constraint": constraint,
    }


def aggregate_attention_window_metrics(
    config_name: str,
    *,
    prompt_token_lens: Sequence[int],
    sink_size: int,
    window_size: int,
) -> Dict[str, Any]:
    """Aggregate per-sample :func:`compute_effective_attention_window`
    output across an evaluation set.

    Returns the mean / min / max / median of
    ``effective_keys_at_last_query`` and
    ``effective_attention_fraction`` plus the constraint label and
    the per-sample list (kept for full transparency of the JSON
    evidence). Empty ``prompt_token_lens`` raises ``ValueError``.
    """
    if not prompt_token_lens:
        raise ValueError("prompt_token_lens must be non-empty")
    per_sample = [
        compute_effective_attention_window(
            config_name,
            seq_len=int(t),
            sink_size=sink_size,
            window_size=window_size,
        )
        for t in prompt_token_lens
    ]
    keys = [s["effective_keys_at_last_query"] for s in per_sample]
    fracs = [s["effective_attention_fraction"] for s in per_sample]

    def _median(xs: List[float]) -> float:
        srt = sorted(xs)
        n = len(srt)
        return srt[n // 2] if n % 2 == 1 else (srt[n // 2 - 1] + srt[n // 2]) / 2

    constraint = per_sample[0]["structural_constraint"]
    return {
        "config": config_name,
        "structural_constraint": constraint,
        "samples_total": len(per_sample),
        "effective_keys_at_last_query_mean": sum(keys) / len(keys),
        "effective_keys_at_last_query_min": min(keys),
        "effective_keys_at_last_query_max": max(keys),
        "effective_keys_at_last_query_median": _median([float(k) for k in keys]),
        "effective_attention_fraction_mean": sum(fracs) / len(fracs),
        "effective_attention_fraction_min": min(fracs),
        "effective_attention_fraction_max": max(fracs),
        "effective_attention_fraction_median": _median(fracs),
        "per_sample": per_sample,
    }


def format_attention_window_summary(metrics: Dict[str, Any]) -> str:
    """Return a one-line human-readable summary of the aggregated
    attention-window metrics.

    Mirrors :func:`format_memory_summary` so runners can print
    per-config attention coverage at the same density as latency
    and recall.
    """
    keys_mean = metrics.get("effective_keys_at_last_query_mean")
    frac_mean = metrics.get("effective_attention_fraction_mean")
    constraint = metrics.get("structural_constraint", "?")
    if keys_mean is None or frac_mean is None:
        return f"attn_window: n/a (constraint={constraint})"
    return (
        f"attn_window: mean_keys={keys_mean:.0f} "
        f"({frac_mean * 100:.2f}% of context)  constraint={constraint}"
    )


def format_memory_summary(snapshot: Dict[str, Any]) -> str:
    """Return a one-line human-readable summary of a memory snapshot.

    Used by runners to print per-config memory at the same density
    as the latency / recall summary lines. Returns a string suitable
    for direct ``print()``-ing; callers prepend their own prefix.
    """
    kind = snapshot.get("device_kind", "?")
    if kind == "cuda":
        peak = snapshot.get("peak_allocated_bytes")
        cur = snapshot.get("current_allocated_bytes")
        total = snapshot.get("device_total_bytes")
        if peak is not None and total is not None and total > 0:
            pct = peak / total * 100
            return (
                f"cuda peak={peak / 1e9:.2f}GB ({pct:.0f}% of "
                f"{total / 1e9:.0f}GB)  current={cur / 1e9:.2f}GB"
            )
        return f"cuda peak={peak} current={cur}"
    if kind == "mps":
        cur = snapshot.get("current_allocated_bytes")
        drv = snapshot.get("driver_allocated_bytes")
        if cur is not None:
            cur_str = f"{cur / 1e9:.2f}GB"
        else:
            cur_str = "n/a"
        if drv is not None:
            drv_str = f"{drv / 1e9:.2f}GB"
        else:
            drv_str = "n/a"
        return f"mps current={cur_str} driver={drv_str} (no peak counter)"
    cur = snapshot.get("current_allocated_bytes")
    if cur is not None:
        return f"cpu rss={cur / 1e9:.2f}GB"
    return f"{kind} (no memory accounting available)"
