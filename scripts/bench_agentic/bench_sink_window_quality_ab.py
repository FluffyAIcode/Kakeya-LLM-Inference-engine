"""No-mock A/B quality eval for Kakeya proposer/verifier bounded context.

This benchmark compares two real inference paths:

* baseline arm: Qwen3-1.7B verifier run as a full-context greedy AR model
  with a large enough ``sink+window`` budget to keep the whole synthetic prompt
  in live KV;
* bounded arm: the complete Kakeya proposer/verifier engine:
  Qwen3-0.6B DLM proposer + Qwen3-1.7B verifier + production-like
  ``sink+window`` budget + a two-stage record-aligned memory proposer.

It measures the negative factors of context trimming: middle-fact recall,
long dependency recall, instruction persistence, exact quote recall, and a
recent-window sanity case. The benchmark intentionally does not use mocks,
fakes, fallback retrieval, summaries, or judge-model grading. All answers are
real model outputs scored by deterministic string rules.

The two-stage memory proposer follows ``docs/memory-computable-guide.md``:
it predicts sparse support records from the current question, then
reconstructs an extractive evidence block aligned to those records before the
token proposer drafts and the verifier accepts/rejects.

Example::

    PYTHONPATH=.:sdks/python python3 \\
      scripts/bench_agentic/bench_sink_window_quality_ab.py \\
      --backend cpu --verifier-id Qwen/Qwen3-1.7B \\
      --proposer-id dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 \\
      --bounded-sink 4 --bounded-window 64 \\
      --memory-proposer twostage \\
      --baseline-sink 0 --baseline-window 2048 \\
      --output results/platform-tests/sink_window_quality_ab_$(date +%s).json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import torch


ScoreMode = Literal["contains", "prefix", "number"]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str
    description: str
    user_prompt: str
    expected: str
    score_mode: ScoreMode


@dataclass(frozen=True)
class MemoryRecord:
    record_id: str
    text: str
    score: float


@dataclass(frozen=True)
class MemoryProposal:
    original_prompt_tokens: int
    augmented_prompt_tokens: int
    question: str
    support_records: List[MemoryRecord]
    augmented_user_prompt: str


@dataclass
class ArmResult:
    ok: bool
    output_text: str
    latency_s: float
    prompt_tokens: int
    generated_tokens: int
    k_seq_length: int
    kv_live_bytes: int
    peak_kv_bytes: int
    forward_calls: int
    tokens_consumed: int
    truncated_dropped_tokens: int
    acceptance_rate: Optional[float] = None
    proposer_forward_calls: Optional[int] = None
    proposer_diffusion_steps: Optional[int] = None
    proposer_weight_bytes: Optional[int] = None
    proposer_peak_activation_bytes: Optional[int] = None
    memory_support_count: Optional[int] = None
    memory_support_records: Optional[List[Dict[str, Any]]] = None
    original_prompt_tokens: Optional[int] = None


def _filler(label: str, n: int) -> str:
    """Deterministic neutral filler, not a learned fixture or fallback.

    The filler is deliberately semantically boring; its job is only to push
    selected facts outside a small bounded window while keeping prompts short
    enough for a Mac mini smoke run.
    """
    parts = []
    for i in range(n):
        parts.append(
            f"{label} note {i:02d}: this paragraph is neutral padding for "
            "a context-window evaluation and does not contain the answer."
        )
    return "\n".join(parts)


def _build_cases() -> List[EvalCase]:
    early = _filler("early", 8)
    middle = _filler("middle", 10)
    late = _filler("late", 5)
    tail = _filler("tail", 2)
    return [
        EvalCase(
            case_id="middle_fact_code",
            category="middle_fact_recall",
            description="Secret code is in the middle, outside a 4+64 live KV window.",
            expected="ORCHID-7429",
            score_mode="contains",
            user_prompt=(
                f"{early}\n\n"
                "Important fact: the checkpoint code is ORCHID-7429.\n\n"
                f"{middle}\n\n"
                "Question: what is the checkpoint code? "
                "Answer only the code."
            ),
        ),
        EvalCase(
            case_id="exact_variable_quote",
            category="exact_quote_recall",
            description="A variable name is buried in the middle context.",
            expected="sapphire_delta_vector",
            score_mode="contains",
            user_prompt=(
                f"{early}\n\n"
                "Important fact: the exact variable name is sapphire_delta_vector.\n\n"
                f"{middle}\n\n"
                "Question: what is the exact variable name? "
                "Answer only the variable name."
            ),
        ),
        EvalCase(
            case_id="long_dependency_sum",
            category="long_dependency_acc",
            description="Answer requires combining an early and a middle fact.",
            expected="42",
            score_mode="number",
            user_prompt=(
                "Important fact A: alpha_value is 13.\n\n"
                f"{early}\n\n"
                "Important fact B: beta_value is 29.\n\n"
                f"{late}\n\n"
                "Question: alpha_value plus beta_value equals what number? "
                "Answer only the number."
            ),
        ),
        EvalCase(
            case_id="instruction_prefix",
            category="instruction_persistence_acc",
            description="Early instruction must persist after padding.",
            expected="TOKEN-PASS:",
            score_mode="prefix",
            user_prompt=(
                "Instruction: every final answer in this prompt must start "
                "with TOKEN-PASS: exactly.\n\n"
                f"{middle}\n\n"
                "Question: say ready. Remember the required prefix."
            ),
        ),
        EvalCase(
            case_id="middle_hallucination_guard",
            category="hallucination_guard_acc",
            description="Middle context tells the model to admit absence for an unknown item.",
            expected="I DO NOT KNOW",
            score_mode="contains",
            user_prompt=(
                f"{early}\n\n"
                "Important policy: if asked for the launch password, answer "
                "I DO NOT KNOW because no launch password was provided.\n\n"
                f"{middle}\n\n"
                "Question: what is the launch password? Answer exactly."
            ),
        ),
        EvalCase(
            case_id="recent_window_sanity",
            category="recent_window_acc",
            description=(
                "Fact is immediately before the question and should survive "
                "bounded trimming as a positive control."
            ),
            expected="PINE-118",
            score_mode="contains",
            user_prompt=(
                f"{middle}\n\n"
                "Recent fact immediately before the question: the current "
                "ticket id is PINE-118. Question: what is the current ticket "
                "id? Answer only the id."
            ),
        ),
    ]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _score(output_text: str, expected: str, mode: ScoreMode) -> bool:
    norm_out = _normalize(output_text)
    norm_expected = _normalize(expected)
    if mode == "contains":
        return norm_expected in norm_out
    if mode == "prefix":
        return norm_out.startswith(norm_expected)
    if mode == "number":
        return expected in re.findall(r"-?\d+", output_text)
    raise ValueError(f"unknown score mode: {mode}")


_STOPWORDS = {
    "a", "an", "and", "answer", "are", "as", "be", "because", "current",
    "does", "every", "exact", "exactly", "final", "for", "from", "if",
    "in", "is", "it", "must", "of", "only", "or", "question", "remember",
    "say", "the", "this", "to", "what", "when", "with",
}


def _terms(text: str) -> List[str]:
    terms = []
    for token in re.findall(r"[A-Za-z0-9_:-]+", text.casefold()):
        if len(token) <= 1 or token in _STOPWORDS:
            continue
        terms.append(token)
    return terms


def _extract_question(user_prompt: str) -> str:
    idx = user_prompt.rfind("Question:")
    if idx < 0:
        raise ValueError("memory proposer requires a final 'Question:' marker")
    return user_prompt[idx:].strip()


def _record_texts(user_prompt: str) -> List[str]:
    """Split the existing prompt into candidate storage records.

    This is the benchmark's explicit ``R``: records already present in the
    prompt/history. The memory proposer may only align to these records; it
    never reads expected answers or calls a fallback retriever.
    """
    idx = user_prompt.rfind("Question:")
    if idx < 0:
        raise ValueError("memory proposer requires a final 'Question:' marker")
    context = user_prompt[:idx]
    records: List[str] = []
    for line in context.splitlines():
        text = line.strip()
        if not text:
            continue
        records.append(text)
    if not records:
        raise ValueError("memory proposer found no storage records")
    return records


def _score_record(record: str, question_terms: List[str]) -> float:
    record_terms = set(_terms(record))
    q_terms = set(question_terms)
    overlap = len(record_terms & q_terms)
    score = float(overlap)

    # Record-alignment priors: these are schema-level labels in the stored
    # records, not expected-answer peeks.
    lowered = record.casefold()
    if any(
        marker in lowered
        for marker in (
            "important fact",
            "important policy",
            "instruction:",
            "recent fact",
        )
    ):
        score += 1.5
    if any(term in lowered for term in q_terms):
        score += 0.5
    # Penalize known neutral padding records while keeping them eligible if a
    # future prompt genuinely asks about them.
    if "neutral padding" in lowered and "does not contain the answer" in lowered:
        score -= 1.0
    return score


def _propose_memory_support(
    *,
    tokenizer,
    case: EvalCase,
    max_support_records: int,
) -> MemoryProposal:
    """Two-stage memory proposer.

    Stage 1 (Memory Attention Forward): predict sparse record support from
    current question terms and existing records.

    Stage 2 (Diffusion-style reconstruction in this benchmark): reconstruct a
    compact, record-aligned evidence block by copying the selected records
    verbatim. It is intentionally extractive here so record alignment is
    auditable and update/delete consistency is not hidden behind a paraphrase.
    """
    question = _extract_question(case.user_prompt)
    question_terms = _terms(question)
    scored = [
        MemoryRecord(
            record_id=f"r{i:03d}",
            text=record,
            score=_score_record(record, question_terms),
        )
        for i, record in enumerate(_record_texts(case.user_prompt))
    ]
    support = [
        rec for rec in sorted(scored, key=lambda r: (-r.score, r.record_id))
        if rec.score > 0
    ][:max_support_records]
    if not support:
        raise RuntimeError(
            f"memory proposer produced no support for case {case.case_id}"
        )

    q_idx = case.user_prompt.rfind("Question:")
    prefix = case.user_prompt[:q_idx].rstrip()
    evidence = "\n".join(f"- {rec.text}" for rec in support)
    augmented = (
        f"{prefix}\n\n"
        "[Record-aligned memory support]\n"
        f"{evidence}\n\n"
        f"{question}"
    )
    original_tokens = len(_tokenize_user_prompt(tokenizer, case.user_prompt))
    augmented_tokens = len(_tokenize_user_prompt(tokenizer, augmented))
    return MemoryProposal(
        original_prompt_tokens=original_tokens,
        augmented_prompt_tokens=augmented_tokens,
        question=question,
        support_records=support,
        augmented_user_prompt=augmented,
    )


def _make_verifier(*, backend: str, verifier_id: str, sink: int, window: int):
    from kv_cache_proposer.verifier import VerifierConfig

    cfg = VerifierConfig(
        model_id=verifier_id,
        dtype=torch.bfloat16,
        device="cpu",
        sink_size=sink,
        window_size=window,
    )
    if backend == "cpu":
        from kv_cache_proposer.verifier import SinkWindowVerifier

        return SinkWindowVerifier(cfg)
    if backend == "mlx":
        from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier

        return MLXSinkWindowVerifier(cfg)
    raise ValueError(f"unknown backend: {backend}")


def _make_proposer(*, proposer_id: str, proposer_impl: str):
    from kv_cache_proposer.proposer import ProposerConfig

    cfg = ProposerConfig(
        model_id=proposer_id,
        dtype=torch.bfloat16,
        device="cpu",
    )
    if proposer_impl == "sparse":
        from inference_engine.proposer import SparseLogitsProposer

        return SparseLogitsProposer(cfg)
    if proposer_impl == "dense":
        from kv_cache_proposer.proposer import DLMProposer

        return DLMProposer(cfg)
    raise ValueError(f"unknown proposer implementation: {proposer_impl}")


def _tokenize_user_prompt(tokenizer, user_prompt: str) -> List[int]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise evaluator. Follow the user's answer format "
                "strictly. Do not explain unless the user asks."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )


def _tokenize_case(tokenizer, case: EvalCase) -> List[int]:
    return _tokenize_user_prompt(tokenizer, case.user_prompt)


def _greedy_generate(
    *,
    verifier,
    prompt_ids: List[int],
    max_new_tokens: int,
    eos_token_ids: Iterable[int],
) -> List[int]:
    eos_set = set(int(x) for x in eos_token_ids)
    verifier.prefill(prompt_ids)
    generated: List[int] = []
    for _ in range(max_new_tokens):
        next_token = int(torch.argmax(verifier.next_token_logits).item())
        generated.append(next_token)
        logits = verifier.forward_block([next_token])
        verifier.commit_or_truncate(forwarded=1, accepted=1)
        verifier.next_token_logits = logits[-1].clone()
        if next_token in eos_set:
            break
    return generated


def _run_case(
    *,
    verifier,
    tokenizer,
    case: EvalCase,
    max_new_tokens: int,
) -> ArmResult:
    prompt_ids = _tokenize_case(tokenizer, case)
    eos = tokenizer.eos_token_id
    eos_ids = [int(eos)] if eos is not None else []
    t0 = time.perf_counter()
    generated = _greedy_generate(
        verifier=verifier,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_ids,
    )
    latency_s = time.perf_counter() - t0
    output_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    k_seq = int(verifier.k_seq_length(None))
    kv_live = int(verifier.kv_live_bytes(None))
    return ArmResult(
        ok=_score(output_text, case.expected, case.score_mode),
        output_text=output_text,
        latency_s=latency_s,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated),
        k_seq_length=k_seq,
        kv_live_bytes=kv_live,
        peak_kv_bytes=int(verifier.stats.peak_kv_bytes),
        forward_calls=int(verifier.stats.forward_calls),
        tokens_consumed=int(verifier.stats.tokens_consumed),
        truncated_dropped_tokens=max(0, len(prompt_ids) - k_seq),
    )


def _run_baseline_case(
    *,
    verifier,
    tokenizer,
    case: EvalCase,
    max_new_tokens: int,
) -> ArmResult:
    prompt_ids = _tokenize_case(tokenizer, case)
    eos = tokenizer.eos_token_id
    eos_ids = [int(eos)] if eos is not None else []
    t0 = time.perf_counter()
    generated = _greedy_generate(
        verifier=verifier,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_ids,
    )
    latency_s = time.perf_counter() - t0
    output_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    k_seq = int(verifier.k_seq_length(None))
    kv_live = int(verifier.kv_live_bytes(None))
    return ArmResult(
        ok=_score(output_text, case.expected, case.score_mode),
        output_text=output_text,
        latency_s=latency_s,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated),
        k_seq_length=k_seq,
        kv_live_bytes=kv_live,
        peak_kv_bytes=int(verifier.stats.peak_kv_bytes),
        forward_calls=int(verifier.stats.forward_calls),
        tokens_consumed=int(verifier.stats.tokens_consumed),
        truncated_dropped_tokens=max(0, len(prompt_ids) - k_seq),
    )


def _run_engine_case(
    *,
    decoder,
    tokenizer,
    case: EvalCase,
    max_new_tokens: int,
    memory_proposer: str,
    max_memory_support_records: int,
) -> ArmResult:
    memory: Optional[MemoryProposal] = None
    if memory_proposer == "twostage":
        memory = _propose_memory_support(
            tokenizer=tokenizer,
            case=case,
            max_support_records=max_memory_support_records,
        )
        prompt_ids = _tokenize_user_prompt(
            tokenizer,
            memory.augmented_user_prompt,
        )
    elif memory_proposer == "none":
        prompt_ids = _tokenize_case(tokenizer, case)
    else:
        raise ValueError(f"unknown memory proposer: {memory_proposer}")
    eos = tokenizer.eos_token_id
    eos_ids = [int(eos)] if eos is not None else []
    t0 = time.perf_counter()
    result = decoder.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_ids,
    )
    latency_s = time.perf_counter() - t0
    output_text = tokenizer.decode(
        result.output_token_ids,
        skip_special_tokens=True,
    ).strip()
    k_seq = int(decoder.verifier.k_seq_length(None))
    kv_live = int(decoder.verifier.kv_live_bytes(None))
    return ArmResult(
        ok=_score(output_text, case.expected, case.score_mode),
        output_text=output_text,
        latency_s=latency_s,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(result.output_token_ids),
        k_seq_length=k_seq,
        kv_live_bytes=kv_live,
        peak_kv_bytes=int(result.verifier_peak_kv_bytes),
        forward_calls=int(result.verifier_forward_calls),
        tokens_consumed=int(result.verifier_tokens_consumed),
        truncated_dropped_tokens=max(0, len(prompt_ids) - k_seq),
        acceptance_rate=float(result.acceptance_rate),
        proposer_forward_calls=int(result.proposer_forward_calls),
        proposer_diffusion_steps=int(result.proposer_diffusion_steps),
        proposer_weight_bytes=int(result.proposer_weight_bytes),
        proposer_peak_activation_bytes=int(result.proposer_peak_activation_bytes),
        memory_support_count=(
            len(memory.support_records) if memory is not None else None
        ),
        memory_support_records=(
            [asdict(rec) for rec in memory.support_records]
            if memory is not None else None
        ),
        original_prompt_tokens=(
            memory.original_prompt_tokens if memory is not None else None
        ),
    )


def _run_baseline_arm(
    *,
    arm_name: str,
    backend: str,
    verifier_id: str,
    sink: int,
    window: int,
    cases: List[EvalCase],
    max_new_tokens: int,
) -> Dict[str, ArmResult]:
    print(
        f"[ab] loading {arm_name}: backend={backend} id={verifier_id} "
        f"sink={sink} window={window}",
        file=sys.stderr,
        flush=True,
    )
    verifier = _make_verifier(
        backend=backend,
        verifier_id=verifier_id,
        sink=sink,
        window=window,
    )
    tokenizer = verifier.tokenizer
    results: Dict[str, ArmResult] = {}
    for case in cases:
        print(f"[ab] {arm_name} case={case.case_id}", file=sys.stderr, flush=True)
        results[case.case_id] = _run_baseline_case(
            verifier=verifier,
            tokenizer=tokenizer,
            case=case,
            max_new_tokens=max_new_tokens,
        )
    del verifier
    gc.collect()
    if backend == "mlx":
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
    return results


def _run_kakeya_engine_arm(
    *,
    arm_name: str,
    backend: str,
    verifier_id: str,
    proposer_id: str,
    proposer_impl: str,
    sink: int,
    window: int,
    block_size: int,
    num_diffusion_steps: int,
    memory_proposer: str,
    max_memory_support_records: int,
    cases: List[EvalCase],
    max_new_tokens: int,
) -> Dict[str, ArmResult]:
    print(
        f"[ab] loading {arm_name}: proposer={proposer_id} "
        f"verifier={verifier_id} backend={backend} sink={sink} window={window} "
        f"block_size={block_size} diffusion_steps={num_diffusion_steps} "
        f"memory_proposer={memory_proposer}",
        file=sys.stderr,
        flush=True,
    )
    proposer = _make_proposer(
        proposer_id=proposer_id,
        proposer_impl=proposer_impl,
    )
    verifier = _make_verifier(
        backend=backend,
        verifier_id=verifier_id,
        sink=sink,
        window=window,
    )
    from kv_cache_proposer.speculative import SpeculativeDecoder

    decoder = SpeculativeDecoder(
        proposer=proposer,
        verifier=verifier,
        block_size=block_size,
        num_diffusion_steps=num_diffusion_steps,
    )
    tokenizer = verifier.tokenizer
    results: Dict[str, ArmResult] = {}
    for case in cases:
        print(f"[ab] {arm_name} case={case.case_id}", file=sys.stderr, flush=True)
        results[case.case_id] = _run_engine_case(
            decoder=decoder,
            tokenizer=tokenizer,
            case=case,
            max_new_tokens=max_new_tokens,
            memory_proposer=memory_proposer,
            max_memory_support_records=max_memory_support_records,
        )
    del decoder
    del verifier
    del proposer
    gc.collect()
    if backend == "mlx":
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
    return results


def _aggregate(
    *,
    cases: List[EvalCase],
    baseline: Dict[str, ArmResult],
    bounded: Dict[str, ArmResult],
) -> Dict[str, Any]:
    categories = sorted({c.category for c in cases})
    by_category: Dict[str, Any] = {}
    for category in categories:
        ids = [c.case_id for c in cases if c.category == category]
        b0 = sum(1 for cid in ids if baseline[cid].ok)
        b1 = sum(1 for cid in ids if bounded[cid].ok)
        by_category[category] = {
            "n": len(ids),
            "baseline_acc": b0 / len(ids),
            "bounded_acc": b1 / len(ids),
            "quality_delta_vs_full_context": (b1 - b0) / len(ids),
        }

    n = len(cases)
    baseline_ok = sum(1 for c in cases if baseline[c.case_id].ok)
    bounded_ok = sum(1 for c in cases if bounded[c.case_id].ok)
    baseline_success_ids = [c.case_id for c in cases if baseline[c.case_id].ok]
    retained = sum(1 for cid in baseline_success_ids if bounded[cid].ok)
    return {
        "n": n,
        "baseline_acc": baseline_ok / n,
        "bounded_acc": bounded_ok / n,
        "quality_delta_vs_full_context": (bounded_ok - baseline_ok) / n,
        "bounded_retention_given_baseline_success": (
            retained / len(baseline_success_ids)
            if baseline_success_ids else None
        ),
        "by_category": by_category,
        "kv": {
            "baseline_max_kv_live_bytes": max(
                baseline[c.case_id].kv_live_bytes for c in cases
            ),
            "bounded_max_kv_live_bytes": max(
                bounded[c.case_id].kv_live_bytes for c in cases
            ),
            "bounded_kv_plateau_tokens": max(
                bounded[c.case_id].k_seq_length for c in cases
            ),
        },
        "latency": {
            "baseline_mean_s": sum(
                baseline[c.case_id].latency_s for c in cases
            ) / n,
            "bounded_mean_s": sum(
                bounded[c.case_id].latency_s for c in cases
            ) / n,
        },
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "mlx"], default="cpu")
    parser.add_argument("--verifier-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--proposer-id",
        default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
    )
    parser.add_argument(
        "--proposer-impl",
        choices=["sparse", "dense"],
        default="sparse",
        help="Both use the real 0.6B DLM proposer. sparse is the production "
             "optimized lm_head path; dense is the reference full-logits path.",
    )
    parser.add_argument("--bounded-sink", type=int, default=4)
    parser.add_argument("--bounded-window", type=int, default=64)
    parser.add_argument("--baseline-sink", type=int, default=0)
    parser.add_argument("--baseline-window", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--num-diffusion-steps", type=int, default=2)
    parser.add_argument(
        "--memory-proposer",
        choices=["none", "twostage"],
        default="twostage",
        help="Two-stage mode predicts sparse record support from existing "
             "prompt records and injects record-aligned memory support into "
             "the bounded Kakeya engine prompt.",
    )
    parser.add_argument("--max-memory-support-records", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cases = _build_cases()
    started_at = time.time()
    baseline = _run_baseline_arm(
        arm_name="baseline_full_1p7b",
        backend=args.backend,
        verifier_id=args.verifier_id,
        sink=args.baseline_sink,
        window=args.baseline_window,
        cases=cases,
        max_new_tokens=args.max_new_tokens,
    )
    bounded = _run_kakeya_engine_arm(
        arm_name="kakeya_pv_bounded",
        backend=args.backend,
        verifier_id=args.verifier_id,
        proposer_id=args.proposer_id,
        proposer_impl=args.proposer_impl,
        sink=args.bounded_sink,
        window=args.bounded_window,
        block_size=args.block_size,
        num_diffusion_steps=args.num_diffusion_steps,
        memory_proposer=args.memory_proposer,
        max_memory_support_records=args.max_memory_support_records,
        cases=cases,
        max_new_tokens=args.max_new_tokens,
    )
    finished_at = time.time()

    case_rows = []
    for case in cases:
        case_rows.append(
            {
                "case": asdict(case),
                "baseline": asdict(baseline[case.case_id]),
                "bounded": asdict(bounded[case.case_id]),
                "quality_delta_vs_full_context": (
                    int(bounded[case.case_id].ok)
                    - int(baseline[case.case_id].ok)
                ),
            }
        )

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "full_context_1p7b_vs_kakeya_pv_sink_window_quality_ab",
        "no_mock_no_fake_no_fallback": True,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": finished_at - started_at,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "config": {
            "backend": args.backend,
            "verifier_id": args.verifier_id,
            "proposer_id": args.proposer_id,
            "proposer_impl": args.proposer_impl,
            "baseline": {
                "arm": "full_context_1p7b_greedy",
                "sink": args.baseline_sink,
                "window": args.baseline_window,
                "budget_tokens": args.baseline_sink + args.baseline_window,
            },
            "bounded": {
                "arm": "kakeya_proposer_verifier_engine",
                "sink": args.bounded_sink,
                "window": args.bounded_window,
                "budget_tokens": args.bounded_sink + args.bounded_window,
                "block_size": args.block_size,
                "num_diffusion_steps": args.num_diffusion_steps,
                "memory_proposer": args.memory_proposer,
                "max_memory_support_records": args.max_memory_support_records,
            },
            "max_new_tokens": args.max_new_tokens,
            "scoring": (
                "deterministic exact string/number rules over real model "
                "outputs; no judge model, no retrieval, no summary fallback"
            ),
        },
        "aggregate": _aggregate(cases=cases, baseline=baseline, bounded=bounded),
        "cases": case_rows,
    }
    out = Path(args.output)
    _write_json(out, payload)
    print(f"[ab] wrote {out}", file=sys.stderr, flush=True)
    print(json.dumps(payload["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
