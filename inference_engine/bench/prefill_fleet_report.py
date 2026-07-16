"""Canonical report schema and aggregation for the two-Mac prefill benchmark."""
from __future__ import annotations

import statistics
from typing import Any, Sequence

PHASES = ("remote_compute", "primary_hot_hit", "allens_cold_restore")
HIT_SOURCES = ("remote_worker", "primary_hot", "allens_offload", "unknown")
_PRIVATE_KEYS = {
    "prompt",
    "token_ids",
    "cache_key",
    "block_hash",
    "payload_sha256",
    "peer_address",
    "source_path",
}


def normalize_stage(stage: dict[str, Any]) -> dict[str, Any]:
    name = stage.get("name")
    if name not in PHASES:
        raise ValueError(f"unknown benchmark phase {name!r}")
    hit_source = stage.get("hit_source", "unknown")
    if hit_source not in HIT_SOURCES:
        raise ValueError(f"unknown hit_source {hit_source!r}")
    output_tokens = int(stage.get("output_tokens", 0))
    prefix_tokens = int(stage.get("prefix_tokens", 0))
    append_s = float(stage.get("append_s", 0.0))
    decode_s = float(stage.get("decode_s", 0.0))
    e2e_s = float(stage.get("e2e_s", 0.0))
    if min(output_tokens, prefix_tokens) < 0 or min(append_s, decode_s, e2e_s) < 0:
        raise ValueError("benchmark token and duration values must be non-negative")
    normalized = dict(stage)
    normalized.update({
        "name": name,
        "hit_source": hit_source,
        "prefix_tokens": prefix_tokens,
        "output_tokens": output_tokens,
        "append_s": append_s,
        "ttft_s": float(stage.get("ttft_s", 0.0)),
        "decode_s": decode_s,
        "e2e_s": e2e_s,
        "prefill_or_restore_tok_s": (
            prefix_tokens / append_s if append_s > 0 else 0.0
        ),
        "decode_tok_s": output_tokens / decode_s if decode_s > 0 else 0.0,
        "generation_latency_ms_per_token": (
            decode_s / output_tokens * 1000.0 if output_tokens > 0 else 0.0
        ),
        "e2e_tok_s": output_tokens / e2e_s if e2e_s > 0 else 0.0,
    })
    assert_public_safe(normalized)
    return normalized


def summarize_stages(stages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    normalized = [normalize_stage(stage) for stage in stages]
    sources = {source: 0 for source in HIT_SOURCES}
    for stage in normalized:
        sources[stage["hit_source"]] += 1
    decode = [stage["decode_tok_s"] for stage in normalized]
    return {
        "stages_total": len(normalized),
        "stages_failed": sum(not stage.get("ok", False) for stage in normalized),
        "ttft_p50_s": _median(stage["ttft_s"] for stage in normalized),
        "prefill_tok_s_p50": _median(
            stage["prefill_or_restore_tok_s"] for stage in normalized
        ),
        "decode_tok_s_p50": statistics.median(decode) if decode else 0.0,
        "e2e_tok_s_p50": _median(stage["e2e_tok_s"] for stage in normalized),
        "generation_latency_ms_p50": _median(
            stage["generation_latency_ms_per_token"] for stage in normalized
        ),
        "bytes_received": sum(
            int(stage.get("delta", {}).get("bytes_received", 0))
            for stage in normalized
        ),
        "hit_source_counts": sources,
    }


def assert_public_safe(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _PRIVATE_KEYS:
                raise ValueError(f"private benchmark field {key!r} is forbidden")
            assert_public_safe(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_public_safe(child)
    elif isinstance(value, str):
        if "/Users/" in value or "169.254." in value:
            raise ValueError("benchmark report contains a private path or address")


def _median(values) -> float:
    items = list(values)
    return float(statistics.median(items)) if items else 0.0
