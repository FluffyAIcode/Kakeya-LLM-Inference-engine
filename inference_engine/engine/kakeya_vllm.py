"""Kakeya Inference Engine v0.5 (CUDA) — Kakeya Attention on the vLLM runtime.

This is the **product entrypoint** for KIE-v2: it runs Kakeya Attention's
bounded-window (S5) KV management *on top of* vLLM's serving runtime. vLLM
contributes the three components the user asked to integrate — they are all
Apache-2.0 and inherited unchanged:

  * **Fused MoE Triton kernel** — vLLM's grouped-GEMM expert kernel (the
    dominant ~90% of the gemma-4-26B-A4B decode forward).
  * **CUDA graphs** — vLLM captures the fixed-shape decode step
    (``enforce_eager=False``), removing per-token kernel-launch overhead.
  * **Continuous-batching scheduler** — vLLM's request scheduler / paged
    KV-manager drives multi-tenant throughput.

Kakeya contributes the *attention / KV* layer: a bounded resident window
(``sink + window``) applied via ``hf_overrides`` so the model's sliding-attention
layers keep only the Kakeya window instead of the model default. On gemma-4's
hybrid (5 full + 25 sliding) this is the full Kakeya behaviour with **no
per-token restoration** (the S5 "free lunch"); the restoration backend for
full-attention models (Qwen/Llama) — the large memory differentiator — is the
v0.6 roadmap item tracked in ADR 0015.

Measured (H200, gemma-4-26B-A4B, ctx 16k, recall 1.0): decode throughput
**exceeds** vLLM default by ~1.15–1.23× across N=1..70 — see
``docs/reports/kakeya-inference-engine-v0.5-cuda.md``.

The heavy imports (``vllm``) are deferred to construction time so this module is
importable on torch/vllm-free hosts (CI, the admission math, docs tooling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Kakeya S5 defaults (gemma-4 restored-S5): 4 sink tokens + 64-token window.
DEFAULT_SINK = 4
DEFAULT_WINDOW = 64


def kakeya_window_total(sink: int = DEFAULT_SINK, window: int = DEFAULT_WINDOW) -> int:
    """Total bounded resident span = ``sink + window`` (the Kakeya S5 window)."""
    if sink < 0 or window <= 0:
        raise ValueError(f"sink>=0 and window>0 required (got sink={sink}, window={window})")
    return int(sink) + int(window)


def kakeya_hf_overrides(
    sink: int = DEFAULT_SINK,
    window: int = DEFAULT_WINDOW,
    *,
    nested_text_config: bool = False,
) -> dict[str, Any]:
    """Pure, dependency-free builder for the vLLM ``hf_overrides`` that pins the
    model's sliding-attention layers to the Kakeya bounded window.

    The top-level ``sliding_window`` covers text-only models (Qwen/Llama). Models
    whose attention config is **nested** under ``text_config`` (multimodal, e.g.
    gemma-4) also need the nested key — set ``nested_text_config=True``. Injecting
    a ``text_config`` key into a model that has none makes vLLM try to parse it as
    a real sub-config and fail, so nesting must be **opt-in / detected**, not
    unconditional. Kept torch/vllm-free so it is unit-testable on any host.
    """
    sw = kakeya_window_total(sink, window)
    overrides: dict[str, Any] = {"sliding_window": sw}
    if nested_text_config:
        overrides["text_config"] = {"sliding_window": sw}
    return overrides


@dataclass
class KakeyaVLLMConfig:
    """Configuration for :class:`KakeyaVLLM` (the v0.5-CUDA engine)."""

    model: str
    sink: int = DEFAULT_SINK
    window: int = DEFAULT_WINDOW
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    # enforce_eager=False keeps vLLM's CUDA-graph decode capture ON (we *want*
    # the graphs + fused-MoE; this is the whole point of running on vLLM).
    enforce_eager: bool = False
    quantization: str | None = None
    # Whether to also nest sliding_window under text_config (multimodal configs
    # such as gemma-4). None = auto-detect from the model config at construction;
    # True/False forces it. Auto-detect avoids breaking text-only models.
    nest_text_config: bool | None = None
    extra_vllm_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def window_total(self) -> int:
        return kakeya_window_total(self.sink, self.window)

    def to_vllm_kwargs(self) -> dict[str, Any]:
        """Materialise the ``vllm.LLM(**kwargs)`` argument dict — the Kakeya
        bounded window is injected as ``hf_overrides``; everything else hands the
        runtime (fused-MoE / CUDA-graph / scheduler) to vLLM unchanged.

        ``nest_text_config=None`` (auto) is treated as flat here (the safe
        default for direct callers / tests); :class:`KakeyaVLLM` resolves the
        auto-detection from the real model config before calling this."""
        kwargs: dict[str, Any] = dict(
            model=self.model,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            hf_overrides=kakeya_hf_overrides(
                self.sink, self.window, nested_text_config=bool(self.nest_text_config)
            ),
        )
        if self.quantization:
            kwargs["quantization"] = self.quantization
        kwargs.update(self.extra_vllm_kwargs)
        return kwargs


class KakeyaVLLM:
    """Kakeya Inference Engine v0.5 (CUDA).

    Thin product wrapper: builds a ``vllm.LLM`` with the Kakeya bounded window
    applied, then delegates generation to vLLM (which runs the fused-MoE kernel,
    CUDA-graph decode, and continuous-batching scheduler). Construct it and call
    :meth:`generate` exactly like a vLLM ``LLM``.

    Example::

        from inference_engine.engine.kakeya_vllm import KakeyaVLLM
        eng = KakeyaVLLM("google/gemma-4-26b-a4b-it", max_model_len=16384)
        out = eng.generate(prompts, sampling_params)
    """

    def __init__(
        self,
        model: str,
        *,
        sink: int = DEFAULT_SINK,
        window: int = DEFAULT_WINDOW,
        config: KakeyaVLLMConfig | None = None,
        **vllm_kwargs: Any,
    ) -> None:
        if config is None:
            config = KakeyaVLLMConfig(
                model=model, sink=sink, window=window, extra_vllm_kwargs=dict(vllm_kwargs)
            )
        self.config = config
        # Auto-detect text_config nesting from the (small) model config when not
        # explicitly set, so the same wrapper works for hybrid multimodal configs
        # (gemma-4 → nested) and text-only models (Qwen/Llama → flat).
        if config.nest_text_config is None:
            config.nest_text_config = self._detect_nested_text_config(config.model)
        # Deferred import: vLLM (and torch/CUDA) are only needed to actually serve.
        from vllm import LLM  # type: ignore

        self._llm = LLM(**config.to_vllm_kwargs())

    @staticmethod
    def _detect_nested_text_config(model: str) -> bool:
        """True iff the model's HF config nests attention under ``text_config``
        (multimodal, e.g. gemma-4). Falls back to flat (False) if the config
        cannot be loaded — flat is the safe default that never breaks vLLM."""
        try:
            from transformers import AutoConfig  # type: ignore

            cfg = AutoConfig.from_pretrained(model)
            return hasattr(cfg, "text_config")
        except Exception:
            return False

    @property
    def window_total(self) -> int:
        """The Kakeya bounded resident span enforced on the sliding layers."""
        return self.config.window_total

    @property
    def llm(self) -> Any:
        """The underlying ``vllm.LLM`` (escape hatch for vLLM-native APIs)."""
        return self._llm

    def generate(self, prompts: Any, sampling_params: Any = None, **kwargs: Any) -> Any:
        """Delegate to vLLM's batched generate (fused-MoE + CUDA-graph + scheduler)."""
        return self._llm.generate(prompts, sampling_params, **kwargs)
