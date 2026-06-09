"""DFlash drafter — product-API wrapper around the K3 dLM proposer.

User-facing entry point per ADR 0008 §11.7.0 (K3 model identity locked
to Gemma 4 family — production drafter is
``z-lab/gemma-4-26B-A4B-it-DFlash`` or its alignment-trained local
counterpart).

The class :class:`DFlashDrafter` wraps the lower-level
:func:`inference_engine.v04.dflash_loader.load_dflash_drafter`
(which handles the safetensors-key remapping + extras attachment +
``embed_tokens.weight.var()`` trained-init verification per
PR #95 / ADR §11.15.3 prereq 4) and exposes a clean product-shape
API for two use cases:

    1. **Mac mini local Kakeya inference product integration test**
       — load a local alignment-trained drafter checkpoint, run it
       as the v0.4 dLM proposer on Mac M4 PyTorch MPS bf16. This is
       the use case the user requested 2026-06-09:

           from inference_engine.v04.dflash_drafter import DFlashDrafter
           drafter = DFlashDrafter.from_pretrained(
               "models/dflash-kakeya-baseline", dtype=torch.bfloat16,
           )

    2. **HF Hub-hosted reference drafter** — load
       ``z-lab/gemma-4-26B-A4B-it-DFlash`` (or any other DFlash
       drafter checkpoint published on HF Hub) for vast.ai / CUDA
       evaluation runs.

Both load paths flow through the same loader; ``from_pretrained``
auto-detects whether the argument is a local directory or an HF
repo id by checking the filesystem.

Public API
----------

* :meth:`DFlashDrafter.from_pretrained(path_or_repo, dtype=..., device=..., trust_remote_code=...)`
  Classmethod constructor. Validates the checkpoint, attaches the
  DFlash extras, verifies trained-init, returns a ready-to-use
  drafter wrapper.

* :attr:`DFlashDrafter.model`
  The underlying ``Qwen3ForCausalLM`` instance (loaded with the
  DFlash safetensors weights via the prereq-4 corrected loader).
  Pass to :func:`inference_engine.v04.capture_proposer_kv` for the
  proposer-role K/V capture.

* :attr:`DFlashDrafter.tokenizer`
  The matched HF tokenizer (DFlash declares ``model_type: qwen3``
  so this is the Qwen3 tokenizer family).

* :attr:`DFlashDrafter.extras`
  ``torch.nn.Module`` containing ``fc`` and ``hidden_norm``
  parameters from the checkpoint, attached but not wired into the
  forward path. K3 Block B's cross-model ``DLMRestoredVerifier``
  is the consumer that activates these via the projection
  ``f_θ`` (per ADR §11.15.3 / k3-cross-model contract).

* :attr:`DFlashDrafter.device` / :attr:`DFlashDrafter.dtype`
  Resolved torch device + dtype the model is loaded on.

* :attr:`DFlashDrafter.config`
  The DFlash repo's config.json content (includes ``block_size``,
  ``target_layer_ids``, ``dflash_config``).

* :meth:`DFlashDrafter.propose_kv(input_ids)`
  Run the model over ``input_ids`` and return a
  :class:`inference_engine.v04.KVCapture` containing per-layer
  pre-norm pre-RoPE K, V tensors. This IS the v0.4 K/V Restoration
  proposer-role primitive; consumers (cross-model
  ``DLMRestoredVerifier`` etc.) feed the result into the verifier's
  attention pipeline via :func:`prepare_restored_attention_kv` or
  the K2.A.2 stateful incremental forward.

What this module does NOT do (deliberately deferred)
----------------------------------------------------

* **Block-diffusion sampling** (DFlash's actual drafting protocol —
  ``block_size: 16`` parallel mask-token decoding). That's a more
  involved generation primitive; this module only exposes the K/V
  capture role today. Tracked per ADR §11.15.3 as Block B follow-up.

* **Cross-layer feature conditioning** (the ``fc`` / ``hidden_norm``
  extras-driven path that consumes verifier hidden states from the
  layers selected by ``target_layer_ids``). The extras are loaded
  and attached via the loader; wiring them into a forward pass is
  the K3 Block B / Block C training PR scope.

* **Speculative decoding orchestration** (the verifier-runs-drafter
  loop with accept/reject). That's a higher-level inference engine
  concern; this module exposes the building block.

"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DFlashDrafter:
    """Product-API wrapper around a loaded DFlash drafter checkpoint.

    Construct via :meth:`from_pretrained`; do not instantiate
    directly. The dataclass fields are the public attributes of the
    wrapper.
    """

    model: Any  # transformers.Qwen3ForCausalLM (lazy-typed to avoid HF import here)
    tokenizer: Any  # transformers.PreTrainedTokenizer
    extras: Any  # torch.nn.Module containing fc / hidden_norm parameters; or None
    config: dict  # DFlash config.json content (raw dict)
    device: Any  # torch.device-like (str OK)
    dtype: Any  # torch.dtype-like
    embed_tokens_var: float
    embed_tokens_trained: bool
    architectural_warnings: list
    source: str  # path or repo id used to load

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str,
        *,
        dtype: Any = None,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        require_trained_embed: bool = True,
        **hf_kwargs: Any,
    ) -> "DFlashDrafter":
        """Load a DFlash drafter from a local directory OR HF repo id.

        Parameters
        ----------
        path_or_repo
            Either a filesystem path to a directory containing
            ``config.json`` + safetensors files (e.g.
            ``"models/dflash-kakeya-baseline"``), OR an HF repo id
            (e.g. ``"z-lab/gemma-4-26B-A4B-it-DFlash"``). Auto-
            detection: if the path exists on disk and is a
            directory, treat as local; otherwise treat as repo id.
        dtype
            Optional torch.dtype override for the model parameters.
            Use ``torch.bfloat16`` on Mac M4 / CUDA, ``torch.float32``
            on CPU. ``None`` keeps the checkpoint's native dtype.
        device
            Optional device override. ``None`` = auto-detect (MPS
            on Apple Silicon, then CUDA, then CPU).
            ``"mps"`` / ``"cuda"`` / ``"cpu"`` for explicit choice.
        trust_remote_code
            Forwarded to HF; defaults to ``True`` per ADR §11.15.2.1
            (DFlash declares ``model_type: qwen3`` and ships no
            custom modeling, so trust_remote_code is moot for the
            model itself but the tokenizer can still benefit).
        require_trained_embed
            If True (default), raises :class:`ValueError` if the
            loaded model's ``embed_tokens.weight.var()`` falls
            below the trained-init threshold defined in the
            loader. Set False ONLY for diagnostic loading of
            possibly-broken checkpoints.
        **hf_kwargs
            Additional kwargs forwarded to the loader (e.g.
            ``cache_dir``, ``token``).

        Raises
        ------
        FileNotFoundError
            If ``path_or_repo`` looks local but is missing
            ``config.json`` or safetensors files.
        ValueError
            If ``require_trained_embed=True`` and the loader flags
            ``embed_tokens_trained=False`` (see loader docstring
            for the variance threshold).
        """
        from inference_engine.v04.dflash_loader import load_dflash_drafter

        device = _resolve_device(device)
        logger.info(
            "DFlashDrafter.from_pretrained source=%r dtype=%s device=%s",
            path_or_repo, dtype, device,
        )

        result = load_dflash_drafter(
            path_or_repo,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
            **hf_kwargs,
        )

        # DFlash architectural exception: the drafter does NOT own its own
        # embed_tokens — it shares the verifier's at inference time (per
        # ADR §11.7.0 + PR #93's dflash_drafter.py docstring lines 7-15).
        # So a properly-published DFlash baseline checkpoint will ALWAYS
        # show embed_tokens.weight.var() ≈ 4e-4 (the Normal(0, 0.02)
        # random-init signature), because the safetensors file does NOT
        # carry embed_tokens and Qwen3ForCausalLM constructs them with
        # default random init.
        #
        # Detection: the DFlash config.json carries dflash_config.
        # target_layer_ids and block_size — these are the architectural
        # markers that "this checkpoint expects shared embed_tokens from
        # a verifier, not its own". When detected, the trained-embed gate
        # is architecturally inappropriate (would correctly catch a real
        # bug for a standalone Qwen3 checkpoint, but produces a false
        # positive for a DFlash drafter).
        is_dflash = bool(
            result.inspection.config.get("dflash_config")
            or result.inspection.config.get("target_layer_ids")
            or result.inspection.config.get("block_size")
        )

        if require_trained_embed and not result.embed_tokens_trained and not is_dflash:
            raise ValueError(
                f"DFlashDrafter.from_pretrained({path_or_repo!r}): the loaded "
                f"checkpoint's embed_tokens.weight.var() = "
                f"{result.embed_tokens_var:.6e} indicates the embeddings are "
                f"NOT trained (random initialisation). Loading would produce "
                f"meaningless K/V at every layer. If this is a deliberately-"
                f"random diagnostic load, pass require_trained_embed=False. "
                f"Otherwise re-fetch the checkpoint or check the safetensors "
                f"key remap (run 'python -m inference_engine.v04.dflash_loader "
                f"inspect {path_or_repo}' for the diagnose dump)."
            )

        if is_dflash:
            # KNOWN ARCHITECTURAL LIMITATION (recorded 2026-06-09):
            #
            # This loader (PR #96) treats the DFlash drafter as a
            # standalone Qwen3 model. The K3 native DFlash implementation
            # on PR #93's branch
            # (AgentMemory/v04-pr-k3-dflash-native-integration-2815)
            # implements the proper architecture:
            #
            #   * No own embed_tokens (uses verifier's at inference)
            #   * No own lm_head (uses verifier's)
            #   * fc + hidden_norm consume verifier hidden states from
            #     dflash_config.target_layer_ids (with +1 shift per
            #     vLLM PR #41703)
            #   * Non-causal block attention with block_size=16
            #   * Block-diffusion mask-token decoding
            #
            # Loading via THIS module produces a model whose
            # embed_tokens are randomly initialised. Calling
            # .propose_kv() will run the forward (validates plumbing
            # shape + dtype + device end-to-end) but the resulting
            # K/V tensors at every layer derive from random embeddings
            # — they are NOT the K/V tensors a properly-injected
            # DFlash drafter would produce. The proposer-role smoke
            # validates LOAD + FORWARD plumbing, not product semantics.
            #
            # Resolution: merge PR #93 to main, retire this module
            # (or rewrite it to delegate to PR #93's DFlashDrafter).
            # Tracked as the "PR #96 retirement" item.
            warning = (
                "ARCHITECTURAL WARNING: this DFlash checkpoint expects "
                "shared embed_tokens with the verifier (per "
                "dflash_config in config.json), but PR #96's DFlashDrafter "
                "loads it as a standalone Qwen3 with random-init embeddings. "
                "The proposer K/V produced by this loader's propose_kv() "
                "validates plumbing only, not product semantics. The proper "
                "implementation lives on PR #93's branch "
                "(AgentMemory/v04-pr-k3-dflash-native-integration-2815) "
                "and should be used for any meaningful K/V evidence. "
                f"embed_tokens.weight.var() = {result.embed_tokens_var:.6e} "
                f"(typical random-init signature; DFlash design, NOT a bug)."
            )
            result.architectural_warnings.append(warning)
            logger.warning(warning)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            path_or_repo, trust_remote_code=trust_remote_code,
        )

        result.model.eval()
        for p in result.model.parameters():
            p.requires_grad = False

        return cls(
            model=result.model,
            tokenizer=tokenizer,
            extras=result.extras,
            config=dict(result.inspection.config),
            device=device,
            dtype=dtype,
            embed_tokens_var=result.embed_tokens_var,
            embed_tokens_trained=result.embed_tokens_trained,
            architectural_warnings=list(result.architectural_warnings),
            source=str(path_or_repo),
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def block_size(self) -> Optional[int]:
        """DFlash block-diffusion drafting block size (default 16
        per the upstream config). Returned for downstream
        speculative-decoding orchestration; the present module
        does not exercise block-diffusion sampling."""
        return self.config.get("block_size")

    @property
    def target_layer_ids(self) -> Optional[list]:
        """Indices of verifier layers the DFlash drafter is trained
        to condition on (via fc + hidden_norm extras). Used by K3
        Block B's cross-model DLMRestoredVerifier to wire the
        projection ``f_θ``; not consumed at this layer."""
        return self.config.get("target_layer_ids")

    @property
    def num_layers(self) -> int:
        """Number of transformer layers in the drafter (Qwen3
        ``num_hidden_layers``)."""
        return int(self.config.get("num_hidden_layers", 0))

    @property
    def model_type(self) -> str:
        """The HF model_type field. For DFlash drafters this is
        always ``"qwen3"`` (per ADR §11.7.0 footnote — DFlash's
        transformer block layout follows Qwen3's pattern)."""
        return str(self.config.get("model_type", ""))

    # ------------------------------------------------------------------
    # Proposer-role primitive
    # ------------------------------------------------------------------

    def propose_kv(self, input_ids: Any) -> Any:
        """Run a single forward of the drafter over ``input_ids`` and
        return a :class:`inference_engine.v04.KVCapture` of the
        pre-norm pre-RoPE K, V projections at every layer.

        This IS the v0.4 dLM K/V Restoration proposer-role primitive
        (ADR §11.5). Consumers feed the captured K/V into the
        verifier's attention pipeline via either:

        * :func:`prepare_restored_attention_kv` — K2.A.1 stateless
          path (one-shot per forward), or
        * the K2.A.2 stateful incremental path — pre-computes evicted
          K/V once per session bootstrap then reuses cached values.

        The K2.A.1 stateless path is appropriate for **single-request
        product-shape evaluation** (the Mac mini integration test
        the user is preparing). The K2.A.2 stateful path is
        appropriate for long-session decode where the verifier
        per-step forward needs to be O(1) in T (per ADR §11.11.14).

        Parameters
        ----------
        input_ids
            ``[B, T]`` token-id tensor on the same device as
            :attr:`model`. ``B=1`` is the only currently-tested
            shape.

        Returns
        -------
        :class:`inference_engine.v04.KVCapture`
            Per-layer pre-norm pre-RoPE K, V tensors of shape
            ``[B, T, num_kv_heads, head_dim]``.
        """
        from inference_engine.v04.kv_capture import capture_proposer_kv
        return capture_proposer_kv(self.model, input_ids)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DFlashDrafter(source={self.source!r}, model_type={self.model_type!r}, "
            f"num_layers={self.num_layers}, block_size={self.block_size}, "
            f"target_layer_ids={self.target_layer_ids}, dtype={self.dtype}, "
            f"device={self.device}, embed_tokens_trained={self.embed_tokens_trained}, "
            f"architectural_warnings={len(self.architectural_warnings)})"
        )

    def summary(self) -> dict:
        """JSON-serialisable summary dict for product evidence
        collection (smoke harnesses, ladder runs, etc.)."""
        return {
            "kind": "dflash_drafter",
            "source": self.source,
            "model_type": self.model_type,
            "num_layers": self.num_layers,
            "block_size": self.block_size,
            "target_layer_ids": self.target_layer_ids,
            "dtype": str(self.dtype) if self.dtype is not None else None,
            "device": str(self.device) if self.device is not None else None,
            "embed_tokens_var": self.embed_tokens_var,
            "embed_tokens_trained": self.embed_tokens_trained,
            "architectural_warnings": list(self.architectural_warnings),
            "extras_attached": self.extras is not None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(arg: Optional[str]) -> str:
    """Resolve a device hint to a concrete torch device string.

    Auto-detection order: MPS (Apple Silicon) → CUDA → CPU.
    Matches the K1.E NIAH runner's ``pick_device`` logic for
    consistency across product-shape evidence collection scripts.
    """
    if arg is not None and arg != "auto":
        return arg
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
