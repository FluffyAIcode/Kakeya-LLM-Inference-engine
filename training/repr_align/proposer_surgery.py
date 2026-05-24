"""Stage 1: proposer surgery for representation alignment.

This module implements the **frozen-artifact** half of ADR 0001 §4
Stage 1: take a verifier's input embedding table and output LM head,
freeze them, and pair them with two small learnable bridge projections
that let a smaller-hidden-dim proposer backbone slot in between.

Why two projections, not one
----------------------------

ADR 0001 §2.2 says "proposer embedding and output projection are
copied verbatim from the verifier and frozen". That is correct as a
parameter-ownership statement, but it elides a dimension question
that comes up the moment you try to wire it together end-to-end:

    * Verifier (`Qwen/Qwen3-1.7B`):  hidden_size = 2048,  V = 151_936
    * Proposer (`dllm-hub Qwen3-0.6B-mdlm`):  hidden_size = 1024, V = 151_936

If we copy the verifier's embedding (a `V x 2048` table) into the
proposer's input slot, the proposer's transformer backbone — built
for `1024`-dim activations — cannot consume it. Symmetrically, the
verifier's `lm_head` (a `V x 2048` matrix) cannot be applied to the
proposer's `1024`-dim hidden states.

EAGLE-3's published implementation handles this with two thin
learnable adapters:

    input path:   token_id  ─►  frozen_embed (V × d_v)  ─►  W_in (d_v → d_q)
                                                            │
                                                            ▼
                                                  proposer backbone (d_q)
                                                            │
                                                            ▼
    output path:  W_out (d_q → d_v)  ─►  frozen_lm_head (V × d_v)  ─►  logits

`W_in` and `W_out` together carry ~2 * d_v * d_q parameters
(≈ 4 M for the Qwen3-1.7B / 0.6B pair). That is negligible compared
to the embedding/lm_head they bracket (~600 M combined), so the
"verifier vocabulary representation is free" property of ADR 0001 §4
still holds — the embedding table itself is the load-bearing piece;
the bridge matrices are linear adjustments around it.

Why a wrapper, not in-place mutation
------------------------------------

The proposer model from dllm-hub is a HuggingFace `AutoModelForMaskedLM`
instance with its own `embed_tokens` and `lm_head` modules already
wired into its `forward()`. Mutating those in-place would couple the
surgery to the specific HF class layout (`model.model.embed_tokens`
vs `model.transformer.wte` etc.) and would make the surgery hard to
reverse for ablations.

Instead we expose a `nn.Module` that holds **only** the frozen
artifacts plus the two bridge projections, and we provide three
explicit method endpoints (`embed`, `project_to_verifier_space`,
`lm_logits`). Stage 3's trainer composes these endpoints with the
proposer backbone (LoRA-adapted) at runtime; the wrapper does not
care which backbone it is paired with, which makes both unit-testing
and ablations straightforward.

What this module is *not*
-------------------------

* It is not a full proposer. It contains no transformer layers. By
  itself it cannot generate tokens.
* It does not load weights from HuggingFace. The classmethod
  `ReprAlignedSurgery.from_verifier_module(...)` accepts an already-
  constructed verifier `nn.Module`; the entry script that drives it
  on real Qwen3 weights ships in Stage 2 alongside data collection.
* It does not insert LoRA adapters into the proposer backbone. That
  belongs to Stage 3's trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SurgeryConfig:
    """Configuration for :class:`ReprAlignedSurgery`.

    Attributes
    ----------
    verifier_hidden_dim:
        Hidden dimension of the verifier (``d_v``). Must equal the
        second dim of both the embedding and lm_head weight tensors.
    proposer_hidden_dim:
        Hidden dimension of the proposer backbone (``d_q``). Determines
        the output dim of ``W_in`` and the input dim of ``W_out``.
    vocab_size:
        Vocabulary size (``V``). Must equal the first dim of both
        the embedding and lm_head weight tensors.
    bridge_init_std:
        Standard deviation for the truncated-normal initialization of
        ``W_in`` and ``W_out``. Default (0.02) matches Qwen3's own
        initializer for new linear layers and gives a stable starting
        point for representation alignment training; deeper analysis
        of init choice is deferred to Stage 3.
    """

    verifier_hidden_dim: int
    proposer_hidden_dim: int
    vocab_size: int
    bridge_init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.verifier_hidden_dim <= 0:
            raise ValueError(
                f"verifier_hidden_dim must be positive, got {self.verifier_hidden_dim}"
            )
        if self.proposer_hidden_dim <= 0:
            raise ValueError(
                f"proposer_hidden_dim must be positive, got {self.proposer_hidden_dim}"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.bridge_init_std <= 0:
            raise ValueError(
                f"bridge_init_std must be positive, got {self.bridge_init_std}"
            )


class ReprAlignedSurgery(nn.Module):
    """Frozen verifier artifacts + learnable bridge projections.

    Parameters
    ----------
    config:
        Static dimensions and init hyperparameters; see
        :class:`SurgeryConfig`.
    embed_weight:
        ``[V, d_v]`` tensor copied from the verifier's input embedding.
        Stored as a frozen ``nn.Embedding`` (``requires_grad=False``).
        The tensor is cloned and detached on construction; mutations
        to the source tensor after construction do not leak in.
    lm_head_weight:
        ``[V, d_v]`` tensor copied from the verifier's output
        projection. Stored as a frozen ``nn.Linear`` weight
        (``requires_grad=False``, no bias). Same cloning semantics as
        ``embed_weight``.

    Shape contract
    --------------
    Both weight tensors must be exactly ``[V, d_v]`` and must have
    matching first/second dimensions; we validate this at
    construction time and raise ``ValueError`` on mismatch (no
    silent broadcasting, no resizing).
    """

    def __init__(
        self,
        config: SurgeryConfig,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config

        self._validate_weight(embed_weight, name="embed_weight")
        self._validate_weight(lm_head_weight, name="lm_head_weight")

        # Detach + clone both weights so the surgery owns its copy and
        # cannot be mutated through the source tensor's graph.
        embed_w = embed_weight.detach().clone()
        head_w = lm_head_weight.detach().clone()

        self.frozen_embed = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.verifier_hidden_dim,
            _weight=embed_w,
        )
        self.frozen_embed.weight.requires_grad_(False)

        self.frozen_lm_head = nn.Linear(
            in_features=config.verifier_hidden_dim,
            out_features=config.vocab_size,
            bias=False,
        )
        with torch.no_grad():
            self.frozen_lm_head.weight.copy_(head_w)
        self.frozen_lm_head.weight.requires_grad_(False)

        # Learnable bridge projections.
        # W_in: d_v -> d_q (consumes verifier-space embeddings, emits
        #                   proposer-space activations to feed the backbone)
        # W_out: d_q -> d_v (consumes proposer-space hidden states, emits
        #                    verifier-space activations for the lm_head)
        # No bias on either: bias would shift representation alignment
        # off zero-mean for no benefit.
        self.W_in = nn.Linear(
            in_features=config.verifier_hidden_dim,
            out_features=config.proposer_hidden_dim,
            bias=False,
        )
        self.W_out = nn.Linear(
            in_features=config.proposer_hidden_dim,
            out_features=config.verifier_hidden_dim,
            bias=False,
        )

        self._init_bridges()

    def _validate_weight(self, weight: torch.Tensor, *, name: str) -> None:
        if weight.dim() != 2:
            raise ValueError(
                f"{name} must be 2-D [vocab_size, verifier_hidden_dim]; "
                f"got shape {tuple(weight.shape)}"
            )
        if weight.shape[0] != self.config.vocab_size:
            raise ValueError(
                f"{name}.shape[0] = {weight.shape[0]} does not match "
                f"config.vocab_size = {self.config.vocab_size}"
            )
        if weight.shape[1] != self.config.verifier_hidden_dim:
            raise ValueError(
                f"{name}.shape[1] = {weight.shape[1]} does not match "
                f"config.verifier_hidden_dim = {self.config.verifier_hidden_dim}"
            )

    def _init_bridges(self) -> None:
        """Truncated-normal init on both bridge projections.

        Std matches Qwen3's own linear-layer initializer (0.02 by
        default) so the surgery starts in a regime the backbone has
        seen during pretraining, which avoids large early-step
        gradients in Stage 3 training.
        """
        std = self.config.bridge_init_std
        nn.init.trunc_normal_(self.W_in.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.W_out.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)

    # ------------------------------------------------------------------
    # Public endpoints — these are what the Stage 3 trainer composes
    # with the proposer backbone.
    # ------------------------------------------------------------------

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token ids -> proposer-space embeddings.

        Equivalent to ``W_in(frozen_embed(input_ids))``. Flowing
        through the frozen embedding does not produce gradient against
        the embedding weights; only ``W_in`` is trained.

        Parameters
        ----------
        input_ids:
            Integer tensor of shape ``[B, L]`` with values in
            ``[0, V)``. Out-of-range ids will trigger CUDA-side
            assert failures or CPU IndexError exactly as a normal
            ``nn.Embedding`` would; we do not pre-validate them in
            this hot path.

        Returns
        -------
        Tensor of shape ``[B, L, proposer_hidden_dim]``.
        """
        verifier_space = self.frozen_embed(input_ids)
        return self.W_in(verifier_space)

    def project_to_verifier_space(self, hidden_q: torch.Tensor) -> torch.Tensor:
        """Proposer-space hidden states -> verifier-space hidden states.

        This is the projected-hidden output that Stage 3's
        representation-alignment loss compares against the verifier's
        own last-layer hidden states.

        Parameters
        ----------
        hidden_q:
            Tensor of shape ``[..., proposer_hidden_dim]``. Any leading
            shape is preserved; only the last dim is projected.

        Returns
        -------
        Tensor of shape ``[..., verifier_hidden_dim]``.
        """
        return self.W_out(hidden_q)

    def lm_logits(self, hidden_q: torch.Tensor) -> torch.Tensor:
        """Proposer-space hidden states -> token-vocabulary logits.

        Equivalent to ``frozen_lm_head(project_to_verifier_space(hidden_q))``.
        The frozen lm_head does not produce gradient against its own
        weights; gradient flows back through ``W_out`` only.

        Parameters
        ----------
        hidden_q:
            Tensor of shape ``[..., proposer_hidden_dim]``.

        Returns
        -------
        Tensor of shape ``[..., vocab_size]``.
        """
        return self.frozen_lm_head(self.project_to_verifier_space(hidden_q))

    def forward(
        self, input_ids: torch.Tensor, hidden_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience forward used by the Stage 3 trainer.

        Returns the triple ``(proposer_input_embeds, projected_hidden,
        logits)`` so the trainer can compute the embedding-time
        backbone input, the representation-alignment loss target, and
        the token-distill loss target in a single forward.

        Note: the *proposer backbone* is NOT called here. The trainer
        runs the backbone between calls to ``embed`` and the rest of
        the pipeline. This forward is a self-consistency helper, not
        the full pipeline.
        """
        proposer_input_embeds = self.embed(input_ids)
        projected_hidden = self.project_to_verifier_space(hidden_q)
        logits = self.frozen_lm_head(projected_hidden)
        return proposer_input_embeds, projected_hidden, logits

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_weights(
        cls,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        proposer_hidden_dim: int,
        bridge_init_std: float = 0.02,
    ) -> "ReprAlignedSurgery":
        """Build a surgery from raw weight tensors.

        Vocabulary size and verifier hidden dim are inferred from
        ``embed_weight.shape``; ``lm_head_weight`` is checked against
        them for consistency.
        """
        if embed_weight.dim() != 2:
            raise ValueError(
                "embed_weight must be 2-D [vocab_size, verifier_hidden_dim]; "
                f"got shape {tuple(embed_weight.shape)}"
            )
        config = SurgeryConfig(
            verifier_hidden_dim=int(embed_weight.shape[1]),
            proposer_hidden_dim=int(proposer_hidden_dim),
            vocab_size=int(embed_weight.shape[0]),
            bridge_init_std=float(bridge_init_std),
        )
        return cls(
            config=config,
            embed_weight=embed_weight,
            lm_head_weight=lm_head_weight,
        )

    @classmethod
    def from_verifier_module(
        cls,
        verifier: nn.Module,
        proposer_hidden_dim: int,
        bridge_init_std: float = 0.02,
        embed_module_path: str = "model.embed_tokens",
        lm_head_module_path: str = "lm_head",
    ) -> "ReprAlignedSurgery":
        """Build a surgery from an instantiated HuggingFace verifier.

        Walks the dotted path ``embed_module_path`` and
        ``lm_head_module_path`` from the verifier root to find the
        embedding and LM-head modules, then extracts their weight
        tensors. Defaults match Qwen3's HF layout
        (``model.embed_tokens`` and ``lm_head`` at the top level).
        Pass alternative paths for verifier families that lay out
        their modules differently.

        Notes on tied embeddings: if the verifier ties input embedding
        and output projection, both paths resolve to weights that are
        the same tensor in memory. We clone-detach independently in
        the constructor, so the surgery's internal frozen modules are
        independent copies regardless. Both stay frozen, so the
        duplication has no training-time correctness implication; the
        cost is one extra ``V × d_v`` worth of memory.
        """
        embed_module = _resolve_dotted_path(verifier, embed_module_path)
        lm_head_module = _resolve_dotted_path(verifier, lm_head_module_path)
        embed_weight = _extract_weight(embed_module, name=embed_module_path)
        lm_head_weight = _extract_weight(lm_head_module, name=lm_head_module_path)
        return cls.from_weights(
            embed_weight=embed_weight,
            lm_head_weight=lm_head_weight,
            proposer_hidden_dim=proposer_hidden_dim,
            bridge_init_std=bridge_init_std,
        )

    # ------------------------------------------------------------------
    # Introspection helpers used by the Stage 3 trainer and by tests.
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> int:
        """Count of parameters with ``requires_grad=True``.

        For a freshly-constructed surgery this returns
        ``2 * d_v * d_q`` — the two bridge projection matrices and
        nothing else.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameters(self) -> int:
        """Count of parameters with ``requires_grad=False``.

        For a freshly-constructed surgery this returns
        ``2 * V * d_v`` — the embedding and the lm_head, both frozen.
        """
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


def _resolve_dotted_path(root: nn.Module, path: str) -> nn.Module:
    """Walk ``path`` (e.g. ``"model.embed_tokens"``) from ``root``.

    Raises ``AttributeError`` with a helpful message that includes
    the offending attribute and the path traversed so far, which
    makes diagnosing layout mismatches between verifier families
    significantly easier than the default ``AttributeError`` from
    ``getattr``.
    """
    obj: object = root
    traversed: list[str] = []
    for part in path.split("."):
        if not hasattr(obj, part):
            traversed_str = ".".join(traversed) or "<root>"
            raise AttributeError(
                f"verifier module at '{traversed_str}' has no attribute "
                f"'{part}' (full path: '{path}'); pass an explicit "
                f"embed_module_path / lm_head_module_path matching "
                f"this verifier's HF layout"
            )
        obj = getattr(obj, part)
        traversed.append(part)
    if not isinstance(obj, nn.Module):
        raise TypeError(
            f"path '{path}' resolved to {type(obj).__name__}, "
            f"expected nn.Module"
        )
    return obj


def _extract_weight(module: nn.Module, *, name: str) -> torch.Tensor:
    """Return the ``weight`` tensor of an embedding or linear module.

    Both ``nn.Embedding`` and ``nn.Linear`` expose their parameter
    matrix under ``.weight``. We accept either; anything else is a
    structural mismatch and we error rather than guess.
    """
    if not hasattr(module, "weight"):
        raise AttributeError(
            f"module at '{name}' (type {type(module).__name__}) has "
            f"no 'weight' attribute; expected nn.Embedding or nn.Linear"
        )
    weight = module.weight
    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"'{name}.weight' is {type(weight).__name__}, "
            f"expected torch.Tensor"
        )
    return weight
