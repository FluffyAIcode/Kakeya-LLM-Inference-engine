"""K3 Block C — `f_θ` K/V projection: drafter K/V → verifier K/V space.

Per ADR 0008 §11.5 (v0.4 GA dLM K/V Restoration architecture), the
verifier maintains only a sink+window local KV cache and accepts
**reconstructed** K/V at every evicted position from the proposer's
transient K/V. In the K3 cross-model setup (drafter = DFlash 0.4B,
verifier = Gemma 4 26B-A4B), the drafter's K/V live in a different
space than the verifier's:

  drafter K, V shape (per layer, per position):
      [num_kv_heads_drafter * head_dim_drafter]      (e.g. 2 * 128 = 256)

  verifier K, V shape (per layer, per position):
      [num_kv_heads_verifier * head_dim_verifier]    (e.g. 8 * 256 = 2048)

`f_θ` is the trainable projection that bridges these spaces. Its
contract: for every position p, take the drafter's K/V at p across
ALL drafter layers (concatenated along the feature dim) and produce
the verifier's K/V at p at EVERY verifier layer.

Architecture (chosen 2026-06-09 for K3 first-iteration training)
----------------------------------------------------------------

Shared encoder + per-verifier-layer decoder, low-rank factorisation:

    drafter_kv_input [B, T, drafter_layers * drafter_kv_dim]
                ↓
    shared encoder Linear(drafter_layers*drafter_kv_dim, rank)
                ↓
    rep [B, T, rank]
                ↓
    per-verifier-layer decoder K: Linear(rank, verifier_kv_dim) × num_verifier_layers
    per-verifier-layer decoder V: Linear(rank, verifier_kv_dim) × num_verifier_layers
                ↓
    output [B, T, num_verifier_layers, num_kv_heads_v, head_dim_v]
        for K, and same shape for V

Total params (default rank=256):
    encoder:  drafter_layers * drafter_kv_dim × rank = 5 * 256 × 256 ≈ 327k
    decoders: 2 (K+V) × num_verifier_layers × rank × verifier_kv_dim
            = 2 × 30 × 256 × 2048 ≈ 31.5M
    Total:    ~31.8M params (vs drafter 430M, verifier 26B → small)

Why this architecture
---------------------

1. **Per-verifier-layer decoders**: each verifier layer has its own
   K/V distribution; one shared output projection is too lossy. 30
   separate decoders give per-layer capacity.

2. **Shared encoder**: forces the drafter K/V representation to
   capture position-level features that generalise across verifier
   layers. Reduces parameter count vs full per-(drafter,verifier)-pair
   matrices (which would be 30 × 5 × 2 × full_dim².

3. **Low-rank**: rank=256 is a tunable. Smaller rank = fewer params
   + faster training but less capacity; larger rank approaches the
   shared encoder being identity. 256 was chosen as the smallest
   rank that keeps verifier_kv_dim/rank ratio reasonable (2048/256=8)
   without crushing capacity at the encoder bottleneck.

4. **Separate K and V decoders**: K and V have different roles
   downstream (Q·K dot product vs attention-weighted sum of V); their
   per-layer distributions differ. Separate decoders capture this.

Training contract (per :mod:`scripts.research.k3_f_theta_train`)
----------------------------------------------------------------

* Inputs: paired (drafter_kv, verifier_kv) over a long-context corpus
  collected by running both models on the same input sequences and
  recording K/V at every layer at every position.

* Loss: MSE between f_θ(drafter_kv) and verifier_kv, averaged over
  layers and positions. Weighted equally across layers; weighting
  schemes are a hyperparameter.

* Optimiser: AdamW with lr=1e-3, weight_decay=0.01.

Loadable checkpoint
-------------------

The trained `f_θ` is saved as a state_dict. The
:class:`FThetaProjection.from_pretrained` classmethod loads from
either a local file or HF hub id. The cross-model DLMRestoredVerifier
(:mod:`inference_engine.v04.cross_model_dlm_verifier`) consumes this
state_dict at construction time.

This module is engine API surface (not research scaffolding), so
imports are minimal and tests cover the shape contract + load/save
+ device dispatch.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclasses.dataclass(frozen=True)
class FThetaConfig:
    """Configuration for :class:`FThetaProjection`.

    Stored alongside the trained state_dict as ``f_theta_config.json``
    so the cross-model verifier can reconstruct the projection at load
    time without inferring shapes from the state_dict alone.

    Heterogeneous verifier KV heads
    -------------------------------
    Production verifiers do not always use a uniform KV-head count
    across layers. Gemma 4 26B-A4B, for example, uses 8 KV heads on its
    sliding-attention layers and 4 KV heads on its full-attention
    layers (head_dim is uniform at 256). ``verifier_layer_kv_heads``
    captures the per-layer count; when ``None`` every layer uses
    ``verifier_num_kv_heads`` (the legacy uniform behaviour, kept for
    backward compatibility and the same-head-count case).
    """

    drafter_num_layers: int            # e.g. DFlash drafter 5 layers
    drafter_num_kv_heads: int          # e.g. DFlash 2 kv heads
    drafter_head_dim: int              # e.g. DFlash 128 head dim
    verifier_num_layers: int           # e.g. Gemma 4 26B-A4B 30 layers
    verifier_num_kv_heads: int         # representative / uniform KV head count
    verifier_head_dim: int             # e.g. Gemma 4 256 head dim
    rank: int = 256                    # encoder bottleneck
    # Per-layer KV head counts (len == verifier_num_layers). None ⇒
    # uniform verifier_num_kv_heads for every layer.
    verifier_layer_kv_heads: Optional[Tuple[int, ...]] = None
    # Per-layer head dims (len == verifier_num_layers). None ⇒ uniform
    # verifier_head_dim. Gemma 4 uses 256 on sliding layers and 512
    # (global_head_dim) on its full-attention layers.
    verifier_layer_head_dims: Optional[Tuple[int, ...]] = None

    @property
    def drafter_kv_dim(self) -> int:
        return self.drafter_num_kv_heads * self.drafter_head_dim

    @property
    def verifier_kv_dim(self) -> int:
        return self.verifier_num_kv_heads * self.verifier_head_dim

    @property
    def layer_kv_heads(self) -> Tuple[int, ...]:
        """Per-layer KV head counts (always length ``verifier_num_layers``)."""
        if self.verifier_layer_kv_heads is None:
            return tuple(
                self.verifier_num_kv_heads
                for _ in range(self.verifier_num_layers)
            )
        return tuple(int(h) for h in self.verifier_layer_kv_heads)

    @property
    def layer_head_dims(self) -> Tuple[int, ...]:
        """Per-layer head dims (always length ``verifier_num_layers``)."""
        if self.verifier_layer_head_dims is None:
            return tuple(
                self.verifier_head_dim
                for _ in range(self.verifier_num_layers)
            )
        return tuple(int(d) for d in self.verifier_layer_head_dims)

    @property
    def layer_kv_dims(self) -> Tuple[int, ...]:
        """Per-layer K (or V) feature dim = kv_heads[i] * head_dim[i]."""
        return tuple(
            h * d for h, d in zip(self.layer_kv_heads, self.layer_head_dims)
        )

    @property
    def encoder_in_features(self) -> int:
        """Concat dim across all drafter layers' K (or V) per position."""
        return self.drafter_num_layers * self.drafter_kv_dim

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if self.verifier_layer_kv_heads is not None:
            d["verifier_layer_kv_heads"] = list(self.verifier_layer_kv_heads)
        if self.verifier_layer_head_dims is not None:
            d["verifier_layer_head_dims"] = list(self.verifier_layer_head_dims)
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "FThetaConfig":
        list_fields = {"verifier_layer_kv_heads", "verifier_layer_head_dims"}
        kwargs: dict = {}
        for k, v in d.items():
            if k in list_fields:
                kwargs[k] = None if v is None else tuple(int(x) for x in v)
            else:
                kwargs[k] = int(v)
        return cls(**kwargs)


class FThetaProjection(nn.Module):
    """`f_θ`: projects drafter K/V into verifier K/V space.

    Forward contract:

      forward_k(drafter_k_concat: torch.Tensor)
        Input shape:  [B, T, drafter_num_layers * drafter_kv_dim]
        Output shape: [B, T, verifier_num_layers, verifier_num_kv_heads, verifier_head_dim]

      forward_v(drafter_v_concat: torch.Tensor)
        Same shapes as forward_k but separate weights (K and V have
        different downstream roles → separate projections).

    Helper :meth:`forward_kv_pack` accepts the unpacked drafter
    KVCapture format (list of 5 [B, T, num_kv_heads_d, head_dim_d]
    tensors) and runs the concat + project + reshape pipeline in one
    call — what the cross-model verifier uses.
    """

    def __init__(self, config: FThetaConfig) -> None:
        super().__init__()
        self.config = config

        # Shared encoder: drafter K/V (concat across drafter layers) → rank-d rep
        self.encoder_k = nn.Linear(
            config.encoder_in_features, config.rank, bias=False,
        )
        self.encoder_v = nn.Linear(
            config.encoder_in_features, config.rank, bias=False,
        )

        # Per-verifier-layer decoders, each sized to its own layer's KV
        # feature dim (heterogeneous KV-head counts are supported).
        self.decoders_k = nn.ModuleList([
            nn.Linear(config.rank, kv_dim, bias=False)
            for kv_dim in config.layer_kv_dims
        ])
        self.decoders_v = nn.ModuleList([
            nn.Linear(config.rank, kv_dim, bias=False)
            for kv_dim in config.layer_kv_dims
        ])

    # -----------------------------------------------------------------
    # Forward primitives
    # -----------------------------------------------------------------

    def _project(
        self,
        drafter_concat: torch.Tensor,
        encoder: nn.Module,
        decoders: nn.ModuleList,
    ) -> List[torch.Tensor]:
        if drafter_concat.dim() != 3:
            raise ValueError(
                f"expected [B, T, encoder_in_features]; got shape "
                f"{tuple(drafter_concat.shape)}"
            )
        if drafter_concat.size(-1) != self.config.encoder_in_features:
            raise ValueError(
                f"last dim {drafter_concat.size(-1)} != "
                f"encoder_in_features {self.config.encoder_in_features}"
            )
        # f_θ weights may be a different dtype than the captured drafter
        # K/V (e.g. f_θ in fp32, drafter in bf16). Cast the input to the
        # encoder's weight dtype so matmul dtypes agree.
        drafter_concat = drafter_concat.to(encoder.weight.dtype)
        rep = encoder(drafter_concat)  # [B, T, rank]
        kv_heads = self.config.layer_kv_heads
        head_dims = self.config.layer_head_dims
        outs: List[torch.Tensor] = []
        for li, dec in enumerate(decoders):
            o = dec(rep)  # [B, T, kv_heads[li] * head_dims[li]]
            B, T, _ = o.shape
            outs.append(o.view(B, T, kv_heads[li], head_dims[li]))
        return outs

    def forward_k(self, drafter_k_concat: torch.Tensor) -> List[torch.Tensor]:
        """Project drafter K (concat across drafter layers) to per-verifier-layer K.

        Parameters
        ----------
        drafter_k_concat
            [B, T, drafter_num_layers * drafter_kv_dim]

        Returns
        -------
        List of ``verifier_num_layers`` tensors, each shape
        ``[B, T, layer_kv_heads[i], verifier_head_dim]`` (per-layer KV
        head counts can differ).
        """
        return self._project(drafter_k_concat, self.encoder_k, self.decoders_k)

    def forward_v(self, drafter_v_concat: torch.Tensor) -> List[torch.Tensor]:
        """V counterpart of :meth:`forward_k`."""
        return self._project(drafter_v_concat, self.encoder_v, self.decoders_v)

    # -----------------------------------------------------------------
    # KVCapture-aware helper
    # -----------------------------------------------------------------

    def forward_kv_pack(
        self,
        drafter_k_per_layer: Sequence[torch.Tensor],
        drafter_v_per_layer: Sequence[torch.Tensor],
    ) -> tuple:
        """Take unpacked KVCapture tensors and project to verifier K/V.

        Parameters
        ----------
        drafter_k_per_layer
            List of ``drafter_num_layers`` tensors, each shape
            ``[B, T, drafter_num_kv_heads, drafter_head_dim]`` (the
            natural KVCapture layout from
            :class:`inference_engine.v04.KVCapture`).

        drafter_v_per_layer
            Same as ``drafter_k_per_layer`` but for V tensors.

        Returns
        -------
        (verifier_k, verifier_v) where each is a list of
        ``verifier_num_layers`` tensors, element ``i`` shaped
        ``[B, T, layer_kv_heads[i], verifier_head_dim]``.
        """
        if len(drafter_k_per_layer) != self.config.drafter_num_layers:
            raise ValueError(
                f"expected {self.config.drafter_num_layers} drafter layers, "
                f"got {len(drafter_k_per_layer)}"
            )
        if len(drafter_v_per_layer) != self.config.drafter_num_layers:
            raise ValueError(
                f"expected {self.config.drafter_num_layers} drafter layers "
                f"for V, got {len(drafter_v_per_layer)}"
            )
        # Concat along the kv-feature dim to get [B, T, drafter_layers * kv_dim]
        # Each layer tensor is [B, T, num_kv_heads, head_dim] → flatten last two
        # dims → [B, T, kv_dim], then concat across layers.
        k_flat = [k.flatten(-2, -1) for k in drafter_k_per_layer]
        v_flat = [v.flatten(-2, -1) for v in drafter_v_per_layer]
        k_concat = torch.cat(k_flat, dim=-1)  # [B, T, drafter_layers * kv_dim]
        v_concat = torch.cat(v_flat, dim=-1)
        return self.forward_k(k_concat), self.forward_v(v_concat)

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save_pretrained(self, output_dir: str | Path) -> None:
        """Save config + state_dict to ``output_dir``.

        Layout::

            output_dir/
                f_theta_config.json     # FThetaConfig.to_json_dict()
                f_theta_weights.pt      # torch state_dict (bf16 by default)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "f_theta_config.json").write_text(
            json.dumps(self.config.to_json_dict(), indent=2),
        )
        torch.save(self.state_dict(), output_dir / "f_theta_weights.pt")

    @classmethod
    def from_pretrained(
        cls, source: str | Path, *, dtype: Any = None, device: Any = None,
    ) -> "FThetaProjection":
        """Load f_θ from a directory containing config + weights.

        ``source`` is a local directory. HF Hub support deferred until
        a public f_θ checkpoint is hosted (training is internal first).
        """
        source = Path(source)
        if not source.is_dir():
            raise FileNotFoundError(
                f"f_θ source must be a directory; got {source}"
            )
        config_path = source / "f_theta_config.json"
        weights_path = source / "f_theta_weights.pt"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"missing {weights_path}")

        config = FThetaConfig.from_json_dict(
            json.loads(config_path.read_text()),
        )
        model = cls(config)
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        if dtype is not None:
            model = model.to(dtype)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model
