# K3 cross-model DLMRestoredVerifier — interface contract

**Status**: design draft (2026-06-09).
**Implementation PR**: not yet opened.
**Successor of**: K1.D `DLMRestoredVerifier` (single-model, identity `f_θ`).

This document specifies the contract for the cross-model
`DLMRestoredVerifier` that the K2.B / K3 implementation PR must
deliver. The current K1.D `DLMRestoredVerifier(model, ...)`
assumes a single shared model instance for both proposer and
verifier roles; K3 production scale uses two independent models
of different architectures (DFlash dLM drafter + Gemma 4 26B AR
verifier) plus a learned per-layer-per-head projection `f_θ`.

The contract here is what the K3 implementation must satisfy.
The implementation itself is the next PR; this document scopes
its acceptance.

## 1. Constructor signature

The cross-model wrapper extends K1.D's constructor:

```python
class DLMRestoredVerifier:
    def __init__(
        self,
        # Existing K1 / K2.A parameter (kept for backward compat):
        model: Optional[nn.Module] = None,            # same-model path
        # New K2.B / K3 parameters:
        proposer_model: Optional[nn.Module] = None,   # dLM drafter
        verifier_model: Optional[nn.Module] = None,   # AR target
        f_theta: Optional["LayerProjection"] = None,  # K/V projection
        layer_alignment: Optional[List[int]] = None,  # drafter L → verifier L
        *,
        sink_size: int = 4,
        window_size: int = 64,
        kv_compressor_factory: Optional[Callable] = None,  # K2.A.1 hook
    ) -> None: ...
```

Three valid configurations:

| `model` | `proposer_model` | `verifier_model` | `f_theta` | semantics |
|---|---|---|---|---|
| not None | None | None | (ignored) | **K1 / K2.A path**: same model, identity projection. Backward-compatible with current K1.D code. |
| None | not None | not None | not None | **K2.B / K3 path**: cross-model with learned projection. |
| any combination of None / both / mismatched | | | | **Error**: `ValueError` with message specifying which path the caller appears to want and what's missing. |

The constructor raises `ValueError` for invalid combinations
**before** attempting any model interrogation, so misuse is
caught at construct time, not buried in a forward.

## 2. `LayerProjection` interface — the `f_θ` adapter

The `f_theta` parameter is an instance of a `LayerProjection`
protocol that handles the drafter→verifier K/V projection:

```python
class LayerProjection(Protocol):
    """Cross-model K/V projection: drafter K, V → verifier K, V.

    Stateless, parameterised, callable. One LayerProjection
    instance covers all (drafter_layer, verifier_layer) pairs
    via the layer_alignment mapping; per-layer or per-head
    parameter-tying is an implementation detail of the chosen
    LayerProjection class.
    """

    @property
    def drafter_kv_shape(self) -> Tuple[int, int, int]:
        """(num_layers, num_kv_heads, head_dim) of the drafter's K/V."""

    @property
    def verifier_kv_shape(self) -> Tuple[int, int, int]:
        """(num_layers, num_kv_heads, head_dim) of the verifier's K/V."""

    def project(
        self,
        drafter_K: torch.Tensor,        # [B, drafter_kv_heads, T, drafter_head_dim]
        drafter_V: torch.Tensor,
        drafter_layer_idx: int,
        verifier_layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project drafter K/V → verifier K/V at the given layer pair.

        Returns (verifier_K, verifier_V) of shape
        [B, verifier_kv_heads, T, verifier_head_dim].

        For K2.B/K3 same-family pairings, project is implemented as a
        per-layer-per-head linear projection trained per
        docs/design/k3-f-theta-training-pipeline.md.
        """

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Trainable parameters; saved with checkpoint."""

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None: ...
```

Two concrete implementations needed:

- **`IdentityLayerProjection`**: passthrough used by K1 / K2.A
  same-model path (`drafter_K == verifier_K` by construction
  when `proposer_model is verifier_model`). No trainable
  parameters. Validates that `drafter_kv_shape == verifier_kv_shape`.
- **`LinearLayerProjection`**: per-(layer, head) linear projection
  `verifier_K = W_K · drafter_K + b_K` (and similar for V),
  where `W_K` shape is `[verifier_kv_heads, verifier_head_dim,
  drafter_kv_heads, drafter_head_dim]` (broadcastable across
  batch and time dims). Trainable; trained per the f_θ training
  pipeline doc.

## 3. `layer_alignment` — drafter layer → verifier layer mapping

DFlash drafter and Gemma 4 verifier have different layer counts:

| K3 model | layer count | head_dim | num_kv_heads (per group) |
|---|---|---|---|
| `z-lab/gemma-4-26B-A4B-it-DFlash` | TBD (likely ~6-12; block-diffusion drafters are deep but few) | TBD | TBD |
| `google/gemma-4-26B-A4B-it` | 30 | per Google spec | per Google spec |

(TBDs above must be filled in by the K3 implementation PR after
loading both models and inspecting `model.config`. The K3
feasibility smoke script writes these into its JSON report.)

`layer_alignment[verifier_layer_idx] -> drafter_layer_idx` is a
list of length `verifier_num_layers`, specifying which drafter
layer's K/V to use as the source for each verifier layer's
restoration. Three viable strategies:

- **`uniform`**: `layer_alignment[i] = (drafter_num_layers - 1) * i // (verifier_num_layers - 1)`.
  Map verifier layers linearly across drafter layers. Simplest;
  works when drafter has more layers than would seem necessary.
- **`pooled`**: `layer_alignment[i] = bucket_average(verifier_layer_groups)`.
  Multiple verifier layers share one drafter layer's K/V;
  averaging across verifier layers is done inside `project()`.
- **`learned`**: a separate small embedding from each
  `verifier_layer_idx` to a soft mixture over drafter layers.
  Trainable as part of `f_θ`. Most flexible, hardest to interpret.

K3 implementation PR should default to `uniform` and provide
`pooled` and `learned` as opt-in via `LayerProjection` subclass.

## 4. `forward()` orchestration

The cross-model forward replaces K1.D's single-model `model.forward()`
with a paired execution:

```python
@torch.no_grad()
def forward(
    self,
    input_ids: torch.Tensor,       # [1, T] (single-batch only in K1/K2/K3)
    *,
    apply_rotary_pos_emb,           # verifier's HF helpers
    eager_attention_forward,
    all_attention_functions=None,
) -> torch.Tensor:
    # Step 1: drafter forward → capture K, V at every drafter layer.
    drafter_capture = capture_proposer_kv(self.proposer_model, input_ids)

    # Step 2: compute evicted positions (same as K1.D).
    evicted = compute_evicted_positions(seq_len, sink, window)

    # Step 3: project drafter K/V → verifier K/V via f_θ at each
    #         (verifier_layer, drafter_layer) pair determined by
    #         layer_alignment. Result: verifier_capture in
    #         verifier-shape, sliceable to evicted positions.
    verifier_capture = self.f_theta.project_capture(
        drafter_capture, layer_alignment=self.layer_alignment,
    )

    # Step 4-5: install patched attention on VERIFIER's layers
    #            (not drafter's), using verifier_capture sliced to
    #            evicted positions. Identical to K1.D from this
    #            point onward.
    with self._restoration_active(
        verifier_capture, evicted, resident,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        eager_attention_forward=eager_attention_forward,
        all_attention_functions=all_attention_functions,
    ):
        outputs = self.verifier_model(input_ids=input_ids, use_cache=False)

    return outputs.logits
```

Critical invariants:

- **`drafter_capture.num_layers == drafter_num_layers`** (drafter's
  own count, NOT verifier's). Existing `KVCapture` works
  unchanged; the projection step in 3 transforms shapes.
- **`verifier_capture.num_layers == verifier_num_layers`** post-projection.
- **The patched attention is installed on the VERIFIER's layers**.
  The drafter's attention is never patched — it just runs its
  own forward and we hook K/V outputs.
- **Tokenizer parity**: both models must share tokenizer (same
  family). `Qwen/Qwen3.5-4B` ↔ `z-lab/Qwen3.5-4B-DFlash` is
  same family; `google/gemma-4-26B-A4B-it` ↔
  `z-lab/gemma-4-26B-A4B-it-DFlash` is same family. Cross-family
  pairing (e.g. Qwen3.5 + Gemma 4) is **not supported** in K2.B/K3
  and would require token re-mapping (out of scope).

## 5. Memory budget

Per ADR 0008 §11.13 sustained vs per-step peak distinction, the
cross-model wrapper must satisfy:

- **Sustained** (between forwards):
  - K1 path: `O(model_weights)` (one model, ≈ 2 GB for Gemma 3-1B)
  - K3 path stateless K2.A.1-equivalent: `O(verifier_weights + drafter_weights + f_theta_weights)`
    ≈ 13 GB (4-bit verifier on Mac) + 0.8 GB (drafter) + ~10 MB (f_θ)
    ≈ 14 GB sustained on Mac M4
- **Per-step peak**: `O(2 × T × hidden_dim)` from K1.D / K2.A.1
  (proposer + verifier full forwards). On Mac M4 24 GB at 4-bit
  verifier, the verifier-side T-scaled component is reduced (4-bit
  storage = smaller activations? — unclear; the activations are
  computed in fp16/bf16 internally even when weights are 4-bit).
  Empirical confirmation needed via the K3 feasibility smoke at
  varying `--prompt-tokens`.

## 6. Backward compatibility

Existing K1.D / K2.A.1 callers (single `model` argument) **must
continue to work bit-for-bit**. The signature change is additive:
new optional kwargs default to None, and the constructor branches
to the K1 path when `model is not None`.

Test: every existing test in
`tests/inference_engine/v04/test_dlm_restored_verifier.py` (31 tests
including the K2.A.1 backward-compat regression guard) **must
continue to pass unchanged** under the cross-model implementation.

## 7. What this contract does NOT specify

- The training procedure for `f_θ`. See
  [k3-f-theta-training-pipeline.md](k3-f-theta-training-pipeline.md).
- The KakeyaLattice composition path (K2.A.1 stateless integration
  is wired through `kv_compressor_factory`; K2.A.2 stateful caching
  is a separate refactor). Per ADR 0008 §11.11.6, K2.B's `f_θ` is
  trained against KL-on cache; this contract is silent on how —
  the K3 implementation PR addresses that wiring.
- Speculative-decoding integration (DFlash's draft-token output).
  K3 ships v0.4 K/V Restoration on top of DFlash; the SD glue
  (taking DFlash's draft tokens, running rejection sampling
  against the verifier with restored attention, etc.) is a
  separate concern from this verifier contract.
- Multi-batch (`B > 1`). K1.D currently asserts `B == 1`; K3
  inherits that constraint until a separate batching PR.

## 8. Acceptance criteria (for the K3 implementation PR)

The K3 implementation PR is accepted when:

1. The constructor signature matches §1 exactly. Invalid
   configurations raise `ValueError` with diagnostic messages
   referencing this document.
2. `IdentityLayerProjection` and `LinearLayerProjection` both
   ship as concrete classes; `LinearLayerProjection` has a
   trainable parameter set saveable via `state_dict()`.
3. `layer_alignment=uniform` is the default; `pooled` and
   `learned` are tested but not the default.
4. K1.D backward compat: all 31 existing tests in
   `test_dlm_restored_verifier.py` pass unchanged.
5. New cross-model tests cover: `IdentityLayerProjection` ==
   K1.D bit-for-bit when `proposer_model is verifier_model`;
   `LinearLayerProjection` with random weights produces
   non-NaN output of correct shape; `layer_alignment` strategies
   `uniform`/`pooled`/`learned` produce different outputs but
   all of correct shape; `ValueError` on invalid configs.
6. K3 feasibility smoke (`scripts/research/k3_feasibility_smoke.py`)
   loads + smoke-forwards the verified DFlash drafter +
   Gemma 4 26B-A4B-it pair, with a JSON evidence report
   confirming load + memory + latency.

## 9. Open questions deferred to implementation

These are flagged but not answered by this contract; the K3
implementation PR may surface architectural surprises requiring
this contract to be amended.

- DFlash drafter's exact layer count, head_dim, num_kv_heads —
  unknown until model loaded. The K3 feasibility smoke writes
  these into its JSON report; K3 implementation PR uses those
  values to parameterise `LinearLayerProjection`.
- Whether DFlash's `k_proj` / `v_proj` modules are accessible
  via the same hook pattern as K1.A. DFlash uses
  `trust_remote_code=True` (custom modeling_*.py); the K1.A
  hook code may need to adapt to a different attention class
  hierarchy.
- f_θ initialization. Per ADR §11.11.6, DFlash drafters are
  pre-trained to condition on target features; `f_θ` may
  initialize close to identity-on-shared-subspace rather than
  random. The K3 implementation PR's f_θ training
  start-of-training behaviour will reveal whether this is true.
- f_θ trainable parameter count budget. At 65:1 scale ratio
  with per-(layer, head) linear projection, the parameter count
  is roughly:
    `verifier_layers × verifier_kv_heads × verifier_head_dim
     × drafter_kv_heads × drafter_head_dim`
  For Gemma 4 26B-A4B (30 layers, ~16 kv_heads, head_dim=256
  per typical Gemma family) × DFlash drafter (TBD shapes), the
  parameter count is ~ 30 × 16 × 256 × 16 × 256 ≈ 500M
  parameters — comparable in size to the drafter itself. May
  need parameter sharing across layers or low-rank factorisation
  to fit a research training budget.
