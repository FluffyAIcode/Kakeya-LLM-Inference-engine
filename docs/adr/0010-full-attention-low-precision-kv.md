# ADR 0010 — Full-attention verifier + low-precision (INT8 / NF4) KV cache

- **Status: OBSOLETE / WITHDRAWN (2026-06-07, never implemented)**
- **Date drafted**: 2026-06-07
- **Date withdrawn**: 2026-06-07 (same day; never left draft)
- **Reason for withdrawal**: This ADR proposed implementing NF4 / INT8 KV
  cache quantization as the v0.4 GA path. Withdrawn the same day because
  [KakeyaLattice](https://github.com/FluffyAIcode/LLM-KV--Cache-compress)
  (D4 / E8 nested-lattice KV codec) **already exists and is strictly
  better** than this draft on every metric the draft cared about:
  - Validated on real vLLM + NVIDIA H200, not theoretical.
  - **Beats Google's TurboQuant 12 / 12** on K-MSE, V-MSE, and |Δppl| at
    matched bit budgets across Qwen3, GLM-4, Gemma 4, DeepSeek.
  - **+26.9 % to +37.8 % compression-ratio advantage** at deployment-
    relevant quality thresholds (|Δppl| ≤ 2 %).
  - Pure per-vector codec, no cross-token state — supports streaming /
    online compression with zero calibration.
  - Distributed as `pip install kakeyalattice` with reference vLLM
    plugin and `transformers.DynamicCache` drop-in subclass.

  Pushing NF4 as the v0.4 GA path while KakeyaLattice already exists
  would be **a regression from the project's existing strength to a
  generic baseline**. This draft is preserved in git history as a
  factual record of a wrong recommendation; do not implement it.
- **Replaced by**: [ADR 0012 — KakeyaLattice KV codec integration](0012-kakeyalattice-kv-codec-integration.md)
  (planned; defines backend coverage, MLX port, v1.4/v1.5 selection,
  and integration with the AR-verifier + DLM-proposer architecture).

---

## Why this withdrawal exists in the doc tree

Open ADRs are read by future contributors as "options under
consideration". An ADR draft that recommends an inferior design,
left lingering as `Proposed`, will eventually be picked up and
acted on by someone who doesn't know about KakeyaLattice. Replacing
the draft with this tombstone makes the obsolete status visible at
the file level — the title bar of any IDE / `cat 0010-*.md` /
`grep -r Status docs/adr/` immediately shows `OBSOLETE`.

The original draft's content (NF4 default, INT8 fallback, outlier-
aware calibration, six-phase implementation plan) is recoverable
from git history if anyone needs to reference what was withdrawn:

```bash
git show <commit-before-withdrawal>:docs/adr/0010-full-attention-low-precision-kv.md
```

It is intentionally not preserved inline because keeping rejected
designs prominently visible in the doc tree creates ambiguity about
project direction.

## Lesson recorded

The agent assistant that drafted this ADR did so because it was
unaware of KakeyaLattice (a separate repository under the same
owner). When recommending a KV-cache compression strategy, the
correct prior should have been "the project owner's existing
production-grade KV codec is the relevant baseline, not generic
NF4 / INT8 from the literature". Future ADR drafts touching KV
compression should explicitly check
`https://github.com/FluffyAIcode/LLM-KV--Cache-compress` for the
current SOTA before proposing alternatives.
