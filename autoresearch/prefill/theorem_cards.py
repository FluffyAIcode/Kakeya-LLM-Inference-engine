"""Deterministic, bounded theorem cards for the pinned Lean environment."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CARD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TheoremCard:
    card_id: str
    theorem_name: str
    exact_type: str
    import_name: str
    source_path: str
    source_hash: str
    environment_hash: str
    applicability_tags: tuple[str, ...]
    required_hypotheses: tuple[str, ...]
    description: str

    @property
    def content_hash(self) -> str:
        return _digest(asdict(self))


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def pinned_environment_hash(project_root: Path) -> str:
    root = Path(project_root)
    return _digest({
        name: (root / name).read_text(encoding="utf-8")
        for name in ("lean-toolchain", "lake-manifest.json", "lakefile.lean")
    })


def _source_hash(root: Path, relative: str) -> str:
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()


_CARD_SPECS = (
    (
        "locally_uniform_limit_holomorphic",
        "TendstoLocallyUniformlyOn.differentiableOn",
        """{E : Type u_1} {ι : Type u_2} [NormedAddCommGroup E] [NormedSpace ℂ E]
{U : Set ℂ} {φ : Filter ι} {F : ι → ℂ → E} {f : ℂ → E} [CompleteSpace E] [φ.NeBot]
(hf : TendstoLocallyUniformlyOn F f φ U)
(hF : ∀ᶠ n in φ, DifferentiableOn ℂ (F n) U) (hU : IsOpen U) :
DifferentiableOn ℂ f U""",
        "Mathlib.Analysis.Complex.LocallyUniformLimit",
        ".lake/packages/mathlib/Mathlib/Analysis/Complex/LocallyUniformLimit.lean",
        ("locally_uniform_limit", "holomorphic_sum", "disk"),
        ("filter_nebot", "local_uniform_convergence", "eventual_holomorphicity", "open_domain"),
        "A locally uniform limit of holomorphic functions on an open set is holomorphic.",
    ),
    (
        "holomorphic_tsum_summable_bound",
        "Complex.differentiableOn_tsum_of_summable_norm",
        """{E : Type u_1} {ι : Type u_2} [NormedAddCommGroup E] [NormedSpace ℂ E]
{U : Set ℂ} {F : ι → ℂ → E} [CompleteSpace E] {u : ι → ℝ} (hu : Summable u)
(hf : ∀ i, DifferentiableOn ℂ (F i) U) (hU : IsOpen U)
(hF_le : ∀ i w, w ∈ U → ‖F i w‖ ≤ u i) :
DifferentiableOn ℂ (fun w => ∑' i, F i w) U""",
        "Mathlib.Analysis.Complex.LocallyUniformLimit",
        ".lake/packages/mathlib/Mathlib/Analysis/Complex/LocallyUniformLimit.lean",
        ("holomorphic_sum", "tsum", "summable_majorant"),
        ("summable_norm_bound", "termwise_holomorphicity", "open_domain"),
        "A summably dominated series of holomorphic terms has a holomorphic sum.",
    ),
    (
        "removable_singularity_continuous",
        "Complex.analyticAt_of_differentiable_on_punctured_nhds_of_continuousAt",
        """{E : Type u} [NormedAddCommGroup E] [NormedSpace ℂ E] [CompleteSpace E]
{f : ℂ → E} {c : ℂ}
(hd : ∀ᶠ z in 𝓝[≠] c, DifferentiableAt ℂ f z)
(hc : ContinuousAt f c) : AnalyticAt ℂ f c""",
        "Mathlib.Analysis.Complex.RemovableSingularity",
        ".lake/packages/mathlib/Mathlib/Analysis/Complex/RemovableSingularity.lean",
        ("removable_singularity", "analytic_extension", "punctured_neighborhood"),
        ("punctured_holomorphicity", "continuity_at_center"),
        "Punctured-neighborhood holomorphicity plus continuity makes the singularity removable.",
    ),
    (
        "removable_singularity_bounded",
        "Complex.differentiableOn_update_limUnder_of_bddAbove",
        """{E : Type u} [NormedAddCommGroup E] [NormedSpace ℂ E] [CompleteSpace E]
{f : ℂ → E} {s : Set ℂ} {c : ℂ} (hc : s ∈ 𝓝 c)
(hd : DifferentiableOn ℂ f (s \\ {c}))
(hb : BddAbove (norm ∘ f '' (s \\ {c}))) :
DifferentiableOn ℂ (Function.update f c ((𝓝[≠] c).limUnder f)) s""",
        "Mathlib.Analysis.Complex.RemovableSingularity",
        ".lake/packages/mathlib/Mathlib/Analysis/Complex/RemovableSingularity.lean",
        ("removable_singularity", "bounded", "holomorphic_extension"),
        ("neighborhood", "punctured_holomorphicity", "bounded_image"),
        "A bounded punctured holomorphic function extends differentiably across the center.",
    ),
    (
        "analytic_identity_principle",
        "AnalyticOnNhd.eqOn_of_preconnected_of_eventuallyEq",
        """{𝕜 E F} [NontriviallyNormedField 𝕜] [NormedAddCommGroup E] [NormedSpace 𝕜 E]
[NormedAddCommGroup F] [NormedSpace 𝕜 F] {f g : E → F} {U : Set E}
(hf : AnalyticOnNhd 𝕜 f U) (hg : AnalyticOnNhd 𝕜 g U) (hU : IsPreconnected U)
{z₀ : E} (h₀ : z₀ ∈ U) (hfg : f =ᶠ[𝓝 z₀] g) : Set.EqOn f g U""",
        "Mathlib.Analysis.Analytic.Uniqueness",
        ".lake/packages/mathlib/Mathlib/Analysis/Analytic/Uniqueness.lean",
        ("identity_principle", "analytic_continuation", "preconnected"),
        ("both_analytic", "preconnected_domain", "local_eventual_equality"),
        "Analytic functions locally equal at one point agree on a preconnected domain.",
    ),
)


def build_theorem_card_index(project_root: Path) -> tuple[TheoremCard, ...]:
    root = Path(project_root)
    environment_hash = pinned_environment_hash(root)
    cards = tuple(TheoremCard(
        card_id=card_id,
        theorem_name=name,
        exact_type=exact_type,
        import_name=import_name,
        source_path=source_path,
        source_hash=_source_hash(root, source_path),
        environment_hash=environment_hash,
        applicability_tags=tuple(tags),
        required_hypotheses=tuple(hypotheses),
        description=description,
    ) for (
        card_id, name, exact_type, import_name, source_path, tags,
        hypotheses, description,
    ) in _CARD_SPECS)
    return tuple(sorted(cards, key=lambda item: item.card_id))


def search_theorem_cards(
    cards: Iterable[TheoremCard],
    tags: Iterable[str],
    *,
    limit: int = 8,
) -> tuple[TheoremCard, ...]:
    wanted = frozenset(str(tag) for tag in tags)
    ranked = sorted(
        (
            (len(wanted.intersection(card.applicability_tags)), card.card_id, card)
            for card in cards
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return tuple(card for score, _card_id, card in ranked if score > 0)[:limit]


def validate_theorem_cards(
    cards: Iterable[TheoremCard],
    *,
    project_root: Path,
    timeout_s: float = 120.0,
) -> None:
    root = Path(project_root)
    expected_environment = pinned_environment_hash(root)
    cards = tuple(cards)
    for card in cards:
        if card.environment_hash != expected_environment:
            raise ValueError(f"STALE_THEOREM_CARD_ENVIRONMENT:{card.card_id}")
        if card.source_hash != _source_hash(root, card.source_path):
            raise ValueError(f"STALE_THEOREM_CARD_SOURCE:{card.card_id}")
    content = "\n".join(
        [*(f"import {name}" for name in sorted({c.import_name for c in cards})), ""],
    ) + "\n".join(f"#check {card.theorem_name}" for card in cards) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", dir=root, encoding="utf-8", delete=False,
    ) as handle:
        handle.write(content)
        path = Path(handle.name)
    try:
        result = subprocess.run(
            ["lake", "env", "lean", str(path)],
            cwd=root, text=True, capture_output=True, timeout=timeout_s, check=False,
        )
    finally:
        path.unlink(missing_ok=True)
    if result.returncode:
        raise ValueError(
            "THEOREM_CARD_ELABORATION_FAILED:"
            + (result.stderr or result.stdout).strip()
        )
