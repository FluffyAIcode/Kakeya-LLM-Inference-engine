"""Host-owned, sourced Strategy seeds for the canonical RH root.

The records in this module are plans and proof obligations, not proof claims.
In particular, finite coefficient or polynomial checks never prove RH.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from autoresearch.prefill.strategy_tournament import (
    LemmaNode,
    PlanClass,
    PlanExecutionStatus,
    StrategyPlan,
)
from autoresearch.prefill.theorem_cards import pinned_environment_hash


RH_ROOT_ID = "RH-C0-7024d428ede1"
RH_ROOT_HASH = (
    "7024d428ede1c873b201a0801e42593ab9afd4e632c16c492678886797602547"
)
FINITE_WARNING = (
    "Finite computations, finite coefficient positivity, and finitely many "
    "hyperbolic Jensen polynomials do not prove the Riemann Hypothesis."
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


@dataclass(frozen=True)
class RHStrategySpec:
    route_id: str
    title: str
    relation_to_rh: str
    relation_source: str
    relation_obligation: str
    definitions: tuple[str, ...]
    first_subgoal: str
    first_subgoal_theorem: str
    theorem_cards: tuple[str, ...]
    dependencies: tuple[str, ...]
    mathlib_support: tuple[str, ...]
    missing_interfaces: tuple[str, ...]
    success_criterion: str
    falsification_criterion: str
    abandonment_criterion: str
    execution_status: str
    finite_warning: str = FINITE_WARNING

    @property
    def content_hash(self) -> str:
        return _digest(asdict(self))


def rh_strategy_specs() -> tuple[RHStrategySpec, ...]:
    return (
        RHStrategySpec(
            route_id="JENSEN_LAGUERRE_POLYA",
            title="Jensen polynomials / Laguerre-Pólya approximation",
            relation_to_rh="EQUIVALENT_WITH_UNPROVED_FORMAL_BRIDGE",
            relation_source="doi:10.1073/pnas.1902572116",
            relation_obligation=(
                "Define the classical completed xi function and its centered "
                "Taylor coefficients; prove that their all-degree/all-shift "
                "Jensen hyperbolicity criterion is equivalent to pinned "
                "Mathlib RiemannHypothesis. Keep coefficient/Jensen "
                "approximation distinct from zero-product approximation."
            ),
            definitions=(
                "Kakeya.RHJensen.riemannXi",
                "Kakeya.RHJensen.xiJensenEntire",
                "Kakeya.RHJensen.xiGamma",
                "Kakeya.RHJensen.jensenPolynomial",
                "Kakeya.RHJensen.Hyperbolic",
                "Kakeya.RHJensen.AllJensenHyperbolic",
            ),
            first_subgoal=(
                "For arbitrary real coefficients, prove the exact "
                "nondegenerate degree-two Jensen real-root/Turán "
                "discriminant equivalence; retain it as finite support only."
            ),
            first_subgoal_theorem=(
                "Kakeya.RHJensen.jensenQuadratic_has_real_roots_iff"
            ),
            theorem_cards=(
                "gorz-2019-jensen",
                "osullivan-2021-xi-lp",
                "mathlib-4.32.0-rc1-riemann",
            ),
            dependencies=(
                "KakeyaLeanGate/RHJensen.lean",
                "docs/research/rh-jensen-source-cards.json",
            ),
            mathlib_support=(
                "completedRiemannZeta", "completedRiemannZeta₀",
                "differentiable_completedZeta₀", "Real.sq_sqrt",
                "Complex.taylorSeries_eq_of_entire'",
                "Polynomial.IsRoot", "Polynomial.Splits",
                "TendstoLocallyUniformlyOn",
            ),
            missing_interfaces=(
                "XI_COEFFICIENT_REALITY_AND_IDENTIFICATION",
                "JENSEN_HYPERBOLIC_ALL_DEGREES_SHIFTS",
                "LAGUERRE_POLYA_CLASS_AND_LIMIT_BRIDGE",
                "JENSEN_CRITERION_IFF_MATHLIB_RH",
            ),
            success_criterion=(
                "Lean accepts the sourced xi/Jensen definitions, finite "
                "degree formulas and Turán criterion; all xi/Jensen/RH "
                "bridges remain explicit proof obligations."
            ),
            falsification_criterion=(
                "Lean rejects the coefficient convention or root formula, or "
                "a sourced xi normalization cannot be related to Mathlib."
            ),
            abandonment_criterion=(
                "Abandon as an RH route if the all-degree/all-shift equivalence "
                "cannot be sourced and formalized non-circularly."
            ),
            execution_status=PlanExecutionStatus.EXECUTABLE.value,
        ),
        RHStrategySpec(
            route_id="LI_COEFFICIENT_POSITIVITY",
            title="Li coefficients positivity criterion",
            relation_to_rh="EQUIVALENT_SOURCE_ONLY",
            relation_source="doi:10.1006/jnth.1997.2137",
            relation_obligation=(
                "Define Li's lambda_n from a sourced xi normalization, prove "
                "well-defined derivatives/zero sums, and formalize positivity "
                "for every n iff pinned Mathlib RiemannHypothesis."
            ),
            definitions=(
                "LI_COEFFICIENT", "RIEMANN_XI_NORMALIZATION",
                "ITERATED_COMPLEX_DERIVATIVE_AT_ONE",
            ),
            first_subgoal=(
                "Elaborate a definition of lambda_n with the exact normalization "
                "from Li (1997), then prove the derivative expression is typed."
            ),
            first_subgoal_theorem="",
            theorem_cards=("source-li-criterion-1997",),
            dependencies=("doi:10.1006/jnth.1997.2137",),
            mathlib_support=(
                "completedRiemannZeta", "completedRiemannZeta₀",
                "differentiable_completedZeta₀",
            ),
            missing_interfaces=(
                "RIEMANN_XI_NORMALIZATION", "LI_COEFFICIENT",
                "LI_POSITIVITY_ALL_N_IFF_MATHLIB_RH",
            ),
            success_criterion=(
                "A sourced lambda_n definition elaborates and the all-n "
                "equivalence obligation is represented without assuming RH."
            ),
            falsification_criterion=(
                "Normalization or convergence hypotheses cannot be matched to "
                "Mathlib's completed zeta declarations."
            ),
            abandonment_criterion=(
                "Remain planning-only until the xi and all-n equivalence "
                "interfaces are sourced and Lean-elaborated."
            ),
            execution_status=PlanExecutionStatus.PLANNING_ONLY.value,
        ),
        RHStrategySpec(
            route_id="WEIL_POSITIVE_QUADRATIC_FORM",
            title="Positive kernel / energy functional criterion",
            relation_to_rh="EQUIVALENT_SOURCE_ONLY",
            relation_source=(
                "A. Weil, Sur les formules explicites de la théorie des "
                "nombres premiers (1952)"
            ),
            relation_obligation=(
                "Formalize Weil's exact explicit-formula quadratic functional "
                "Q_W(g)=W(g*g*) on the sourced admissible test-function domain, "
                "and prove positive semidefiniteness on that domain iff pinned "
                "Mathlib RiemannHypothesis."
            ),
            definitions=(
                "WEIL_TEST_FUNCTION_DOMAIN", "MULTIPLICATIVE_CONVOLUTION",
                "TRANSPOSE_CONJUGATE", "WEIL_EXPLICIT_FORMULA_FUNCTIONAL",
            ),
            first_subgoal=(
                "Define the exact test-function domain and involution/convolution "
                "interfaces, then type the Hermitian quadratic form Q_W."
            ),
            first_subgoal_theorem="",
            theorem_cards=("source-weil-positivity-1952",),
            dependencies=(
                "Weil-1952-explicit-formula",
                "Guinand-Weil-explicit-formula",
            ),
            mathlib_support=(
                "ContinuousMap", "MeasureTheory.Integral",
                "Convolution", "starRingEnd",
            ),
            missing_interfaces=(
                "WEIL_TEST_FUNCTION_DOMAIN",
                "WEIL_EXPLICIT_FORMULA_FUNCTIONAL",
                "WEIL_POSITIVITY_IFF_MATHLIB_RH",
            ),
            success_criterion=(
                "The sourced domain and exact Q_W elaborate, with the RH "
                "implication/equivalence retained as an explicit obligation."
            ),
            falsification_criterion=(
                "Any proposed kernel lacks the sourced explicit-formula identity "
                "or changes the admissible positivity domain."
            ),
            abandonment_criterion=(
                "Reject generic positivity searches; remain planning-only until "
                "one exact sourced kernel/form/domain is formalized."
            ),
            execution_status=PlanExecutionStatus.PLANNING_ONLY.value,
        ),
    )


def build_rh_strategy_plans(project_root: Path) -> tuple[StrategyPlan, ...]:
    environment = pinned_environment_hash(project_root)
    plans = []
    for index, spec in enumerate(rh_strategy_specs(), 1):
        target_ref = (
            "lean:" + spec.first_subgoal_theorem
            if spec.execution_status == PlanExecutionStatus.EXECUTABLE.value
            else "planning:" + spec.route_id
        )
        unresolved = (
            () if spec.execution_status == PlanExecutionStatus.EXECUTABLE.value
            else spec.missing_interfaces
        )
        dependency_ids = spec.dependencies
        target_complexity = 20 + index
        definition_auditor_hash = hashlib.sha256(
            b"host-rh-strategy-seed-v1"
        ).hexdigest()
        body = {
            "schema_version": 1,
            "seed_hash": spec.content_hash,
            "plan_class": PlanClass.REDUCTION_TO_KNOWN_RESULT.value,
            "target_ref": target_ref,
            "required_definition_ids": spec.definitions,
            "theorem_card_ids": spec.theorem_cards,
            "dependency_ids": dependency_ids,
            "falsification_test_id": "FALSIFY_" + spec.route_id,
            "success_criterion_id": "SUCCESS_" + spec.route_id,
            "abandonment_criterion_id": "ABANDON_" + spec.route_id,
            "parent_obligation_ref": RH_ROOT_ID,
            "parent_complexity": 100,
            "target_complexity": target_complexity,
            "source_move_id": "MOVE_REDUCE_" + spec.route_id,
            "environment_hash": environment,
            "evidence_refs": (
                spec.relation_source, "mathlib:" + environment,
                "seed:" + spec.content_hash,
            ),
            "unresolved_definition_ids": unresolved,
            "definition_gap_ids": tuple("gap:" + item for item in unresolved),
            "definition_auditor_hash": definition_auditor_hash,
            "execution_status": spec.execution_status,
            "restriction_ids": ("FINITE_COMPUTATION_DOES_NOT_PROVE_RH",),
        }
        content_hash = _digest(body)
        plans.append(StrategyPlan(
            plan_id="RHSP-" + content_hash[:20],
            plan_class=PlanClass.REDUCTION_TO_KNOWN_RESULT.value,
            target_ref=target_ref,
            required_definition_ids=spec.definitions,
            theorem_card_ids=spec.theorem_cards,
            dependency_ids=dependency_ids,
            lemma_graph=(LemmaNode(
                lemma_id="RHL-" + content_hash[:16],
                dependency_ids=dependency_ids,
                target_ref=target_ref,
                complexity=target_complexity,
            ),),
            falsification_test_id="FALSIFY_" + spec.route_id,
            success_criterion_id="SUCCESS_" + spec.route_id,
            abandonment_criterion_id="ABANDON_" + spec.route_id,
            expected_information_gain=6 - index,
            assumption_ids=(),
            restriction_ids=("FINITE_COMPUTATION_DOES_NOT_PROVE_RH",),
            parent_obligation_ref=RH_ROOT_ID,
            parent_complexity=100,
            target_complexity=target_complexity,
            risk=index,
            source_move_id="MOVE_REDUCE_" + spec.route_id,
            environment_hash=environment,
            evidence_refs=(
                spec.relation_source, "mathlib:" + environment,
                "seed:" + spec.content_hash,
            ),
            unresolved_definition_ids=unresolved,
            definition_gap_ids=tuple("gap:" + item for item in unresolved),
            definition_auditor_hash=definition_auditor_hash,
            proposition_transformation_ref="",
            case_partition_ids=(),
            execution_status=spec.execution_status,
            content_hash=content_hash,
        ))
    return tuple(plans)
