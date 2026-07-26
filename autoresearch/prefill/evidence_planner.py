"""Host-owned evidence graph and bounded, non-omniscient proof planning."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Iterable, Mapping

from autoresearch.prefill.creative_decomposition import CandidateSet, MoveCandidate
from autoresearch.prefill.theorem_cards import TheoremCard


PLANNER_VERSION = 1
OPERATOR_EVENT = "host_feasibility_evidence_planner_v1"


class NodeKind(str, Enum):
    GAP = "GAP"
    EVIDENCE = "EVIDENCE"
    THEOREM_CARD = "THEOREM_CARD"
    MOVE = "MOVE"


class EdgeKind(str, Enum):
    RESOLVES_GAP = "resolves_gap"
    REQUIRES = "requires"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CASE_OF = "case_of"
    REDUCES_TO = "reduces_to"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    ADVISORY = "ADVISORY"
    UNRESOLVED = "UNRESOLVED"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    kind: str
    verification_status: str
    provenance_refs: tuple[str, ...]
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    kind: str
    source_id: str
    target_id: str
    provenance_refs: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceGapGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    content_hash: str

    @property
    def by_id(self) -> dict[str, GraphNode]:
        return {node.node_id: node for node in self.nodes}


@dataclass(frozen=True)
class ProofPlanNode:
    plan_node_id: str
    graph_node_id: str
    dependency_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanScore:
    unresolved_gap_coverage: int
    verified_theorem_support: int
    dependency_depth: int
    strict_complexity_reduction: int
    novelty_information_gain: int
    risk: int

    @property
    def ordering_key(self) -> tuple[int, ...]:
        return (
            -self.unresolved_gap_coverage,
            -self.verified_theorem_support,
            self.dependency_depth,
            -self.strict_complexity_reduction,
            -self.novelty_information_gain,
            self.risk,
        )


@dataclass(frozen=True)
class ProofPlan:
    plan_id: str
    nodes: tuple[ProofPlanNode, ...]
    resolved_gap_ids: tuple[str, ...]
    theorem_card_ids: tuple[str, ...]
    case_exhaustiveness_claims: tuple[str, ...]
    score: PlanScore
    score_explanation: Mapping[str, int]
    graph_hash: str
    content_hash: str


@dataclass(frozen=True)
class HostEvidenceContext:
    satisfied_precondition_ids: tuple[str, ...]
    satisfied_theorem_hypothesis_ids: tuple[str, ...]
    available_dependency_artifact_ids: tuple[str, ...]
    unresolved_gap_ids: tuple[str, ...]


_DEFINITION_PRECONDITIONS: Mapping[str, tuple[str, ...]] = {
    "density_defined": ("DEF_SEQUENCE_DENSITY", "DEF_CRITICAL_DENSITY"),
    "growth_notation_defined": ("DEF_GROWTH_ORDER",),
    "positive_radius": ("DEF_POLE_NEIGHBORHOOD",),
    "registered_definition_gap": (),
}


def host_evidence_context(
    artifacts: Mapping[str, object],
    *,
    artifact_hashes: Mapping[str, str],
) -> HostEvidenceContext:
    """Derive executable facts only from persisted, verified artifacts."""
    definitions: set[str] = set()
    missing: set[str] = set()
    hypotheses: set[str] = set()
    for artifact in artifacts.values():
        payload = asdict(artifact) if hasattr(artifact, "__dataclass_fields__") else artifact
        if not isinstance(payload, Mapping):
            continue
        for item in payload.get("definitions", ()):
            if isinstance(item, Mapping):
                definitions.add(str(item.get("definition_id", "")))
        for item in payload.get("missing_definitions", ()):
            if isinstance(item, Mapping):
                missing.add(str(item.get("definition_id", "")))
        hypotheses.update(str(item) for item in payload.get(
            "verified_hypothesis_ids", (),
        ))
    satisfied = {
        precondition
        for precondition, requirements in _DEFINITION_PRECONDITIONS.items()
        if requirements and set(requirements) <= definitions
    }
    if missing:
        satisfied.add("registered_definition_gap")
    return HostEvidenceContext(
        tuple(sorted(satisfied)),
        tuple(sorted(hypotheses)),
        tuple(sorted(artifact_hashes.values())),
        tuple(sorted(f"gap:definition:{item}" for item in missing if item)),
    )


def _edge(
    kind: EdgeKind,
    source_id: str,
    target_id: str,
    provenance_refs: Iterable[str],
) -> GraphEdge:
    provenance = tuple(sorted(set(provenance_refs)))
    edge_id = "edge:" + _digest((kind.value, source_id, target_id, provenance))
    return GraphEdge(edge_id, kind.value, source_id, target_id, provenance)


def build_evidence_gap_graph(
    *,
    artifacts: Mapping[str, object],
    artifact_hashes: Mapping[str, str],
    advisory_artifacts: Mapping[str, Mapping[str, object]],
    theorem_cards: Iterable[TheoremCard],
    candidate_set: CandidateSet,
    failure_reason_codes: Iterable[str] = (),
) -> EvidenceGapGraph:
    nodes: dict[str, GraphNode] = {}
    edges: dict[str, GraphEdge] = {}
    missing_ids: list[str] = []
    for role, artifact in sorted(artifacts.items()):
        payload = asdict(artifact) if hasattr(artifact, "__dataclass_fields__") else artifact
        if not isinstance(payload, Mapping):
            continue
        provenance = artifact_hashes.get(role, "")
        for item in payload.get("missing_definitions", ()):
            if not isinstance(item, Mapping):
                continue
            definition_id = str(item.get("definition_id", ""))
            if not definition_id:
                continue
            gap_id = f"gap:definition:{definition_id}"
            missing_ids.append(gap_id)
            nodes[gap_id] = GraphNode(
                gap_id, NodeKind.GAP.value, VerificationStatus.UNRESOLVED.value,
                (provenance,), {"definition_id": definition_id, "source_role": role},
            )
        for item in payload.get("definitions", ()):
            if not isinstance(item, Mapping):
                continue
            definition_id = str(item.get("definition_id", ""))
            evidence_id = f"evidence:{provenance}:definition:{definition_id}"
            nodes[evidence_id] = GraphNode(
                evidence_id, NodeKind.EVIDENCE.value,
                VerificationStatus.VERIFIED.value, (provenance,),
                {"definition_id": definition_id, "source_role": role},
            )
            gap_id = f"gap:definition:{definition_id}"
            nodes.setdefault(gap_id, GraphNode(
                gap_id, NodeKind.GAP.value, VerificationStatus.UNRESOLVED.value,
                (provenance,), {"definition_id": definition_id},
            ))
            support = _edge(EdgeKind.SUPPORTS, evidence_id, gap_id, (provenance,))
            edges[support.edge_id] = support
    for digest, payload in sorted(advisory_artifacts.items()):
        evidence_id = f"evidence:{digest}:advisory"
        nodes[evidence_id] = GraphNode(
            evidence_id, NodeKind.EVIDENCE.value,
            VerificationStatus.ADVISORY.value, (digest,),
            {"premise": False, "source_role": payload.get("role", "")},
        )
    for code in sorted(set(failure_reason_codes)):
        gap_id = f"gap:failure:{code}"
        nodes[gap_id] = GraphNode(
            gap_id, NodeKind.GAP.value, VerificationStatus.UNRESOLVED.value,
            (code,), {"reason_code": code},
        )
    for card in sorted(theorem_cards, key=lambda item: item.card_id):
        node_id = f"theorem:{card.card_id}"
        nodes[node_id] = GraphNode(
            node_id, NodeKind.THEOREM_CARD.value,
            VerificationStatus.VERIFIED.value, (card.content_hash,),
            {"required_hypotheses": card.required_hypotheses},
        )
    for candidate in candidate_set.candidates:
        base_id = f"move:{candidate.candidate_id}"
        nodes[base_id] = _move_node(base_id, candidate, ())
        for precondition in candidate.move.precondition_ids:
            gap_id = f"gap:precondition:{precondition}"
            nodes.setdefault(gap_id, GraphNode(
                gap_id, NodeKind.GAP.value, VerificationStatus.UNRESOLVED.value,
                (), {"precondition_id": precondition},
            ))
            required = _edge(EdgeKind.REQUIRES, base_id, gap_id, ())
            edges[required.edge_id] = required
        if candidate.move.resolves_gap_ids:
            targets = tuple(
                f"gap:precondition:{item}"
                for item in candidate.move.resolves_gap_ids
            )
            for target in targets:
                case_id = f"{base_id}:case:{target}"
                nodes[case_id] = _move_node(case_id, candidate, (target,))
                case_edge = _edge(EdgeKind.CASE_OF, case_id, base_id, ())
                resolve_edge = _edge(
                    EdgeKind.RESOLVES_GAP, case_id, target,
                    nodes[target].provenance_refs if target in nodes else (),
                )
                edges[case_edge.edge_id] = case_edge
                edges[resolve_edge.edge_id] = resolve_edge
    canonical = {
        "planner_version": PLANNER_VERSION,
        "nodes": [asdict(nodes[key]) for key in sorted(nodes)],
        "edges": [asdict(edges[key]) for key in sorted(edges)],
    }
    return EvidenceGapGraph(
        tuple(nodes[key] for key in sorted(nodes)),
        tuple(edges[key] for key in sorted(edges)),
        _digest(canonical),
    )


def _move_node(
    node_id: str,
    candidate: MoveCandidate,
    target_gap_ids: tuple[str, ...],
) -> GraphNode:
    return GraphNode(
        node_id, NodeKind.MOVE.value, VerificationStatus.VERIFIED.value,
        (candidate.candidate_hash,),
        {
            "candidate_id": candidate.candidate_id,
            "candidate_hash": candidate.candidate_hash,
            "target_gap_ids": target_gap_ids,
            "metric_parent": candidate.move.metric.parent,
            "metric_child": candidate.move.metric.child,
            "theorem_card_ids": candidate.theorem_card_ids,
            "dependency_depth": candidate.dependency_depth,
            "requires_parent_case_split": candidate.move.requires_parent_case_split,
        },
    )


def generate_proof_plans(
    graph: EvidenceGapGraph,
    *,
    unresolved_gap_ids: Iterable[str],
    beam_width: int = 16,
) -> tuple[ProofPlan, ...]:
    """Deterministic bounded backward chaining over registered graph edges."""
    unresolved = set(unresolved_gap_ids)
    by_id = graph.by_id
    resolves: dict[str, set[str]] = {}
    for edge in graph.edges:
        if edge.kind == EdgeKind.RESOLVES_GAP.value:
            resolves.setdefault(edge.source_id, set()).add(edge.target_id)
    plans: list[ProofPlan] = []
    for move_id, covered in sorted(resolves.items()):
        if move_id not in by_id:
            continue
        node = by_id[move_id]
        actual_coverage = tuple(sorted(covered.intersection(unresolved)))
        if not actual_coverage:
            continue
        attrs = node.attributes
        reduction = int(attrs.get("metric_parent", 0)) - int(
            attrs.get("metric_child", 0),
        )
        theorem_ids = tuple(str(item) for item in attrs.get("theorem_card_ids", ()))
        score = PlanScore(
            len(actual_coverage),
            len(theorem_ids),
            int(attrs.get("dependency_depth", 0)),
            reduction,
            1,
            int(bool(attrs.get("requires_parent_case_split", False))),
        )
        plan_node = ProofPlanNode("step:1", move_id, ())
        explanation = {
            "unresolved_gap_coverage": score.unresolved_gap_coverage,
            "verified_theorem_support": score.verified_theorem_support,
            "dependency_depth": score.dependency_depth,
            "strict_complexity_reduction": score.strict_complexity_reduction,
            "novelty_information_gain": score.novelty_information_gain,
            "risk": score.risk,
        }
        canonical = {
            "graph_hash": graph.content_hash,
            "nodes": [asdict(plan_node)],
            "resolved_gap_ids": actual_coverage,
            "theorem_card_ids": theorem_ids,
            "case_exhaustiveness_claims": (),
            "score": asdict(score),
        }
        content_hash = _digest(canonical)
        plans.append(ProofPlan(
            "plan:" + content_hash[:16], (plan_node,), actual_coverage,
            theorem_ids, (), score, explanation, graph.content_hash, content_hash,
        ))
    plans.sort(key=lambda item: (*item.score.ordering_key, item.plan_id))
    return tuple(plans[:beam_width])


def validate_proof_plan(graph: EvidenceGapGraph, plan: ProofPlan) -> None:
    if plan.graph_hash != graph.content_hash:
        raise ValueError("STALE_PROOF_PLAN_GRAPH_HASH")
    graph_nodes = graph.by_id
    plan_nodes = {node.plan_node_id: node for node in plan.nodes}
    if len(plan_nodes) != len(plan.nodes):
        raise ValueError("DUPLICATE_PROOF_PLAN_NODE")
    for node in plan.nodes:
        if node.graph_node_id not in graph_nodes:
            raise ValueError("UNREGISTERED_PROOF_PLAN_GRAPH_NODE")
        if any(item not in plan_nodes for item in node.dependency_node_ids):
            raise ValueError("PROOF_PLAN_DEPENDENCY_CLOSURE")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("CYCLIC_PROOF_PLAN")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in plan_nodes[node_id].dependency_node_ids:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in plan_nodes:
        visit(node_id)
    if any(
        graph_nodes[node.graph_node_id].kind != NodeKind.MOVE.value
        for node in plan.nodes
    ):
        raise ValueError("PROOF_PLAN_NODE_NOT_EXECUTABLE_MOVE")
    if any(
        graph_nodes[node.graph_node_id].verification_status
        != VerificationStatus.VERIFIED.value
        for node in plan.nodes
    ):
        raise ValueError("UNVERIFIED_PROOF_PLAN_NODE")
    for claim in plan.case_exhaustiveness_claims:
        case_node = graph_nodes.get(claim)
        if case_node is None or not bool(case_node.attributes.get("exhaustive", False)):
            raise ValueError("CASE_EXHAUSTIVENESS_NOT_VERIFIED")
    for node in plan.nodes:
        attrs = graph_nodes[node.graph_node_id].attributes
        if int(attrs.get("metric_child", 0)) >= int(attrs.get("metric_parent", 0)):
            raise ValueError("PROOF_PLAN_NOT_STRICTLY_SIMPLER")
        for card_id in attrs.get("theorem_card_ids", ()):
            card = graph_nodes.get(f"theorem:{card_id}")
            if card is None or card.verification_status != VerificationStatus.VERIFIED.value:
                raise ValueError("THEOREM_CARD_NOT_VERIFIED")


def first_executable_node(plan: ProofPlan) -> ProofPlanNode:
    completed: set[str] = set()
    for node in plan.nodes:
        if set(node.dependency_node_ids) <= completed:
            return node
        completed.add(node.plan_node_id)
    raise ValueError("NO_EXECUTABLE_PROOF_PLAN_NODE")


def replan_after_failure(
    graph: EvidenceGapGraph,
    *,
    failed_plan_node_id: str,
    reason_code: str,
    unresolved_gap_ids: Iterable[str],
) -> tuple[EvidenceGapGraph, tuple[ProofPlan, ...]]:
    evidence_id = f"evidence:failure:{_digest((failed_plan_node_id, reason_code))}"
    gap_id = f"gap:failure:{reason_code}"
    nodes = (*graph.nodes, GraphNode(
        evidence_id, NodeKind.EVIDENCE.value, VerificationStatus.VERIFIED.value,
        (failed_plan_node_id,), {"reason_code": reason_code},
    ), GraphNode(
        gap_id, NodeKind.GAP.value, VerificationStatus.UNRESOLVED.value,
        (evidence_id,), {"reason_code": reason_code},
    ))
    contradiction = _edge(
        EdgeKind.CONTRADICTS, evidence_id, failed_plan_node_id,
        (failed_plan_node_id,),
    )
    canonical = {
        "planner_version": PLANNER_VERSION,
        "nodes": [asdict(item) for item in sorted(nodes, key=lambda item: item.node_id)],
        "edges": [asdict(item) for item in sorted(
            (*graph.edges, contradiction), key=lambda item: item.edge_id,
        )],
    }
    updated = EvidenceGapGraph(
        tuple(sorted(nodes, key=lambda item: item.node_id)),
        tuple(sorted((*graph.edges, contradiction), key=lambda item: item.edge_id)),
        _digest(canonical),
    )
    blocked = {
        edge.target_id for edge in updated.edges
        if edge.kind == EdgeKind.CONTRADICTS.value
    }
    filtered = replace(
        updated,
        nodes=tuple(node for node in updated.nodes if node.node_id not in blocked),
    )
    plans = generate_proof_plans(
        filtered, unresolved_gap_ids=(*unresolved_gap_ids, gap_id),
    )
    return updated, plans
