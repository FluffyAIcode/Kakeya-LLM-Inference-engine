"""Typed mathematical IR and deterministic host-owned Lean compiler."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable, Mapping


MATH_IR_VERSION = 3
REGISTRY_VERSION = 3
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_RAW_SOURCE = re.compile(
    r"```|:=\s*by|(?:^|\s)(?:theorem|lemma|import|axiom)\s+|"
    r"\\(?:frac|sum|forall|exists|mathbb|text|begin|end)\b|\$",
    re.IGNORECASE,
)


class GateStatus(str, Enum):
    PASSED = "PASSED"
    ADAPTER_BLOCKED = "ADAPTER_BLOCKED"
    INTEGRATION_BLOCKED = "INTEGRATION_BLOCKED"
    SEMANTIC_BACKJUMP = "SEMANTIC_BACKJUMP"
    MATHEMATICAL_REJECTION = "MATHEMATICAL_REJECTION"


class TypedIRError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        line: int = 0,
        node_id: str = "",
        owner: str = "decomposer",
        status: GateStatus = GateStatus.INTEGRATION_BLOCKED,
    ) -> None:
        self.code = str(code)
        self.line = int(line)
        self.node_id = str(node_id)
        self.owner = str(owner)
        self.status = status
        super().__init__(
            f"{status.value}:{code}:owner={owner}:line={line}:"
            f"node={node_id}:{message}"
        )


@dataclass(frozen=True)
class TypeSpec:
    type_id: str
    lean_name: str


@dataclass(frozen=True)
class SymbolSpec:
    symbol_id: str
    lean_name: str
    argument_types: tuple[str, ...]
    result_type: str


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    arity: int
    argument_types: tuple[str, ...]
    result_type: str
    lean_token: str


TYPE_REGISTRY: dict[str, TypeSpec] = {
    spec.type_id: spec for spec in (
        TypeSpec("Prop", "Prop"),
        TypeSpec("Nat", "ℕ"),
        TypeSpec("Int", "ℤ"),
        TypeSpec("Real", "ℝ"),
        TypeSpec("Complex", "ℂ"),
        TypeSpec("NatToComplex", "ℕ → ℂ"),
        TypeSpec("ComplexToComplex", "ℂ → ℂ"),
        TypeSpec("NatToComplexFunction", "ℕ → ℂ → ℂ"),
        TypeSpec("SetComplex", "Set ℂ"),
    )
}
SYMBOL_REGISTRY: dict[str, SymbolSpec] = {
    spec.symbol_id: spec for spec in (
        SymbolSpec("density", "density", ("NatToComplex",), "Real"),
        SymbolSpec(
            "localConvergence",
            "localConvergence",
            ("NatToComplex", "Complex", "Int", "Real"),
            "Prop",
        ),
        SymbolSpec(
            "growthConstraint",
            "growthConstraint",
            ("ComplexToComplex", "Nat"),
            "Prop",
        ),
        SymbolSpec("True", "True", (), "Prop"),
        SymbolSpec("False", "False", (), "Prop"),
        SymbolSpec(
            "polesOutsideDisk", "polesOutsideDisk",
            ("SetComplex", "Complex", "Real"), "Prop",
        ),
        SymbolSpec(
            "localUniformConvergenceOnDisk", "localUniformConvergenceOnDisk",
            ("NatToComplexFunction", "ComplexToComplex", "Complex", "Real"),
            "Prop",
        ),
        SymbolSpec(
            "termsHolomorphicOnDisk", "termsHolomorphicOnDisk",
            ("NatToComplexFunction", "Complex", "Real"), "Prop",
        ),
        SymbolSpec(
            "holomorphicSumOnDisk", "holomorphicSumOnDisk",
            ("ComplexToComplex", "Complex", "Real"), "Prop",
        ),
        SymbolSpec(
            "agreesWithSimplePoleOnPuncturedDisk",
            "agreesWithSimplePoleOnPuncturedDisk",
            ("ComplexToComplex", "Complex", "Complex", "Real"), "Prop",
        ),
        SymbolSpec("nonzeroComplex", "nonzeroComplex", ("Complex",), "Prop"),
        SymbolSpec("positiveRadius", "positiveRadius", ("Real",), "Prop"),
    )
}
OPERATOR_REGISTRY: dict[str, OperatorSpec] = {
    spec.operator_id: spec for spec in (
        OperatorSpec("not", 1, ("Prop",), "Prop", "¬"),
        OperatorSpec("and", 2, ("Prop", "Prop"), "Prop", "∧"),
        OperatorSpec("or", 2, ("Prop", "Prop"), "Prop", "∨"),
        OperatorSpec("implies", 2, ("Prop", "Prop"), "Prop", "→"),
        OperatorSpec("eq_real", 2, ("Real", "Real"), "Prop", "="),
        OperatorSpec("gt_real", 2, ("Real", "Real"), "Prop", ">"),
        OperatorSpec("lt_real", 2, ("Real", "Real"), "Prop", "<"),
        OperatorSpec("ge_real", 2, ("Real", "Real"), "Prop", "≥"),
        OperatorSpec("le_real", 2, ("Real", "Real"), "Prop", "≤"),
        OperatorSpec("eq_nat", 2, ("Nat", "Nat"), "Prop", "="),
    )
}


@dataclass(frozen=True)
class Binder:
    binder_id: str
    type_id: str


@dataclass(frozen=True)
class ExprNode:
    node_id: str
    kind: str
    ref: str
    arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class MathIR:
    theorem_id: str
    binders: tuple[Binder, ...]
    nodes: tuple[ExprNode, ...]
    premises: tuple[str, ...]
    conclusion: str
    dependency_ids: tuple[str, ...] = ()
    schema_version: int = MATH_IR_VERSION
    registry_version: int = REGISTRY_VERSION


@dataclass(frozen=True)
class ValidatedMathIR:
    math_ir: MathIR
    node_types: Mapping[str, str]
    content_hash: str


@dataclass(frozen=True)
class LeanCompilation:
    theorem_id: str
    declaration_source: str
    declaration_hash: str
    proposition_hash: str
    math_ir_hash: str
    imports_hash: str


@dataclass(frozen=True)
class ProofStep:
    tactic_id: str
    argument_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProofPlan:
    theorem_id: str
    proposition_hash: str
    steps: tuple[ProofStep, ...]
    schema_version: int = 1


@dataclass(frozen=True)
class TypedIRCandidate:
    choice_id: str
    transformation_id: str
    summary_id: str
    typed_payload: tuple[str, ...]
    typed_ir_hash: str
    candidate_hash: str


@dataclass(frozen=True)
class TypedIRCandidateRegistry:
    target_ref: str
    viewpoint: str
    dependency_ids: tuple[str, ...]
    candidates: tuple[TypedIRCandidate, ...]
    registry_hash: str

    @property
    def choices(self) -> tuple[str, ...]:
        return tuple(candidate.choice_id for candidate in self.candidates)

    def resolve(self, choice_id: str) -> TypedIRCandidate:
        for candidate in self.candidates:
            if candidate.choice_id == choice_id:
                return candidate
        raise TypedIRError(
            "INVALID_DECOMPOSITION_CHOICE",
            f"choice {choice_id!r} is not in candidate registry {self.registry_hash}",
            owner="decomposer",
            status=GateStatus.ADAPTER_BLOCKED,
        )


TACTIC_REGISTRY: dict[str, tuple[int, str]] = {
    "intro": (1, "intro {0}"),
    "exact": (1, "exact {0}"),
    "apply": (1, "apply {0}"),
    "assumption": (0, "assumption"),
    "constructor": (0, "constructor"),
    "left": (0, "left"),
    "right": (0, "right"),
    "trivial": (0, "trivial"),
    "contradiction": (0, "contradiction"),
}


_DECOMPOSITION_TEMPLATES: tuple[
    tuple[str, str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    ...,
] = (
    (
        "A",
        "DENSITY_LOWER_BOUND",
        "DENSITY_THRESHOLD_MOVE",
        ("NatToComplex", "Real"),
        ("density",),
        ("gt_real",),
        (
            "binder sequence NatToComplex",
            "binder threshold Real",
            "var sequenceNode sequence",
            "var thresholdNode threshold",
            "symbol densityNode density sequenceNode",
            "op resultNode gt_real densityNode thresholdNode",
            "conclusion resultNode",
        ),
    ),
    (
        "B",
        "LOCAL_CONVERGENCE_OBLIGATION",
        "LOCAL_CONVERGENCE_MOVE",
        ("NatToComplex", "Complex", "Int", "Real"),
        ("localConvergence",),
        (),
        (
            "binder sequence NatToComplex",
            "binder center Complex",
            "binder index Int",
            "binder radius Real",
            "var sequenceNode sequence",
            "var centerNode center",
            "var indexNode index",
            "var radiusNode radius",
            "symbol resultNode localConvergence sequenceNode centerNode indexNode radiusNode",
            "conclusion resultNode",
        ),
    ),
    (
        "C",
        "GROWTH_CONSTRAINT_OBLIGATION",
        "GROWTH_CONSTRAINT_MOVE",
        ("ComplexToComplex", "Nat"),
        ("growthConstraint",),
        (),
        (
            "binder function ComplexToComplex",
            "binder degree Nat",
            "var functionNode function",
            "var degreeNode degree",
            "symbol resultNode growthConstraint functionNode degreeNode",
            "conclusion resultNode",
        ),
    ),
    (
        "D",
        "LOCAL_TO_GROWTH_BRIDGE",
        "REGISTERED_BRIDGE_MOVE",
        (
            "NatToComplex", "Complex", "Int", "Real",
            "ComplexToComplex", "Nat",
        ),
        ("localConvergence", "growthConstraint"),
        ("implies",),
        (
            "binder sequence NatToComplex",
            "binder center Complex",
            "binder index Int",
            "binder radius Real",
            "binder function ComplexToComplex",
            "binder degree Nat",
            "var sequenceNode sequence",
            "var centerNode center",
            "var indexNode index",
            "var radiusNode radius",
            "var functionNode function",
            "var degreeNode degree",
            "symbol localNode localConvergence sequenceNode centerNode indexNode radiusNode",
            "symbol growthNode growthConstraint functionNode degreeNode",
            "op resultNode implies localNode growthNode",
            "conclusion resultNode",
        ),
    ),
    (
        "CASE_SPLIT",
        "CASE_SPLIT",
        "CASE_PARTITION_MOVE",
        (),
        ("True", "False"),
        ("or",),
        (
            "symbol leftCase True",
            "symbol rightCase False",
            "op partition or leftCase rightCase",
            "conclusion partition",
        ),
    ),
    (
        "RESTRICT_DOMAIN",
        "RESTRICT_DOMAIN",
        "DISK_DOMAIN_RESTRICTION_MOVE",
        ("Real",),
        ("positiveRadius",),
        (),
        (
            "binder radius Real",
            "var radiusNode radius",
            "symbol restricted positiveRadius radiusNode",
            "conclusion restricted",
        ),
    ),
    (
        "REMOVE_IRRELEVANT_ASSUMPTION",
        "REMOVE_IRRELEVANT_ASSUMPTION",
        "ASSUMPTION_PRUNING_MOVE",
        (),
        ("True",),
        (),
        (
            "symbol reduced True",
            "conclusion reduced",
        ),
    ),
    (
        "HOLOMORPHIC_EXTENSION",
        "HOLOMORPHIC_EXTENSION",
        "LOCAL_HOLOMORPHIC_SUM_MOVE",
        ("ComplexToComplex", "Complex", "Real"),
        ("holomorphicSumOnDisk",),
        (),
        (
            "binder sum ComplexToComplex",
            "binder center Complex",
            "binder radius Real",
            "var sumNode sum",
            "var centerNode center",
            "var radiusNode radius",
            "symbol result holomorphicSumOnDisk sumNode centerNode radiusNode",
            "conclusion result",
        ),
    ),
    (
        "SINGULARITY_CONTRADICTION",
        "SINGULARITY_CONTRADICTION",
        "LOCAL_HOLOMORPHICITY_SPECIAL_CASE_LEMMA",
        (
            "NatToComplexFunction", "ComplexToComplex", "SetComplex",
            "Complex", "Real",
        ),
        (
            "polesOutsideDisk", "localUniformConvergenceOnDisk",
            "termsHolomorphicOnDisk", "holomorphicSumOnDisk",
            "agreesWithSimplePoleOnPuncturedDisk", "nonzeroComplex",
            "positiveRadius", "False",
        ),
        (),
        (
            "binder terms NatToComplexFunction",
            "binder sum ComplexToComplex",
            "binder poles SetComplex",
            "binder center Complex",
            "binder radius Real",
            "binder residue Complex",
            "var termsNode terms",
            "var sumNode sum",
            "var polesNode poles",
            "var centerNode center",
            "var radiusNode radius",
            "var residueNode residue",
            "symbol polesOutside polesOutsideDisk polesNode centerNode radiusNode",
            "symbol localUniform localUniformConvergenceOnDisk termsNode sumNode centerNode radiusNode",
            "symbol termsHolomorphic termsHolomorphicOnDisk termsNode centerNode radiusNode",
            "symbol sumHolomorphic holomorphicSumOnDisk sumNode centerNode radiusNode",
            "symbol poleAgreement agreesWithSimplePoleOnPuncturedDisk sumNode centerNode residueNode radiusNode",
            "symbol residueNonzero nonzeroComplex residueNode",
            "symbol radiusPositive positiveRadius radiusNode",
            "symbol contradiction False",
            "premise polesOutside",
            "premise localUniform",
            "premise termsHolomorphic",
            "premise sumHolomorphic",
            "premise poleAgreement",
            "premise residueNonzero",
            "premise radiusPositive",
            "conclusion contradiction",
        ),
    ),
)


def build_decomposer_candidate_registry(
    *,
    target_ref: str,
    viewpoint: str,
    dependency_ids: Iterable[str] = (),
    allowed_transformations: Iterable[str] | None = None,
) -> TypedIRCandidateRegistry:
    """Build a finite host-owned menu; the model emits only one choice code."""
    dependencies = tuple(sorted({str(item) for item in dependency_ids if item}))
    allowed = (
        {str(item) for item in allowed_transformations}
        if allowed_transformations is not None
        else {template[1] for template in _DECOMPOSITION_TEMPLATES}
    )
    candidates: list[TypedIRCandidate] = []
    for (
        choice_id,
        transformation_id,
        summary_id,
        required_types,
        required_symbols,
        required_operators,
        body,
    ) in _DECOMPOSITION_TEMPLATES:
        if transformation_id not in allowed:
            continue
        if (
            any(item not in TYPE_REGISTRY for item in required_types)
            or any(item not in SYMBOL_REGISTRY for item in required_symbols)
            or any(item not in OPERATOR_REGISTRY for item in required_operators)
        ):
            continue
        payload = (*body, *(f"dependency {item}" for item in dependencies))
        validated = validate_math_ir(parse_math_ir(payload))
        canonical = {
            "choice_id": choice_id,
            "transformation_id": transformation_id,
            "summary_id": summary_id,
            "target_ref": str(target_ref),
            "viewpoint": str(viewpoint),
            "dependency_ids": dependencies,
            "typed_ir": asdict(validated.math_ir),
            "typed_ir_hash": validated.content_hash,
        }
        candidate_hash = hashlib.sha256(json.dumps(
            canonical, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        candidates.append(TypedIRCandidate(
            choice_id,
            transformation_id,
            summary_id,
            tuple(payload),
            validated.content_hash,
            candidate_hash,
        ))
    manifest = {
        "target_ref": str(target_ref),
        "viewpoint": str(viewpoint),
        "dependency_ids": dependencies,
        "candidates": [asdict(candidate) for candidate in candidates],
        "math_registry_hash": registry_hash(),
    }
    manifest_hash = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return TypedIRCandidateRegistry(
        str(target_ref),
        str(viewpoint),
        dependencies,
        tuple(candidates),
        manifest_hash,
    )


def registry_hash() -> str:
    payload = {
        "version": REGISTRY_VERSION,
        "types": [asdict(item) for item in TYPE_REGISTRY.values()],
        "symbols": [asdict(item) for item in SYMBOL_REGISTRY.values()],
        "operators": [asdict(item) for item in OPERATOR_REGISTRY.values()],
        "tactics": TACTIC_REGISTRY,
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def parse_math_ir(
    lines: Iterable[str],
    *,
    theorem_id: str = "hostGeneratedTheorem",
) -> MathIR:
    """Parse the registered line DSL; this is adapter work, not proof search."""
    binders: list[Binder] = []
    nodes: list[ExprNode] = []
    premises: list[str] = []
    conclusion = ""
    dependencies: list[str] = []
    for line_number, raw in enumerate(lines, 1):
        text = str(raw).strip()
        if not text:
            continue
        if _RAW_SOURCE.search(text):
            raise TypedIRError(
                "RAW_SOURCE_FORBIDDEN",
                "Math IR contains source-language or raw LaTeX syntax",
                line=line_number,
                status=GateStatus.ADAPTER_BLOCKED,
            )
        parts = text.split()
        command = parts[0]
        try:
            if command == "binder" and len(parts) == 3:
                binders.append(Binder(parts[1], parts[2]))
            elif command == "var" and len(parts) == 3:
                nodes.append(ExprNode(parts[1], "var", parts[2]))
            elif command == "symbol" and len(parts) >= 3:
                nodes.append(ExprNode(parts[1], "symbol", parts[2], tuple(parts[3:])))
            elif command == "op" and len(parts) >= 3:
                nodes.append(ExprNode(parts[1], "operator", parts[2], tuple(parts[3:])))
            elif command == "premise" and len(parts) == 2:
                premises.append(parts[1])
            elif command == "conclusion" and len(parts) == 2:
                if conclusion:
                    raise ValueError("duplicate conclusion")
                conclusion = parts[1]
            elif command == "dependency" and len(parts) == 2:
                dependencies.append(parts[1])
            else:
                raise ValueError("unknown command or wrong arity")
        except ValueError as exc:
            raise TypedIRError(
                "MALFORMED_IR_LINE", str(exc), line=line_number,
                status=GateStatus.ADAPTER_BLOCKED,
            ) from exc
    return MathIR(
        theorem_id,
        tuple(binders),
        tuple(nodes),
        tuple(premises),
        conclusion,
        tuple(dependencies),
    )


def _canonical_ir(math_ir: MathIR) -> bytes:
    return json.dumps(
        asdict(math_ir),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def validate_dependency_graph(graph: Mapping[str, Iterable[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(item: str) -> None:
        if item in visiting:
            raise TypedIRError(
                "DEPENDENCY_CYCLE", f"dependency cycle at {item}",
                owner="decomposer",
                status=GateStatus.SEMANTIC_BACKJUMP,
            )
        if item in visited:
            return
        visiting.add(item)
        for dependency in graph.get(item, ()):
            if dependency not in graph:
                raise TypedIRError(
                    "UNKNOWN_DEPENDENCY",
                    f"{item} references unknown dependency {dependency}",
                    owner="decomposer",
                    status=GateStatus.SEMANTIC_BACKJUMP,
                )
            visit(str(dependency))
        visiting.remove(item)
        visited.add(item)

    for node in graph:
        visit(str(node))


def validate_math_ir(math_ir: MathIR) -> ValidatedMathIR:
    if math_ir.schema_version != MATH_IR_VERSION:
        raise TypedIRError("STALE_IR_VERSION", "unsupported Math IR version")
    if math_ir.registry_version != REGISTRY_VERSION:
        raise TypedIRError("STALE_REGISTRY", "unsupported registry version")
    if not _IDENTIFIER.fullmatch(math_ir.theorem_id):
        raise TypedIRError("INVALID_THEOREM_ID", "invalid theorem ID")
    binder_types: dict[str, str] = {}
    for binder in math_ir.binders:
        if not _IDENTIFIER.fullmatch(binder.binder_id):
            raise TypedIRError("INVALID_BINDER", binder.binder_id)
        if binder.binder_id in binder_types:
            raise TypedIRError("DUPLICATE_BINDER", binder.binder_id)
        if binder.type_id not in TYPE_REGISTRY:
            raise TypedIRError(
                "UNKNOWN_TYPE", binder.type_id, node_id=binder.binder_id,
                owner="decomposer", status=GateStatus.SEMANTIC_BACKJUMP,
            )
        binder_types[binder.binder_id] = binder.type_id
    node_map: dict[str, ExprNode] = {}
    for node in math_ir.nodes:
        if not _IDENTIFIER.fullmatch(node.node_id) or node.node_id in node_map:
            raise TypedIRError("INVALID_OR_DUPLICATE_NODE", node.node_id)
        node_map[node.node_id] = node
    node_types: dict[str, str] = {}
    visiting: set[str] = set()

    def infer(node_id: str) -> str:
        if node_id in node_types:
            return node_types[node_id]
        if node_id in visiting:
            raise TypedIRError("EXPRESSION_CYCLE", "cyclic node", node_id=node_id)
        try:
            node = node_map[node_id]
        except KeyError as exc:
            raise TypedIRError(
                "UNKNOWN_NODE", node_id, node_id=node_id,
                owner="decomposer", status=GateStatus.SEMANTIC_BACKJUMP,
            ) from exc
        visiting.add(node_id)
        if node.kind == "var":
            if node.arguments or node.ref not in binder_types:
                raise TypedIRError(
                    "BINDER_SCOPE", f"unknown binder {node.ref}", node_id=node_id,
                    owner="decomposer", status=GateStatus.SEMANTIC_BACKJUMP,
                )
            result = binder_types[node.ref]
        elif node.kind == "symbol":
            try:
                symbol = SYMBOL_REGISTRY[node.ref]
            except KeyError as exc:
                raise TypedIRError(
                    "UNKNOWN_SYMBOL", node.ref, node_id=node_id,
                    owner="decomposer", status=GateStatus.SEMANTIC_BACKJUMP,
                ) from exc
            actual = tuple(infer(argument) for argument in node.arguments)
            if actual != symbol.argument_types:
                raise TypedIRError(
                    "SYMBOL_TYPE_MISMATCH",
                    f"{node.ref} expects {symbol.argument_types}, got {actual}",
                    node_id=node_id,
                    owner="decomposer",
                    status=GateStatus.SEMANTIC_BACKJUMP,
                )
            result = symbol.result_type
        elif node.kind == "operator":
            try:
                operator = OPERATOR_REGISTRY[node.ref]
            except KeyError as exc:
                raise TypedIRError(
                    "UNKNOWN_OPERATOR", node.ref, node_id=node_id,
                    owner="decomposer", status=GateStatus.SEMANTIC_BACKJUMP,
                ) from exc
            if len(node.arguments) != operator.arity:
                raise TypedIRError("OPERATOR_ARITY", node.ref, node_id=node_id)
            actual = tuple(infer(argument) for argument in node.arguments)
            if actual != operator.argument_types:
                raise TypedIRError(
                    "OPERATOR_TYPE_MISMATCH",
                    f"{node.ref} expects {operator.argument_types}, got {actual}",
                    node_id=node_id,
                    owner="decomposer",
                    status=GateStatus.SEMANTIC_BACKJUMP,
                )
            result = operator.result_type
        else:
            raise TypedIRError("UNKNOWN_NODE_KIND", node.kind, node_id=node_id)
        visiting.remove(node_id)
        node_types[node_id] = result
        return result

    roots = (*math_ir.premises, math_ir.conclusion)
    if not math_ir.conclusion:
        raise TypedIRError("MISSING_CONCLUSION", "conclusion is required")
    for root in roots:
        if infer(root) != "Prop":
            raise TypedIRError(
                "NON_PROPOSITION_ROOT", root, node_id=root,
                owner="decomposer", status=GateStatus.SEMANTIC_BACKJUMP,
            )
    unreachable = set(node_map) - set(node_types)
    if unreachable:
        raise TypedIRError(
            "UNREACHABLE_NODES", ", ".join(sorted(unreachable)),
            owner="decomposer", status=GateStatus.SEMANTIC_BACKJUMP,
        )
    return ValidatedMathIR(
        math_ir,
        node_types,
        hashlib.sha256(_canonical_ir(math_ir)).hexdigest(),
    )


def _render_expr(node_id: str, node_map: Mapping[str, ExprNode]) -> str:
    node = node_map[node_id]
    if node.kind == "var":
        return node.ref
    if node.kind == "symbol":
        symbol = SYMBOL_REGISTRY[node.ref]
        if not node.arguments:
            return symbol.lean_name
        arguments = " ".join(f"({_render_expr(arg, node_map)})" for arg in node.arguments)
        return f"{symbol.lean_name} {arguments}"
    operator = OPERATOR_REGISTRY[node.ref]
    rendered = [_render_expr(arg, node_map) for arg in node.arguments]
    if operator.arity == 1:
        return f"({operator.lean_token} {rendered[0]})"
    return f"({rendered[0]} {operator.lean_token} {rendered[1]})"


def build_lean_declaration(validated: ValidatedMathIR) -> LeanCompilation:
    """Deterministically build Lean; no model-authored source enters here."""
    math_ir = validated.math_ir
    node_map = {node.node_id: node for node in math_ir.nodes}
    binders = " ".join(
        f"({binder.binder_id} : {TYPE_REGISTRY[binder.type_id].lean_name})"
        for binder in math_ir.binders
    )
    proposition_parts = [
        *(_render_expr(item, node_map) for item in math_ir.premises),
        _render_expr(math_ir.conclusion, node_map),
    ]
    proposition = " → ".join(f"({item})" for item in proposition_parts)
    declaration_core = f"theorem {math_ir.theorem_id}"
    if binders:
        declaration_core += f" {binders}"
    declaration_core += f" : {proposition}"
    source = declaration_core + " := by"
    proposition_hash = hashlib.sha256(proposition.encode()).hexdigest()
    declaration_hash = hashlib.sha256(declaration_core.encode()).hexdigest()
    imports_hash = hashlib.sha256(
        b"import KakeyaLeanGate\nset_option autoImplicit false"
    ).hexdigest()
    return LeanCompilation(
        math_ir.theorem_id,
        source,
        declaration_hash,
        proposition_hash,
        validated.content_hash,
        imports_hash,
    )


def verify_proposition_equivalence(
    validated: ValidatedMathIR,
    compilation: LeanCompilation,
) -> None:
    rebuilt = build_lean_declaration(validated)
    if (
        compilation.math_ir_hash != validated.content_hash
        or compilation.proposition_hash != rebuilt.proposition_hash
        or compilation.declaration_hash != rebuilt.declaration_hash
        or compilation.declaration_source != rebuilt.declaration_source
    ):
        raise TypedIRError(
            "PROPOSITION_HASH_MISMATCH",
            "Math IR and Lean declaration are not content-equivalent",
            status=GateStatus.INTEGRATION_BLOCKED,
            owner="host_compiler",
        )


def parse_proof_plan(
    lines: Iterable[str],
    *,
    theorem_id: str,
    proposition_hash: str,
) -> ProofPlan:
    steps: list[ProofStep] = []
    for line_number, raw in enumerate(lines, 1):
        text = str(raw).strip()
        if not text:
            continue
        if _RAW_SOURCE.search(text):
            raise TypedIRError(
                "RAW_PROOF_SOURCE_FORBIDDEN", "proof plan contains source syntax",
                line=line_number, owner="proof_search",
                status=GateStatus.ADAPTER_BLOCKED,
            )
        parts = text.split()
        tactic_id = parts[0]
        try:
            arity, _template = TACTIC_REGISTRY[tactic_id]
        except KeyError as exc:
            raise TypedIRError(
                "UNSUPPORTED_TACTIC", tactic_id, line=line_number,
                owner="proof_search", status=GateStatus.MATHEMATICAL_REJECTION,
            ) from exc
        arguments = tuple(parts[1:])
        if len(arguments) != arity:
            raise TypedIRError(
                "TACTIC_ARITY", tactic_id, line=line_number,
                owner="proof_search", status=GateStatus.MATHEMATICAL_REJECTION,
            )
        if any(not _IDENTIFIER.fullmatch(item) for item in arguments):
            raise TypedIRError(
                "UNSAFE_TACTIC_ARGUMENT", tactic_id, line=line_number,
                owner="proof_search", status=GateStatus.ADAPTER_BLOCKED,
            )
        steps.append(ProofStep(tactic_id, arguments))
    if not steps:
        raise TypedIRError(
            "EMPTY_PROOF_PLAN", "proof plan is empty", owner="proof_search",
            status=GateStatus.MATHEMATICAL_REJECTION,
        )
    return ProofPlan(theorem_id, proposition_hash, tuple(steps))


def render_proof_plan(plan: ProofPlan, compilation: LeanCompilation) -> str:
    if (
        plan.theorem_id != compilation.theorem_id
        or plan.proposition_hash != compilation.proposition_hash
    ):
        raise TypedIRError(
            "PROOF_TARGET_MISMATCH", "proof plan targets another proposition",
            owner="proof_search", status=GateStatus.MATHEMATICAL_REJECTION,
        )
    rendered = []
    for step in plan.steps:
        arity, template = TACTIC_REGISTRY[step.tactic_id]
        if len(step.argument_ids) != arity:
            raise TypedIRError("TACTIC_ARITY", step.tactic_id)
        rendered.append("  " + template.format(*step.argument_ids))
    return compilation.declaration_source + "\n" + "\n".join(rendered)
