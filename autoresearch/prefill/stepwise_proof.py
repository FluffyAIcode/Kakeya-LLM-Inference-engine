"""Crash-safe, action-ID-only Lean proof search."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping

from autoresearch.prefill.research_contract import ResearchContract


PROOF_SEARCH_VERSION = 1


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


class ActionKind(str, Enum):
    INTRO = "INTRO"
    EXACT = "EXACT"
    APPLY = "APPLY"
    ASSUMPTION = "ASSUMPTION"
    CONSTRUCTOR = "CONSTRUCTOR"
    SIMP = "SIMP"
    REWRITE = "REWRITE"
    CASES = "CASES"


class FeedbackCode(str, Enum):
    ACCEPTED = "ACCEPTED"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    TACTIC_FAILED = "TACTIC_FAILED"
    EXPECTED_SUBGOAL_MISMATCH = "EXPECTED_SUBGOAL_MISMATCH"
    ADAPTER_ERROR = "ADAPTER_ERROR"
    HOST_ERROR = "HOST_ERROR"
    SYNTAX_ERROR = "SYNTAX_ERROR"


ZERO_BUDGET_CODES = {
    FeedbackCode.ADAPTER_ERROR.value,
    FeedbackCode.HOST_ERROR.value,
    FeedbackCode.SYNTAX_ERROR.value,
}


@dataclass(frozen=True)
class ProofGoal:
    goal_id: str
    proposition_hash: str
    local_context_ids: tuple[str, ...]
    subgoal_signature: str


@dataclass(frozen=True)
class ProofAction:
    action_id: str
    kind: str
    operand_ids: tuple[str, ...]
    theorem_card_id: str = ""
    novelty: int = 0


@dataclass(frozen=True)
class ActionSelection:
    goal_id: str
    action_id: str
    operand_ids: tuple[str, ...]
    substitution_map_ids: tuple[tuple[str, str], ...]
    expected_subgoal_signatures: tuple[str, ...]


@dataclass(frozen=True)
class ProofStep:
    proof_step_id: str
    goal_id: str
    selected_lemma_or_action_id: str
    substitution_map_ids: tuple[tuple[str, str], ...]
    expected_subgoal_signatures: tuple[str, ...]
    rendered_ast: str
    resulting_subgoal_signatures: tuple[str, ...]
    lean_feedback_code: str


@dataclass(frozen=True)
class LeanStepResult:
    accepted: bool
    feedback_code: str
    subgoal_signatures: tuple[str, ...]
    message_ref: str = ""


@dataclass
class ProofSearchState:
    contract_id: str
    theorem_id: str
    proposition_hash: str
    environment_hash: str
    open_goals: list[ProofGoal]
    accepted_steps: list[ProofStep] = field(default_factory=list)
    rejected_feedback: list[dict[str, object]] = field(default_factory=list)
    semantic_failures: int = 0
    proof_budget: int = 32
    tokens_consumed: int = 0
    status: str = "SEARCHING"
    schema_version: int = 1


@dataclass(frozen=True)
class LeanExecutionContext:
    project_root: Path
    declaration_source: str
    import_names: tuple[str, ...] = ("KakeyaLeanGate",)
    timeout_s: float = 120.0


def new_search_state(
    contract: ResearchContract,
    goals: Iterable[ProofGoal],
    *,
    proof_budget: int = 32,
) -> ProofSearchState:
    if not contract.theorem_id or not contract.proposition_hash:
        raise ValueError("PROOF_SEARCH_REFUSES_UNELABORATED_GOAL")
    goals = list(goals)
    if not goals or any(
        goal.proposition_hash != contract.proposition_hash for goal in goals[:1]
    ):
        raise ValueError("PROOF_SEARCH_REQUIRES_CURRENT_LEAN_PROOF_STATE")
    return ProofSearchState(
        contract.contract_id,
        contract.theorem_id,
        contract.proposition_hash,
        contract.environment_hash,
        goals,
        proof_budget=proof_budget,
    )


def enumerate_applicable_actions(
    state: ProofSearchState,
    *,
    local_context_ids: Iterable[str],
    theorem_card_to_operand_id: Mapping[str, str],
    constructor_ids: Iterable[str] = (),
    rewrite_ids: Iterable[str] = (),
) -> tuple[ProofAction, ...]:
    if not state.open_goals:
        return ()
    actions = [
        ProofAction("A_INTRO", ActionKind.INTRO.value, ()),
        ProofAction("A_ASSUMPTION", ActionKind.ASSUMPTION.value, ()),
        ProofAction("A_SIMP", ActionKind.SIMP.value, ()),
    ]
    for index, operand in enumerate(sorted(set(local_context_ids)), 1):
        actions.append(ProofAction(
            f"A_EXACT_{index}", ActionKind.EXACT.value, (operand,), novelty=1,
        ))
    for index, (card_id, operand) in enumerate(
        sorted(theorem_card_to_operand_id.items()), 1,
    ):
        actions.append(ProofAction(
            f"A_APPLY_{index}", ActionKind.APPLY.value, (operand,),
            theorem_card_id=card_id, novelty=2,
        ))
    for index, constructor in enumerate(sorted(set(constructor_ids)), 1):
        actions.append(ProofAction(
            f"A_CONSTRUCTOR_{index}", ActionKind.CONSTRUCTOR.value,
            (constructor,),
        ))
    for index, rewrite in enumerate(sorted(set(rewrite_ids)), 1):
        actions.append(ProofAction(
            f"A_REWRITE_{index}", ActionKind.REWRITE.value, (rewrite,),
        ))
    return tuple(actions)


def validate_selection(
    selection: ActionSelection,
    actions: Iterable[ProofAction],
    state: ProofSearchState,
) -> ProofAction:
    if not state.open_goals or selection.goal_id != state.open_goals[0].goal_id:
        raise ValueError("SELECTION_NOT_AT_FIRST_UNPROVED_SUBGOAL")
    registry = {action.action_id: action for action in actions}
    action = registry.get(selection.action_id)
    if action is None:
        raise ValueError("UNREGISTERED_PROOF_ACTION_ID")
    if tuple(selection.operand_ids) != action.operand_ids:
        raise ValueError("UNREGISTERED_ACTION_OPERAND_ID")
    substitutions = dict(selection.substitution_map_ids)
    if len(substitutions) != len(selection.substitution_map_ids):
        raise ValueError("DUPLICATE_SUBSTITUTION_ID")
    return action


def render_lean_ast(
    action: ProofAction,
    *,
    operand_sources: Mapping[str, str],
    substitution_sources: Mapping[str, str],
    substitution_map_ids: Iterable[tuple[str, str]] = (),
) -> str:
    """Render a tiny registered tactic AST; no model text reaches Lean."""
    operands = []
    for operand_id in action.operand_ids:
        if operand_id not in operand_sources:
            raise ValueError("UNKNOWN_OPERAND_SOURCE")
        operands.append(operand_sources[operand_id])
    substitutions = []
    for variable_id, value_id in substitution_map_ids:
        if variable_id not in substitution_sources:
            raise ValueError("UNKNOWN_SUBSTITUTION_VARIABLE")
        if value_id not in substitution_sources:
            raise ValueError("UNKNOWN_SUBSTITUTION_VALUE")
        substitutions.append(
            f"({substitution_sources[variable_id]} := "
            f"{substitution_sources[value_id]})"
        )
    suffix = (" " + " ".join(substitutions)) if substitutions else ""
    kind = ActionKind(action.kind)
    if kind == ActionKind.INTRO:
        return "intro"
    if kind == ActionKind.ASSUMPTION:
        return "assumption"
    if kind == ActionKind.SIMP:
        return "simp"
    if kind == ActionKind.CONSTRUCTOR:
        return "constructor"
    if kind == ActionKind.EXACT:
        return f"exact {operands[0]}{suffix}"
    if kind == ActionKind.APPLY:
        return f"apply {operands[0]}{suffix}"
    if kind == ActionKind.REWRITE:
        return f"rw [{operands[0]}]"
    if kind == ActionKind.CASES:
        return f"cases {operands[0]}"
    raise ValueError("UNRENDERABLE_PROOF_ACTION")


def attempt_step(
    state: ProofSearchState,
    selection: ActionSelection,
    actions: Iterable[ProofAction],
    *,
    operand_sources: Mapping[str, str],
    substitution_sources: Mapping[str, str],
    lean_executor: Callable[[str, ProofSearchState], LeanStepResult],
    token_count: int = 0,
) -> LeanStepResult:
    action = validate_selection(selection, actions, state)
    rendered = render_lean_ast(
        action,
        operand_sources=operand_sources,
        substitution_sources=substitution_sources,
        substitution_map_ids=selection.substitution_map_ids,
    )
    result = lean_executor(rendered, state)
    state.tokens_consumed += max(0, int(token_count))
    if result.accepted:
        if (
            selection.expected_subgoal_signatures
            and selection.expected_subgoal_signatures
            != result.subgoal_signatures
        ):
            result = LeanStepResult(
                False, FeedbackCode.EXPECTED_SUBGOAL_MISMATCH.value,
                result.subgoal_signatures, result.message_ref,
            )
        else:
            canonical = {
                "contract_id": state.contract_id,
                "prior_steps": [step.proof_step_id for step in state.accepted_steps],
                "goal_id": selection.goal_id,
                "action_id": action.action_id,
                "substitutions": selection.substitution_map_ids,
                "expected": selection.expected_subgoal_signatures,
                "rendered_ast": rendered,
                "resulting": result.subgoal_signatures,
            }
            step_id = "PS-" + _digest(canonical)[:20]
            state.accepted_steps.append(ProofStep(
                step_id, selection.goal_id, action.action_id,
                selection.substitution_map_ids,
                selection.expected_subgoal_signatures, rendered,
                result.subgoal_signatures, FeedbackCode.ACCEPTED.value,
            ))
            # Lean returns the complete current subgoal vector, so it replaces
            # (rather than appends to) the host's prior proof-state snapshot.
            state.open_goals = [
                ProofGoal(
                    f"{selection.goal_id}.{index}",
                    state.proposition_hash, (),
                    signature,
                )
                for index, signature in enumerate(result.subgoal_signatures, 1)
            ]
            state.status = "PROVED" if not state.open_goals else "SEARCHING"
            return result
    state.rejected_feedback.append({
        "goal_id": selection.goal_id,
        "action_id": selection.action_id,
        "feedback_code": result.feedback_code,
        "message_ref": result.message_ref,
    })
    if result.feedback_code not in ZERO_BUDGET_CODES:
        state.semantic_failures += 1
        if state.semantic_failures >= state.proof_budget:
            state.status = "BUDGET_EXHAUSTED"
    return result


def persist_search_state(path: Path, state: ProofSearchState) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    durable = asdict(state)
    # Rejections are transient typed feedback to the same search state; only
    # Lean-accepted formal steps enter the durable proof artifact.
    durable["rejected_feedback"] = []
    payload = json.dumps(durable, sort_keys=True, indent=2)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def load_search_state(path: Path) -> ProofSearchState | None:
    path = Path(path).expanduser()
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["open_goals"] = [ProofGoal(
        goal_id=item["goal_id"],
        proposition_hash=item["proposition_hash"],
        local_context_ids=tuple(item["local_context_ids"]),
        subgoal_signature=item["subgoal_signature"],
    ) for item in raw["open_goals"]]
    raw["accepted_steps"] = [ProofStep(
        proof_step_id=item["proof_step_id"],
        goal_id=item["goal_id"],
        selected_lemma_or_action_id=item["selected_lemma_or_action_id"],
        substitution_map_ids=tuple(
            tuple(pair) for pair in item["substitution_map_ids"]
        ),
        expected_subgoal_signatures=tuple(
            item["expected_subgoal_signatures"],
        ),
        rendered_ast=item["rendered_ast"],
        resulting_subgoal_signatures=tuple(
            item["resulting_subgoal_signatures"],
        ),
        lean_feedback_code=item["lean_feedback_code"],
    ) for item in raw["accepted_steps"]]
    state = ProofSearchState(**raw)
    # The ordered open_goals list makes resume deterministic at index zero.
    return state


def beam_rank(
    candidates: Iterable[tuple[ProofSearchState, ProofAction]],
    *,
    beam_width: int,
) -> tuple[tuple[ProofSearchState, ProofAction], ...]:
    return tuple(sorted(
        candidates,
        key=lambda item: (
            len(item[0].open_goals),
            item[0].semantic_failures,
            -int(bool(item[1].theorem_card_id)),
            -item[1].novelty,
            item[1].action_id,
        ),
    )[:beam_width])


def tokens_per_accepted_step(state: ProofSearchState) -> float:
    return (
        state.tokens_consumed / len(state.accepted_steps)
        if state.accepted_steps else 0.0
    )


def lean_step_executor(
    context: LeanExecutionContext,
) -> Callable[[str, ProofSearchState], LeanStepResult]:
    """Build an immediate real-Lean executor for one deterministic AST step."""
    declaration = context.declaration_source.rstrip()
    if ":= by" not in declaration or "\n" in declaration:
        raise ValueError("LEAN_EXECUTOR_REQUIRES_ELABORATED_DECLARATION_SCAFFOLD")

    def execute(rendered_ast: str, state: ProofSearchState) -> LeanStepResult:
        tactics = [
            *(step.rendered_ast for step in state.accepted_steps),
            rendered_ast,
        ]
        source = "\n".join((
            *(f"import {name}" for name in context.import_names),
            "",
            declaration,
            *(f"  {tactic}" for tactic in tactics),
            "",
        ))
        root = Path(context.project_root)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", dir=root, encoding="utf-8", delete=False,
        ) as handle:
            handle.write(source)
            path = Path(handle.name)
        try:
            result = subprocess.run(
                ["lake", "env", "lean", str(path)],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=context.timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return LeanStepResult(
                False, FeedbackCode.HOST_ERROR.value, (),
                "host:" + _digest(str(exc))[:16],
            )
        finally:
            path.unlink(missing_ok=True)
        output = (result.stderr or "") + "\n" + (result.stdout or "")
        if result.returncode == 0:
            return LeanStepResult(True, FeedbackCode.ACCEPTED.value, ())
        if "unsolved goals" in output:
            goals = tuple(
                " ".join(block.split())
                for block in re.findall(
                    r"(?:case\s+\S+\s*)?(⊢.*?)(?=\n\n|$)",
                    output,
                    re.DOTALL,
                )
            )
            return LeanStepResult(
                True, FeedbackCode.ACCEPTED.value,
                goals or ("UNSOLVED_GOAL",),
            )
        lowered = output.lower()
        code = (
            FeedbackCode.SYNTAX_ERROR.value
            if "unexpected token" in lowered or "parser" in lowered
            else FeedbackCode.TYPE_MISMATCH.value
            if "type mismatch" in lowered
            else FeedbackCode.TACTIC_FAILED.value
        )
        return LeanStepResult(
            False, code, (), "lean:" + _digest(output)[:16],
        )

    return execute
