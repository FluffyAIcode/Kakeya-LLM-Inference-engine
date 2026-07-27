"""Sourced, Lean-elaborated bootstrap for the Riemann Hypothesis root."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from autoresearch.prefill.orchestration_state import (
    BlockedEventType,
    BlockedExitEvent,
    OrchestrationCheckpoint,
    ProofState,
    apply_blocked_exit_event,
    persist_validated_artifact,
    save_checkpoint,
)
from autoresearch.prefill.target_context import activate_target_context
from autoresearch.prefill.theorem_cards import pinned_environment_hash


ROOT_BOOTSTRAP_SCHEMA_VERSION = 1
MATHLIB_REVISION = "360da6fa66c1273b76b6b2d8c5666fd5ac2e3b56"
MATHLIB_IMPORT = "Mathlib.NumberTheory.LSeries.RiemannZeta"
MATHLIB_SOURCE = (
    ".lake/packages/mathlib/Mathlib/NumberTheory/LSeries/RiemannZeta.lean"
)
CANONICAL_DECLARATION = "RiemannHypothesis"
CANONICAL_PROPOSITION = "RiemannHypothesis"
EXPANDED_PROPOSITION = """∀ (s : ℂ), riemannZeta s = 0 →
  (¬ ∃ n : ℕ, s = -2 * (n + 1)) →
  s ≠ 1 →
  s.re = 1 / 2"""


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


@dataclass(frozen=True)
class RootSourceCard:
    card_id: str
    declaration_name: str
    exact_type: str
    import_name: str
    source_path: str
    source_sha256: str
    source_revision: str
    source_repository: str
    provenance_kind: str
    content_hash: str


@dataclass(frozen=True)
class RootCandidate:
    candidate_id: str
    proposition: str
    binders: tuple[str, ...]
    nontrivial_zero_condition: str
    critical_line_semantics: str
    import_name: str
    source_card_ids: tuple[str, ...]
    proposition_hash: str
    elaborated: bool = False
    lean_output_hash: str = ""
    accepted: bool = False
    rejection_codes: tuple[str, ...] = ()
    equivalence_theorem: str = ""


@dataclass(frozen=True)
class RootBootstrapResult:
    root_id: str
    proposition_hash: str
    certificate_hash: str
    ledger_version: int
    source_cards: tuple[RootSourceCard, ...]
    candidates: tuple[RootCandidate, ...]
    changed: bool


def build_root_source_cards(project_root: Path) -> tuple[RootSourceCard, ...]:
    """Bind actual declarations to the exact pinned Mathlib source bytes."""
    root = Path(project_root)
    manifest = json.loads((root / "lake-manifest.json").read_text())
    package = next(
        item for item in manifest["packages"] if item.get("name") == "mathlib"
    )
    if package.get("rev") != MATHLIB_REVISION:
        raise ValueError("ROOT_SOURCE_MATHLIB_REVISION_MISMATCH")
    source = root / MATHLIB_SOURCE
    text = source.read_text(encoding="utf-8")
    required_source = (
        "def riemannZeta := hurwitzZetaEven 0",
        "def completedRiemannZeta (s : ℂ) : ℂ",
        "def completedRiemannZeta₀ (s : ℂ) : ℂ",
        "theorem riemannZeta_neg_two_mul_nat_add_one",
        "def RiemannHypothesis : Prop :=",
        "¬∃ n : ℕ, s = -2 * (n + 1)",
    )
    if any(item not in text for item in required_source):
        raise ValueError("ROOT_SOURCE_DECLARATION_DRIFT")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    specs = (
        ("rh-riemann-zeta", "riemannZeta", "ℂ → ℂ"),
        ("rh-completed-zeta", "completedRiemannZeta", "ℂ → ℂ"),
        ("rh-completed-zeta-entire", "completedRiemannZeta₀", "ℂ → ℂ"),
        (
            "rh-trivial-zero-family",
            "riemannZeta_neg_two_mul_nat_add_one",
            "∀ n : ℕ, riemannZeta (-2 * (n + 1)) = 0",
        ),
        ("rh-canonical-proposition", "RiemannHypothesis", "Prop"),
    )
    cards = []
    for card_id, name, exact_type in specs:
        body = {
            "card_id": card_id,
            "declaration_name": name,
            "exact_type": exact_type,
            "import_name": MATHLIB_IMPORT,
            "source_path": MATHLIB_SOURCE,
            "source_sha256": source_hash,
            "source_revision": MATHLIB_REVISION,
            "source_repository": package["url"],
            "provenance_kind": "PINNED_LOCAL_MATHLIB_DECLARATION",
        }
        cards.append(RootSourceCard(**body, content_hash=_digest(body)))
    return tuple(cards)


def _candidate_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "candidate_id": "RH-ROOT-MATHLIB",
            "proposition": CANONICAL_PROPOSITION,
            "binders": ("definition-owned:RiemannHypothesis",),
            "nontrivial_zero_condition": (
                "riemannZeta s = 0; excludes s = -2*(n+1); excludes s = 1"
            ),
            "critical_line_semantics": "Complex.re s = (1 / 2 : ℝ)",
            "source_card_ids": (
                "rh-canonical-proposition", "rh-riemann-zeta",
                "rh-trivial-zero-family",
            ),
            "accepted": True,
            "equivalence_theorem": "",
        },
        {
            "candidate_id": "RH-ROOT-EXPANDED",
            "proposition": EXPANDED_PROPOSITION,
            "binders": ("s : ℂ", "n : ℕ"),
            "nontrivial_zero_condition": (
                "riemannZeta s = 0; ¬∃ n, s = -2*(n+1); s ≠ 1"
            ),
            "critical_line_semantics": "Complex.re s = (1 / 2 : ℝ)",
            "source_card_ids": (
                "rh-canonical-proposition", "rh-riemann-zeta",
                "rh-trivial-zero-family",
            ),
            "accepted": True,
            "equivalence_theorem": "kakeya_rh_expanded_iff_canonical",
        },
        {
            "candidate_id": "RH-ROOT-CRITICAL-STRIP",
            "proposition": (
                "∀ (s : ℂ), riemannZeta s = 0 → 0 < s.re → s.re < 1 → "
                "s.re = 1 / 2"
            ),
            "binders": ("s : ℂ",),
            "nontrivial_zero_condition": "0 < Complex.re s ∧ Complex.re s < 1",
            "critical_line_semantics": "Complex.re s = (1 / 2 : ℝ)",
            "source_card_ids": ("rh-riemann-zeta",),
            "accepted": False,
            "rejection_codes": ("UNPROVED_EQUIVALENCE_TO_CANONICAL_ROOT",),
        },
        {
            "candidate_id": "RH-ROOT-COMPLETED-LAMBDA",
            "proposition": (
                "∀ (s : ℂ), completedRiemannZeta s = 0 → "
                "s ≠ 0 → s ≠ 1 → s.re = 1 / 2"
            ),
            "binders": ("s : ℂ",),
            "nontrivial_zero_condition": (
                "completedRiemannZeta s = 0; excludes its poles 0 and 1"
            ),
            "critical_line_semantics": "Complex.re s = (1 / 2 : ℝ)",
            "source_card_ids": ("rh-completed-zeta",),
            "accepted": False,
            "rejection_codes": (
                "COMPLETED_LAMBDA_NOT_COMPLETED_XI",
                "UNPROVED_ZERO_EQUIVALENCE_TO_CANONICAL_ROOT",
            ),
        },
        {
            "candidate_id": "RH-ROOT-TRIVIAL-ZEROS-INCLUDED",
            "proposition": (
                "∀ (s : ℂ), riemannZeta s = 0 → s ≠ 1 → s.re = 1 / 2"
            ),
            "binders": ("s : ℂ",),
            "nontrivial_zero_condition": "missing trivial-zero exclusion",
            "critical_line_semantics": "Complex.re s = (1 / 2 : ℝ)",
            "source_card_ids": ("rh-riemann-zeta",),
            "accepted": False,
            "rejection_codes": ("TRIVIAL_ZEROS_NOT_EXCLUDED",),
        },
        {
            "candidate_id": "RH-ROOT-VACUOUS",
            "proposition": (
                "∀ (s : ℂ), riemannZeta s = 0 ∧ riemannZeta s ≠ 0 → "
                "s.re = 1 / 2"
            ),
            "binders": ("s : ℂ",),
            "nontrivial_zero_condition": "contradictory antecedent",
            "critical_line_semantics": "Complex.re s = (1 / 2 : ℝ)",
            "source_card_ids": ("rh-riemann-zeta",),
            "accepted": False,
            "rejection_codes": ("VACUOUS_ANTECEDENT",),
        },
        {
            "candidate_id": "RH-ROOT-COMPLETED-XI",
            "proposition": "∀ (s : ℂ), riemannXi s = 0 → s.re = 1 / 2",
            "binders": ("s : ℂ",),
            "nontrivial_zero_condition": "riemannXi s = 0",
            "critical_line_semantics": "Complex.re s = (1 / 2 : ℝ)",
            "source_card_ids": (),
            "accepted": False,
            "rejection_codes": ("UNDEFINED_COMPLETED_XI_SYMBOL",),
        },
    )


def elaborate_root_candidates(
    project_root: Path,
    *,
    timeout_s: float = 120.0,
) -> tuple[RootCandidate, ...]:
    """Compile every branch independently; failed branches remain auditable."""
    root = Path(project_root)
    results = []
    for spec in _candidate_specs():
        proposition = str(spec["proposition"])
        declaration = (
            f"import {MATHLIB_IMPORT}\n"
            f"def RootCandidate : Prop := {proposition}\n"
            "#check RootCandidate\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", dir=root, encoding="utf-8", delete=False,
        ) as handle:
            handle.write(declaration)
            path = Path(handle.name)
        try:
            completed = subprocess.run(
                ["lake", "env", "lean", str(path)],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        finally:
            path.unlink(missing_ok=True)
        elaborated = completed.returncode == 0
        rejection_codes = tuple(spec.get("rejection_codes", ()))
        if not elaborated:
            rejection_codes = tuple(sorted({
                *rejection_codes, "LEAN_ELABORATION_FAILED",
            }))
        accepted = bool(spec.get("accepted")) and elaborated
        results.append(RootCandidate(
            candidate_id=spec["candidate_id"],
            proposition=proposition,
            binders=tuple(spec["binders"]),
            nontrivial_zero_condition=spec["nontrivial_zero_condition"],
            critical_line_semantics=spec["critical_line_semantics"],
            import_name=MATHLIB_IMPORT,
            source_card_ids=tuple(spec["source_card_ids"]),
            proposition_hash=hashlib.sha256(proposition.encode()).hexdigest(),
            elaborated=elaborated,
            lean_output_hash=hashlib.sha256(
                (completed.stdout + completed.stderr).encode()
            ).hexdigest(),
            accepted=accepted,
            rejection_codes=rejection_codes,
            equivalence_theorem=spec.get("equivalence_theorem", ""),
        ))
    canonical = results[0]
    if not canonical.accepted:
        raise ValueError("CANONICAL_RH_ROOT_DID_NOT_ELABORATE")
    return tuple(results)


def validate_root_sources(
    project_root: Path,
) -> tuple[tuple[RootSourceCard, ...], tuple[RootCandidate, ...]]:
    cards = build_root_source_cards(project_root)
    candidates = elaborate_root_candidates(project_root)
    checks = "\n".join(
        [f"import {MATHLIB_IMPORT}"]
        + [f"#check {card.declaration_name}" for card in cards]
        + [
            "example : KakeyaRiemannHypothesisExpanded ↔ "
            "KakeyaRiemannHypothesisRoot := "
            "kakeya_rh_expanded_iff_canonical"
        ]
    )
    checks = checks.replace(
        f"import {MATHLIB_IMPORT}",
        "import KakeyaLeanGate.RiemannHypothesisRoot",
        1,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", dir=project_root,
        encoding="utf-8", delete=False,
    ) as handle:
        handle.write(checks + "\n")
        path = Path(handle.name)
    try:
        completed = subprocess.run(
            ["lake", "env", "lean", str(path)],
            cwd=project_root,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    finally:
        path.unlink(missing_ok=True)
    if completed.returncode:
        raise ValueError("ROOT_SOURCE_CARD_ELABORATION_FAILED")
    return cards, candidates


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _append_transaction(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def bootstrap_rh_root(
    *,
    project_root: Path,
    ledger_path: Path,
    checkpoint_path: Path,
    checkpoint: OrchestrationCheckpoint,
    ledger: Mapping[str, Any],
) -> RootBootstrapResult:
    """Create RH-C0 and clear ROOT_UNAVAILABLE in a replay-safe transaction."""
    cards, candidates = validate_root_sources(project_root)
    selected = candidates[0]
    root_id = "RH-C0-" + selected.proposition_hash[:12]
    obligations = [dict(item) for item in ledger.get("obligations", ())]
    existing = next(
        (item for item in obligations if item.get("obligation_id") == root_id),
        None,
    )
    if existing is not None:
        if (
            existing.get("lean_signature_hash") != selected.proposition_hash
            or ledger.get("backjump_target_id")
        ):
            raise ValueError("ROOT_BOOTSTRAP_REPLAY_STATE_MISMATCH")
        certificate_hash = str(existing["root_bootstrap_certificate_hash"])
        return RootBootstrapResult(
            root_id, selected.proposition_hash, certificate_hash,
            int(ledger["version"]), cards, candidates, False,
        )
    if ledger.get("backjump_target_id") != "ROOT_UNAVAILABLE":
        raise ValueError("ROOT_BOOTSTRAP_REQUIRES_ROOT_UNAVAILABLE")
    if checkpoint.proof_state != ProofState.BLOCKED:
        raise ValueError("ROOT_BOOTSTRAP_REQUIRES_BLOCKED_CHECKPOINT")
    rh_c1_before = next(
        dict(item) for item in obligations if item.get("obligation_id") == "RH-C1"
    )
    new_version = int(ledger.get("version", 0)) + 1
    source_index_hash = _digest([asdict(card) for card in cards])
    branch_index_hash = _digest([asdict(item) for item in candidates])
    signed_body = {
        "schema_version": ROOT_BOOTSTRAP_SCHEMA_VERSION,
        "event_type": "RH_ROOT_BOOTSTRAP",
        "root_id": root_id,
        "proposition_hash": selected.proposition_hash,
        "selected_candidate_id": selected.candidate_id,
        "source_index_hash": source_index_hash,
        "branch_index_hash": branch_index_hash,
        "mathlib_revision": MATHLIB_REVISION,
        "ledger_id": str(ledger["ledger_id"]),
        "ledger_version_before": int(ledger["version"]),
        "ledger_version_after": new_version,
        "cleared_backjump": "ROOT_UNAVAILABLE",
        "quarantined_obligation_hash": _digest(rh_c1_before),
    }
    certificate_hash = _digest(signed_body)
    lean_signature = (
        "theorem rh_root : RiemannHypothesis := by\n  sorry"
    )
    template = {key: "" for key in rh_c1_before}
    template.update({
        "obligation_id": root_id,
        "statement": CANONICAL_PROPOSITION,
        "status": "UNRESOLVED",
        "parent_id": "",
        "last_run_id": "host:" + certificate_hash[:20],
        "last_evidence": (
            "Pinned Mathlib RiemannHypothesis declaration elaborated; "
            f"source-index sha256:{source_index_hash}."
        ),
        "formal_status": "FORMALIZED",
        "lean_signature": lean_signature,
        "lean_signature_hash": selected.proposition_hash,
        "decomposition_role_run_ids": {},
        "dependency_labels": [],
        "dependency_ids": [],
        "public_assumptions": [],
        "source_card_ids": [card.card_id for card in cards],
        "root_candidate_ids": [item.candidate_id for item in candidates],
        "proposition_hash": selected.proposition_hash,
        "root_bootstrap_certificate_hash": certificate_hash,
    })
    obligations.append(template)
    ledger_after = dict(ledger)
    ledger_after.update({
        "obligations": obligations,
        "version": new_version,
        "backjump_target_id": "",
    })
    environment_hash = pinned_environment_hash(project_root)
    context, _ = activate_target_context(
        checkpoint_path,
        checkpoint,
        target_obligation_id=root_id,
        statement=CANONICAL_PROPOSITION,
        environment_hash=environment_hash,
        strategy_plan_hash="STRATEGY_PENDING",
        evidence={
            "EVIDENCE_TARGET_STATEMENT": CANONICAL_PROPOSITION,
            "root_source_index_hash": source_index_hash,
            "root_candidate_index_hash": branch_index_hash,
            "root_bootstrap_certificate_hash": certificate_hash,
            "mathlib_revision": MATHLIB_REVISION,
        },
        definition_ids=tuple(card.declaration_name for card in cards),
        candidate_ids=tuple(item.candidate_id for item in candidates),
        theorem_card_ids=tuple(card.card_id for card in cards),
    )
    checkpoint.ledger_version = new_version
    checkpoint.ledger_id = str(ledger["ledger_id"])
    checkpoint.root_goal_sha256 = selected.proposition_hash
    checkpoint.elaborated_theorem_id = "KakeyaRiemannHypothesisRoot"
    checkpoint.proposition_hash = selected.proposition_hash
    checkpoint.lean_declaration_hash = hashlib.sha256(
        (Path(project_root) / "KakeyaLeanGate/RiemannHypothesisRoot.lean").read_bytes()
    ).hexdigest()
    auditor_payload = {
        "schema_version": 1,
        "target_obligation_id": root_id,
        "target_context_hash": context.binding.context_hash,
        "parent_statement_hash": context.binding.parent_statement_hash,
        "strategy_plan_hash": "STRATEGY_PENDING",
        "environment_hash": environment_hash,
        "definitions": [{
            "definition_id": card.declaration_name,
            "source_card_id": card.card_id,
            "source_hash": card.source_sha256,
        } for card in cards],
        "missing_definitions": [],
        "source_index_hash": source_index_hash,
        "candidate_index_hash": branch_index_hash,
        "root_specific": True,
    }
    auditor = persist_validated_artifact(
        checkpoint_path,
        checkpoint,
        role="definition_auditor",
        payload=auditor_payload,
        dependencies=[],
        source_run_id="host:rh-root-bootstrap:" + certificate_hash[:20],
        save=False,
    )
    checkpoint.definition_audit_outcome = "COMPLETE"
    checkpoint.definition_audit_fingerprint = auditor.sha256
    event = BlockedExitEvent(
        event_id=certificate_hash,
        event_type=BlockedEventType.NEW_STRATEGY_TRIGGER.value,
        reason="sourced Lean-elaborated RH root available",
        target_state=ProofState.STRATEGY_TOURNAMENT.value,
        reset_role=ProofState.STRATEGY_TOURNAMENT.value,
        metadata={
            "strategy_trigger": "RH_ROOT_BOOTSTRAP",
            "root_id": root_id,
            "proposition_hash": selected.proposition_hash,
            "certificate": signed_body,
            "signature_algorithm": "SHA256-CONTENT-ADDRESS",
        },
    )
    apply_blocked_exit_event(checkpoint, event)
    checkpoint.recovery_events.append({
        **signed_body,
        "event_id": certificate_hash,
        "certificate_hash": certificate_hash,
        "signature_algorithm": "SHA256-CONTENT-ADDRESS",
        "created_at": time.time(),
    })
    transaction_id = "rh-root-bootstrap:" + certificate_hash
    transaction_path = checkpoint_path.with_name(
        "proof_root_bootstrap.journal.jsonl"
    )
    prepared = {
        "kind": "root_bootstrap_transaction",
        "phase": "PREPARED",
        "transaction_id": transaction_id,
        "ledger_sha256": _digest(ledger_after),
        "target_context_hash": checkpoint.target_context_hash,
        "certificate_hash": certificate_hash,
    }
    _append_transaction(transaction_path, prepared)
    _atomic_write_json(ledger_path, ledger_after)
    save_checkpoint(checkpoint_path, checkpoint)
    rh_c1_after = next(
        item for item in ledger_after["obligations"]
        if item.get("obligation_id") == "RH-C1"
    )
    if rh_c1_after != rh_c1_before:
        raise AssertionError("RH-C1 quarantine provenance changed")
    _append_transaction(transaction_path, {
        **prepared,
        "phase": "COMMITTED",
        "committed_at": time.time(),
    })
    return RootBootstrapResult(
        root_id, selected.proposition_hash, certificate_hash,
        new_version, cards, candidates, True,
    )
