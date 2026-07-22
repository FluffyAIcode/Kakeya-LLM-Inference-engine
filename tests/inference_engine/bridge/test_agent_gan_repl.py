import hashlib
import io
import json
import re
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from autoresearch.prefill.lean_gate import (
    LeanSignatureResult,
    lean_theorem_signature_hash,
    validate_lean_proof,
)
from autoresearch.prefill.semantic_decompose import SemanticUnitTooLarge
from scripts.agent_gan_repl import (
    PrefillHeartbeat,
    CriticIssueBatch,
    PremiseAudit,
    PremiseDefense,
    PremiseReview,
    DefinitionAudit,
    CounterexampleReport,
    DecompositionProposal,
    FormalizationBundle,
    ProofAttempt,
    DefenseReport,
    JudgeDecision,
    ProofObligation,
    ProofObligationLedger,
    ReplCheckpoint,
    ReplPhase,
    TimestampedTee,
    TokenPrinter,
    _gate_failure,
    _json_artifact,
    _stage,
    _telemetry_request,
    build_critic_messages,
    build_generator_messages,
    extract_obligation_history,
    enforce_prefill_token_budget,
    install_signal_protection,
    is_runtime_artifact_prompt,
    consume_critic_issue_batch,
    apply_critic_verdicts,
    audit_ledger_semantic_duplicates,
    build_autoresearch_verdict,
    create_child_obligations,
    certified_decomposition_requested,
    format_critic_issue_injection,
    format_proof_ledger,
    generator_issue_coverage,
    decide_premise_review,
    extract_premise_suspicions,
    load_checkpoint,
    load_pending_critic_issues,
    load_proof_ledger,
    pending_obligations,
    parse_repl_command,
    parse_premise_audit,
    parse_premise_defense,
    parse_certified_artifact,
    recover_checkpoint_from_log,
    save_critic_issue_batch,
    save_decomposition_manifest,
    save_proof_ledger,
    save_checkpoint,
    run_isolated_premise_review,
    run_certified_decomposition,
    persist_verified_decomposition,
    validate_evidence_artifact,
)


def test_json_artifact_repairs_invalid_latex_escapes_losslessly():
    artifact = _json_artifact(
        r'{"statement":"sequence \{z_n\} has density \rho"}',
    )
    assert artifact == {
        "statement": r"sequence \{z_n\} has density \rho",
    }


class Tokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(96 + token) for token in token_ids)


def _valid_lean_signature(suffix=""):
    return LeanSignatureResult(
        f"theorem frontier{suffix} (p : Prop) : p := by sorry",
        f"lean-signature-hash{suffix}",
        True,
    )


def _premise_suspicion_text(invalidation="PREMISE_SUSPECTED"):
    claim = {
        "schema_version": 1,
        "quantifier": "FOR_ALL",
        "variables": ["x"],
        "domain": "REAL",
        "lhs": "x*x",
        "relation": "==",
        "rhs": "x",
    }
    return f"""
### ISSUE_VERDICT ROOT-A
Status: DISPROVED
Invalidation: {invalidation}
Premise refuted: For every real x, x squared equals x.
Evidence type: FINITE_COUNTEREXAMPLE
Evidence artifact: {json.dumps({"claim": claim}, separators=(",", ":"))}
Evidence: Substitution of the explicit finite value x=2 gives four on the left and two on the right, contradicting universal equality.
Missing lemma: none
"""


def _audit_text(status="CONFIRMED", confidence="0.95", evidence_type="FINITE_COUNTEREXAMPLE"):
    suspicion = extract_premise_suspicions(
        _premise_suspicion_text(),
        {"ROOT-A"},
    )["ROOT-A"]
    artifact = {
        "claim_hash": suspicion.claim_hash,
        "claim": suspicion.claim_schema,
        "witness": {"x": 2},
    }
    return f"""
### PREMISE_AUDIT ROOT-A
Status: {status}
Evidence type: {evidence_type}
Evidence source: host-checkable substitution x=2
Confidence: {confidence}
Artifact: {json.dumps(artifact, separators=(",", ":"))}
Analysis: The universal quantifier is contradicted by this explicit domain element.
"""


def _defense_text(status="NOT_RESCUED"):
    return f"""
### PREMISE_DEFENSE ROOT-A
Status: {status}
Correction: Restricting x to zero or one would rescue a different proposition.
Failure reason: The stated universal real-domain premise has no such restriction.
Evidence: The proposed restriction changes the exact domain and therefore cannot rescue the original quantified premise.
"""


def _certificate_runner(
    *,
    child_signature="theorem childReduction : True := by sorry",
    proof_source="theorem reduction (h : True) : True := by exact h",
    cycle=False,
    parent_hash_override="",
    judge_decision="ACCEPT",
    counterexample_case=None,
    multi_child=False,
):
    calls = []
    parent_source = "theorem parentTarget : True := by sorry"
    reduction_source = "theorem reduction (h : True) : True := by sorry"
    child_statement = (
        "For every fixed compact disk, prove an explicit uniform boundary "
        "inequality for the analytic approximants."
    )

    def runner(role, messages, expected_run_id):
        calls.append((role, messages, expected_run_id))
        package = json.loads(messages[-1]["content"])
        common = {
            "target_obligation_id": package["target_obligation_id"],
            "parent_statement_hash": package["parent_statement_hash"],
            "root_goal_hash": package["root_goal_hash"],
            "producer_role": role,
            "producer_run_id": expected_run_id,
            "upstream_artifact_hashes": package[
                "upstream_artifact_hashes"
            ],
        }
        if role == "definition_auditor":
            heading = "DEFINITION_AUDIT"
            specific = {
                "definitions": [{"symbol": "K", "domain": "compact disks"}],
                "missing_definitions": [],
            }
        elif role == "counterexample_worker":
            heading = "COUNTEREXAMPLE_REPORT"
            specific = {
                "status": (
                    "COUNTEREXAMPLE_FOUND"
                    if counterexample_case else "NO_COUNTEREXAMPLE"
                ),
                "cases": [counterexample_case] if counterexample_case else [],
            }
        elif role == "decomposer":
            heading = "DECOMPOSITION_PROPOSAL"
            children = [{
                "label": "L1",
                "statement": child_statement,
                "kind": "LEMMA",
            }]
            if multi_child:
                children.append({
                    "label": "L2",
                    "statement": (
                        "For every boundary point, prove a separate explicit "
                        "continuity inequality for the analytic approximants."
                    ),
                    "kind": "LEMMA",
                })
            specific = {
                "parent_statement": package["parent_statement"],
                "children": children,
                "dependency_edges": [["L1", "L1"]] if cycle else [],
                "public_assumptions": [],
                "reduction_labels": [
                    child["label"] for child in children
                ],
                "reduction_contract": "L1 implies the exact parent.",
            }
        elif role == "formalizer":
            heading = "FORMALIZATION_BUNDLE"
            parent_hash = (
                parent_hash_override
                or lean_theorem_signature_hash(parent_source)
            )
            specific = {
                "math_ir": {
                    "parent_signature_hash": parent_hash,
                    "parent_proposition_hash": hashlib.sha256(
                        b"True",
                    ).hexdigest(),
                    "child_labels": ["L1"],
                    "public_assumptions": [],
                    "reduction_labels": ["L1"],
                },
                "parent_signature_source": parent_source,
                "parent_signature_hash": parent_hash,
                "parent_newly_formalized": True,
                "children": [{
                    "label": "L1",
                    "statement": child_statement,
                    "lean_signature": child_signature,
                    "lean_signature_hash": lean_theorem_signature_hash(
                        child_signature,
                    ),
                }],
                "reduction_theorem_source": reduction_source,
                "reduction_signature_hash": lean_theorem_signature_hash(
                    reduction_source,
                ),
            }
        elif role == "prover":
            heading = "PROOF_ATTEMPT"
            specific = {
                "status": "PROVED",
                "reduction_theorem_source": proof_source,
            }
        elif role == "adversarial_proponent":
            heading = "DEFENSE_REPORT"
            specific = {
                "status": "DEFENDED",
                "issues": [],
                "repairs": [],
            }
        else:
            heading = "JUDGE_DECISION"
            specific = {
                "decision": judge_decision,
                "reason": "Host manifest reviewed.",
            }
        text = (
            f"### {heading}\nArtifact: "
            + json.dumps({**common, **specific}, separators=(",", ":"))
        )
        return text, expected_run_id

    return runner, calls


def _fake_signature_validator(source, *, project_root):
    del project_root
    if "badChild" in source:
        return LeanSignatureResult(
            source,
            "",
            False,
            status="TYPECHECK_FAILED",
            error="bad child",
        )
    return LeanSignatureResult(
        source,
        lean_theorem_signature_hash(source),
        True,
    )


def _fake_proof_validator(source, *, project_root):
    del project_root
    if re.search(r"\b(?:sorry|admit)\b", source):
        return LeanSignatureResult(
            source,
            "",
            False,
            status="UNSAFE_REJECTED",
            error="incomplete proof",
        )
    return LeanSignatureResult(
        source,
        hashlib.sha256(source.encode()).hexdigest(),
        True,
        status="PROVED",
    )


def test_timestamped_tee_preserves_terminal_and_flushes_log(tmp_path):
    terminal = io.StringIO()
    timestamps = iter(("t1", "t2", "t3"))
    path = tmp_path / "agent.log"
    tee = TimestampedTee(
        terminal,
        path,
        timestamp_fn=lambda: next(timestamps),
    )
    tee.write("generator> hel")
    tee.write("lo\nnext line\n")
    tee.log_only("[input] prove RH")
    tee.close_log()
    assert terminal.getvalue() == "generator> hello\nnext line\n"
    assert path.read_text() == (
        "[t1] generator> hello\n"
        "[t2] next line\n"
        "[t3] [input] prove RH\n"
    )


def test_timestamped_tee_shutdown_restores_streams_and_flush_is_safe(
    tmp_path,
    monkeypatch,
):
    terminal = io.StringIO()
    tee = TimestampedTee(terminal, tmp_path / "agent.log")
    monkeypatch.setattr(sys, "stdout", tee)
    monkeypatch.setattr(sys, "stderr", tee)
    tee.close_log()
    assert sys.stdout is terminal
    assert sys.stderr is terminal
    tee.flush()
    tee.write("after-close")
    assert terminal.getvalue() == "after-close"


def test_token_printer_streams_only_new_suffix(capsys):
    printer = TokenPrinter(Tokenizer(), "generator")
    printer([1])
    printer([1, 2])
    printer.finish()
    assert capsys.readouterr().out == "generator> ab\n"


def test_repl_stage_is_redacted_and_passes_cache_gate():
    warm = {
        "prefix_tokens": 10,
        "e2e_s": 2,
        "delta": {
            "remote_jobs": 1,
            "remote_hits": 1,
            "tokens_reused": 10,
            "tokens_computed": 0,
            "fallbacks": 0,
            "remote_job_failures": 0,
        },
    }
    actual = {
        "prefix_tokens": 10,
        "output_tokens": 2,
        "append_s": 0.1,
        "ttft_s": 0.2,
        "decode_s": 0.3,
        "e2e_s": 0.4,
        "stop_reason": "eos",
        "complete": True,
        "delta": {
            "local_hits": 1,
            "remote_jobs": 0,
            "tokens_computed": 0,
            "fallbacks": 0,
        },
    }
    stage = _stage("generator", warm, actual, "private output")
    assert stage["ok"]
    assert stage["output_chars"] == 14
    assert len(stage["output_hash"]) == 64
    assert "output" not in stage
    assert not _stage(
        "generator",
        warm,
        {**actual, "complete": False, "stop_reason": "client_safety_limit"},
        "cut off",
    )["ok"]


def test_external_sigterm_is_ignored_until_user_quits(monkeypatch, capsys):
    installed = {}
    monkeypatch.setattr(
        signal,
        "signal",
        lambda number, handler: installed.update({number: handler}),
    )
    install_signal_protection()
    installed[signal.SIGTERM](signal.SIGTERM, None)
    output = capsys.readouterr().out
    assert "ignored external signal" in output
    assert "Type /quit to approve shutdown" in output


def test_shell_supervisor_restarts_signal_exits_only():
    source = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "run_agent_gan_repl.sh"
    ).read_text()
    assert "trap" in source and "TERM HUP" in source
    assert '"$status" -eq 143' in source
    assert "restarting in 2s" in source


def test_prefill_heartbeat_reports_elapsed_progress(capsys):
    with PrefillHeartbeat(
        "Critic",
        interval_s=0.01,
        stats_provider=lambda: {
            "remote_job_tokens_computed": 256,
            "remote_job_tokens_total": 1024,
        },
    ):
        time.sleep(0.025)
    output = capsys.readouterr().out
    assert "Critic Prefill:" in output
    assert "256/1024 tokens (25.0%)" in output
    assert "ETA" in output


def test_stage_includes_full_context_metrics():
    warm = {
        "prefix_tokens": 10,
        "e2e_s": 1,
        "delta": {
            "remote_jobs": 1,
            "remote_hits": 1,
            "tokens_reused": 10,
            "tokens_computed": 0,
            "fallbacks": 0,
            "remote_job_failures": 0,
        },
    }
    actual = {
        "prefix_tokens": 10,
        "output_tokens": 1,
        "append_s": 0.1,
        "ttft_s": 0.2,
        "decode_s": 0.3,
        "e2e_s": 0.4,
        "stop_reason": "eos",
        "complete": True,
        "delta": {
            "local_hits": 1,
            "remote_jobs": 0,
            "tokens_computed": 0,
            "fallbacks": 0,
        },
    }
    stage = _stage(
        "critic",
        warm,
        actual,
        "ok",
        extra_metrics={
            "generator_full_tokens": 100,
            "critic_context_tokens": 100,
            "critic_omitted_tokens": 0,
            "review_scope": "full",
            "critic_protocol": "goal_anchored_recursive_gan_v3",
        },
    )
    assert stage["critic_context_tokens"] == 100
    assert stage["critic_omitted_tokens"] == 0
    assert stage["review_scope"] == "full"
    assert stage["critic_protocol"] == "goal_anchored_recursive_gan_v3"


def test_telemetry_timeout_warns_without_stopping_inference(
    monkeypatch,
    capsys,
):
    def timeout(*_args, **_kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("scripts.agent_gan_repl._json_request", timeout)
    assert _telemetry_request("http://dashboard/metrics") is None
    output = capsys.readouterr().out
    assert "telemetry-warning" in output
    assert "inference will continue" in output


def test_gate_failure_exposes_reuse_counters():
    error = _gate_failure(
        "Generator",
        {"delta": {"local_hits": 0, "remote_hits": 0}},
        {"delta": {"local_hits": 0, "fallbacks": 1}},
    )
    message = str(error)
    assert "Generator KV gate failed" in message
    assert "'remote_hits': 0" in message
    assert "'fallbacks': 1" in message


def test_interactive_prompts_are_deterministic_for_kv_reuse():
    kwargs = {
        "steering": "continue the zero-free-region branch",
        "previous_generator": "previous complete argument",
        "previous_critic": "previous complete correction",
        "proof_ledger": "PROOF OBLIGATION LEDGER id=rh-ledger",
    }
    generator_a = build_generator_messages("prove RH", **kwargs)
    generator_b = build_generator_messages("prove RH", **kwargs)
    critic_a = build_critic_messages(
        "prove RH",
        "complete generator response",
        steering=kwargs["steering"],
        proof_ledger=kwargs["proof_ledger"],
        stop_reason="eos",
        complete=True,
    )
    critic_b = build_critic_messages(
        "prove RH",
        "complete generator response",
        steering=kwargs["steering"],
        proof_ledger=kwargs["proof_ledger"],
        stop_reason="eos",
        complete=True,
    )
    assert generator_a == generator_b
    assert critic_a == critic_b
    combined = repr(generator_a + critic_a)
    assert "Internal run" not in combined
    assert "IMMUTABLE RESEARCH GOAL" in combined
    assert "previous complete correction" in combined
    assert "PROOF OBLIGATION LEDGER" in combined
    assert "Goal Alignment: ALIGNED" in combined
    assert "Goal Alignment: DRIFTED" in combined
    assert "open problem" in combined
    assert "recursive adversarial proof analyst" in combined
    assert "Never output a numeric score" in combined
    assert "Decomposition Loop" in combined
    assert "If the response stops at" in combined
    assert "attack that stopping claim" in combined
    assert "Ignore prizes, money, prestige" in combined
    assert "smallest unresolved frontier" in combined
    assert "sample, summarize, simplify" in combined
    assert "LEAN_SIGNATURE" not in combined


def test_generator_history_is_scoped_to_target_leaf():
    history = """
### ISSUE_RESPONSE RH-C1
Correction: unrelated operator branch.

### ISSUE_VERDICT RH-C1
Status: UNRESOLVED
Missing lemma: unrelated operator lemma.

### ISSUE_RESPONSE RH-C2-child
Correction: target convergence branch.

### ISSUE_VERDICT RH-C2-child
Status: UNRESOLVED
Missing lemma: target compact convergence lemma.
"""
    scoped = extract_obligation_history(history, "RH-C2-child")
    assert "target convergence branch" in scoped
    assert "target compact convergence lemma" in scoped
    assert "RH-C1" not in scoped
    messages = build_generator_messages(
        "prove RH",
        previous_generator=history,
        previous_critic=history,
        target_obligation_id="RH-C2-child",
    )
    prompt = messages[-1]["content"]
    assert "target convergence branch" in prompt
    assert "unrelated operator branch" not in prompt


def test_generator_and_critic_share_exact_one_step_interface_and_output():
    interface = json.dumps({
        "root_goal_hash": "root-hash",
        "target_obligation_id": "ROOT-L1",
        "target_statement": "Exact bounded target statement.",
        "interface_hash": "interface-hash",
    }, separators=(",", ":"))
    generator = build_generator_messages(
        "full root prose must not be active",
        steering="Attempt exactly one boundary case.",
        target_obligation_id="ROOT-L1",
        proof_step_interface=interface,
    )
    generator_prompt = generator[-1]["content"]
    assert interface in generator_prompt
    assert "### ISSUE_RESPONSE ROOT-L1" in generator[0]["content"]
    assert "### ISSUE_RESPONSE <ID>" not in generator[0]["content"]
    assert "full root prose" not in generator_prompt
    exact_output = (
        "### ISSUE_RESPONSE ROOT-L1\n"
        "Correction: exact bytes π.\n"
        "Derivation: one bounded step.\n"
        "Remaining gap: none"
    )
    critic = build_critic_messages(
        "full root prose must not be active",
        exact_output,
        proof_step_interface=interface,
        stop_reason="eos",
        complete=True,
    )
    critic_prompt = critic[-1]["content"]
    assert interface in critic_prompt
    assert exact_output in critic_prompt
    assert critic_prompt.count(exact_output) == 1
    assert "full root prose" not in critic_prompt
    assert "Audit exactly one certified proof step" in critic[0]["content"]
    assert "Build a proof-obligation tree" not in critic[0]["content"]


def test_generator_repairs_real_namespaced_single_target_id():
    target = (
        "RH-C2-0ef53a217d-25557e489d-4f025934ee-3110912e68-"
        "763645cd6b-40ef83e052-b80cd1343b-3be9e8e78f-fbee0281ff-"
        "e4fff64467"
    )
    malformed = (
        "rh-rigorous-obligations-v1-rh-C2-0ef53a217d-25557e489d-"
        "4f025934ee-311091268-763645cd6b-40ef83e052-b80cd1343b-"
        "3be9e8e78f-fbee0281ff-e4fff64467"
    )
    covered, missing = generator_issue_coverage(
        f"### ISSUE_RESPONSE {malformed}\nCorrection: repaired",
        [ProofObligation(target, "Exact target.")],
    )
    assert covered == {target}
    assert missing == set()


def test_generator_rejects_unrelated_and_ambiguous_ids():
    target = ProofObligation(
        "RH-C2-aaaaaaaaaa-bbbbbbbbbb-cccccccccc",
        "First target.",
    )
    covered, missing = generator_issue_coverage(
        "### ISSUE_RESPONSE rh-c2-invented-unrelated-label",
        [target],
    )
    assert covered == set()
    assert missing == {target.obligation_id}

    alternatives = [
        ProofObligation(
            "RH-C2-aaaaaaaaaa-bbbbbbbbbb-111111111a",
            "Alternative A.",
        ),
        ProofObligation(
            "RH-C2-aaaaaaaaaa-bbbbbbbbbb-111111111b",
            "Alternative B.",
        ),
    ]
    covered, missing = generator_issue_coverage(
        "### ISSUE_RESPONSE rh-c2-aaaaaaaaaa-bbbbbbbbbb-111111111c",
        alternatives,
    )
    assert covered == set()
    assert missing == {item.obligation_id for item in alternatives}


def test_prefill_budget_rejects_whole_input_without_truncation():
    token_ids = list(range(7))
    enforce_prefill_token_budget("Generator", token_ids, 7)
    try:
        enforce_prefill_token_budget("Critic", token_ids, 6)
    except ValueError as exc:
        assert "without truncation" in str(exc)
        assert "7 > 6" in str(exc)
    else:
        raise AssertionError("over-budget Prefill must be rejected")
    assert token_ids == list(range(7))


def test_runtime_output_cannot_replace_research_goal():
    for text in (
        "critic> ### Central Claim",
        "[metrics] KV hit=100%",
        "[allens] Critic Prefill: 30s",
        "prompt> ",
        "Traceback (most recent call last):",
    ):
        assert is_runtime_artifact_prompt(text)
    assert not is_runtime_artifact_prompt("证明黎曼猜想")


def test_command_state_machine_requires_explicit_ready_commands():
    assert parse_repl_command(
        "证明黎曼猜想",
        ReplPhase.WAITING_FOR_GOAL,
    ).action == "new"
    assert parse_repl_command("/continue", ReplPhase.READY).action == "continue"
    steering = parse_repl_command(
        "/steer analyze the explicit formula",
        ReplPhase.READY,
    )
    assert steering.action == "steer"
    assert steering.payload == "analyze the explicit formula"
    assert parse_repl_command("/new new goal", ReplPhase.READY).payload == (
        "new goal"
    )
    assert parse_repl_command("/quit", ReplPhase.READY).action == "quit"
    for raw, phase in (
        ("Operator output fragment", ReplPhase.READY),
        ("critic> copied output", ReplPhase.READY),
        ("/continue", ReplPhase.WAITING_FOR_GOAL),
        ("/continue", ReplPhase.RUNNING),
        ("/steer", ReplPhase.READY),
        ("/new", ReplPhase.READY),
    ):
        try:
            parse_repl_command(raw, phase)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected rejection for {raw!r} in {phase}")


def test_continuous_auto_loop_is_default_and_exception_pauses():
    source = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "agent_gan_repl.py"
    ).read_text()
    assert 'parser.set_defaults(auto_loop=True)' in source
    assert "select.select(" in source
    assert 'raw_input = "/continue"' in source
    assert '"/continue queued"' in source
    assert "auto_loop_active = False" in source
    assert "[auto-loop-paused]" in source
    assert '"--no-auto-loop"' in source
    assert "extract_lean_signature_blocks" not in source


def test_checkpoint_round_trip_is_private(tmp_path):
    path = tmp_path / "state.json"
    expected = ReplCheckpoint(
        research_goal="prove RH",
        previous_generator="full generator",
        previous_critic="full critic",
        last_run_id="br_ok",
    )
    save_checkpoint(path, expected)
    assert load_checkpoint(path) == expected
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_checkpoint(tmp_path / "missing.json") is None


def test_critic_issue_inbox_retries_until_successful_consumption(tmp_path):
    path = tmp_path / "critic-inbox.json"
    batch = CriticIssueBatch(
        issue_id="math-review-1",
        issues=[
            "Do not identify -zeta'/zeta with xi.",
            "Prove zero convergence before invoking Hurwitz.",
        ],
    )
    save_critic_issue_batch(path, batch)
    loaded = load_pending_critic_issues(path)
    assert loaded == batch
    assert path.stat().st_mode & 0o777 == 0o600
    injection = format_critic_issue_injection(loaded)
    assert "math-review-1" in injection
    assert "1. Do not identify" in injection
    assert "2. Prove zero convergence" in injection
    # Merely loading/injecting does not consume the issues.
    assert load_pending_critic_issues(path).status == "pending"
    consume_critic_issue_batch(path, loaded, "br_success")
    assert load_pending_critic_issues(path) is None
    persisted = json.loads(path.read_text())
    assert persisted["status"] == "consumed"
    assert persisted["consumed_by_run"] == "br_success"


def test_proof_ledger_carries_unresolved_and_closes_valid_verdicts(tmp_path):
    path = tmp_path / "proof-ledger.json"
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger-v1",
        obligations=[
            ProofObligation("RH-C1", "Separate xi from -zeta'/zeta."),
            ProofObligation("RH-C2", "Prove zero convergence."),
        ],
    )
    save_proof_ledger(path, ledger)
    loaded = load_proof_ledger(path)
    assert loaded == ledger
    assert path.stat().st_mode & 0o777 == 0o600
    assert [item.obligation_id for item in pending_obligations(loaded)] == [
        "RH-C1",
        "RH-C2",
    ]
    rendered = format_proof_ledger(loaded)
    assert "### ISSUE_RESPONSE <ID>" in rendered
    assert "LEAN_SIGNATURE" not in rendered
    covered, missing = generator_issue_coverage(
        "### ISSUE_RESPONSE RH-C1\nCorrection: fixed",
        pending_obligations(loaded),
    )
    assert covered == {"RH-C1"}
    assert missing == {"RH-C2"}
    critic = """
### ISSUE_VERDICT RH-C1
Status: PROVED
Evidence: This is a sufficiently detailed derivation that distinguishes the two analytic objects.
Missing lemma: none

### ISSUE_VERDICT RH-C2
Status: UNRESOLVED
Evidence: No locally uniform convergence theorem was supplied.
Missing lemma: A valid zero-convergence theorem.
"""
    verdicts = apply_critic_verdicts(loaded, critic, "br_review")
    assert verdicts == {"RH-C1": "PROVED", "RH-C2": "UNRESOLVED"}
    assert [item.obligation_id for item in pending_obligations(loaded)] == [
        "RH-C2",
    ]
    save_proof_ledger(path, loaded)
    assert load_proof_ledger(path).version == 2


def test_proof_ledger_rejects_weak_or_missing_closure():
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[
            ProofObligation("RH-C1", "Issue one"),
            ProofObligation("RH-C2", "Issue two"),
        ],
    )
    critic = """
### ISSUE_VERDICT RH-C1
Status: PROVED
Evidence: too short
Missing lemma: none
"""
    verdicts = apply_critic_verdicts(ledger, critic, "br_weak")
    assert verdicts == {
        "RH-C1": "UNRESOLVED",
        "RH-C2": "UNRESOLVED",
    }
    assert len(pending_obligations(ledger)) == 2


def test_premise_invalidation_quarantines_subtree_and_backjumps():
    root = ProofObligation("ROOT", "Establish the main reduction.")
    target = ProofObligation(
        "ROOT-A",
        "For every real x, x squared equals x.",
        parent_id="ROOT",
    )
    descendant = ProofObligation(
        "ROOT-A-1",
        "Use kernel positivity to derive the bound.",
        parent_id="ROOT-A",
    )
    alternate = ProofObligation(
        "ROOT-B",
        "Construct an alternate sign-changing kernel.",
        parent_id="ROOT",
    )
    ledger = ProofObligationLedger(
        ledger_id="premise-recovery",
        obligations=[root, target, descendant, alternate],
    )
    critic = _premise_suspicion_text()
    suspicion = extract_premise_suspicions(
        critic,
        {"ROOT-A"},
    )["ROOT-A"]
    review = decide_premise_review(
        parse_premise_audit(_audit_text(), "ROOT-A", "audit-1"),
        parse_premise_defense(
            _defense_text(),
            "ROOT-A",
            "defense-1",
        ),
        project_root=Path(__file__).resolve().parents[3],
        suspicion=suspicion,
    )
    assert apply_critic_verdicts(
        ledger,
        critic,
        "br_premise",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": review},
    ) == {"ROOT-A": "DISPROVED"}
    assert target.invalidation_kind == "PREMISE_INVALIDATED"
    assert descendant.status == "QUARANTINED"
    assert descendant.quarantine_root_id == "ROOT-A"
    assert descendant.quarantine_run_id == "br_premise"
    assert alternate.status == "UNRESOLVED"
    assert ledger.backjump_target_id == "ROOT"
    assert len(ledger.no_go_lessons) == 1
    assert ledger.no_go_lessons[0].claim_hash == suspicion.claim_hash
    assert ledger.no_go_lessons[0].auditor_run_id == "audit-1"
    assert ledger.no_go_lessons[0].reversible_status == "ACTIVE"
    assert pending_obligations(ledger) == [alternate]
    recovery_prompt = format_proof_ledger(ledger, [alternate])
    assert "For every real x, x squared equals x" in recovery_prompt
    assert "never assume, rename, or propose" in recovery_prompt
    apply_critic_verdicts(
        ledger,
        critic,
        "br_repeat",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": review},
    )
    assert len(ledger.no_go_lessons) == 1
    apply_critic_verdicts(
        ledger,
        """
### ISSUE_VERDICT ROOT-A-1
Status: PROVED
Evidence: This attempted late verdict must not reopen a terminal quarantined descendant under any circumstance.
Missing lemma: none
""",
        "br_late",
        {"ROOT-A-1"},
    )
    assert descendant.status == "QUARANTINED"
    assert descendant.quarantine_run_id == "br_premise"


def test_weak_premise_verdict_does_not_cascade():
    target = ProofObligation("ROOT-A", "Assume positivity.")
    child = ProofObligation("ROOT-A-1", "Apply positivity.", parent_id="ROOT-A")
    ledger = ProofObligationLedger("weak-premise", [target, child])
    critic = """
### ISSUE_VERDICT ROOT-A
Status: DISPROVED
Invalidation: PREMISE
Premise refuted:
Evidence: This evidence is deliberately long enough, but no explicit premise is named for host validation.
Missing lemma: none
"""
    verdicts = apply_critic_verdicts(ledger, critic, "br_weak", {"ROOT-A"})
    assert verdicts == {"ROOT-A": "UNRESOLVED"}
    assert target.invalidation_kind == ""
    assert child.status == "UNRESOLVED"
    assert ledger.no_go_lessons == []
    weak_closure = _premise_suspicion_text().replace(
        "Substitution of the explicit finite value x=2 gives four on the left and two on the right, contradicting universal equality.",
        "too short",
    )
    apply_critic_verdicts(
        ledger,
        weak_closure,
        "br_weak_closure",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": PremiseReview(
            "INCONCLUSIVE",
            False,
            reason="workers unavailable",
        )},
    )
    assert target.status == "UNRESOLVED"
    assert target.invalidation_kind == ""


def test_valid_suspicion_is_temporary_until_independent_upgrade():
    target = ProofObligation("ROOT-A", "For every real x, x squared equals x.")
    child = ProofObligation("ROOT-A-1", "Use equality.", parent_id="ROOT-A")
    ledger = ProofObligationLedger("suspected", [target, child])
    verdicts = apply_critic_verdicts(
        ledger,
        _premise_suspicion_text(),
        "br_suspect",
        {"ROOT-A"},
    )
    assert verdicts == {"ROOT-A": "UNRESOLVED"}
    assert target.invalidation_kind == "PREMISE_SUSPECTED"
    assert target.premise_review_status == "SUSPECTED"
    assert child.status == "UNRESOLVED"
    assert child.temporary_quarantine_root_id == "ROOT-A"
    assert ledger.no_go_lessons == []
    assert pending_obligations(ledger) == [child]


def test_old_direct_premise_input_is_only_audited_suspicion():
    suspicions = extract_premise_suspicions(
        _premise_suspicion_text("PREMISE"),
        {"ROOT-A"},
    )
    assert suspicions["ROOT-A"].evidence_type == "FINITE_COUNTEREXAMPLE"
    ledger = ProofObligationLedger(
        "legacy-transcript",
        [ProofObligation("ROOT-A", "Universal equality.")],
    )
    apply_critic_verdicts(
        ledger,
        _premise_suspicion_text("PREMISE"),
        "legacy-run",
        {"ROOT-A"},
    )
    assert ledger.obligations[0].status == "UNRESOLVED"
    assert ledger.obligations[0].invalidation_kind == "PREMISE_SUSPECTED"
    assert ledger.no_go_lessons == []


def test_audit_and_defense_parsers_and_verified_outcome_matrix(tmp_path):
    audit = parse_premise_audit(_audit_text(), "ROOT-A", "audit-run")
    defense = parse_premise_defense(
        _defense_text(),
        "ROOT-A",
        "defense-run",
    )
    assert audit.status == "CONFIRMED"
    assert defense.status == "NOT_RESCUED"
    suspicion = extract_premise_suspicions(
        _premise_suspicion_text(),
        {"ROOT-A"},
    )["ROOT-A"]
    verified, detail = validate_evidence_artifact(
        audit,
        suspicion=suspicion,
        project_root=tmp_path,
    )
    assert verified and "4.0 == 2.0 as False" in detail
    decision = decide_premise_review(
        audit,
        defense,
        project_root=tmp_path,
        suspicion=suspicion,
    )
    assert decision.status == "PREMISE_INVALIDATED"
    assert decision.verified
    assert decision.auditor_run_id == "audit-run"
    mismatched_audit = PremiseAudit(
        **{
            **audit.__dict__,
            "artifact": {
                **audit.artifact,
                "claim_hash": "0" * 64,
            },
        },
    )
    assert decide_premise_review(
        mismatched_audit,
        defense,
        project_root=tmp_path,
        suspicion=suspicion,
    ).status == "INCONCLUSIVE"
    assert decide_premise_review(
        parse_premise_audit(
            _audit_text(status="NOT_CONFIRMED"),
            "ROOT-A",
        ),
        defense,
        project_root=tmp_path,
    ).status == "NOT_CONFIRMED"
    assert decide_premise_review(
        audit,
        parse_premise_defense(
            _defense_text(status="RESCUED"),
            "ROOT-A",
        ),
        project_root=tmp_path,
    ).status == "RESCUED"
    assert decide_premise_review(
        parse_premise_audit(
            _audit_text(status="INCONCLUSIVE"),
            "ROOT-A",
        ),
        defense,
        project_root=tmp_path,
    ).status == "INCONCLUSIVE"
    assert decide_premise_review(
        parse_premise_audit(
            _audit_text(confidence="0.5"),
            "ROOT-A",
        ),
        defense,
        project_root=tmp_path,
    ).status == "INCONCLUSIVE"
    assert decide_premise_review(
        audit,
        parse_premise_defense(
            _defense_text(status="INCONCLUSIVE"),
            "ROOT-A",
        ),
        project_root=tmp_path,
    ).status == "INCONCLUSIVE"

    symbolic_suspicion = extract_premise_suspicions(
        _premise_suspicion_text().replace(
            "Evidence type: FINITE_COUNTEREXAMPLE",
            "Evidence type: SYMBOLIC_CONTRADICTION",
        ),
        {"ROOT-A"},
    )["ROOT-A"]
    symbolic = PremiseAudit(
        "ROOT-A",
        "CONFIRMED",
        "SYMBOLIC_CONTRADICTION",
        "expanded universal identity",
        0.9,
        {
            "claim_hash": symbolic_suspicion.claim_hash,
            "claim": symbolic_suspicion.claim_schema,
            "witness": {"x": 2},
        },
        "The claimed symbolic identity fails under the exact witness.",
    )
    assert validate_evidence_artifact(
        symbolic,
        suspicion=symbolic_suspicion,
        project_root=tmp_path,
    )[0]


def test_arbitrary_true_arithmetic_cannot_invalidate_unrelated_premise(
    tmp_path,
):
    critic = """
### ISSUE_VERDICT ROOT-A
Status: DISPROVED
Invalidation: PREMISE_SUSPECTED
Premise refuted: An unrelated analytic continuation premise is false.
Evidence type: FINITE_COUNTEREXAMPLE
Evidence artifact: {"claim":{"schema_version":1,"quantifier":"FOR_ALL","variables":["x"],"domain":"INTEGER","lhs":"x","relation":"!=","rhs":"2"}}
Evidence: The model attaches a true arithmetic observation to an unrelated natural-language premise and calls it a counterexample.
Missing lemma: none
"""
    suspicion = extract_premise_suspicions(
        critic,
        {"ROOT-A"},
    )["ROOT-A"]
    audit = PremiseAudit(
        "ROOT-A",
        "CONFIRMED",
        "FINITE_COUNTEREXAMPLE",
        "unrelated arithmetic",
        0.99,
        {
            "claim_hash": suspicion.claim_hash,
            "claim": suspicion.claim_schema,
            "witness": {"x": 1},
        },
        "The attached relation is true, not a counterexample.",
    )
    verified, reason = validate_evidence_artifact(
        audit,
        suspicion=suspicion,
        project_root=tmp_path,
    )
    assert not verified
    assert "as True" in reason
    decision = decide_premise_review(
        audit,
        parse_premise_defense(_defense_text(), "ROOT-A"),
        project_root=tmp_path,
        suspicion=suspicion,
    )
    assert decision.status == "INCONCLUSIVE"
    ledger = ProofObligationLedger(
        "unrelated-arithmetic",
        [ProofObligation("ROOT-A", "Unrelated analytic continuation premise.")],
    )
    apply_critic_verdicts(
        ledger,
        critic,
        "unrelated-run",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": decision},
    )
    assert ledger.obligations[0].invalidation_kind == "APPROACH_FAILED"
    assert ledger.no_go_lessons == []
    tampered = PremiseAudit(
        **{
            **audit.__dict__,
            "artifact": {
                **audit.artifact,
                "claim_hash": "tampered",
            },
        },
    )
    assert not validate_evidence_artifact(
        tampered,
        suspicion=suspicion,
        project_root=tmp_path,
    )[0]
    missing_witness = PremiseAudit(
        **{
            **audit.__dict__,
            "artifact": {
                **audit.artifact,
                "witness": {},
            },
        },
    )
    assert not validate_evidence_artifact(
        missing_witness,
        suspicion=suspicion,
        project_root=tmp_path,
    )[0]
    unknown_variable = critic.replace(
        '"rhs":"2"',
        '"rhs":"y"',
    )
    assert extract_premise_suspicions(
        unknown_variable,
        {"ROOT-A"},
    ) == {}


def test_unverified_theorem_reference_fails_open(tmp_path):
    pinned_text = """
### ISSUE_VERDICT ROOT-A
Status: DISPROVED
Invalidation: PREMISE_SUSPECTED
Premise refuted: A cited theorem contradicts the target premise.
Evidence type: PINNED_THEOREM
Evidence artifact: {"claim":{"reference":"Example Theorem 2.1","assumptions":["exact assumption A"]}}
Evidence: The citation purports to conflict with the premise under the exact listed assumption, but requires registry verification.
Missing lemma: none
"""
    suspicion = extract_premise_suspicions(
        pinned_text,
        {"ROOT-A"},
    )["ROOT-A"]
    audit = PremiseAudit(
        "ROOT-A",
        "CONFIRMED",
        "PINNED_THEOREM",
        "Example Theorem 2.1",
        0.99,
        {"claim_hash": suspicion.claim_hash, "claim": suspicion.claim_schema},
        "The cited result appears relevant.",
    )
    decision = decide_premise_review(
        audit,
        parse_premise_defense(_defense_text(), "ROOT-A"),
        project_root=tmp_path,
        suspicion=suspicion,
    )
    assert decision.status == "INCONCLUSIVE"
    assert not decision.verified
    assert "no trusted local theorem registry" in decision.reason


def test_lean_proof_without_safe_negation_wrapper_fails_open(tmp_path):
    signature_hash = "lean-target-hash"
    lean_text = f"""
### ISSUE_VERDICT ROOT-A
Status: DISPROVED
Invalidation: PREMISE_SUSPECTED
Premise refuted: The formalized target proposition has a constructive negation.
Evidence type: LEAN_PROOF
Evidence artifact: {{"claim":{{"schema_version":1,"contract":"NEGATION_OF_TARGET_SIGNATURE","lean_signature_hash":"{signature_hash}"}}}}
Evidence: A complete Lean proof is proposed against the exact host-recorded target signature, subject to safe wrapper validation.
Missing lemma: none
"""
    suspicion = extract_premise_suspicions(
        lean_text,
        {"ROOT-A"},
        {"ROOT-A": signature_hash},
    )["ROOT-A"]
    audit = PremiseAudit(
        "ROOT-A",
        "CONFIRMED",
        "LEAN_PROOF",
        "generated Lean theorem",
        0.99,
        {
            "claim_hash": suspicion.claim_hash,
            "claim": suspicion.claim_schema,
            "lean_signature_hash": signature_hash,
            "contract": "NEGATION_OF_TARGET_SIGNATURE",
            "source": "theorem contradiction : False := by trivial",
        },
        "Lean artifact supplied.",
        "audit-lean",
    )

    verified, reason = validate_evidence_artifact(
        audit,
        suspicion=suspicion,
        project_root=tmp_path,
    )
    assert not verified
    assert "cannot be safely transformed" in reason
    assert extract_premise_suspicions(
        lean_text,
        {"ROOT-A"},
        {"ROOT-A": "different-host-signature-hash"},
    ) == {}
    tampered = PremiseAudit(
        **{
            **audit.__dict__,
            "artifact": {
                **audit.artifact,
                "lean_signature_hash": "tampered",
            },
        },
    )
    assert not validate_evidence_artifact(
        tampered,
        suspicion=suspicion,
        project_root=tmp_path,
    )[0]
    rejected = validate_lean_proof(
        "theorem incomplete : True := by sorry",
        project_root=tmp_path,
    )
    assert not rejected.ok
    assert rejected.status == "UNSAFE_REJECTED"


def test_complete_minimal_lean_reduction_proof_is_accepted():
    result = validate_lean_proof(
        "theorem certifiedReduction (h : True) : True := by exact h",
        project_root=Path(__file__).resolve().parents[3],
    )
    assert result.ok
    assert result.status == "PROVED"


def test_worker_roles_are_isolated_ordered_and_fail_open(tmp_path):
    suspicion = extract_premise_suspicions(
        _premise_suspicion_text(),
        {"ROOT-A"},
    )["ROOT-A"]
    calls = []

    def runner(role, messages):
        calls.append((role, messages))
        if role == "premise_auditor":
            return _audit_text(), "audit-run"
        assert _audit_text().strip() in messages[-1]["content"]
        assert suspicion.claim_hash in messages[-1]["content"]
        return _defense_text(), "defense-run"

    audit, defense, transcripts = run_isolated_premise_review(
        "prove target",
        suspicion,
        runner,
    )
    assert [role for role, _ in calls] == [
        "premise_auditor",
        "adversarial_proponent",
    ]
    assert calls[0][1] is not calls[1][1]
    assert "COMPLETE ISOLATED AUDITOR OUTPUT" not in calls[0][1][-1]["content"]
    assert audit.run_id == "audit-run"
    assert defense.run_id == "defense-run"
    assert transcripts["auditor"] == _audit_text()

    failed_calls = []

    def failing_runner(role, _messages):
        failed_calls.append(role)
        raise TimeoutError(f"{role} timed out")

    failed_audit, failed_defense, failed_transcripts = (
        run_isolated_premise_review(
            "prove target",
            suspicion,
            failing_runner,
        )
    )
    assert failed_calls == ["premise_auditor", "adversarial_proponent"]
    assert failed_audit is None and failed_defense is None
    assert "EXECUTION FAILED" in failed_transcripts["auditor"]
    failed_decision = decide_premise_review(
        failed_audit,
        failed_defense,
        project_root=tmp_path,
    )
    assert failed_decision.status == "INCONCLUSIVE"
    failed_ledger = ProofObligationLedger(
        "worker-failure",
        [ProofObligation("ROOT-A", "Failed attempted approach.")],
    )
    apply_critic_verdicts(
        failed_ledger,
        _premise_suspicion_text(),
        "failed-review-run",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": failed_decision},
    )
    assert failed_ledger.obligations[0].status == "DISPROVED"
    assert failed_ledger.obligations[0].invalidation_kind == "APPROACH_FAILED"
    assert failed_ledger.no_go_lessons == []


def test_rescued_review_reverses_quarantine_and_no_go():
    root = ProofObligation("ROOT", "Main unresolved reduction.")
    target = ProofObligation("ROOT-A", "Universal equality.", parent_id="ROOT")
    child = ProofObligation("ROOT-A-1", "Dependent lemma.", parent_id="ROOT-A")
    ledger = ProofObligationLedger("reversible", [root, target, child])
    confirmed = PremiseReview(
        "PREMISE_INVALIDATED",
        True,
        0.95,
        "FINITE_COUNTEREXAMPLE",
        "host arithmetic substitution",
        "audit-1",
        "defense-1",
    )
    apply_critic_verdicts(
        ledger,
        _premise_suspicion_text(),
        "br_confirm",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": confirmed},
    )
    assert child.status == "QUARANTINED"
    rescued = PremiseReview(
        "RESCUED",
        False,
        0.9,
        "FINITE_COUNTEREXAMPLE",
        "domain check",
        "audit-2",
        "defense-2",
        "The original domain excluded x=2.",
    )
    apply_critic_verdicts(
        ledger,
        _premise_suspicion_text(),
        "br_rescue",
        {"ROOT-A"},
        premise_reviews={"ROOT-A": rescued},
    )
    assert target.status == "DISPROVED"
    assert target.invalidation_kind == "APPROACH_FAILED"
    assert target.premise_review_status == "RESCUED"
    assert target.premise_auditor_run_id == "audit-2"
    assert child.status == "UNRESOLVED"
    assert child.quarantine_reversible_status == "REVERSED"
    assert child.temporary_quarantine_root_id == ""
    assert ledger.no_go_lessons[0].reversible_status == "REVERSED"
    assert ledger.backjump_target_id == ""


def test_all_nonupgrade_reviews_close_only_approach_without_no_go():
    for review_status in ("NOT_CONFIRMED", "RESCUED", "INCONCLUSIVE"):
        root = ProofObligation("ROOT", "Sound parent.")
        target = ProofObligation(
            "ROOT-A",
            "For every real x, x squared equals x.",
            parent_id="ROOT",
        )
        child = ProofObligation(
            "ROOT-A-1",
            "Historical alternate descendant.",
            parent_id="ROOT-A",
        )
        ledger = ProofObligationLedger(
            f"fallback-{review_status}",
            [root, target, child],
        )
        apply_critic_verdicts(
            ledger,
            _premise_suspicion_text(),
            f"run-{review_status}",
            {"ROOT-A"},
            premise_reviews={"ROOT-A": PremiseReview(
                review_status,
                False,
                0.6,
                "FINITE_COUNTEREXAMPLE",
                "independent worker result",
                f"audit-{review_status}",
                f"defense-{review_status}",
                f"{review_status} fallback",
            )},
        )
        assert target.status == "DISPROVED"
        assert target.invalidation_kind == "APPROACH_FAILED"
        assert target.premise_review_status == review_status
        assert target.premise_review_reason == f"{review_status} fallback"
        assert child.status == "UNRESOLVED"
        assert child.temporary_quarantine_root_id == ""
        assert child.quarantine_root_id == ""
        assert ledger.no_go_lessons == []
        assert ledger.backjump_target_id == ""


def test_approach_invalidation_does_not_quarantine_descendants_or_siblings():
    target = ProofObligation("ROOT-A", "Try a contour-shift proof.")
    child = ProofObligation(
        "ROOT-A-1",
        "Try a different contour under the same target.",
        parent_id="ROOT-A",
    )
    sibling = ProofObligation("ROOT-B", "Try a spectral proof.")
    ledger = ProofObligationLedger("approach-only", [target, child, sibling])
    critic = """
### ISSUE_VERDICT ROOT-A
Status: DISPROVED
Evidence: The attempted contour crosses an uncontrolled pole, so this derivation fails even though the target statement may still hold.
Missing lemma: none
"""
    apply_critic_verdicts(ledger, critic, "br_approach", {"ROOT-A"})
    assert target.invalidation_kind == "APPROACH_FAILED"
    assert child.status == "UNRESOLVED"
    assert sibling.status == "UNRESOLVED"
    assert {item.obligation_id for item in pending_obligations(ledger)} == {
        "ROOT-A-1",
        "ROOT-B",
    }


def test_no_go_lesson_rejects_semantically_repeated_child():
    ledger = ProofObligationLedger(
        "no-go",
        [ProofObligation("ROOT", "Find a valid replacement argument.")],
    )
    premise = "For every real x, x squared equals x."
    source = ProofObligation("OLD", premise)
    old_ledger = ProofObligationLedger("old", [source])
    apply_critic_verdicts(
        old_ledger,
        _premise_suspicion_text().replace("ROOT-A", "OLD"),
        "br_old",
        {"OLD"},
        premise_reviews={"OLD": PremiseReview(
            "PREMISE_INVALIDATED",
            True,
            0.95,
            "FINITE_COUNTEREXAMPLE",
            "host arithmetic substitution",
            "audit-old",
            "defense-old",
        )},
    )
    ledger.no_go_lessons = old_ledger.no_go_lessons
    rejections = []
    created = create_child_obligations(
        ledger,
        """
### ISSUE_VERDICT ROOT
Status: UNRESOLVED
Evidence: A replacement argument is still required.
Missing lemma: Prove that for every real x, x squared equals x.
""",
        "br_new",
        {"ROOT"},
        rejections,
        {"ROOT": _valid_lean_signature()},
        certified_only=False,
    )
    assert created == []
    assert "no-go premise" in rejections[0]


def test_v1_ledger_loads_without_recovery_metadata(tmp_path):
    path = tmp_path / "legacy-ledger.json"
    path.write_text(json.dumps({
        "ledger_id": "legacy",
        "obligations": [{
            "obligation_id": "ROOT",
            "statement": "Legacy unresolved statement.",
            "status": "UNRESOLVED",
            "parent_id": "",
        }],
        "no_go_lessons": [{
            "claim_hash": "legacy-hash",
            "refuted_premise": "Legacy premise.",
            "evidence": "Legacy evidence.",
            "source_obligation_id": "ROOT",
            "run_id": "legacy-run",
        }],
        "version": 1,
        "schema_version": 1,
    }))
    ledger = load_proof_ledger(path)
    assert ledger.no_go_lessons[0].confidence == 0.0
    assert ledger.no_go_lessons[0].reversible_status == "ACTIVE"
    assert ledger.backjump_target_id == ""
    assert ledger.obligations[0].invalidation_kind == ""
    assert ledger.obligations[0].decomposition_certificate_hash == ""
    assert ledger.obligations[0].dependency_ids == []
    assert ledger.obligations[0].public_assumptions == []


def test_free_form_missing_lemma_cannot_persist_child():
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[ProofObligation("RH-C2", "Prove zero convergence.")],
    )
    critic = """
### ISSUE_VERDICT RH-C2
**Status:** UNRESOLVED
**Evidence:** The proposed convergence argument does not control zeros on compact subsets.
**Missing lemma:** Prove locally uniform convergence on every compact subset of the critical strip.
"""
    apply_critic_verdicts(ledger, critic, "br_first")
    rejections = []
    created = create_child_obligations(
        ledger,
        critic,
        "br_first",
        {"RH-C2"},
        rejections,
        {"RH-C2": _valid_lean_signature()},
    )
    assert created == []
    assert len(ledger.obligations) == 1
    assert "verified decomposition certificate" in rejections[0]


def test_valid_certified_decomposition_runs_seven_roles_and_persists(tmp_path):
    parent = ProofObligation(
        "ROOT",
        "Establish the global convergence theorem for analytic approximants.",
    )
    ledger = ProofObligationLedger("certified", [parent])
    runner, calls = _certificate_runner()
    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        runner,
        project_root=tmp_path,
        orchestration_id="orch-1",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert [call[0] for call in calls] == [
        "definition_auditor",
        "counterexample_worker",
        "decomposer",
        "formalizer",
        "prover",
        "adversarial_proponent",
        "judge",
    ]
    assert len({id(call[1]) for call in calls}) == 7
    for index, (_role, messages, expected_run_id) in enumerate(calls):
        package = json.loads(messages[-1]["content"])
        assert package["producer_run_id"] == expected_run_id
        if index:
            assert package["upstream_artifact_hashes"]
    packages = {
        role: json.loads(messages[-1]["content"])
        for role, messages, _run_id in calls
    }
    assert set(
        packages["formalizer"]["validated_upstream_artifacts"],
    ) == {"decomposer"}
    assert set(
        packages["prover"]["validated_upstream_artifacts"],
    ) == {"formalizer"}
    assert "definition_auditor" not in (
        packages["prover"]["validated_upstream_artifacts"]
    )
    assert result.verified
    assert ledger.obligations == [parent]
    created = persist_verified_decomposition(
        ledger,
        "ROOT",
        result,
        "run-certified",
    )
    assert len(created) == 1
    assert created[0].obligation_id.startswith("ROOT-")
    assert created[0].decomposition_certificate_hash
    assert created[0].reduction_theorem_status == "PROVED"
    assert created[0].certificate_reversible_status == "ACTIVE"
    assert parent.formal_status == "FORMALIZED"
    manifest = tmp_path / "reviews" / "manifest.json"
    save_decomposition_manifest(manifest, {
        "verified": result.verified,
        "transcripts": result.transcripts,
        "artifact_hashes": result.artifact_hashes,
    })
    assert manifest.stat().st_mode & 0o777 == 0o600


def test_certificate_parser_rejects_tampered_bindings_for_every_role(
    tmp_path,
):
    ledger = ProofObligationLedger(
        "bindings",
        [ProofObligation(
            "ROOT",
            "Establish the global convergence theorem for analytic approximants.",
        )],
    )
    runner, calls = _certificate_runner()
    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        runner,
        project_root=tmp_path,
        orchestration_id="orch-bind",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    headings = [
        "DEFINITION_AUDIT",
        "COUNTEREXAMPLE_REPORT",
        "DECOMPOSITION_PROPOSAL",
        "FORMALIZATION_BUNDLE",
        "PROOF_ATTEMPT",
        "DEFENSE_REPORT",
        "JUDGE_DECISION",
    ]
    for heading, (role, messages, expected_run_id) in zip(headings, calls):
        package = json.loads(messages[-1]["content"])
        upstream = package["upstream_artifact_hashes"]
        artifact, error = parse_certified_artifact(
            result.transcripts[role],
            heading,
            target_obligation_id="ROOT",
            parent_statement_hash="tampered",
            root_goal_hash=package["root_goal_hash"],
            producer_run_id=expected_run_id,
            upstream_artifact_hashes=upstream,
        )
        assert artifact is None
        assert "tampered" in error


def test_certificate_timeout_or_malformed_role_never_mutates_ledger(tmp_path):
    ledger = ProofObligationLedger(
        "fail-open",
        [ProofObligation(
            "ROOT",
            "Establish the global convergence theorem for analytic approximants.",
        )],
    )
    before = asdict(ledger)
    runner, _calls = _certificate_runner()

    def timeout_runner(role, messages, expected_run_id):
        if role == "counterexample_worker":
            raise TimeoutError("worker timeout")
        return runner(role, messages, expected_run_id)

    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        timeout_runner,
        project_root=tmp_path,
        orchestration_id="orch-timeout",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert not result.verified
    assert persist_verified_decomposition(
        ledger,
        "ROOT",
        result,
        "run-timeout",
    ) == []
    assert asdict(ledger) == before

    def oversized_runner(_role, _messages, _expected_run_id):
        raise SemanticUnitTooLarge("formal artifact", 2053, 2052)

    oversized = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        oversized_runner,
        project_root=tmp_path,
        orchestration_id="orch-oversized",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert not oversized.verified
    assert "SEMANTIC_UNIT_TOO_LARGE" in oversized.errors[0]
    assert asdict(ledger) == before


def test_certificate_graph_proof_parent_child_and_judge_gates(tmp_path):
    cases = [
        ({"cycle": True}, "acyclic"),
        ({"proof_source": "theorem reduction (h : True) : True := by sorry"}, "complete reduction proof"),
        ({"child_signature": "theorem badChild : True := by sorry"}, "child L1 signature failed"),
        ({"judge_decision": "REJECT"}, "Judge decision"),
        ({"multi_child": True}, "exactly one child"),
        ({
            "counterexample_case": {
                "evidence_type": "PINNED_THEOREM",
                "reference": "Unsupported Theorem 1",
            },
        }, "no verified evidence"),
        ({
            "counterexample_case": {
                "evidence_type": "FINITE_COUNTEREXAMPLE",
                "evidence_source": "host arithmetic evaluator",
                "claim": {
                    "schema_version": 1,
                    "quantifier": "FOR_ALL",
                    "variables": ["x"],
                    "domain": "REAL",
                    "lhs": "x*x",
                    "relation": "==",
                    "rhs": "x",
                },
                "witness": {"x": 2},
            },
        }, "verified counterexample refutes the parent"),
    ]
    for index, (options, expected_error) in enumerate(cases):
        ledger = ProofObligationLedger(
            f"invalid-{index}",
            [ProofObligation(
                "ROOT",
                "Establish the global convergence theorem for analytic approximants.",
            )],
        )
        runner, _calls = _certificate_runner(**options)
        result = run_certified_decomposition(
            ledger,
            "ROOT",
            "Immutable root goal",
            runner,
            project_root=tmp_path,
            orchestration_id=f"orch-invalid-{index}",
            signature_validator=_fake_signature_validator,
            proof_validator=_fake_proof_validator,
        )
        assert not result.verified
        assert any(expected_error in error for error in result.errors)
        assert len(ledger.obligations) == 1

    existing_source = "theorem boundParent : True := by sorry"
    existing = ProofObligation(
        "ROOT",
        "Establish the global convergence theorem for analytic approximants.",
        formal_status="FORMALIZED",
        lean_signature=existing_source,
        lean_signature_hash=lean_theorem_signature_hash(existing_source),
    )
    ledger = ProofObligationLedger("parent-mismatch", [existing])
    runner, _calls = _certificate_runner(parent_hash_override="wrong-hash")
    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        runner,
        project_root=tmp_path,
        orchestration_id="orch-parent-mismatch",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert not result.verified
    assert any("parent signature" in error for error in result.errors)


def test_judge_cannot_override_failed_host_graph_gate(tmp_path):
    ledger = ProofObligationLedger(
        "judge-host",
        [ProofObligation(
            "ROOT",
            "Establish the global convergence theorem for analytic approximants.",
        )],
    )
    runner, _calls = _certificate_runner(cycle=True, judge_decision="ACCEPT")
    result = run_certified_decomposition(
        ledger,
        "ROOT",
        "Immutable root goal",
        runner,
        project_root=tmp_path,
        orchestration_id="orch-judge",
        signature_validator=_fake_signature_validator,
        proof_validator=_fake_proof_validator,
    )
    assert result.artifacts["judge"].decision == "ACCEPT"
    assert not result.verified
    assert not result.validation["host_gates_passed"]


def test_certified_trigger_skips_proved_and_detects_frontiers():
    unresolved = """
### ISSUE_VERDICT ROOT
Status: UNRESOLVED
Evidence: A precise reduction is still missing from the attempted argument.
Missing lemma: Prove a compact boundary inequality.
"""
    proved = """
### ISSUE_VERDICT ROOT
Status: PROVED
Evidence: This sufficiently detailed complete derivation closes the exact target obligation.
Missing lemma: none
"""
    assert certified_decomposition_requested(unresolved, "", {"ROOT"})
    assert not certified_decomposition_requested(proved, "", {"ROOT"})
    assert certified_decomposition_requested(
        "",
        "### ISSUE_RESPONSE ROOT\nRemaining gap: Define the exact topology.",
        {"ROOT"},
    )


def test_host_generated_autoresearch_verdict_uses_new_child_frontier():
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[
            ProofObligation("RH-C2", "Prove zero convergence."),
            ProofObligation(
                "RH-C2-child",
                "Prove locally uniform convergence on compact subsets.",
                parent_id="RH-C2",
                decomposition_certificate_hash="certificate-hash",
                reduction_theorem_status="PROVED",
            ),
        ],
    )

    class Candidate:
        CANDIDATE_ID = "candidate-v4"
        TARGET_OBLIGATION_ID = "RH-C2"

    verdict = build_autoresearch_verdict(
        Candidate,
        ledger,
        {"RH-C2": "UNRESOLVED"},
        [ledger.obligations[1]],
    )
    assert verdict["outcome"] == "DECOMPOSED"
    assert verdict["created_obligation_ids"] == ["RH-C2-child"]
    ledger.obligations[1].decomposition_certificate_hash = ""
    assert build_autoresearch_verdict(
        Candidate,
        ledger,
        {"RH-C2": "UNRESOLVED"},
        [ledger.obligations[1]],
    )["outcome"] == "INCONCLUSIVE"


def test_critic_leaf_table_cannot_bypass_certificate():
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[
            ProofObligation("RH-C1", "Operator construction."),
            ProofObligation("RH-C2", "Zero convergence."),
        ],
    )
    critic = """
### Leaf Obligation Ledger
| ID | Status | Evidence/Argument | Missing Lemma/Requirement |
| :--- | :--- | :--- | :--- |
| **RH-C2-1** | UNRESOLVED | No compact convergence proof. | Prove locally uniform convergence on compact subsets. |
| **RH-C2-2** | UNRESOLVED | Hurwitz does not cover poles. | Prove a singular-limit zero-counting theorem. |
| **RH-C1-1** | UNRESOLVED | Unrelated operator issue. | Construct a self-adjoint operator. |
"""
    applied = apply_critic_verdicts(
        ledger,
        critic,
        "br_table",
        {"RH-C2"},
        None,
    )
    created = create_child_obligations(
        ledger,
        critic,
        "br_table",
        {"RH-C2"},
        None,
        {
            "RH-C2": [
                _valid_lean_signature("1"),
                _valid_lean_signature("2"),
            ],
        },
    )
    assert applied == {"RH-C2": "UNRESOLVED"}
    assert created == []
    assert ledger.obligations[0].last_run_id == ""
    assert pending_obligations(ledger) == ledger.obligations


def test_corrupted_critic_id_binds_to_single_host_target():
    target = (
        "RH-C2-0ef53a217d-25557e489d-4f025934ee-3110912e68-"
        "763645cd6b-40ef83e052-b80cd1343b"
    )
    corrupted = (
        "RH-C2-0ef53a217d-25557e489d-4f025934ee-3110912e68-"
        "763645cd6b-40ef83e052-b80cd13rum-40ef83e052-b80cd1343b"
    )
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[ProofObligation(target, "Sectorial compatibility.")],
    )
    repairs = []
    verdicts = apply_critic_verdicts(
        ledger,
        f"""
### ISSUE_VERDICT {corrupted}
**Status:** UNRESOLVED
**Evidence:** The local density inference is not justified by the global growth class.
**Missing lemma:** Prove a quantitative sectorial density bound from the indicator function.
""",
        "br_corrupt",
        {target},
        repairs,
    )
    assert verdicts == {target: "UNRESOLVED"}
    assert repairs == [(corrupted, target)]
    assert "local density inference" in ledger.obligations[0].last_evidence


def test_cycle_and_invented_ids_cannot_create_children():
    root = ProofObligation(
        "RH-C2-root",
        'The "Angular-Growth Decoupling Lemma": prove angular separation.',
    )
    target = ProofObligation(
        "RH-C2-root-child",
        'The "Sectorial Compatibility Lemma": prove a sectorial bound.',
        parent_id=root.obligation_id,
    )
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[root, target],
    )
    critic = f"""
### ISSUE_VERDICT {target.obligation_id}
**Status:** UNRESOLVED
**Evidence:** The proposed local density lower bound remains unsupported.
**Missing lemma:** The "Angular-Growth Decoupling Lemma": prove angular separation.

### Leaf Obligation Ledger
| ID | Status | Evidence | Missing Lemma |
| RH-Cty-0-alpha | UNRESOLVED | invented | Prove an invented temporary lemma. |
"""
    rejections = []
    created = create_child_obligations(
        ledger,
        critic,
        "br_cycle",
        {target.obligation_id},
        rejections,
        certified_only=False,
    )
    assert created == []
    assert len(ledger.obligations) == 2
    assert any("existing obligation" in item for item in rejections)


def test_variable_renamed_parent_child_is_retroactively_rejected():
    parent = ProofObligation(
        "RH-C2-gap",
        (
            "Prove that for a fixed genus p there exists a critical density "
            "rho_c such that a zero sequence with density rho > rho_c cannot "
            "converge to a local pole of order m without forcing the global "
            "growth order above p."
        ),
    )
    child = ProofObligation(
        "RH-C2-equivalence",
        (
            "Establish the relationship between local accumulation rate delta "
            "at s0 and the global exponent of convergence lambda: if local "
            "density constructs a singularity of order m, global growth must "
            "satisfy lambda >= function(rho,m)."
        ),
        parent_id=parent.obligation_id,
    )
    grandchild = ProofObligation(
        "RH-C2-saturation",
        (
            "Show local density delta imposes a lower bound on global order "
            "rho through a density-order saturation mapping."
        ),
        parent_id=child.obligation_id,
    )
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[parent, child, grandchild],
    )
    rejected = audit_ledger_semantic_duplicates(ledger)
    assert rejected[0][0:2] == (child.obligation_id, parent.obligation_id)
    assert child.status == "REJECTED_DUPLICATE"
    assert grandchild.status == "REJECTED_DUPLICATE"
    assert pending_obligations(ledger) == [parent]


def test_recover_complete_checkpoint_from_timestamped_log(tmp_path):
    path = tmp_path / "agent.log"
    path.write_text(
        "\n".join((
            "[t] [goal] anchored: prove RH",
            "[t] [inference-start] time=t run=br_good goal=hash",
            "[t] generator> first generator line",
            "[t] second generator line",
            "[t] [allens] Critic Prefill: 100 tokens...",
            "[t] critic> first critic line",
            "[t] second critic line",
            "[t] [premise-suspected] id=ROOT-A",
            "[t] premise_auditor> complete auditor transcript",
            "[t] adversarial_proponent> complete defense transcript",
            "[t] [metrics] run=br_good",
        )),
    )
    recovered = recover_checkpoint_from_log(path, "br_good")
    assert recovered.research_goal == "prove RH"
    assert recovered.previous_generator == (
        "first generator line\nsecond generator line"
    )
    assert recovered.previous_critic == "first critic line\nsecond critic line"
    assert recovered.last_run_id == "br_good"
