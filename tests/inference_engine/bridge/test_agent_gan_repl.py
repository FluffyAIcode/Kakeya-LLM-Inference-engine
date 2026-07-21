import io
import json
import signal
import sys
import time
from pathlib import Path

from scripts.agent_gan_repl import (
    PrefillHeartbeat,
    CriticIssueBatch,
    ProofObligation,
    ProofObligationLedger,
    ReplCheckpoint,
    ReplPhase,
    TimestampedTee,
    TokenPrinter,
    _gate_failure,
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
    format_critic_issue_injection,
    format_proof_ledger,
    generator_issue_coverage,
    load_checkpoint,
    load_pending_critic_issues,
    load_proof_ledger,
    pending_obligations,
    parse_repl_command,
    recover_checkpoint_from_log,
    save_critic_issue_batch,
    save_proof_ledger,
    save_checkpoint,
)


class Tokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(96 + token) for token in token_ids)


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


def test_missing_lemma_creates_deduplicated_child_and_selects_leaf():
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
    created = create_child_obligations(
        ledger,
        critic,
        "br_first",
        {"RH-C2"},
    )
    assert len(created) == 1
    assert created[0].parent_id == "RH-C2"
    assert pending_obligations(ledger) == created
    assert create_child_obligations(
        ledger,
        critic,
        "br_repeat",
        {"RH-C2"},
    ) == []


def test_host_generated_autoresearch_verdict_uses_new_child_frontier():
    ledger = ProofObligationLedger(
        ledger_id="rh-ledger",
        obligations=[
            ProofObligation("RH-C2", "Prove zero convergence."),
            ProofObligation(
                "RH-C2-child",
                "Prove locally uniform convergence on compact subsets.",
                parent_id="RH-C2",
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


def test_critic_leaf_table_creates_all_target_children_only():
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
    )
    created = create_child_obligations(
        ledger,
        critic,
        "br_table",
        {"RH-C2"},
    )
    assert applied == {"RH-C2": "UNRESOLVED"}
    assert len(created) == 2
    assert all(item.parent_id == "RH-C2" for item in created)
    assert ledger.obligations[0].last_run_id == ""
    assert pending_obligations(ledger) == [
        ledger.obligations[0],
        *created,
    ]


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
