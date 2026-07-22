import json
import pytest

from autoresearch.prefill.semantic_decompose import (
    SemanticUnitTooLarge,
    admit_token_ids,
    downstream_output_cap,
)
from autoresearch.prefill.supervisor import (
    append_result,
    best_kept,
    build_host_candidate,
    build_strategy_contract,
    build_strategy_prompt,
    build_strategy_research_state,
    check_runtime_health,
    extract_gan_failure_reason,
    infrastructure_failure_fingerprint,
    parse_research_verdict,
    read_results,
    repair_candidate_schema,
    render_candidate,
    should_keep,
    StrategyPrefillHeartbeat,
    StrategyPrefillBudgetExceeded,
    strategy_trigger_reason,
    _extract_json,
    _pending_leaf_ids,
    validate_candidate,
)
from pathlib import Path


def _candidate():
    return {
        "candidate_id": "trial",
        "target_obligation_id": "RH-C1",
        "hypothesis": "Construct and attack one explicit operator.",
        "generator_directive": "Define the operator.",
        "critic_directive": "Falsify the operator.",
        "prefill_compute_chunk_tokens": 256,
        "snapshot_mode": "final_only",
        "max_segment_seconds": 300.0,
        "require_full_context": True,
        "allow_fallback": False,
    }


def test_candidate_render_is_executable_and_strict(tmp_path):
    candidate = _candidate()
    validate_candidate(candidate)
    path = tmp_path / "candidate.py"
    path.write_text(render_candidate(candidate))
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    assert namespace["CANDIDATE_ID"] == "trial"
    assert namespace["PREFILL_COMPUTE_CHUNK_TOKENS"] == 256
    bad = {**candidate, "allow_fallback": True}
    try:
        validate_candidate(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("fallback candidate must be rejected")


def test_infrastructure_failure_fingerprint_is_stable_and_specific():
    failed = {
        "research_outcome": "EVALUATION_FAILED",
        "error": "RuntimeError: GAN benchmark is not completed: failed",
    }
    assert infrastructure_failure_fingerprint(failed)
    assert infrastructure_failure_fingerprint(failed) == (
        infrastructure_failure_fingerprint({
            **failed,
            "error": "  RUNTIMEERROR:  GAN benchmark is not completed: failed ",
        })
    )
    assert infrastructure_failure_fingerprint({
        **failed,
        "error": "different failure",
    }) != infrastructure_failure_fingerprint(failed)
    assert infrastructure_failure_fingerprint({
        **failed,
        "research_outcome": "FALSIFIED",
    }) == ""


def test_gan_failure_reason_preserves_semantic_error():
    output = (
        "[inference-failed] time=now run=br_1 "
        "error=SemanticResponseIncomplete: "
        "SEMANTIC_RESPONSE_INCOMPLETE: Generator stopped before EOS\n"
    )
    assert extract_gan_failure_reason(output) == (
        "SemanticResponseIncomplete: SEMANTIC_RESPONSE_INCOMPLETE: "
        "Generator stopped before EOS"
    )
    assert extract_gan_failure_reason("no structured failure") == ""
    assert infrastructure_failure_fingerprint({
        "research_outcome": "EVALUATION_FAILED",
        "error": "",
    }) == ""


def test_strategy_schema_repair_prefers_current_branch_leaf():
    current = _candidate()
    ledger = {"obligations": [
        {
            "obligation_id": "RH-C1",
            "statement": "Operator construction.",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
        {
            "obligation_id": "RH-C1-child",
            "statement": "Prove the operator is self-adjoint.",
            "status": "UNRESOLVED",
            "parent_id": "RH-C1",
        },
        {
            "obligation_id": "RH-C2",
            "statement": "Zero convergence.",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
    ]}
    repaired, fields = repair_candidate_schema(
        {
            "candidate_id": "trial-child",
            "hypothesis": "The proposed domain yields a symmetric operator.",
        },
        current=current,
        ledger=ledger,
    )
    assert repaired["target_obligation_id"] == "RH-C1-child"
    assert repaired["prefill_compute_chunk_tokens"] == 256
    assert "RH-C1-child" in repaired["generator_directive"]
    assert "strictly smaller" in repaired["critic_directive"]
    assert set(fields) == {
        "target_obligation_id",
        "generator_directive",
        "critic_directive",
        "prefill_compute_chunk_tokens",
    }


def test_strategy_schema_repair_accepts_uppercase_and_alias_keys():
    repaired, fields = repair_candidate_schema(
        {
            "CANDIDATE_ID": "alias-trial",
            "TARGET": "RH-C2",
            "HYPOTHESIS": "Test convergence.",
            "GENERATOR_PROMPT": "Construct the approximation.",
            "CRITIC_PROMPT": "Falsify the approximation.",
            "CHUNK_TOKENS": "128",
        },
        current=_candidate(),
        ledger={"obligations": [{
            "obligation_id": "RH-C2",
            "statement": "Zero convergence.",
            "status": "UNRESOLVED",
            "parent_id": "",
        }]},
    )
    validate_candidate({
        **repaired,
        "snapshot_mode": "final_only",
        "require_full_context": True,
        "allow_fallback": False,
    })
    assert repaired["candidate_id"] == "alias-trial"
    assert repaired["prefill_compute_chunk_tokens"] == 256
    assert set(fields) == {
        "candidate_id",
        "target_obligation_id",
        "hypothesis",
        "generator_directive",
        "critic_directive",
        "prefill_compute_chunk_tokens",
    }


def test_strategy_schema_repair_flattens_nested_hypothesis_and_plan():
    repaired, fields = repair_candidate_schema(
        {
            "candidate_id": "candidate-v2-c2-sub-01",
            "hypothesis": {
                "statement": "Construct regularized analytic continuations.",
                "target_obligation": "RH-C2-child",
            },
            "plan": {
                "steps": [
                    "Define a regularization kernel.",
                    "Attempt to prove compact convergence.",
                ],
            },
        },
        current={**_candidate(), "target_obligation_id": "RH-C2"},
        ledger={"obligations": [
            {
                "obligation_id": "RH-C2",
                "statement": "Zero convergence.",
                "status": "UNRESOLVED",
                "parent_id": "",
            },
            {
                "obligation_id": "RH-C2-child",
                "statement": "Prove regularized compact convergence.",
                "status": "UNRESOLVED",
                "parent_id": "RH-C2",
            },
        ]},
    )
    assert repaired["hypothesis"] == (
        "Construct regularized analytic continuations."
    )
    assert repaired["target_obligation_id"] == "RH-C2-child"
    assert "Step 1: Define a regularization kernel." in (
        repaired["generator_directive"]
    )
    assert isinstance(repaired["hypothesis"], str)
    assert {
        "hypothesis",
        "target_obligation_id",
        "generator_directive",
        "critic_directive",
        "prefill_compute_chunk_tokens",
    }.issubset(set(fields))


def test_strategy_parser_converts_prose_plan_and_ignores_latex_braces():
    output = r"""
The next step is to address the pending leaf: **RH-C2-wrong**.

### Plan for Next Run
1. **Objective**: Formulate the mathematical framework for the Zero-Exclusion Lemma.
2. **Mathematical Strategy**:
   * Investigate $\tilde{F}_N(s)$ using Rouché's Theorem.
   * Determine conditions for $Z(\tilde{F}_N,D) \to P(F,D)$.
3. **Constraint**: Do not assume RH.

**Targeting Leaf**: `RH-C2-correct`
"""
    candidate = _extract_json(output)
    assert candidate["strategy_parse_mode"] == "prose"
    assert candidate["target_obligation_id"] == "RH-C2-correct"
    assert "Zero-Exclusion Lemma" in candidate["hypothesis"]
    assert len(candidate["plan"]["steps"]) >= 2


def test_strategy_parser_accepts_python_literal_candidate():
    candidate = _extract_json(
        "```python\n{'candidate_id': 'trial', 'hypothesis': 'test'}\n```",
    )
    assert candidate["candidate_id"] == "trial"
    assert candidate["strategy_parse_mode"] == "python-literal"


def test_strategy_repairs_invalid_json_latex_escapes():
    candidate = _extract_json(
        r'''```json
{"candidate_id":"trial","hypothesis":"sequence \{z_n\} has density \rho"}
```''',
    )
    assert candidate["hypothesis"] == r"sequence \{z_n\} has density \rho"
    assert candidate["strategy_parse_mode"] == "json-escape-repaired"


def test_keep_requires_novel_mathematical_advancement():
    baseline = {
        "proof_obligations_unresolved": "5",
        "metric_cold_critic_prefill_s": "500",
    }
    assert should_keep({
        "accepted": True,
        "research_outcome": "SUPPORTED",
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 4,
        "metric_cold_critic_prefill_s": 900,
    }, baseline)
    assert should_keep({
        "accepted": True,
        "research_outcome": "FALSIFIED",
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 5,
        "metric_cold_critic_prefill_s": 900,
    }, baseline)
    assert should_keep({
        "accepted": True,
        "research_outcome": "DECOMPOSED",
        "created_obligation_ids": ["RH-C1-child"],
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 5,
        "metric_cold_critic_prefill_s": 900,
    }, baseline)
    assert not should_keep({
        "accepted": True,
        "research_outcome": "DECOMPOSED",
        "created_obligation_ids": [],
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 5,
        "metric_cold_critic_prefill_s": 1,
    }, baseline)
    assert not should_keep({
        "accepted": True,
        "research_outcome": "SUPPORTED",
        "created_obligation_ids": [],
        "hypothesis_novel": True,
        "proof_obligations_total": 1,
        "proof_obligations_covered": 0,
        "proof_obligations_unresolved": 4,
        "metric_cold_critic_prefill_s": 1,
    }, baseline)
    assert not should_keep({
        "accepted": True,
        "research_outcome": "INCONCLUSIVE",
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 5,
        "metric_cold_critic_prefill_s": 1,
    }, baseline)
    assert not should_keep({
        "accepted": False,
        "research_outcome": "SUPPORTED",
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 0,
        "metric_cold_critic_prefill_s": 1,
    }, baseline)


def test_parse_research_verdict_uses_last_complete_critic_block():
    output = """
critic> ### AUTORESEARCH_VERDICT
Candidate ID: candidate-v3
Outcome: FALSIFIED
Evidence: The proposed positivity implication fails for the explicit test function at n=7.
New frontier: Characterize the admissible test functions for which the implication remains valid.
"""
    verdict = parse_research_verdict(output, "candidate-v3")
    assert verdict["outcome"] == "FALSIFIED"
    assert "admissible test functions" in verdict["new_frontier"]


def test_parse_host_generated_research_verdict_event():
    output = (
        '[autoresearch-verdict] {"candidate_id":"candidate-v4",'
        '"outcome":"DECOMPOSED",'
        '"evidence":"The Critic isolated a concrete missing convergence lemma.",'
        '"new_frontier":"Prove locally uniform convergence on compact subsets."}'
    )
    verdict = parse_research_verdict(output, "candidate-v4")
    assert verdict["outcome"] == "DECOMPOSED"


def test_pending_leaf_ids_excludes_unresolved_parents():
    ledger = {"obligations": [
        {"obligation_id": "RH-C1", "status": "UNRESOLVED", "parent_id": ""},
        {
            "obligation_id": "RH-C1-child",
            "status": "UNRESOLVED",
            "parent_id": "RH-C1",
        },
        {"obligation_id": "RH-C2", "status": "UNRESOLVED", "parent_id": ""},
    ]}
    assert _pending_leaf_ids(ledger) == ["RH-C1-child", "RH-C2"]


def test_pending_leaf_ids_excludes_stale_premise_descendants():
    ledger = {"obligations": [
        {
            "obligation_id": "ROOT",
            "status": "DISPROVED",
            "invalidation_kind": "PREMISE",
            "parent_id": "",
        },
        {
            "obligation_id": "ROOT-stale",
            "status": "UNRESOLVED",
            "parent_id": "ROOT",
        },
        {
            "obligation_id": "SOUND",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
    ]}
    assert _pending_leaf_ids(ledger) == ["SOUND"]


def test_host_candidate_targets_deepest_current_branch_leaf():
    current = {**_candidate(), "target_obligation_id": "RH-C2"}
    ledger = {"obligations": [
        {
            "obligation_id": "RH-C1",
            "statement": "Unrelated leaf.",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
        {
            "obligation_id": "RH-C2",
            "statement": "Root convergence claim.",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
        {
            "obligation_id": "RH-C2-child",
            "statement": "Construct a compact convergence counterexample.",
            "status": "UNRESOLVED",
            "parent_id": "RH-C2",
            "last_evidence": "The previous approximation failed at a pole.",
        },
    ]}
    candidate = build_host_candidate(current, ledger)
    assert candidate["target_obligation_id"] == "RH-C2-child"
    assert candidate["hypothesis"] == (
        "Construct a compact convergence counterexample."
    )
    assert "do not rename" in candidate["generator_directive"]
    assert candidate["prefill_compute_chunk_tokens"] == 256


def test_host_candidate_rolls_back_to_nearest_valid_ancestor():
    rejected_target = "RH-C2-duplicate-child"
    current = {
        **_candidate(),
        "target_obligation_id": rejected_target,
    }
    ledger = {"obligations": [
        {
            "obligation_id": "RH-C1",
            "statement": "Unrelated leaf.",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
        {
            "obligation_id": "RH-C2-gap",
            "statement": "Falsify the density-singularity implication.",
            "status": "UNRESOLVED",
            "parent_id": "",
        },
        {
            "obligation_id": rejected_target,
            "statement": "Renamed density-singularity implication.",
            "status": "REJECTED_DUPLICATE",
            "parent_id": "RH-C2-gap",
        },
    ]}
    candidate = build_host_candidate(current, ledger)
    assert candidate["target_obligation_id"] == "RH-C2-gap"


def test_host_candidate_uses_recorded_premise_backjump_target():
    current = {**_candidate(), "target_obligation_id": "ROOT-bad"}
    ledger = {
        "backjump_target_id": "ROOT",
        "no_go_lessons": [{
            "refuted_premise": "Every admissible kernel is positive.",
            "source_obligation_id": "ROOT-bad",
        }],
        "obligations": [
            {
                "obligation_id": "ROOT",
                "statement": "Find a sound replacement reduction.",
                "status": "UNRESOLVED",
                "parent_id": "",
            },
            {
                "obligation_id": "ROOT-bad",
                "statement": "Use universal kernel positivity.",
                "status": "DISPROVED",
                "invalidation_kind": "PREMISE",
                "parent_id": "ROOT",
            },
            {
                "obligation_id": "UNRELATED",
                "statement": "Unrelated unresolved branch.",
                "status": "UNRESOLVED",
                "parent_id": "",
            },
        ],
    }
    candidate = build_host_candidate(current, ledger)
    assert candidate["target_obligation_id"] == "ROOT"
    assert "Every admissible kernel is positive" in (
        candidate["generator_directive"]
    )


def test_strategy_is_triggered_only_by_events(tmp_path):
    progress = {
        "kept": "True",
        "research_outcome": "DECOMPOSED",
        "created_obligation_ids": '["child"]',
    }
    inconclusive = {
        "kept": "False",
        "research_outcome": "INCONCLUSIVE",
        "created_obligation_ids": "[]",
    }
    assert strategy_trigger_reason(
        [progress, inconclusive, inconclusive],
        stagnation_rounds=3,
    ) == ""
    assert strategy_trigger_reason(
        [progress, inconclusive, inconclusive, inconclusive],
        stagnation_rounds=3,
    ) == "stagnation-3"
    assert strategy_trigger_reason(
        [{"kept": "True", "research_outcome": "FALSIFIED"}],
        stagnation_rounds=3,
    ) == "branch-falsified"
    assert strategy_trigger_reason(
        [{
            "kept": "True",
            "research_outcome": "FALSIFIED",
            "invalidation_kind": "PREMISE_INVALIDATED",
        }],
        stagnation_rounds=3,
    ) == "premise-invalidated"
    assert strategy_trigger_reason(
        [{
            "kept": "False",
            "research_outcome": "INCONCLUSIVE",
            "invalidation_kind": "PREMISE_SUSPECTED",
        }],
        stagnation_rounds=3,
    ) == ""
    assert strategy_trigger_reason(
        [{
            "kept": "True",
            "research_outcome": "INCONCLUSIVE",
            "invalidation_kind": "APPROACH_FAILED",
        }],
        stagnation_rounds=3,
    ) == "branch-falsified"
    trigger = tmp_path / "request_strategy"
    trigger.write_text("replan")
    assert strategy_trigger_reason(
        [],
        stagnation_rounds=3,
        trigger_file=trigger,
    ) == "manual-trigger-file"


def test_strategy_budget_error_preserves_exact_admission_counts():
    error = StrategyPrefillBudgetExceeded(11935, 8448)
    assert error.token_count == 11935
    assert error.max_tokens == 8448
    assert "without truncation: 11935 > 8448" in str(error)


def test_semantic_admission_and_dynamic_output_reserve_never_slice():
    token_ids = list(range(2053))
    with pytest.raises(
        SemanticUnitTooLarge,
        match="SEMANTIC_UNIT_TOO_LARGE",
    ):
        admit_token_ids(
            "indivisible target",
            token_ids,
            configured_prefill_tokens=6144,
            max_retained_tokens=2052,
        )
    assert token_ids == list(range(2053))
    assert downstream_output_cap(
        max_retained_tokens=2052,
        fixed_downstream_tokens=1700,
        configured_output_tokens=1000,
        control_reserve_tokens=32,
    ) == 320


def test_strategy_candidate_rejects_multi_step_plan():
    with pytest.raises(ValueError, match="exactly one step"):
        validate_candidate({
            **_candidate(),
            "plan": {"steps": ["first", "second"]},
        })


def test_strategy_state_keeps_exact_one_step_interface_only():
    ledger = {"obligations": [
        {
            "obligation_id": "RH-C1",
            "statement": "unrelated root",
            "status": "UNRESOLVED",
            "parent_id": "",
            "last_evidence": "unrelated evidence",
        },
        {
            "obligation_id": "RH-C2",
            "statement": "exact root statement",
            "status": "UNRESOLVED",
            "parent_id": "",
            "last_evidence": "exact root evidence",
        },
        {
            "obligation_id": "RH-C2-child",
            "statement": "exact child statement",
            "status": "UNRESOLVED",
            "parent_id": "RH-C2",
            "last_evidence": "exact child evidence",
        },
    ]}
    results = (
        "candidate_id\ttarget_obligation_id\tresearch_outcome\t"
        "research_evidence\tnew_frontier\tkept\terror\t"
        "hypothesis_sha256\n"
        "c1\tRH-C1\tINCONCLUSIVE\tunrelated result\tx\tFalse\t\th1\n"
        "c2\tRH-C2-child\tDECOMPOSED\texact result\tfrontier\tTrue\t\th2\n"
    )
    state = build_strategy_research_state(
        current={**_candidate(), "target_obligation_id": "RH-C2"},
        ledger=ledger,
        results_text=results,
    )
    interface = state["proof_step_interface"]
    assert interface["target_obligation_id"] == "RH-C2-child"
    assert interface["target_statement"] == "exact child statement"
    assert interface["current_target_evidence"] == "exact child evidence"
    assert interface["parent_interface"]["statement_hash"]
    serialized = str(state)
    assert "exact child evidence" in serialized
    assert "exact root statement" not in serialized
    assert "exact root evidence" not in serialized
    assert "exact result" not in serialized
    assert "unrelated root" not in serialized
    assert "unrelated result" not in serialized


def test_strategy_state_carries_lossless_no_go_lessons():
    premise = "Every admissible kernel is positive."
    evidence = (
        "The explicit admissible polynomial kernel changes sign while "
        "satisfying every required boundary condition."
    )
    ledger = {
        "backjump_target_id": "ROOT",
        "no_go_lessons": [{
            "claim_hash": "abc123",
            "refuted_premise": premise,
            "evidence": evidence,
            "source_obligation_id": "ROOT-bad",
            "run_id": "br_refute",
        }],
        "obligations": [{
            "obligation_id": "ROOT",
            "statement": "Find a replacement reduction.",
            "status": "UNRESOLVED",
            "parent_id": "",
        }, {
            "obligation_id": "ROOT-bad",
            "statement": premise,
            "status": "DISPROVED",
            "invalidation_kind": "PREMISE",
            "parent_id": "ROOT",
        }],
    }
    state = build_strategy_research_state(
        current={**_candidate(), "target_obligation_id": "ROOT-bad"},
        ledger=ledger,
        results_text="",
    )
    lessons = state["proof_step_interface"]["active_no_go_lessons"]
    assert lessons[0]["refuted_premise"] == premise
    assert lessons[0]["evidence"] == evidence


def test_strategy_state_keeps_target_unit_and_hashes_history():
    shared = "Complete exact evidence that must appear once without truncation."
    ledger = {"obligations": [
        {
            "obligation_id": "RH-C2",
            "statement": "Root statement.",
            "status": "UNRESOLVED",
            "parent_id": "",
            "last_evidence": shared,
        },
    ]}
    results = (
        "candidate_id\ttarget_obligation_id\tresearch_outcome\t"
        "research_evidence\tnew_frontier\tkept\terror\t"
        "hypothesis_sha256\n"
        f"c2\tRH-C2\tINCONCLUSIVE\t{shared}\tRoot statement."
        "\tFalse\t\th2\n"
    )
    state = build_strategy_research_state(
        current={**_candidate(), "target_obligation_id": "RH-C2"},
        ledger=ledger,
        results_text=results,
    )
    serialized = str(state)
    assert serialized.count(shared) == 1
    assert serialized.count("Root statement.") == 1
    interface = state["proof_step_interface"]
    assert interface["current_target_evidence"] == shared
    archive = interface["archive_manifest"]
    assert archive["record_count"] == 1
    assert len(archive["ordered_records_sha256"]) == 64


def test_strategy_state_exposes_latest_semantic_failure_for_smaller_step():
    ledger = {"obligations": [{
        "obligation_id": "RH-C2",
        "statement": "Exact current target.",
        "status": "UNRESOLVED",
        "parent_id": "",
    }]}
    error = (
        "SemanticResponseIncomplete: SEMANTIC_RESPONSE_INCOMPLETE: "
        "Generator stopped before EOS after 320 tokens"
    )
    results = (
        "target_obligation_id\thypothesis_sha256\tresearch_outcome\terror\n"
        f"RH-C2\thash-1\tEVALUATION_FAILED\t{error}\n"
    )
    state = build_strategy_research_state(
        current={**_candidate(), "target_obligation_id": "RH-C2"},
        ledger=ledger,
        results_text=results,
    )
    latest = state["proof_step_interface"]["archive_manifest"][
        "latest_failure"
    ]
    assert latest["kind"] == "SEMANTIC_RESPONSE_INCOMPLETE"
    assert latest["role"] == "Generator"
    assert latest["response_tokens"] == 320
    assert "error" not in latest


def test_strategy_prompt_bounded_interface_for_11_nodes_and_28_runs():
    obligations = []
    for index in range(11):
        obligations.append({
            "obligation_id": f"RH-N{index}",
            "statement": (
                f"Complete exact ancestry statement {index}: "
                + "mathematical condition " * 12
            ),
            "status": "UNRESOLVED",
            "parent_id": f"RH-N{index - 1}" if index else "",
            "last_evidence": (
                f"ancestry evidence {index} " + "detail " * 80
            ),
        })
    header = (
        "timestamp\texperiment_id\trun_id\tcandidate_id\t"
        "target_obligation_id\thypothesis_sha256\tresearch_outcome\t"
        "invalidation_kind\tresearch_evidence\tnew_frontier\tkept\terror\n"
    )
    rows = []
    for index in range(28):
        critical = index % 7 == 0
        rows.append("\t".join((
            str(index),
            f"experiment-{index}",
            f"run-{index}",
            f"candidate-{index}",
            "RH-N10",
            f"hypothesis-{index % 4}",
            "FALSIFIED" if critical else "INCONCLUSIVE",
            "",
            f"evidence-{index}-" + "exact mathematical evidence " * 45,
            f"frontier-{index}-" + "exact frontier statement " * 30,
            "True" if critical else "False",
            "" if index % 5 else "worker timeout fingerprint",
        )))
    results = header + "\n".join(rows) + "\n"
    ledger = {"obligations": obligations}
    current = {**_candidate(), "target_obligation_id": "RH-N0"}
    state = build_strategy_research_state(
        current=current,
        ledger=ledger,
        results_text=results,
    )
    interface = state["proof_step_interface"]
    assert interface["target_obligation_id"] == "RH-N10"
    assert "Complete exact ancestry statement 10" in (
        interface["target_statement"]
    )
    assert "Complete exact ancestry statement 9" not in str(state)
    archive = interface["archive_manifest"]
    assert archive["record_count"] == 28
    assert len(archive["ordered_records_sha256"]) == 64
    serialized = json.dumps(state, ensure_ascii=False)
    assert "evidence-27-" not in serialized
    assert "evidence-1-" not in serialized
    program = (
        Path(__file__).resolve().parents[3]
        / "autoresearch"
        / "prefill"
        / "program.md"
    ).read_text()
    contract = build_strategy_contract(program)
    assert "target_obligation_id must equal TARGET_LEAF_ID" in contract
    assert "no fallback" in contract
    prompt = build_strategy_prompt(
        program=program,
        current=current,
        results_text=results,
        ledger=ledger,
    )
    assert "\n\nPROGRAM:\n" not in prompt
    proxy_tokens = (len(prompt.encode("utf-8")) + 3) // 4
    assert proxy_tokens <= 2052


def test_results_are_append_only_and_best_is_selected(tmp_path):
    path = tmp_path / "results.tsv"
    common = {
        "timestamp": 1,
        "experiment_id": "e1",
        "run_id": "br_1",
        "candidate_id": "c1",
        "target_obligation_id": "RH-C1",
        "constraints_pass": True,
        "accepted": True,
        "kept": True,
        "metric_cold_critic_prefill_s": 500,
        "baseline_metric_s": "",
        "proof_obligations_total": 5,
        "proof_obligations_covered": 5,
        "proof_obligations_unresolved": 5,
        "compute_chunk_tokens": 256,
        "candidate_sha256": "a",
        "report_path": "/tmp/r1.json",
    }
    append_result(path, common)
    append_result(path, {
        **common,
        "timestamp": 2,
        "experiment_id": "e2",
        "candidate_id": "c2",
        "metric_cold_critic_prefill_s": 450,
    })
    rows = read_results(path)
    assert len(rows) == 2
    assert best_kept(rows)["candidate_id"] == "c2"


def test_append_result_migrates_legacy_results_header(tmp_path):
    path = tmp_path / "results.tsv"
    path.write_text("timestamp\tcandidate_id\n1\tlegacy\n")
    append_result(path, {
        "timestamp": 2,
        "candidate_id": "new",
        "hypothesis_sha256": "sha",
        "research_outcome": "FALSIFIED",
    })
    rows = read_results(path)
    assert rows[0]["candidate_id"] == "legacy"
    assert rows[0]["research_outcome"] == ""
    assert rows[1]["hypothesis_sha256"] == "sha"


def test_runtime_health_check_is_read_only(monkeypatch):
    ports = []
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor._wait_port",
        lambda host, port: ports.append((host, port)),
    )
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor._json_request",
        lambda _url: {"online_nodes": 1, "kv_hit_rate": 0.75},
    )
    summary = check_runtime_health("169.254.27.104:53051")
    assert summary["kv_hit_rate"] == 0.75
    assert ports == [
        ("169.254.27.104", 53051),
        ("127.0.0.1", 51051),
        ("127.0.0.1", 8090),
    ]


def test_strategy_prefill_heartbeat_reports_delta(monkeypatch, capsys):
    heartbeat = StrategyPrefillHeartbeat(interval_s=0.01)
    heartbeat._baseline = {
        "remote_jobs": 10,
        "remote_job_tokens_total": 300,
        "remote_job_tokens_computed": 300,
        "remote_hits": 2,
        "tokens_reused": 20,
    }
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor._json_request",
        lambda _url: {"prefill": {
            "remote_jobs": 11,
            "remote_job_tokens_total": 300,
            "remote_job_tokens_computed": 128,
            "remote_hits": 3,
            "tokens_reused": 84,
        }},
    )
    heartbeat._emit()
    output = capsys.readouterr().out
    assert "128/300 tokens (42.7%)" in output
    assert "remote_hits=1 reused=64" in output


def test_supervisor_preserves_runtime_and_cache_across_iterations():
    source = (
        Path(__file__).resolve().parents[3]
        / "autoresearch"
        / "prefill"
        / "supervisor.py"
    ).read_text()
    body = source[source.index("def run_iteration"):source.index("def main")]
    assert body.index("audit_ledger_semantic_duplicates(") < body.index(
        "previous_ledger = _backup",
    )
    assert body.index("check_runtime_health(") < body.index(
        "strategy_trigger_reason(",
    )
    assert "phase=runtime-health-check" in body
    assert "phase=deterministic-candidate" in body
    assert "mode=gemma trigger=" in body
    assert "except StrategyPrefillBudgetExceeded" in body
    assert "phase=strategy-deferred-budget" in body
    assert "except SemanticResponseIncomplete" in body
    assert "phase=strategy-deferred-semantic" in body
    assert "if not gan_completed:" in body
    assert "phase=completed-run-preserved" in body
    assert "deploy_candidate" not in source
    assert "clear_primary_cache" not in source
    assert "launchctl" not in source
    assert "bootout" not in source
    assert "results_text[-" not in source
    propose_body = source[
        source.index("def propose_candidate"):
        source.index("def check_runtime_health")
    ]
    assert propose_body.index("admit_token_ids(") < propose_body.index(
        "session.append(ids)",
    )
    run_body = source[
        source.index("def run_gan_experiment"):
        source.index("def read_results")
    ]
    assert '"--max-retained-tokens"' in run_body


def test_gan_subprocess_output_is_streamed_not_captured():
    source = (
        Path(__file__).resolve().parents[3]
        / "autoresearch"
        / "prefill"
        / "supervisor.py"
    ).read_text()
    body = source[
        source.index("def run_gan_experiment"):
        source.index("def read_results")
    ]
    assert "subprocess.Popen(" in body
    assert "stderr=subprocess.STDOUT" in body
    assert 'print(line, end="", flush=True)' in body
    assert "capture_output=True" not in body
