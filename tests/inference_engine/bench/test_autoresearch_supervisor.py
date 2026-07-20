from autoresearch.prefill.supervisor import (
    append_result,
    best_kept,
    deploy_candidate,
    parse_research_verdict,
    read_results,
    repair_candidate_schema,
    render_candidate,
    should_keep,
    StrategyPrefillHeartbeat,
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
    assert repaired["prefill_compute_chunk_tokens"] == 128
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
        "hypothesis_novel": True,
        "proof_obligations_unresolved": 5,
        "metric_cold_critic_prefill_s": 900,
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


def test_worker_deploy_waits_for_unload_and_readiness(monkeypatch):
    captured = []

    class Result:
        stdout = "--prefill-compute-chunk-tokens 128"

    def fake_run(command, **kwargs):
        captured.append((command, kwargs))
        return Result()

    monkeypatch.setattr(
        "autoresearch.prefill.supervisor._wait_port",
        lambda *_args: None,
    )
    monkeypatch.setattr("subprocess.run", fake_run)
    deploy_candidate("allens", 128)
    remote = captured[0][0][-1]
    assert captured[0][1]["check"] is True
    assert "worker service did not unload" in remote
    assert "launchctl bootstrap" in remote
    assert "nc -G 2 -z 127.0.0.1 53051" in remote
    assert "a[i+1]='128'" in remote


def test_strategy_prefill_heartbeat_reports_delta(monkeypatch, capsys):
    heartbeat = StrategyPrefillHeartbeat(interval_s=0.01)
    heartbeat._baseline = {
        "remote_job_tokens_total": 100,
        "remote_job_tokens_computed": 100,
        "remote_hits": 2,
        "tokens_reused": 20,
    }
    monkeypatch.setattr(
        "autoresearch.prefill.supervisor._json_request",
        lambda _url: {"prefill": {
            "remote_job_tokens_total": 300,
            "remote_job_tokens_computed": 228,
            "remote_hits": 3,
            "tokens_reused": 84,
        }},
    )
    heartbeat._emit()
    output = capsys.readouterr().out
    assert "128/200 tokens (64.0%)" in output
    assert "remote_hits=1 reused=64" in output


def test_supervisor_predeploys_before_real_strategy_proposal():
    source = (
        Path(__file__).resolve().parents[3]
        / "autoresearch"
        / "prefill"
        / "supervisor.py"
    ).read_text()
    body = source[source.index("def run_iteration"):source.index("def main")]
    assert body.index("deploy_candidate(args.worker_ssh, previous_chunk)") < (
        body.index("proposed = propose_candidate")
    )
    assert "phase=predeploy-current" in body
    assert "phase=strategy-proposal real-gemma" in body
    assert "if not gan_completed:" in body
    assert "phase=completed-run-preserved" in body


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
