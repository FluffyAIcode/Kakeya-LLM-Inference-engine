from autoresearch.prefill.supervisor import (
    append_result,
    best_kept,
    deploy_candidate,
    parse_research_verdict,
    read_results,
    render_candidate,
    should_keep,
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
