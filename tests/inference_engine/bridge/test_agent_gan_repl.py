import signal
import time
from pathlib import Path

from scripts.agent_gan_repl import (
    PrefillHeartbeat,
    TokenPrinter,
    _gate_failure,
    _stage,
    _telemetry_request,
    build_critic_messages,
    build_generator_messages,
    install_signal_protection,
)


class Tokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(96 + token) for token in token_ids)


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
    with PrefillHeartbeat("Critic", interval_s=0.01):
        time.sleep(0.025)
    output = capsys.readouterr().out
    assert "Critic Prefill still running" in output


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
            "critic_protocol": "recursive_proof_decomposition_v2",
        },
    )
    assert stage["critic_context_tokens"] == 100
    assert stage["critic_omitted_tokens"] == 0
    assert stage["review_scope"] == "full"
    assert stage["critic_protocol"] == "recursive_proof_decomposition_v2"


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
    generator_a = build_generator_messages("prove RH")
    generator_b = build_generator_messages("prove RH")
    critic_a = build_critic_messages(
        "prove RH",
        "complete generator response",
        stop_reason="eos",
        complete=True,
    )
    critic_b = build_critic_messages(
        "prove RH",
        "complete generator response",
        stop_reason="eos",
        complete=True,
    )
    assert generator_a == generator_b
    assert critic_a == critic_b
    combined = repr(generator_a + critic_a)
    assert "Internal run" not in combined
    assert "open problem" in combined
    assert "recursive adversarial proof analyst" in combined
    assert "Never output a numeric score" in combined
    assert "Decomposition Loop" in combined
    assert "If the response stops at" in combined
    assert "attack that stopping claim" in combined
    assert "Ignore prizes, money, prestige" in combined
    assert "smallest unresolved frontier" in combined
    assert "sample, summarize, simplify" in combined
