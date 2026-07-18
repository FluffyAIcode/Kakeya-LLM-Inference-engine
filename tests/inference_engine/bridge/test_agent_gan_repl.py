import io
import signal
import time
from pathlib import Path

from scripts.agent_gan_repl import (
    PrefillHeartbeat,
    ReplCheckpoint,
    ReplPhase,
    TimestampedTee,
    TokenPrinter,
    _gate_failure,
    _stage,
    _telemetry_request,
    build_critic_messages,
    build_generator_messages,
    install_signal_protection,
    is_runtime_artifact_prompt,
    load_checkpoint,
    parse_repl_command,
    recover_checkpoint_from_log,
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
    }
    generator_a = build_generator_messages("prove RH", **kwargs)
    generator_b = build_generator_messages("prove RH", **kwargs)
    critic_a = build_critic_messages(
        "prove RH",
        "complete generator response",
        steering=kwargs["steering"],
        stop_reason="eos",
        complete=True,
    )
    critic_b = build_critic_messages(
        "prove RH",
        "complete generator response",
        steering=kwargs["steering"],
        stop_reason="eos",
        complete=True,
    )
    assert generator_a == generator_b
    assert critic_a == critic_b
    combined = repr(generator_a + critic_a)
    assert "Internal run" not in combined
    assert "IMMUTABLE RESEARCH GOAL" in combined
    assert "previous complete correction" in combined
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
