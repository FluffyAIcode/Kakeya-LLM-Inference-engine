from scripts.agent_gan_repl import TokenPrinter, _stage


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
