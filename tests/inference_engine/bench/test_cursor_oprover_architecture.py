from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.prefill.cursor_strategy import (
    STRATEGY_PROVIDER_UNAVAILABLE,
    STRATEGY_RUN_FAILED,
    CursorStrategyAdapter,
    StrategyProviderError,
    compile_memo_to_plan_id,
)
from autoresearch.prefill.architecture_v9 import run_architecture_v9_entry
from autoresearch.prefill.orchestration_state import (
    OrchestrationCheckpoint,
    ProofState,
)
from autoresearch.prefill.strategy_tournament import StrategyEvent
from autoresearch.prefill.model_residency import (
    ModelResidencyScheduler,
    ResidencyError,
    ResidencyPhase,
)
from autoresearch.prefill.oprover_advisor import (
    OFFICIAL_REVISION,
    OProverConfig,
    OProverProofAdvisor,
    UnavailableOProofs,
)


class FakeSDK:
    def __init__(self, *, models=("gpt-test",), failures=(), status="finished"):
        self.models = models
        self.failures = list(failures)
        self.status = status
        self.cwds = []
        self.prompts = []

    def list_models(self, api_key):
        return [SimpleNamespace(id=item) for item in self.models]

    def prompt(self, prompt, *, api_key, model_id, cwd):
        self.cwds.append(Path(cwd))
        self.prompts.append(prompt)
        if self.failures:
            raise self.failures.pop(0)
        assert (Path(cwd) / "evidence.json").stat().st_mode & 0o777 == 0o400
        return SimpleNamespace(
            status=self.status,
            result="Use the strongest branch.\nSELECTED_PLAN_ID: PLAN-1",
            agent_id="agent-1",
            id="run-1",
        )


@pytest.mark.parametrize(
    ("key", "model", "classification"),
    [
        ("", "gpt-test", "CONFIG_MISSING_API_KEY"),
        ("key", "", "CONFIG_MISSING_MODEL_ID"),
    ],
)
def test_cursor_strategy_missing_config_fails_closed(key, model, classification):
    with pytest.raises(StrategyProviderError) as caught:
        CursorStrategyAdapter(
            api_key=key, model_id=model, sdk=FakeSDK(),
        ).advise(evidence={}, registered_plan_ids=("PLAN-1",))
    assert caught.value.code == STRATEGY_PROVIDER_UNAVAILABLE
    assert caught.value.classification == classification


def test_cursor_discovers_model_and_disposes_read_only_snapshot():
    sdk = FakeSDK()
    memo, telemetry = CursorStrategyAdapter(
        api_key="key", model_id="gpt-test", sdk=sdk,
    ).advise(evidence={"goal": "safe"}, registered_plan_ids=("PLAN-1",))
    assert compile_memo_to_plan_id(memo, ("PLAN-1",)) == "PLAN-1"
    assert telemetry.run_id == "run-1"
    assert telemetry.memo_hash
    assert not sdk.cwds[0].exists()
    assert "CURSOR_API_KEY" not in sdk.prompts[0]


def test_host_plan_compiler_requires_explicit_registered_selection():
    memo = SimpleNamespace(
        text="Mention PLAN-1.\nSELECTED_PLAN_ID: PLAN-10",
    )
    assert compile_memo_to_plan_id(
        memo, ("PLAN-1", "PLAN-10"),
    ) == "PLAN-10"
    memo.text += "\nSELECTED_PLAN_ID: PLAN-1"
    with pytest.raises(ValueError, match="EXACTLY_ONE"):
        compile_memo_to_plan_id(memo, ("PLAN-1", "PLAN-10"))


def test_cursor_rejects_unavailable_model_and_executed_failure():
    with pytest.raises(StrategyProviderError) as missing:
        CursorStrategyAdapter(
            api_key="key", model_id="missing", sdk=FakeSDK(),
        ).advise(evidence={}, registered_plan_ids=("PLAN-1",))
    assert missing.value.classification == "MODEL_NOT_AVAILABLE"
    with pytest.raises(StrategyProviderError) as failed:
        CursorStrategyAdapter(
            api_key="key", model_id="gpt-test",
            sdk=FakeSDK(status="error"),
        ).advise(evidence={}, registered_plan_ids=("PLAN-1",))
    assert failed.value.code == STRATEGY_RUN_FAILED


def test_cursor_retries_only_retryable_and_honors_retry_after():
    error = RuntimeError("network")
    error.is_retryable = True
    error.retry_after = "3"
    sleeps = []
    adapter = CursorStrategyAdapter(
        api_key="key", model_id="gpt-test",
        sdk=FakeSDK(failures=(error,)), sleeper=sleeps.append,
    )
    adapter.advise(evidence={}, registered_plan_ids=("PLAN-1",))
    assert sleeps == [3.0]


class FakeProcesses:
    def __init__(self):
        self.gemma = True
        self.oprover = False
        self.events = []
        self.allens = True

    def quiesce_and_snapshot(self): self.events.append("snapshot")
    def stop_gemma(self):
        self.events.append("stop_gemma"); self.gemma = False; return 101
    def gemma_is_resident(self): return self.gemma
    def load_oprover(self):
        assert not self.gemma
        self.events.append("load_oprover"); self.oprover = True; return 202
    def oprover_is_resident(self): return self.oprover
    def unload_oprover(self, pid):
        assert pid == 202
        self.events.append("unload_oprover"); self.oprover = False
    def restore_gemma(self):
        assert not self.oprover
        self.events.append("restore_gemma"); self.gemma = True; return 303
    def gemma_healthy(self): return self.gemma
    def allens_healthy(self): return self.allens


def scheduler(tmp_path, pm, headroom=lambda: True):
    return ModelResidencyScheduler(
        state_path=tmp_path / "residency.json",
        process_manager=pm,
        headroom_check=headroom,
        model_id="m-a-p/OProver-8B",
        model_revision=OFFICIAL_REVISION,
        tokenizer_id="m-a-p/OProver-8B",
        gemma_cache_namespace="gemma-local4-v1",
        oprover_cache_namespace="oprover-cd9ffd-q5",
    )


def test_residency_full_transition_and_mutual_exclusion(tmp_path):
    pm = FakeProcesses()
    assert scheduler(tmp_path, pm).run_exclusive(lambda: "advice") == "advice"
    assert pm.events == [
        "snapshot", "stop_gemma", "load_oprover", "unload_oprover",
        "restore_gemma",
    ]
    state = json.loads((tmp_path / "residency.json").read_text())
    assert state["phase"] == ResidencyPhase.GEMMA_SERVING.value
    assert state["active_model"] == "gemma"
    assert state["owner_pid"] == 0
    assert state["oprover_pid"] == 0
    assert not pm.oprover


def test_residency_headroom_and_advice_crash_restore_gemma(tmp_path):
    pm = FakeProcesses()
    with pytest.raises(ResidencyError, match="HEADROOM"):
        scheduler(tmp_path, pm, lambda: False).run_exclusive(lambda: None)
    assert pm.gemma and not pm.oprover
    pm = FakeProcesses()
    with pytest.raises(RuntimeError, match="crash"):
        scheduler(tmp_path / "crash", pm).run_exclusive(
            lambda: (_ for _ in ()).throw(RuntimeError("crash"))
        )
    assert pm.gemma and not pm.oprover


def test_residency_separate_cache_namespace_is_hard_gate(tmp_path):
    with pytest.raises(ValueError, match="CACHE_NAMESPACE"):
        ModelResidencyScheduler(
            state_path=tmp_path / "state", process_manager=FakeProcesses(),
            headroom_check=lambda: True, model_id="x", model_revision="r",
            tokenizer_id="t", gemma_cache_namespace="same",
            oprover_cache_namespace="same",
        )


def test_residency_recovery_is_idempotent_and_redacts_errors(tmp_path):
    pm = FakeProcesses()
    instance = scheduler(tmp_path, pm)
    assert instance.run_exclusive(lambda: "ok") == "ok"
    first = json.loads((tmp_path / "residency.json").read_text())
    assert instance.run_exclusive(lambda: "ok") == "ok"
    second = json.loads((tmp_path / "residency.json").read_text())
    assert second["journal_sequence"] > first["journal_sequence"]
    assert second["owner_pid"] == 0
    with pytest.raises(RuntimeError):
        instance.run_exclusive(
            lambda: (_ for _ in ()).throw(
                RuntimeError("api_key=must-not-survive")
            )
        )
    state = json.loads((tmp_path / "residency.json").read_text())
    assert "must-not-survive" not in state["error_code"]
    assert state["phase"] == ResidencyPhase.GEMMA_SERVING.value


class FakeOProver:
    def generate(self, **kwargs):
        assert kwargs["candidate_count"] == 3
        return ("by exact h", "invalid", "by assumption")


def make_model_bundle(model: Path, *, bits: int = 5) -> str:
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "quantization": {"bits": bits},
    }))
    manifest = json.dumps({
        "repo_id": "m-a-p/OProver-8B",
        "revision": OFFICIAL_REVISION,
    }, sort_keys=True).encode()
    (model / "source-manifest.json").write_bytes(manifest)
    (model / "model.safetensors").write_bytes(b"fixture")
    return "sha256:" + hashlib.sha256(manifest).hexdigest()


def test_oprover_rejects_unelaborated_and_persists_verified_only(tmp_path):
    model = tmp_path / "model"
    manifest_hash = make_model_bundle(model)
    seen_scratch = []

    def verify(candidate, scratch):
        seen_scratch.append(scratch)
        return (candidate != "invalid", ("A_EXACT_H",))

    advisor = OProverProofAdvisor(
        config=OProverConfig(
            model_path=model, candidate_count=3, quantization="q5",
            source_checksum_manifest=manifest_hash,
        ),
        runtime=FakeOProver(),
        artifact_dir=tmp_path / "artifacts",
        lean_verify=verify,
        retrieval=UnavailableOProofs(),
    )
    with pytest.raises(ValueError, match="UNELABORATED"):
        advisor.advise(
            theorem_id="", proposition_hash="", lean_goal="",
            local_context=(),
        )
    advice = advisor.advise(
        theorem_id="Known.id", proposition_hash="a" * 64,
        lean_goal="h : P ⊢ P", local_context=("h : P",),
        theorem_card_refs=("TC-id",),
    )
    assert len(advice) == 2
    assert all(item.action_ids == ("A_EXACT_H",) for item in advice)
    assert all(Path(item.artifact_path).is_file() for item in advice)
    assert all(not path.exists() for path in seen_scratch)
    assert "by exact h" not in Path(advice[0].artifact_path).read_text()


def test_oprover_config_pins_revision_and_quant(tmp_path):
    with pytest.raises(ValueError, match="UNPINNED"):
        OProverConfig(tmp_path, revision="main").validate()
    with pytest.raises(ValueError, match="Q4_OR_Q5"):
        OProverConfig(tmp_path, quantization="fp16").validate()


def test_architecture_v9_blocks_without_cursor_config_and_never_falls_back(
    tmp_path,
):
    checkpoint = OrchestrationCheckpoint(
        state=ProofState.STRATEGY_TOURNAMENT.value,
    )
    result = run_architecture_v9_entry(
        tmp_path / "checkpoint.json",
        checkpoint,
        project_root=Path(__file__).resolve().parents[3],
        target_ref="target",
        parent_obligation_ref="ROOT",
        parent_complexity=10,
        event_type=StrategyEvent.INITIAL_BRANCH,
        event_id="INITIAL_BRANCH:test",
        strategy_adapter=CursorStrategyAdapter(
            api_key="", model_id="", sdk=FakeSDK(),
        ),
    )
    assert result.adapter_status == "INTEGRATION_BLOCKED"
    assert result.strategy_run_status == STRATEGY_PROVIDER_UNAVAILABLE
    assert result.validated_artifacts == {}
    assert "Gemma" not in result.blocked_reason


def test_production_entrypoints_cannot_call_legacy_architecture_or_generator():
    root = Path(__file__).resolve().parents[3]
    supervisor = (root / "autoresearch/prefill/supervisor.py").read_text()
    repl = (root / "scripts/agent_gan_repl.py").read_text()
    assert "run_architecture_v7_entry" not in supervisor
    assert "run_architecture_v7_entry" not in repl
    assert 'role="generator"' not in supervisor
