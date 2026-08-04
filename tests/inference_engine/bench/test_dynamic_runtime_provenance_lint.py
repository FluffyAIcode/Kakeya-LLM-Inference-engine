from scripts.check_dynamic_runtime_provenance import audit_python


def test_dynamic_provenance_lint_rejects_hardcoded_critic_reuse(tmp_path):
    source = tmp_path / "unsafe.py"
    source.write_text(
        'payload = {"critic_reused": True}\n'
        'print("critic_reused=true")\n'
    )

    findings, _classifications = audit_python(source)

    assert {finding[2] for finding in findings} == {
        "literal true",
        "literal true log/text",
    }


def test_dynamic_provenance_lint_allows_fail_closed_and_typed_transition(
    tmp_path,
):
    source = tmp_path / "safe.py"
    source.write_text(
        'payload = {"critic_reused": False}\n'
        'checkpoint.transition(state, "reason", strategy_reused=True)\n'
    )

    findings, classifications = audit_python(source)

    assert findings == []
    assert classifications["derived_or_fail_closed_reuse"] == 1
