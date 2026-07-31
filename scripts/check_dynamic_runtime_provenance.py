#!/usr/bin/env python3
"""Reject high-confidence hardcoded dynamic runtime/provenance claims."""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATHS = (
    ROOT / "autoresearch/prefill",
    ROOT / "inference_engine/network",
    ROOT / "scripts/agent_gan_repl.py",
    ROOT / "scripts/agent_gan_inference_demo.py",
    ROOT / "deploy/cloudflare-worker/src",
)
AUDITED_DYNAMIC_KEYS = frozenset({
    "inference_started", "committed", "status", "phase", "role",
    "architecture_version", "schema_version", "ledger_version",
    "supervisor_pid", "writer_pid", "pid", "model_id", "revision",
    "cache_id", "candidate_count", "progress", "route", "fallback",
    "proof_obligations_total", "proof_obligations_covered",
    "proof_obligations_unresolved",
})
SAFE_TRANSITION_CALLS = frozenset({
    "transition",
    "begin_decomposition_iteration",
    "apply_blocked_exit_event",
})
REUSE_TRUE_TEXT = re.compile(r"\b[a-z][a-z0-9_]*_reused\s*=\s*true\b", re.I)


def _files():
    for configured in PRODUCTION_PATHS:
        if configured.is_file():
            yield configured
        elif configured.exists():
            yield from configured.rglob("*.py")
            yield from configured.rglob("*.js")


def _constant_key(node):
    return node.value if isinstance(node, ast.Constant) else None


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def audit_python(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    findings = []
    classifications = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key_node, value_node in zip(node.keys, node.values):
                key = _constant_key(key_node)
                if not isinstance(key, str):
                    continue
                value = (
                    value_node.value
                    if isinstance(value_node, ast.Constant) else None
                )
                if key.endswith("_reused"):
                    classifications["derived_or_fail_closed_reuse"] += 1
                    if value is True:
                        findings.append((node.lineno, key, "literal true"))
                elif key in AUDITED_DYNAMIC_KEYS and value is not None:
                    if key in {"schema_version", "architecture_version"}:
                        classifications["stable_schema_constant"] += 1
                    elif key == "inference_started" and value is False:
                        classifications["fail_closed_idle_default"] += 1
                    else:
                        classifications["typed_event_or_fixture_literal"] += 1
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value_node = node.value
            value = (
                value_node.value
                if isinstance(value_node, ast.Constant) else None
            )
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            for target in targets:
                name = (
                    target.attr if isinstance(target, ast.Attribute)
                    else target.id if isinstance(target, ast.Name)
                    else ""
                )
                if name.endswith("_reused") and value is True:
                    findings.append((node.lineno, name, "literal true assignment"))
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            for keyword in node.keywords:
                if (
                    keyword.arg
                    and keyword.arg.endswith("_reused")
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    and not (
                        keyword.arg == "strategy_reused"
                        and call_name in SAFE_TRANSITION_CALLS
                    )
                ):
                    findings.append((
                        node.lineno,
                        keyword.arg,
                        "literal true argument",
                    ))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and REUSE_TRUE_TEXT.search(node.value)
        ):
            findings.append((
                node.lineno,
                "*_reused",
                "literal true log/text",
            ))
    return findings, classifications


def audit_javascript(path):
    findings = []
    classifications = Counter()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if REUSE_TRUE_TEXT.search(line):
            findings.append((line_number, "*_reused", "literal true log/text"))
    return findings, classifications


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", action="store_true")
    args = parser.parse_args()
    findings = []
    classifications = Counter()
    files = list(dict.fromkeys(_files()))
    for path in files:
        if path == Path(__file__).resolve():
            continue
        if path.suffix == ".py":
            current, counts = audit_python(path)
        else:
            current, counts = audit_javascript(path)
        classifications.update(counts)
        findings.extend(
            (str(path.relative_to(ROOT)), line, field, reason)
            for line, field, reason in current
        )
    payload = {
        "files_scanned": len(files) - 1,
        "prohibited_findings": len(findings),
        "classifications": dict(sorted(classifications.items())),
        "findings": findings,
    }
    if args.audit_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif findings:
        for path, line, field, reason in findings:
            print(f"{path}:{line}: {field}: {reason}")
    else:
        print(
            "dynamic runtime/provenance lint passed "
            f"({payload['files_scanned']} production files)"
        )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
