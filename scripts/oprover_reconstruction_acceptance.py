#!/usr/bin/env python3
"""Controlled canary-gated Pass@K OProver reconstruction acceptance.

This is an explicit hardware acceptance tool, not an ordinary CI test.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from autoresearch.prefill.independent_reconstruction import (
    ProviderCandidate,
    ReconstructionStatus,
    build_reconstruction_package,
    run_pass_at_k,
)
from autoresearch.prefill.model_residency import ModelResidencyScheduler
from autoresearch.prefill.oprover_advisor import OFFICIAL_REVISION
from scripts.oprover_residency_smoke import LaunchctlProcessManager


class HttpOProverProvider:
    def __init__(self, *, model_dir: Path, port: int, timeout: int) -> None:
        self.model_dir = model_dir
        self.port = port
        self.timeout = timeout

    def generate(self, *, prompt: str, seed: int, max_tokens: int):
        body = json.dumps({
            "model": str(self.model_dir),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 1,
            "top_p": 0.999,
            "seed": seed,
        }).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        choice = payload["choices"][0]
        message = choice.get("message", {})
        text = str(
            message.get("content") or message.get("reasoning_content") or ""
        )
        usage = payload.get("usage", {})
        return ProviderCandidate(
            text=text,
            stop_reason=str(choice.get("finish_reason", "unknown")),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        )


CANARIES = {
    "small-local-hypothesis": """import Mathlib
theorem canarySmall (P : Prop) (h : P) : P := by
  exact h
""",
    "project-local-dependency": """import Mathlib
def canaryValue : Nat := 7
theorem canaryDependency : canaryValue = 7 := by rfl
theorem canaryLocalDependency : canaryValue = 7 := by
  exact canaryDependency
""",
    "namespace-import": """import Mathlib.Data.Nat.Prime.Basic
namespace Canary.Namespace
theorem importedNamespace (n : Nat) (h : Nat.Prime n) : 2 <= n := by
  exact h.two_le
end Canary.Namespace
""",
    "multi-step": """import Mathlib
theorem canaryMultiStep (a b c : Nat) (hab : a = b) (hbc : b = c) :
    a = c := by
  calc
    a = b := hab
    _ = c := hbc
""",
}

CANARY_TARGETS = {
    "small-local-hypothesis": "canarySmall",
    "project-local-dependency": "canaryLocalDependency",
    "namespace-import": "Canary.Namespace.importedNamespace",
    "multi-step": "canaryMultiStep",
}


def _summary(result) -> dict:
    body = asdict(result)
    for attempt in body["attempts"]:
        # Compiler diagnostics are required, but temporary absolute paths are not.
        attempt["lean_output"] = attempt["lean_output"].replace(
            tempfile.gettempdir(), "<TMP>",
        )
    return body


def _run_queue(args, provider: HttpOProverProvider) -> dict:
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    report = {
        "schema_version": 2,
        "started_at": time.time(),
        "model_id": "m-a-p/OProver-8B",
        "model_revision": OFFICIAL_REVISION,
        "quantization": args.quant,
        "canaries": [],
        "entries": [],
    }
    with tempfile.TemporaryDirectory(prefix="kakeya-oprover-canaries-") as raw:
        canary_root = Path(raw)
        for index, (key, source_text) in enumerate(CANARIES.items()):
            source = canary_root / f"{index}.lean"
            source.write_text(source_text, encoding="utf-8")
            package = build_reconstruction_package(
                source_path=source,
                project_root=canary_root,
                theorem_id=CANARY_TARGETS[key],
                environment_hash=queue["environment_hash"],
            )
            result = run_pass_at_k(
                package=package,
                provider=provider,
                project_root=args.project_root,
                k=args.canary_k,
                base_seed=args.base_seed + index * args.seed_stride,
                max_tokens=args.max_tokens,
            )
            report["canaries"].append({"key": key, **_summary(result)})
            if result.status != ReconstructionStatus.INDEPENDENTLY_VERIFIED.value:
                report["status"] = "CANARY_GATE_FAILED"
                report["finished_at"] = time.time()
                return report

    for index, entry in enumerate(queue["entries"]):
        package = build_reconstruction_package(
            source_path=args.project_root / entry["source_path"],
            project_root=args.project_root,
            theorem_id=entry["target"],
            environment_hash=queue["environment_hash"],
            retrieval_refs=tuple(entry.get("retrieval_refs", ())),
            retrieval_context=tuple(entry.get("retrieval_context", ())),
            prompt_context_chars=args.prompt_context_chars,
        )
        if package.source_hash != entry["source_root_hash"]:
            raise RuntimeError(f"SOURCE_HASH_MISMATCH:{entry['key']}")
        if package.theorem_hash != entry["proposition_hash"]:
            # The legacy queue's proposition hash used a different canonicalizer.
            # Bind both values instead of silently rewriting the retained queue.
            legacy_proposition_hash = entry["proposition_hash"]
        else:
            legacy_proposition_hash = ""
        result = run_pass_at_k(
            package=package,
            provider=provider,
            project_root=args.project_root,
            k=args.k,
            base_seed=args.base_seed + (index + len(CANARIES)) * args.seed_stride,
            max_tokens=args.max_tokens,
        )
        report["entries"].append({
            "key": entry["key"],
            "target": entry["target"],
            "source_path": entry["source_path"],
            "source_hash": package.source_hash,
            "environment_hash": package.environment_hash,
            "theorem_hash": package.theorem_hash,
            "legacy_proposition_hash": legacy_proposition_hash,
            "dependency_prefix_hash": package.dependency_prefix_hash,
            "compilation_context_chars": len(package.preserved_context),
            "prompt_context_chars": len(package.prompt_context),
            "retrieval_refs": package.retrieval_refs,
            **_summary(result),
        })
    report["status"] = "COMPLETED"
    report["finished_at"] = time.time()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--quant", choices=("q4", "q5"), default="q4")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--canary-k", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=20260801)
    parser.add_argument("--seed-stride", type=int, default=1009)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--prompt-context-chars", type=int, default=48_000)
    args = parser.parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.project_root = args.project_root.expanduser().resolve()
    args.queue = args.queue.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.state_path = args.state_path.expanduser().resolve()

    manager = LaunchctlProcessManager(
        model_dir=args.model_dir, port=args.port, quant=args.quant,
    )
    provider = HttpOProverProvider(
        model_dir=args.model_dir, port=args.port, timeout=args.request_timeout,
    )
    scheduler = ModelResidencyScheduler(
        state_path=args.state_path,
        process_manager=manager,
        headroom_check=lambda: not manager.gemma_is_resident(),
        model_id="m-a-p/OProver-8B",
        model_revision=OFFICIAL_REVISION,
        tokenizer_id="m-a-p/OProver-8B",
        gemma_cache_namespace="gemma-local4-v1",
        oprover_cache_namespace=f"oprover-cd9ffd-{args.quant}",
    )
    report = scheduler.run_exclusive(lambda: _run_queue(args, provider))
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
    )
    args.output.chmod(0o600)
    print(json.dumps({
        "status": report["status"],
        "canary_count": len(report["canaries"]),
        "entry_count": len(report["entries"]),
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if report["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
