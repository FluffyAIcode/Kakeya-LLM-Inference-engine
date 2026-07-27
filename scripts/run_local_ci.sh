#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-.:sdks/python}"

# This script is the canonical local CI subset. GitHub Actions invokes this
# exact file, then adds Docker, package, proto, TypeScript, and Mac integration
# jobs around it.
"$PYTHON" -m py_compile \
  autoresearch/prefill/architecture_v7.py \
  autoresearch/prefill/architecture_v9.py \
  autoresearch/prefill/atomic_definition.py \
  autoresearch/prefill/creative_decomposition.py \
  autoresearch/prefill/cursor_strategy.py \
  autoresearch/prefill/definition_registry.py \
  autoresearch/prefill/definition_resolution.py \
  autoresearch/prefill/evidence_planner.py \
  autoresearch/prefill/host_compiler.py \
  autoresearch/prefill/lean_gate.py \
  autoresearch/prefill/live_status.py \
  autoresearch/prefill/math_ir.py \
  autoresearch/prefill/model_residency.py \
  autoresearch/prefill/oprover_advisor.py \
  autoresearch/prefill/orchestration_state.py \
  autoresearch/prefill/research_contract.py \
  autoresearch/prefill/root_bootstrap.py \
  autoresearch/prefill/semantic_decompose.py \
  autoresearch/prefill/stepwise_proof.py \
  autoresearch/prefill/strategy_tournament.py \
  autoresearch/prefill/supervisor.py \
  autoresearch/prefill/theorem_cards.py \
  autoresearch/prefill/typed_interface_resolution.py \
  autoresearch/prefill/typed_transport.py \
  scripts/agent_gan_inference_demo.py \
  scripts/agent_gan_repl.py \
  scripts/check_typed_role_contracts.py \
  scripts/bootstrap_riemann_hypothesis_root.py \
  scripts/migrate_cursor_strategy_oprover_advisor_v1.py \
  scripts/migrate_definition_auditor_routing_provenance_v1.py \
  scripts/oprover_production_preflight.py \
  scripts/oprover_residency_smoke.py

"$PYTHON" scripts/check_typed_role_contracts.py
node --check deploy/cloudflare-worker/src/index.js
node --check deploy/cloudflare-worker/src/page.js
node --test deploy/cloudflare-worker/test_execution_state.mjs

lake build

"$PYTHON" -m coverage erase
"$PYTHON" -m coverage run -m pytest \
  tests/inference_engine/server/ \
  tests/inference_engine/memory/ \
  tests/inference_engine/scheduler/ \
  tests/inference_engine/pipeline/ \
  tests/inference_engine/session/ \
  tests/inference_engine/bench/ \
  tests/inference_engine/setup/ \
  tests/inference_engine/bridge/ \
  tests/inference_engine/distributed/ \
  tests/inference_engine/network/ \
  tests/sdk/python/ \
  tests/training/repr_align/ \
  tests/backends/mlx/test_env.py \
  --junitxml=junit.xml \
  -v

COVERAGE_INCLUDE='inference_engine/server/auth.py,inference_engine/server/config.py,inference_engine/server/errors.py,inference_engine/server/grpc_app.py,inference_engine/server/metrics.py,inference_engine/server/schemas.py,inference_engine/server/proto_gen/**/*.py,inference_engine/memory/*,inference_engine/bridge/*,inference_engine/distributed/*,inference_engine/network/*,inference_engine/scheduler/config.py,inference_engine/scheduler/session.py,inference_engine/pipeline/*,inference_engine/session/store.py,inference_engine/setup/*,sdks/python/kakeya/__init__.py,sdks/python/kakeya/errors.py,training/repr_align/*'

"$PYTHON" -m coverage report \
  --include="$COVERAGE_INCLUDE" \
  --fail-under=100
"$PYTHON" -m coverage xml \
  -o coverage.xml \
  --include="$COVERAGE_INCLUDE"
