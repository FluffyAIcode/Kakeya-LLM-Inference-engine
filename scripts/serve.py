"""HTTP server launcher (deprecated shim, post PR-D2 of ADR 0008).

Boots the **deprecated** OpenAI-compatible HTTP shim over a real
verifier. Per ADR 0008 §2.7 the HTTP shim is feature-frozen and
slated for retirement; new integrations should use the gRPC server
(``scripts/start_grpc_runtime_server.py``) instead.

PR-D2 (this revision):
  - The shim no longer wraps a SpeculativeEngine. Each
    /v1/chat/completions request runs as a single-shot session
    against the verifier directly: prefill -> generate -> close.
  - Speculative decoding (proposer + verifier) is NOT exercised on
    the HTTP path. Pure AR. Performance roughly matches
    transformers-vanilla.
  - Every response carries ``Deprecation: true`` and a ``Sunset``
    header.

Usage:
    PYTHONPATH=. python3 scripts/serve.py --backend cpu \\
        --verifier-id Qwen/Qwen3-0.6B
    PYTHONPATH=. python3 scripts/serve.py --backend mlx \\
        --verifier-id mlx-community/Qwen3-1.7B-4bit \\
        --host 0.0.0.0 --port 8000

This script is exempt from unit-test coverage (CLI plumbing around
already-tested library code).
"""

from __future__ import annotations

import argparse
import sys

import torch
import uvicorn

from inference_engine.server.app import create_app
from inference_engine.server.config import ServerConfig
from kv_cache_proposer.verifier import VerifierConfig


def _build_verifier(
    *,
    backend: str,
    verifier_id: str,
    sink_size: int,
    window_size: int,
):
    cfg = VerifierConfig(
        model_id=verifier_id,
        dtype=torch.bfloat16, device="cpu",
        sink_size=sink_size, window_size=window_size,
    )

    if backend == "cpu":
        from kv_cache_proposer.verifier import SinkWindowVerifier
        return SinkWindowVerifier(cfg)
    if backend == "mlx":
        from inference_engine.backends.mlx.env import probe_environment
        env = probe_environment()
        if not env.is_available:
            print(f"[serve] MLX unavailable: {env.failure_reason}",
                  file=sys.stderr)
            sys.exit(2)
        from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier
        return MLXSinkWindowVerifier(cfg)
    raise SystemExit(f"unknown backend: {backend}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["mlx", "cpu", "mixed"], default="mlx")
    ap.add_argument("--verifier-id", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--log-level", default=None)
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    # PR-D2: HTTP shim is pure-AR now. The proposer-related flags
    # (block_size, num_diffusion_steps) are accepted but ignored
    # for backward CLI compatibility; remove in v0.4 once
    # downstream scripts are updated.
    ap.add_argument("--block-size", type=int, default=16,
                    help="Ignored post PR-D2; kept for CLI compat.")
    ap.add_argument("--num-diffusion-steps", type=int, default=2,
                    help="Ignored post PR-D2; kept for CLI compat.")
    ap.add_argument("--model-id-label", default=None,
                    help="OpenAI-API ``model`` field returned by /v1/models. "
                         "Defaults to the verifier id.")
    ap.add_argument("--max-concurrent", type=int, default=None,
                    help="Maximum concurrent inference sessions admitted by "
                         "the scheduler. Defaults to 1 (single-user mode).")
    ap.add_argument("--admission-policy", choices=["reject", "queue"],
                    default=None,
                    help="reject (default) returns HTTP 429 immediately when "
                         "the scheduler is full; queue blocks for up to "
                         "--queue-max-wait-s seconds.")
    ap.add_argument("--queue-max-wait-s", type=float, default=None,
                    help="Only honored under --admission-policy queue. "
                         "0 (default) means wait forever.")
    ap.add_argument("--api-key", action="append", default=None,
                    help="Bearer-token API key required for /v1/* routes. "
                         "Pass multiple times to authorize multiple keys. "
                         "Omit to run without auth (single-user dev mode). "
                         "Alternative: KAKEYA_API_KEYS env var (CSV).")
    args = ap.parse_args()

    from inference_engine.scheduler.config import AdmissionPolicy

    base_config = ServerConfig.from_env()
    config = ServerConfig(
        host=args.host or base_config.host,
        port=args.port or base_config.port,
        default_max_new_tokens=base_config.default_max_new_tokens,
        request_timeout_s=base_config.request_timeout_s,
        model_id_label=(
            args.model_id_label
            or args.verifier_id
            or base_config.model_id_label
        ),
        log_level=args.log_level or base_config.log_level,
        max_concurrent=(
            args.max_concurrent
            if args.max_concurrent is not None
            else base_config.max_concurrent
        ),
        admission_policy=(
            AdmissionPolicy(args.admission_policy)
            if args.admission_policy is not None
            else base_config.admission_policy
        ),
        queue_max_wait_s=(
            args.queue_max_wait_s
            if args.queue_max_wait_s is not None
            else base_config.queue_max_wait_s
        ),
        api_keys=(
            frozenset(args.api_key) if args.api_key
            else base_config.api_keys
        ),
    )

    print(
        f"[serve] DEPRECATED HTTP shim "
        f"backend={args.backend} verifier={args.verifier_id} "
        f"host={config.host} port={config.port}\n"
        f"[serve] migrate to gRPC: "
        f"scripts/start_grpc_runtime_server.py",
        file=sys.stderr, flush=True,
    )
    verifier = _build_verifier(
        backend=args.backend,
        verifier_id=args.verifier_id,
        sink_size=args.sink_size,
        window_size=args.window_size,
    )
    app = create_app(
        verifier,
        config,
        model_id_label=config.model_id_label,
    )
    uvicorn.run(app, host=config.host, port=config.port,
                log_level=config.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
