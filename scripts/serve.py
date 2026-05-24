"""HTTP server launcher (E2).

Boots the speculative-decoding engine and serves it via uvicorn over
the OpenAI-compatible API defined in ``inference_engine.server``.

Usage:
    PYTHONPATH=. python3 scripts/serve.py --backend mlx \\
        --verifier-id Qwen/Qwen3-1.7B
    PYTHONPATH=. python3 scripts/serve.py --backend mlx \\
        --verifier-id mlx-community/Qwen3-1.7B-4bit \\
        --host 0.0.0.0 --port 8000

This script is exempt from unit-test coverage (CLI plumbing around
already-tested library code, same convention as ``run_demo.py`` and
``chat.py``). Its correctness is verified by the system-test PR and
by ad-hoc local invocation.
"""

from __future__ import annotations

import argparse
import sys

import torch
import uvicorn

from inference_engine.server.app import create_app
from inference_engine.server.config import ServerConfig
from inference_engine.server.engine import SpeculativeEngine
from kv_cache_proposer.proposer import ProposerConfig
from kv_cache_proposer.speculative import SpeculativeDecoder
from kv_cache_proposer.verifier import VerifierConfig


def _build_engine(
    *,
    backend: str,
    verifier_id: str,
    sink_size: int,
    window_size: int,
    block_size: int,
    num_diffusion_steps: int,
    model_id_label: str,
) -> SpeculativeEngine:
    proposer_cfg = ProposerConfig(dtype=torch.bfloat16, device="cpu")
    verifier_cfg = VerifierConfig(
        model_id=verifier_id,
        dtype=torch.bfloat16, device="cpu",
        sink_size=sink_size, window_size=window_size,
    )

    if backend == "cpu":
        from inference_engine.proposer import SparseLogitsProposer
        from kv_cache_proposer.verifier import SinkWindowVerifier
        proposer = SparseLogitsProposer(proposer_cfg)
        verifier = SinkWindowVerifier(verifier_cfg)
    elif backend == "mlx":
        from inference_engine.backends.mlx.env import probe_environment
        env = probe_environment()
        if not env.is_available:
            print(f"[serve] MLX unavailable: {env.failure_reason}",
                  file=sys.stderr)
            sys.exit(2)
        from inference_engine.backends.mlx.proposer import MLXSparseLogitsProposer
        from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier
        proposer = MLXSparseLogitsProposer(proposer_cfg)
        verifier = MLXSinkWindowVerifier(verifier_cfg)
    elif backend == "mixed":
        from inference_engine.backends.mlx.env import probe_environment
        env = probe_environment()
        if not env.is_available:
            print(f"[serve] MLX unavailable: {env.failure_reason}",
                  file=sys.stderr)
            sys.exit(2)
        from inference_engine.proposer import SparseLogitsProposer
        from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier
        proposer = SparseLogitsProposer(proposer_cfg)
        verifier = MLXSinkWindowVerifier(verifier_cfg)
    else:
        raise SystemExit(f"unknown backend: {backend}")

    decoder = SpeculativeDecoder(
        proposer=proposer,
        verifier=verifier,
        block_size=block_size,
        num_diffusion_steps=num_diffusion_steps,
    )
    return SpeculativeEngine(
        decoder=decoder,
        tokenizer=verifier.tokenizer,
        model_id_label=model_id_label,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["mlx", "cpu", "mixed"], default="mlx")
    ap.add_argument("--verifier-id", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--log-level", default=None)
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-diffusion-steps", type=int, default=2)
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
        f"[serve] backend={args.backend} verifier={args.verifier_id} "
        f"host={config.host} port={config.port}",
        file=sys.stderr, flush=True,
    )
    engine = _build_engine(
        backend=args.backend,
        verifier_id=args.verifier_id,
        sink_size=args.sink_size,
        window_size=args.window_size,
        block_size=args.block_size,
        num_diffusion_steps=args.num_diffusion_steps,
        model_id_label=config.model_id_label,
    )
    app = create_app(engine, config)
    uvicorn.run(app, host=config.host, port=config.port,
                log_level=config.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
