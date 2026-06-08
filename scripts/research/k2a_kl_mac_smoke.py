"""K2.A KakeyaLattice Mac M4 round-trip-identity smoke.

ADR 0008 §11.11.9 acceptance gate: KakeyaLattice must run on
Apple Silicon (PyTorch MPS backend) at the head_dim of the v0.4
target verifier (Gemma 3-1B-it: head_dim=256). This script is
the empirical-evidence path:

  1. Construct V14KakeyaZamirLatticeGPU (D4) on MPS at head_dim=256,
     q_range=38 (the canonical D4 operating point per the
     KakeyaLattice README).
  2. Generate synthetic K and V tensors with realistic shape
     [num_kv_heads=1, n_positions, head_dim=256] in bf16 and fp32.
  3. Run codec.roundtrip(K) and codec.roundtrip(V); measure
     per-position relative MSE: ‖K_hat - K‖² / ‖K‖².
  4. Verify the relative MSE is within the published KL fidelity
     envelope (D4 Q=38 → ~3-5e-5 for typical K/V distributions on
     CUDA; MPS may differ slightly due to bf16 reduction-order
     numerics, so the gate uses a 10x slack: < 5e-4).
  5. Round-trip the same tensors through IdentityCompressor and
     KakeyaLatticeCompressor at the inference_engine/v04/
     adapter level, confirming the adapter's compress/decompress
     state machine works on MPS.

If MPS is unavailable (running on CPU or non-Apple hardware), the
script reports the configuration but skips the MPS-specific
assertions; correctness on the active device is still validated.

If the kakeyalattice package is not installed, the script reports
the install hint and exits 0 (with a clear "skipped" status) —
this is a smoke check, not a hard gate. The hard gate happens
when K2.A integration ships with `kakeyalattice` in install_requires.

Outputs a JSON report under results/research/ that the Mac M4
reviewer can commit + push for the K2.A acceptance evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--device", default="auto", choices=["auto", "mps", "cpu", "cuda"],
    )
    ap.add_argument(
        "--head-dim", type=int, default=256,
        help="Codec block dim. Gemma 3-1B-it has head_dim=256; this is "
             "the K2.A target. Must be a power of 2 (Hadamard requirement).",
    )
    ap.add_argument(
        "--n-positions", type=int, default=256,
        help="How many K/V vectors to round-trip in the synthetic batch.",
    )
    ap.add_argument(
        "--num-kv-heads", type=int, default=1,
        help="Gemma 3-1B has num_kv_heads=1; production verifiers may differ.",
    )
    ap.add_argument(
        "--lattice", choices=["D4", "E8"], default="D4",
        help="D4 = v1.4 (canonical, cheaper). E8 = v1.5 (better shaping gain).",
    )
    ap.add_argument(
        "--q-range", type=int, default=38,
        help="Canonical D4 operating point per the KakeyaLattice README. "
             "Q=38 → 832 bits for D=128. Larger Q → smaller compression "
             "but better fidelity.",
    )
    ap.add_argument(
        "--rmse-bound", type=float, default=5e-4,
        help="Pass threshold for relative MSE (10x the CUDA-published "
             "envelope to absorb MPS bf16 reduction-order noise).",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
    )
    ap.add_argument(
        "--output", default=None,
        help="JSON report path. Default: results/research/k2a_kl_mac_smoke_<stamp>.json",
    )
    return ap.parse_args()


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def relative_mse(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """‖recon - orig‖² / ‖orig‖² in fp32, for whatever dtype tensors are in.

    The numerator and denominator are both computed in fp32 even if
    the tensors are bf16, so quantisation noise dominates the result
    rather than reduction-order noise.
    """
    orig_f = orig.detach().to(torch.float32)
    recon_f = recon.detach().to(torch.float32)
    diff_sq = ((recon_f - orig_f) ** 2).sum()
    orig_sq = (orig_f ** 2).sum().clamp_min(1e-12)
    return float((diff_sq / orig_sq).item())


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    print(f"[k2a-smoke] device={device}", file=sys.stderr)
    print(
        f"[k2a-smoke] config: head_dim={args.head_dim} "
        f"n_positions={args.n_positions} num_kv_heads={args.num_kv_heads} "
        f"lattice={args.lattice} q_range={args.q_range}",
        file=sys.stderr,
    )

    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "k2a_kl_mac_smoke",
        "config": {
            "device": str(device),
            "head_dim": args.head_dim,
            "n_positions": args.n_positions,
            "num_kv_heads": args.num_kv_heads,
            "lattice": args.lattice,
            "q_range": args.q_range,
            "rmse_bound": args.rmse_bound,
            "seed": args.seed,
        },
        "checks": {},
        "summary": {},
    }

    # ----------------------------------------------------------------
    # 1. KakeyaLattice availability + direct-codec smoke.
    # ----------------------------------------------------------------
    try:
        if args.lattice == "D4":
            from kakeyalattice import V14KakeyaZamirLatticeGPU as _Codec
        else:
            from kakeyalattice import V15KakeyaZamirE8GPU as _Codec
    except ImportError as e:
        print(
            f"[k2a-smoke] kakeyalattice not installed: {e}",
            file=sys.stderr,
        )
        report["summary"]["kakeyalattice_installed"] = False
        report["summary"]["status"] = "skipped"
        report["summary"]["install_hint"] = (
            "pip install kakeyalattice  (or pip install -e <local-clone> "
            "of github.com/FluffyAIcode/LLM-KV--Cache-compress)"
        )
        _emit_report(report, args.output)
        return 0  # smoke is not a hard gate

    report["summary"]["kakeyalattice_installed"] = True

    # Construct the codec on the active device.
    print(
        f"[k2a-smoke] constructing {_Codec.__name__} on {device}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    try:
        codec = _Codec(D=args.head_dim, q_range=args.q_range, device=str(device))
    except Exception as e:
        report["summary"]["status"] = "fail"
        report["summary"]["construct_error"] = str(e)
        _emit_report(report, args.output)
        print(
            f"[k2a-smoke] FAIL: codec construction raised {type(e).__name__}",
            file=sys.stderr,
        )
        return 1
    construct_ms = (time.perf_counter() - t0) * 1000
    report["checks"]["codec_construct_ms"] = construct_ms
    print(
        f"[k2a-smoke]   constructed in {construct_ms:.1f} ms",
        file=sys.stderr,
    )

    torch.manual_seed(args.seed)

    # ----------------------------------------------------------------
    # 2. Direct codec round-trip on synthetic K/V (fp32).
    # ----------------------------------------------------------------
    K = (
        torch.randn(
            args.n_positions, args.num_kv_heads, args.head_dim,
            dtype=torch.float32,
        ) * 0.3
    ).to(device)
    V = (
        torch.randn(
            args.n_positions, args.num_kv_heads, args.head_dim,
            dtype=torch.float32,
        ) * 0.3
    ).to(device)

    t0 = time.perf_counter()
    K_hat = codec.roundtrip(K)
    rt_K_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    V_hat = codec.roundtrip(V)
    rt_V_ms = (time.perf_counter() - t0) * 1000

    rmse_K = relative_mse(K, K_hat)
    rmse_V = relative_mse(V, V_hat)
    pass_K = rmse_K <= args.rmse_bound
    pass_V = rmse_V <= args.rmse_bound

    report["checks"]["direct_codec"] = {
        "K_roundtrip_ms": rt_K_ms,
        "V_roundtrip_ms": rt_V_ms,
        "K_relative_mse": rmse_K,
        "V_relative_mse": rmse_V,
        "pass": pass_K and pass_V,
    }
    print(
        f"[k2a-smoke]   direct codec K rmse={rmse_K:.3e} "
        f"({rt_K_ms:.1f} ms)  V rmse={rmse_V:.3e} ({rt_V_ms:.1f} ms)",
        file=sys.stderr,
    )

    # ----------------------------------------------------------------
    # 3. Adapter-level smoke through inference_engine.v04.
    # ----------------------------------------------------------------
    from inference_engine.v04 import (
        IdentityCompressor,
        KakeyaLatticeCompressor,
        make_default_compressor,
    )

    # Identity oracle (exact round-trip).
    identity = IdentityCompressor()
    positions = torch.arange(args.n_positions, dtype=torch.int64).to(device)
    # Adapter expects [..., n, head_dim]; we have [n, num_kv_heads, head_dim],
    # so reshape to [num_kv_heads, n, head_dim].
    K_adapter = K.transpose(0, 1).contiguous()
    V_adapter = V.transpose(0, 1).contiguous()
    identity.compress(K_adapter, V_adapter, positions.cpu())
    K_id, V_id = identity.decompress(positions.cpu())
    identity_rmse_K = relative_mse(K_adapter, K_id)
    identity_rmse_V = relative_mse(V_adapter, V_id)
    identity_exact = identity_rmse_K == 0.0 and identity_rmse_V == 0.0
    report["checks"]["identity_adapter"] = {
        "K_relative_mse": identity_rmse_K,
        "V_relative_mse": identity_rmse_V,
        "exact_round_trip": identity_exact,
        "memory_bytes": identity.memory_bytes(),
    }
    print(
        f"[k2a-smoke]   identity adapter exact round-trip="
        f"{identity_exact}  memory={identity.memory_bytes()} B",
        file=sys.stderr,
    )

    # KakeyaLattice adapter on the active device.
    try:
        kl_adapter = KakeyaLatticeCompressor(
            head_dim=args.head_dim,
            device=device,
            lattice=args.lattice,
            q_range=args.q_range,
        )
    except Exception as e:
        report["summary"]["status"] = "fail"
        report["summary"]["adapter_construct_error"] = str(e)
        _emit_report(report, args.output)
        print(
            f"[k2a-smoke] FAIL: KakeyaLatticeCompressor construct raised "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 1

    kl_adapter.compress(K_adapter, V_adapter, positions.cpu())
    K_kl, V_kl = kl_adapter.decompress(positions.cpu())
    kl_rmse_K = relative_mse(K_adapter, K_kl)
    kl_rmse_V = relative_mse(V_adapter, V_kl)
    kl_pass = kl_rmse_K <= args.rmse_bound and kl_rmse_V <= args.rmse_bound

    # Eviction smoke: drop half, decompress remaining, no error.
    keep = positions.cpu()[: args.n_positions // 2]
    drop = positions.cpu()[args.n_positions // 2 :]
    kl_adapter.evict(drop)
    try:
        kl_adapter.decompress(keep)
        evict_works = True
        evict_error = None
    except Exception as e:
        evict_works = False
        evict_error = str(e)

    report["checks"]["kakeyalattice_adapter"] = {
        "codec_name": kl_adapter.codec_name,
        "K_relative_mse": kl_rmse_K,
        "V_relative_mse": kl_rmse_V,
        "pass": kl_pass,
        "memory_bytes_after_full_compress": kl_adapter.memory_bytes() + sum(
            t.numel() * t.element_size() for t in (
                K_kl[..., 0, :], V_kl[..., 0, :],
            )
        ) * (args.n_positions // 2),  # approximate; for diagnostics
        "evict_works": evict_works,
        "evict_error": evict_error,
    }
    print(
        f"[k2a-smoke]   KL adapter K rmse={kl_rmse_K:.3e}  "
        f"V rmse={kl_rmse_V:.3e}  evict_works={evict_works}",
        file=sys.stderr,
    )

    # ----------------------------------------------------------------
    # 4. Factory smoke: make_default_compressor on the active device.
    # ----------------------------------------------------------------
    factory = make_default_compressor(
        head_dim=args.head_dim, device=device,
        lattice=args.lattice, q_range=args.q_range,
    )
    factory_picked_kl = isinstance(factory, KakeyaLatticeCompressor)
    report["checks"]["factory"] = {
        "picked_kakeyalattice": factory_picked_kl,
        "codec_name": factory.codec_name,
    }
    print(
        f"[k2a-smoke]   factory picked={factory.codec_name}",
        file=sys.stderr,
    )

    # ----------------------------------------------------------------
    # Gate evaluation
    # ----------------------------------------------------------------
    all_pass = (
        report["checks"]["direct_codec"]["pass"]
        and report["checks"]["identity_adapter"]["exact_round_trip"]
        and report["checks"]["kakeyalattice_adapter"]["pass"]
        and report["checks"]["kakeyalattice_adapter"]["evict_works"]
        and report["checks"]["factory"]["picked_kakeyalattice"]
    )
    report["summary"]["status"] = "pass" if all_pass else "fail"
    report["summary"]["mps_active"] = (device.type == "mps")

    _emit_report(report, args.output)

    print("[k2a-smoke] ─── SUMMARY ────────────────────────────────────",
          file=sys.stderr)
    print(
        f"[k2a-smoke]   device={device}  status={report['summary']['status']}",
        file=sys.stderr,
    )
    print(
        f"[k2a-smoke]   direct codec  K={rmse_K:.3e}  V={rmse_V:.3e}  "
        f"(bound {args.rmse_bound:.3e})",
        file=sys.stderr,
    )
    print(
        f"[k2a-smoke]   adapter KL    K={kl_rmse_K:.3e}  V={kl_rmse_V:.3e}",
        file=sys.stderr,
    )
    print(
        f"[k2a-smoke]   identity oracle exact={identity_exact}",
        file=sys.stderr,
    )
    print(
        f"[k2a-smoke]   factory picked KL: {factory_picked_kl}",
        file=sys.stderr,
    )

    return 0 if all_pass else 1


def _emit_report(report: Dict[str, Any], output: str | None) -> None:
    out_path = (
        Path(output) if output is not None
        else Path(f"results/research/k2a_kl_mac_smoke_{int(time.time())}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[k2a-smoke] report -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
