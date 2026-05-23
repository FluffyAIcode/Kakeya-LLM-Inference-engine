"""Streaming chat REPL.

A single-turn or multi-turn chat client over the speculative decoding
engine. Tokens are printed to stdout **as they're committed** via the
``on_token`` callback added in this commit, so:

  * Long answers no longer feel "stuck" — characters appear in real time.
  * The caller sees the partial answer immediately if they Ctrl-C.
  * EOS naturally terminates the response (default ``--max-new-tokens=1024``
    gives the model ample room; anything that needs more than that is a
    rare case worth raising explicitly).

Backend selection (one of the three configurations from
``bench_mlx_speculative.py``):

  ``--backend mlx``  : MLX verifier + MLX sparse proposer (full Apple Silicon
                       path; recommended on Mac, requires Metal)
  ``--backend cpu``  : PyTorch CPU verifier + PyTorch CPU sparse proposer
                       (the Phase B baseline; works anywhere)
  ``--backend mixed``: MLX verifier + PyTorch CPU sparse proposer
                       (cross-backend; useful for diagnosing)

Examples:
    # interactive: type prompts, get streamed answers, blank line to quit
    PYTHONPATH=. python3 scripts/chat.py --backend mlx

    # single-shot with stdin piped in
    echo 'Why is the sky blue?' | PYTHONPATH=. python3 scripts/chat.py \\
        --backend mlx --once

    # specify a longer cap if you genuinely need a multi-paragraph answer
    PYTHONPATH=. python3 scripts/chat.py --backend mlx --max-new-tokens 2048
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

import torch


def _eos_ids(tokenizer):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        ids.append(int(im_end))
    return list(set(ids))


def _build_decoder(backend: str, sink: int, window: int, block_size: int,
                   num_diffusion_steps: int):
    """Construct (decoder, tokenizer) for the chosen backend.

    Verifier-side tokenizer is the source of truth for chat templating
    and EOS id resolution; both verifiers (CPU PyTorch, MLX) load from
    Qwen/Qwen3-1.7B which has the same Qwen3 tokenizer.
    """
    from kv_cache_proposer.proposer import ProposerConfig
    from kv_cache_proposer.speculative import SpeculativeDecoder
    from kv_cache_proposer.verifier import SinkWindowVerifier, VerifierConfig

    proposer_cfg = ProposerConfig(dtype=torch.bfloat16, device="cpu")
    verifier_cfg = VerifierConfig(
        dtype=torch.bfloat16, device="cpu",
        sink_size=sink, window_size=window,
    )

    if backend == "cpu":
        from inference_engine.proposer import SparseLogitsProposer
        proposer = SparseLogitsProposer(proposer_cfg)
        verifier = SinkWindowVerifier(verifier_cfg)
    elif backend == "mlx":
        from inference_engine.backends.mlx.env import probe_environment
        env = probe_environment()
        if not env.is_available:
            print(f"[chat] MLX unavailable: {env.failure_reason}",
                  file=sys.stderr)
            sys.exit(2)
        from inference_engine.backends.mlx.proposer import MLXSparseLogitsProposer
        from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier
        print(f"[chat] env: {env.render()}", file=sys.stderr, flush=True)
        proposer = MLXSparseLogitsProposer(proposer_cfg)
        verifier = MLXSinkWindowVerifier(verifier_cfg)
    elif backend == "mixed":
        from inference_engine.backends.mlx.env import probe_environment
        env = probe_environment()
        if not env.is_available:
            print(f"[chat] MLX unavailable: {env.failure_reason}",
                  file=sys.stderr)
            sys.exit(2)
        from inference_engine.proposer import SparseLogitsProposer
        from inference_engine.backends.mlx.verifier import MLXSinkWindowVerifier
        proposer = SparseLogitsProposer(proposer_cfg)
        verifier = MLXSinkWindowVerifier(verifier_cfg)
    else:  # pragma: no cover - argparse guard already restricts choices
        raise SystemExit(f"unknown backend: {backend}")

    decoder = SpeculativeDecoder(
        proposer=proposer,
        verifier=verifier,
        block_size=block_size,
        num_diffusion_steps=num_diffusion_steps,
    )
    return decoder, verifier.tokenizer


def _stream_response(
    decoder,
    tokenizer,
    history: List[dict],
    max_new_tokens: int,
    eos_set,
    *,
    on_chunk=None,
) -> dict:
    """Generate one assistant turn and stream it to stdout.

    Returns a dict with the full text + timing. ``on_chunk`` (if given)
    fires once per emitted text chunk (i.e. per decoded committed
    token); useful when wrapping this script as a library.
    """
    prompt_ids = tokenizer.apply_chat_template(
        history,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        enable_thinking=False,
    )

    # Streaming detokenizer: decode each new token in context of the
    # prefix so byte-pair merges align with what the user expects to
    # see. We track the running decoded length and print only the
    # delta on each callback.
    decoded_so_far = ""
    emitted_token_ids: List[int] = []
    n_emitted = [0]

    def on_token(tok_id: int) -> bool:
        nonlocal decoded_so_far
        emitted_token_ids.append(int(tok_id))
        full = tokenizer.decode(
            emitted_token_ids, skip_special_tokens=True
        )
        delta = full[len(decoded_so_far):]
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
            if on_chunk is not None:
                on_chunk(delta)
            decoded_so_far = full
        n_emitted[0] += 1
        return False  # never request stop from here; max_new_tokens / EOS handles it

    t0 = time.perf_counter()
    result = decoder.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_set,
        on_token=on_token,
    )
    elapsed = time.perf_counter() - t0
    sys.stdout.write("\n")
    sys.stdout.flush()

    # Diagnose what stopped the generation: EOS, max_new_tokens, or
    # callback-requested stop. The user has been hit by max_new_tokens
    # truncation before; surface it explicitly.
    last_tok = result.output_token_ids[-1] if result.output_token_ids else None
    stopped_on_eos = last_tok in eos_set if last_tok is not None else False
    stopped_on_max = (
        not stopped_on_eos
        and len(result.output_token_ids) >= max_new_tokens
    )
    return {
        "text": tokenizer.decode(
            result.output_token_ids, skip_special_tokens=True
        ),
        "n_tokens": len(result.output_token_ids),
        "wall_time_s": elapsed,
        "tok_per_s": len(result.output_token_ids) / max(elapsed, 1e-9),
        "acceptance_rate": result.acceptance_rate,
        "stopped_on_eos": stopped_on_eos,
        "stopped_on_max_new_tokens": stopped_on_max,
    }


_DEFAULT_SYSTEM = (
    "You are a helpful, concise assistant. Answer the user's question "
    "directly without unnecessary preamble. End your response naturally; "
    "do not pad or repeat."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backend", choices=["mlx", "cpu", "mixed"], default="mlx",
        help="MLX requires Apple Silicon; cpu works anywhere.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="Generation cap. Long answers can need 500-1500.")
    ap.add_argument("--sink-size", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument(
        "--block-size", type=int, default=16,
        help="L; per the param sweep on M4, L=16 K=2 is fastest for the "
             "current proposer (acceptance saturates around 0.07 regardless "
             "of L, so larger blocks amortize fixed overhead better).",
    )
    ap.add_argument(
        "--num-diffusion-steps", type=int, default=2,
        help="K; per the param sweep, K=2 is ~2.3x faster than K=10 with "
             "negligible acceptance change at this proposer's quality. "
             "Bump to 4-8 only if you observe acceptance drop on a "
             "specific prompt.",
    )
    ap.add_argument("--system", default=_DEFAULT_SYSTEM,
                    help="System prompt; default tells model to be concise + EOS naturally.")
    ap.add_argument("--once", action="store_true",
                    help="Read a single prompt from stdin, generate, exit. "
                         "If interactive stdin (no pipe) this is the same as "
                         "running interactively for one turn.")
    args = ap.parse_args()

    print(f"[chat] backend={args.backend}  max_new_tokens={args.max_new_tokens}",
          file=sys.stderr, flush=True)
    print(f"[chat] block_size={args.block_size}  K={args.num_diffusion_steps}  "
          f"sink={args.sink_size}  window={args.window_size}",
          file=sys.stderr, flush=True)
    print("[chat] loading models ...", file=sys.stderr, flush=True)
    decoder, tokenizer = _build_decoder(
        args.backend,
        sink=args.sink_size,
        window=args.window_size,
        block_size=args.block_size,
        num_diffusion_steps=args.num_diffusion_steps,
    )
    eos_set = _eos_ids(tokenizer)
    print(f"[chat] ready. eos_ids={eos_set}", file=sys.stderr, flush=True)

    history: List[dict] = [{"role": "system", "content": args.system}]
    rounds = 0
    while True:
        try:
            if sys.stdin.isatty() and not args.once:
                sys.stderr.write("\nyou> ")
                sys.stderr.flush()
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            print("\n[chat] interrupted", file=sys.stderr)
            break
        if not line:
            break
        prompt = line.strip()
        if not prompt:
            if rounds == 0:
                continue
            break
        history.append({"role": "user", "content": prompt})
        sys.stderr.write("assistant> ")
        sys.stderr.flush()
        info = _stream_response(
            decoder, tokenizer, history,
            args.max_new_tokens, eos_set,
        )
        history.append({"role": "assistant", "content": info["text"]})
        rounds += 1
        # End-of-turn diagnostic: was the response cut by max_new_tokens?
        flag = ""
        if info["stopped_on_max_new_tokens"]:
            flag = "  [WARN: hit max_new_tokens; raise --max-new-tokens for longer answers]"
        elif info["stopped_on_eos"]:
            flag = "  [EOS]"
        print(
            f"[chat] {info['n_tokens']} tokens in {info['wall_time_s']:.2f}s "
            f"= {info['tok_per_s']:.2f} tok/s, acc={info['acceptance_rate']:.3f}{flag}",
            file=sys.stderr, flush=True,
        )
        if args.once:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
