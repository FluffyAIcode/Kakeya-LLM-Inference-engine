# Kakeya + Cursor

Cursor is a popular AI-augmented IDE that ships with built-in OpenAI
client support and allows pointing at custom OpenAI-compatible
endpoints. This guide walks through using a locally-running Kakeya
server as Cursor's inference backend so your code edits, chat, and
inline completions run entirely on-device.

## Why route Cursor through Kakeya

- **Privacy**: code never leaves your machine.
- **Cost**: no per-token cloud bill.
- **Speed (with v0.3+ alignment)**: personal LoRA learns your
  codebase patterns; speculative decoding becomes profitable on
  your day-to-day queries.
- **Concurrent windows**: Cursor's "compose" + "chat" + agent
  panel often fire requests simultaneously. Kakeya's scheduler
  admits them in parallel under `--max-concurrent`.

## Server-side

Cursor sometimes opens 3-4 simultaneous connections (chat panel +
inline completions + agent + indexer). Size the pool accordingly:

```bash
PYTHONPATH=. python3 scripts/serve.py \
    --backend mlx \
    --max-concurrent 6 \
    --admission-policy queue \
    --queue-max-wait-s 30 \
    --host 127.0.0.1 --port 8000 \
    --api-key sk-cursor-$(openssl rand -hex 8)
```

Note the host: keep it on `127.0.0.1` if you only want Cursor on
the same machine to use it. Use `0.0.0.0` only if you intend to
share the server across machines on your LAN.

## Cursor configuration

1. Open Cursor → **Settings** (⌘ + ,) → **Cursor Settings** →
   **Models**.
2. Enable **Override OpenAI Base URL**.
3. Paste the URL: `http://127.0.0.1:8000/v1`.
4. Paste the API key from the `--api-key` flag above.
5. Add a custom model name: `kakeya-v1` (must match the server's
   `--model-id-label`, default `kakeya-v1`).
6. Enable the model and toggle **Verify** — Cursor will hit
   `/v1/models` and confirm.

## Test path

After Verify succeeds:

1. Open the chat panel (⌘ + L) and ask any question. The response
   should stream in. Latency should be a few seconds for a one-line
   answer on M4 24 GB.
2. Open multiple chat panels (⌘ + N for new chat) and ask
   different questions in each — they should all respond
   concurrently. Watch the Kakeya server logs to confirm:
   ```
   INFO ... POST /v1/chat/completions ... 200
   INFO ... POST /v1/chat/completions ... 200    # parallel
   INFO ... POST /v1/chat/completions ... 200    # parallel
   ```
3. (After v0.3.0 alignment training) Use Cursor's inline edit
   (⌘ + K) on a Rust or Python project for ~1 hour. Personal LoRA
   accumulates and converges to your codebase patterns. Acceptance
   rate visible at `/metrics` should rise from ~0.10 toward ~0.40+.

## Known limitations

- **No embeddings endpoint yet.** Cursor uses an embeddings model
  for codebase indexing. Kakeya does not expose `/v1/embeddings`
  in v0.2.x; codebase indexing must use a separate provider
  (cloud OpenAI, or a local embeddings server). `mlx_lm.server`
  has limited embeddings support; running both side-by-side on
  different ports works.
- **Sampling params ignored.** Cursor sets `temperature` and
  similar; Kakeya is greedy. Output is deterministic per prompt
  given the same verifier weights.
- **Context length**. v0.2.x default sink+window=68 is tight for
  IDE workflows where Cursor sometimes sends 8–20k token prompts.
  Bump on the server side:
  ```
  --sink-size 32 --window-size 4096
  ```
  This trades some KV memory for context.

## Cursor-side troubleshooting

- **HTTP 429 in Cursor logs**: pool exhausted. Either bump
  `--max-concurrent`, switch to `queue` policy, or close some
  Cursor windows.
- **Verify fails**: check `curl -sS http://127.0.0.1:8000/healthz`
  works and that the API key is correct (`Bearer ...` format).
- **Slow inline completions**: this is expected on v0.2.x without
  alignment training (acceptance ~0.10). v0.3.0 with personal
  alignment should improve this materially.
