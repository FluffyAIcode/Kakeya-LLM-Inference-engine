# Kakeya + Open WebUI / LM Studio

GUI clients for local LLMs are how non-developer users interact
with the engine. Both Open WebUI (self-hosted, browser) and
LM Studio (desktop app) treat any OpenAI-compatible URL as a
first-class backend.

## Server-side

```bash
PYTHONPATH=. python3 scripts/serve.py \
    --backend mlx \
    --max-concurrent 2 \
    --admission-policy queue \
    --host 127.0.0.1 --port 8000
```

For GUI clients, `--max-concurrent 2` is usually enough — most GUIs
do not multiplex requests aggressively. If you let multiple
browser tabs talk to one Kakeya, bump it.

## Open WebUI configuration

Open WebUI is the dominant self-hosted ChatGPT-style UI for local
models. Run it via Docker (the official path) and point it at
Kakeya.

```bash
docker run -d -p 3000:8080 \
    -v open-webui:/app/backend/data \
    --name open-webui \
    --restart always \
    ghcr.io/open-webui/open-webui:main
```

In the UI:

1. Sign in / create the first admin user.
2. **Admin Settings** → **Connections** → **OpenAI API**.
3. Set **API Base URL**: `http://host.docker.internal:8000/v1`
   (use `http://127.0.0.1:8000/v1` if running Open WebUI natively
   without Docker).
4. Set **API Key**: whatever you passed to `--api-key`. If you
   didn't pass one, leave this field as-is or paste any non-empty
   string (Open WebUI requires non-empty).
5. Click **Verify Connection**. The model list (just `kakeya-v1`)
   should appear.
6. **Admin Settings** → **Models** → enable `kakeya-v1`.

Now any chat in Open WebUI routes to Kakeya. Streaming, multi-turn,
and chat history all work.

## LM Studio configuration

LM Studio is the desktop alternative. As of LM Studio 0.3.x:

1. Click the **Discover** tab → **Add Custom OpenAI-compatible
   Server**.
2. **Server URL**: `http://127.0.0.1:8000`.
3. **API Key**: from `--api-key`. Optional if not set.
4. **Model**: `kakeya-v1` (matches `--model-id-label`).
5. Save → switch to chat tab → select `kakeya-v1` from the model
   dropdown.

## What works / what doesn't

| Feature in GUI | Kakeya support |
| --- | --- |
| Streaming chat | yes (SSE) |
| Multi-turn conversation history | yes |
| System prompt customization | yes |
| Stop generation mid-stream | yes (lifecycle clean) |
| Sampling parameter sliders (`temperature`, `top_p`) | accepted but no-op (greedy) |
| Function / tool calling UI | accepted via API, not always exposed by GUI |
| Multimodal (image upload) | no — text only |
| Embeddings (for RAG) | no — use external embeddings provider |

## Multi-window scenario

Open WebUI lets you open multiple browser tabs each holding a
separate chat. With `--max-concurrent 4` on the server, four tabs
can fire requests in parallel and Kakeya admits all four. This
demonstrates the multi-tenancy advantage in a non-developer-facing
setting:

```
Tab 1: "Help me understand transformer attention."
Tab 2: "Translate this paragraph to Mandarin."
Tab 3: "Suggest a name for my dog."
Tab 4: "Debug this Python error."
```

All four start streaming responses concurrently. With
`mlx_lm.server` (single-tenant), Tabs 2-4 wait for Tab 1 to
finish.

## Caveats

- **Open WebUI's RAG / web search** features call out to other
  endpoints; route those separately, not through Kakeya.
- **Kakeya does not expose `/v1/embeddings`** as of v0.2.x. Open
  WebUI's "Documents" feature requires embeddings; either use the
  built-in Open WebUI embedding model or point it at a separate
  local embeddings server (`mlx_lm.server` has limited support).
- **LM Studio's "load multiple models" feature** is not relevant —
  Kakeya runs one verifier per process. To run multiple verifiers,
  run multiple Kakeya processes on different ports.
