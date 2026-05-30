# Kakeya Integrations

Drop-in guides for the most common ways to consume the Kakeya engine
from agent frameworks and chat clients. Every integration speaks the
OpenAI-compatible API exposed by `scripts/serve.py`, so plugging
Kakeya into an existing application is typically a 1–3 line change to
the client config.

The discriminator between Kakeya and other local-OpenAI-compat
servers (`mlx_lm.server`, `llama.cpp` server, Ollama, LM Studio) is
**multi-agent concurrent execution**: Kakeya's scheduler admits
multiple sessions in parallel under `--max-concurrent N`, with
configurable admission policies and per-session lifecycle. Each
integration guide below shows how that capability surfaces in the
respective framework.

## Matrix

| Framework / Client | OpenAI base URL config | Multi-agent supported | Tool calls | Streaming | Guide |
| --- | --- | --- | --- | --- | --- |
| **LangChain** | `ChatOpenAI(base_url=...)` | yes (asyncio.gather) | yes | yes | [langchain.md](langchain.md) |
| **CrewAI** | `LLM(base_url=...)` | yes (Crew with N agents) | yes | yes | [crewai.md](crewai.md) |
| **Microsoft AutoGen** | `OpenAIChatCompletionClient(base_url=...)` | yes (GroupChat) | yes | yes | [autogen.md](autogen.md) |
| **Cursor** | Settings → Override OpenAI Base URL | yes (multiple windows) | yes | yes | [cursor-bridge.md](cursor-bridge.md) |
| **Open WebUI / LM Studio** | OpenAI URL field | one chat at a time | yes | yes | [openwebui.md](openwebui.md) |

## Common server-side setup

All integrations assume a Kakeya server is running. The minimal
multi-agent-ready invocation:

```bash
PYTHONPATH=. python3 scripts/serve.py \
    --backend mlx \
    --verifier-id Qwen/Qwen3-1.7B \
    --max-concurrent 4 \
    --admission-policy queue \
    --queue-max-wait-s 30 \
    --host 127.0.0.1 --port 8000
```

The `--max-concurrent 4` is the load-bearing flag: it tells the
scheduler to admit up to 4 simultaneous sessions. Combined with
`--admission-policy queue`, additional clients block in a fair FIFO
until a slab frees up rather than failing with HTTP 429.

For production deployments add an API key:

```bash
PYTHONPATH=. python3 scripts/serve.py ... \
    --api-key sk-prod-$(openssl rand -hex 16)
```

Every integration guide below uses `sk-test-1` as the API key
placeholder; replace with your real key.

## Compatibility notes that apply to every framework

These hold regardless of which framework you use; documenting once
here saves duplicating in every guide:

- **`temperature` / `top_p` / `stop` accepted but ignored.** Kakeya
  decoder is greedy temperature-0 by design (ADR 0001 §2.2). Setting
  these does not error, but does not change output. If you need
  temperature > 0, this engine is the wrong choice today.
- **`stream: true` works.** SSE format follows the OpenAI spec
  including the literal `data: [DONE]` terminator. OpenAI client
  libraries parse it without modification.
- **`tools` / `function_call` accepted at the schema layer.** The
  underlying verifier (Qwen3) decides whether to emit a tool call.
  No native grammar-constrained sampling is enforced; reliability of
  JSON output depends on the verifier's training.
- **Pool-full → HTTP 429.** Under default `--admission-policy reject`
  a busy server returns `429 Too Many Requests` with an OpenAI-format
  error envelope. Most frameworks retry on 429 by default. Switch to
  `queue` admission policy if you prefer the client to wait silently.
- **`/healthz` always public, no auth required.** Useful for
  liveness probes and load balancer health checks.
- **`/metrics` always public, no auth required.** Prometheus
  exposition format, ready to scrape.

## Why these five frameworks

The five chosen for v0.3.0 cover the practical surface of "running a
local agent or chat assistant in 2026":

- **LangChain**: dominant agent / RAG / chain orchestration library;
  reaches the largest user base via community ecosystem.
- **CrewAI**: most prominent multi-agent collaboration framework;
  showcases multi-agent concurrency directly.
- **AutoGen**: Microsoft's research-focused multi-agent framework;
  often the first stop for people building research agents.
- **Cursor**: dominant AI-assisted IDE; demonstrates Kakeya as a
  local backend for IDE workflows.
- **Open WebUI / LM Studio**: GUI clients many users prefer over
  CLI; demonstrates Kakeya works for non-developer users too.

Adding more integrations (Ollama proxy bridge, vercel/ai-sdk,
Continue.dev, etc.) is welcome but not v0.3.0 critical-path.
