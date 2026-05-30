# Kakeya + LangChain

LangChain is the most common entry point for agent and RAG
orchestration on local LLMs. Kakeya works as a drop-in OpenAI-
compatible backend.

## Server-side

Run Kakeya as documented in [`README.md`](README.md), tuning
`--max-concurrent` to the number of agents you'll run in parallel:

```bash
PYTHONPATH=. python3 scripts/serve.py \
    --backend mlx \
    --max-concurrent 4 \
    --admission-policy queue \
    --api-key sk-test-1
```

## Client-side

```python
# pip install langchain langchain-openai
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-test-1",
    model="kakeya-v1",   # whatever --model-id-label you set on the server
    timeout=120.0,
)

response = llm.invoke([
    ("system", "You are a careful assistant."),
    ("human",  "Explain speculative decoding in one paragraph."),
])
print(response.content)
```

Streaming:

```python
async for chunk in llm.astream([("human", "Stream a poem.")]):
    print(chunk.content, end="", flush=True)
```

## Multi-agent concurrent execution (the discriminator vs `mlx_lm.server`)

This is what Kakeya does that single-tenant servers do not: three
agents researching three topics in parallel.

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-test-1",
    model="kakeya-v1",
)

async def agent(role_prompt: str, task: str) -> str:
    result = await llm.ainvoke([
        ("system", role_prompt),
        ("human",  task),
    ])
    return result.content

async def main():
    answers = await asyncio.gather(
        agent("You are a Python expert.",
              "Critique this snippet: def foo(x): return x*2"),
        agent("You are a Rust expert.",
              "How do I write a panic-safe drop?"),
        agent("You are a researcher.",
              "Summarize transformer attention in 3 bullets."),
    )
    for i, a in enumerate(answers):
        print(f"--- agent {i} ---\n{a}\n")

asyncio.run(main())
```

With Kakeya at `--max-concurrent 4`, all three calls run in parallel
and the wall-time is approximately `max(per-agent time)` plus
admission-control overhead. Against `mlx_lm.server` (single-tenant),
the same code runs sequentially: wall-time is approximately
`sum(per-agent time)`.

## Tool calling

LangChain's `bind_tools` works as expected:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"Weather in {city}: sunny, 22°C."

llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke([
    ("human", "What's the weather in Tokyo?"),
])
print(response.tool_calls)
```

JSON output reliability depends on the verifier (Qwen3) — Kakeya's
greedy decoding (ADR 0001 §2.2) makes it bit-deterministic, so once
a JSON tool call works on a given prompt, it will always work.

## Caveats

- `temperature` / `top_p` are accepted in `ChatOpenAI(...)` config
  but ignored by the engine (greedy by design).
- LangChain's default retry-on-429 behavior interacts well with
  Kakeya's `--admission-policy reject`. Under `queue` policy, no
  retries are needed.
- For LangGraph multi-agent workflows, the same pattern applies:
  use `asyncio.gather` (or LangGraph's parallel branches) to run
  multiple agents and let Kakeya's scheduler handle admission.
