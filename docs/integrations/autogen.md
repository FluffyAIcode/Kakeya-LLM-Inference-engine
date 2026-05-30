# Kakeya + Microsoft AutoGen

Microsoft's AutoGen is a research-focused multi-agent framework
where agents communicate through a `GroupChat`. It pairs naturally
with Kakeya's scheduler — multiple `AssistantAgent` instances each
take their turn against one shared backend.

## Server-side

```bash
PYTHONPATH=. python3 scripts/serve.py \
    --backend mlx \
    --max-concurrent 4 \
    --admission-policy queue \
    --api-key sk-test-1
```

## Client-side

AutoGen v0.4+ uses the `autogen-agentchat` + `autogen-ext` packages.

```python
# pip install autogen-agentchat autogen-ext[openai]
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent

# Construct the OpenAI-compat client pointing at Kakeya
client = OpenAIChatCompletionClient(
    model="kakeya-v1",
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-test-1",
    # AutoGen requires model_info to be set when using a custom
    # base_url because it does not know our model's capabilities.
    model_info={
        "function_calling": True,
        "json_output": True,
        "vision": False,
        "family": "qwen3",
    },
)

assistant = AssistantAgent(
    name="assistant",
    model_client=client,
    system_message="You are a thoughtful helper.",
)
```

Single-turn smoke:

```python
import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

async def main():
    response = await assistant.on_messages(
        [TextMessage(content="What is speculative decoding?", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(response.chat_message.content)

asyncio.run(main())
```

## Multi-agent group chat (the discriminator vs `mlx_lm.server`)

A planner + coder + reviewer trio collaborating on a small task.
GroupChat dispatches the next-speaker decision; with three
`AssistantAgent` objects backing onto the same Kakeya server, the
scheduler handles their concurrent admission.

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient(
    model="kakeya-v1",
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-test-1",
    model_info={"function_calling": True, "json_output": True,
                "vision": False, "family": "qwen3"},
)

planner = AssistantAgent(
    "planner", model_client=client,
    system_message="Decompose the user task into 2-3 concrete steps.",
)
coder = AssistantAgent(
    "coder", model_client=client,
    system_message="Write Python implementing the planned steps.",
)
reviewer = AssistantAgent(
    "reviewer", model_client=client,
    system_message="Critique the code; suggest one concrete improvement.",
)

team = RoundRobinGroupChat(
    [planner, coder, reviewer],
    termination_condition=MaxMessageTermination(6),
)

async def main():
    async for msg in team.run_stream(
        task="Write a function that computes Fibonacci numbers."
    ):
        print(f"[{msg.source}] {msg.content[:200]}")

asyncio.run(main())
```

In a `RoundRobinGroupChat` only one agent speaks at a time, so the
multi-tenancy advantage shows up when you run **multiple
GroupChats in parallel** (e.g., to A/B-test different prompts).

For genuine concurrent agent workloads, run independent
`run_stream` invocations in parallel:

```python
async def main():
    tasks = [team.run_stream(task=f"Topic {i}") for i in range(3)]
    # AutoGen's run_stream returns an async iterator; wrap each in a
    # consumer task and gather the results.
    consumers = [asyncio.create_task(_consume(t)) for t in tasks]
    await asyncio.gather(*consumers)
```

## Caveats

- AutoGen v0.4+ requires `model_info` when using non-default
  `base_url`; the snippet above shows the minimal four fields.
- `temperature` and other sampling params are accepted but ignored
  by Kakeya (greedy by design).
- AutoGen's `JSONOutputMode` works because Kakeya passes through
  the `response_format` field; reliability depends on the
  underlying verifier.
- For long-running multi-agent loops (`MaxMessageTermination(50)`),
  Kakeya's sink+window KV keeps memory bounded — this is exactly
  the scenario the architecture was designed for.
