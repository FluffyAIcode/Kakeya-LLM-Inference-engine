# Kakeya + CrewAI

CrewAI orchestrates teams of agents that collaborate on complex
tasks. Each agent has a role, a goal, and access to tools; the
crew runs them sequentially or in parallel. Kakeya is a natural
backend because the multi-agent execution model lines up directly
with our scheduler's admission control.

## Server-side

```bash
PYTHONPATH=. python3 scripts/serve.py \
    --backend mlx \
    --max-concurrent 5 \
    --admission-policy queue \
    --queue-max-wait-s 60 \
    --api-key sk-test-1
```

`--max-concurrent 5` covers a typical 3–5-agent crew with one or two
slabs to spare.

## Client-side

```python
# pip install crewai
from crewai import Agent, Task, Crew, LLM

llm = LLM(
    model="openai/kakeya-v1",   # crewai expects the openai/ prefix
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-test-1",
    timeout=120,
)

# Single-agent sanity check
researcher = Agent(
    role="Researcher",
    goal="Surface 3 key facts about a given topic.",
    backstory="You are skeptical and source-driven.",
    llm=llm,
    verbose=True,
)
task = Task(
    description="Research transformer attention.",
    expected_output="Three concise bullets.",
    agent=researcher,
)
crew = Crew(agents=[researcher], tasks=[task])
print(crew.kickoff())
```

## Multi-agent crew (the discriminator vs `mlx_lm.server`)

Three agents collaborating on a code-review task. Each has a
distinct role; CrewAI dispatches them in parallel where the
dependency graph allows.

```python
from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(
    model="openai/kakeya-v1",
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-test-1",
)

# Three specialized agents
analyzer = Agent(role="Static Analyzer",
                 goal="Identify bugs and code smells.",
                 backstory="You think like a Rust borrow checker.",
                 llm=llm)
tester = Agent(role="Test Author",
               goal="Write unit tests covering edge cases.",
               backstory="You hate shipping untested code.",
               llm=llm)
reviewer = Agent(role="Reviewer",
                 goal="Synthesize feedback from analyzer and tester.",
                 backstory="You write clear PR comments.",
                 llm=llm)

snippet = "def divide(a, b): return a / b"

# Two parallel tasks feed into a third synthesis task
analyze_task = Task(
    description=f"Find bugs in: {snippet}",
    expected_output="List of issues.",
    agent=analyzer,
)
test_task = Task(
    description=f"Write 3 pytest tests for: {snippet}",
    expected_output="Python pytest code.",
    agent=tester,
)
review_task = Task(
    description="Combine the above into PR feedback.",
    expected_output="Markdown PR comment.",
    agent=reviewer,
    context=[analyze_task, test_task],   # depends on both
)

crew = Crew(
    agents=[analyzer, tester, reviewer],
    tasks=[analyze_task, test_task, review_task],
    process=Process.sequential,   # CrewAI parallelizes within stages
)
result = crew.kickoff()
print(result)
```

CrewAI executes `analyze_task` and `test_task` concurrently because
they have no dependency; `review_task` runs after both finish. With
Kakeya's scheduler, those two parallel calls share GPU time fairly
under admission control. Against `mlx_lm.server`, they would
serialize and the wall-time would roughly double.

## Caveats

- CrewAI's `LLM` config has many sampling-related fields
  (`temperature`, `top_p`, `frequency_penalty`, `seed`). Kakeya
  ignores all sampling params (greedy by design). Set them or
  don't — output is the same.
- For very long crews (≥ 10 agents) bump `--max-concurrent` to
  match. Otherwise late-admitted agents wait on `queue`.
- Tool calling via CrewAI's `tools=[...]` parameter works. JSON
  reliability depends on the verifier; Qwen3-1.7B is reasonable
  out of the box for simple schemas, less so for nested ones.
