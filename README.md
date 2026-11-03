# agent-frameworks

A hands-on tutorial and comparison of three agent frameworks: **LangGraph**, **CrewAI** and
**Strands Agents**. Each framework gets a set of component notebooks, then builds the same two
agents against one shared contract, so the results can be compared like for like.

## The two common agents

- **ShipFast support agent** (`contracts/agent_contracts/shipfast/SPEC.md`): triage, a tool-using
  specialist that follows a refund policy, a human approval gate for refunds over $100, and
  per-thread memory. 8 scenarios with exact expected outcomes.
- **Research agent** (`contracts/agent_contracts/research/`): research-and-report with a critic loop
  over fixed search results. Spec not written yet.

## Layout

```
contracts/            framework-neutral package (agent_contracts): specs, data, tools, schemas, metrics
langgraph-tutorial/   notebooks/ + lg_agents/ + run_eval.py, own venv
crewai-tutorial/      notebooks/ + crew_agents/ + run_eval.py, own venv
strands-tutorial/     notebooks/ + strands_agents_impl/ + run_eval.py, own venv
results/<agent>/      <framework>.json written by each run_eval.py
comparison/           compare.ipynb reading results/
COMPARISON.md         the final verdict
DECISIONS.md          what was decided, and what was rejected
```

## Setup

```bash
brew install ollama && ollama serve   # in its own terminal; no API key is needed
ollama pull qwen3:8b                  # the base model, about 5 GB
ollama create qwen3-agents -f contracts/ollama/Modelfile   # the shared model with its settings
cp .env.example .env                  # optional overrides only
cd langgraph-tutorial && uv sync      # same in crewai-tutorial/ and strands-tutorial/
```

Each tutorial has its own environment on purpose: CrewAI and LangGraph dependencies conflict.
Every tutorial installs `contracts/` as an editable path dependency, so there is one copy of the
specs, data and tool code.
