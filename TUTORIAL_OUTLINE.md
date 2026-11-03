# Tutorial outline (all three frameworks)

Approved by Jared on 2026-11-03 (DECISIONS.md 13). Every tutorial has the same seven component
notebooks, with the same names, in the same order, so a reader can compare frameworks notebook by
notebook. The capstone notebooks that build ShipFast and the research agent come after them (Otoma
tasks 6 to 8).

The order follows the six ideas on the crash-course slide "What Every Framework Gives You" (agent,
tools, state and memory, orchestration, guardrails and approval, tracing), with the model and tools
taught before the agent, because an agent is the two of them together.

| # | Notebook | What it teaches | LangGraph | CrewAI | Strands |
|---|---|---|---|---|---|
| 01 | `01_model` | Connect to `qwen3-agents`, one call, structured output (ShipFast `Triage`) | `ChatOllama`, structured output | `LLM("ollama/...")` | `OllamaModel` |
| 02 | `02_tools` | Wrap the shared tools, one correct tool call | `@tool`, `bind_tools` | `@tool` | `@tool` |
| 03 | `03_agent` | Model + instructions + tools, the agent loop and its stop condition | `create_agent` | `Agent` + `Task` + `Crew` | `Agent` |
| 04 | `04_state_and_memory` | Per-thread memory: the ShipFast follow-up turn | checkpointer, thread id | crew memory | conversation history, sessions |
| 05 | `05_orchestration` | Workflow vs agent, a bounded loop (rehearses the research critique loop) | `StateGraph`, conditional edges | `Flow`: `@start`, `@listen`, `@router` | multi-agent patterns |
| 06 | `06_guardrails_and_approval` | Pause before a refund over $100, resume on approve or reject | `interrupt` + `Command` | `@human_feedback`, `@persist` | interrupts before a tool call |
| 07 | `07_tracing_and_eval` | Capture tokens and tool calls into the shared harness, run a small scored eval | callbacks | event listeners | hooks, metrics |

Rules for every notebook:

- Runs top to bottom on a clean kernel, on the reference Mac, in a few minutes at most.
- Uses the ShipFast data, except 05 and 07, which use the research fixtures.
- Imports reusable code from the build's package (`lg_agents/`, `crew_agents/`,
  `strands_agents_impl/`) so the code taught is the code tested.
- Ends with a short "How the others do this" note, the raw material for the comparison.

Still to verify against the installed versions before writing those notebooks: Strands multi-agent
patterns and sessions (04, 05), and CrewAI memory, which needs an embedding model; keeping it local
means a small Ollama embedding model, to be decided when notebook 04 is written.
