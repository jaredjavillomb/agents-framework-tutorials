# CLAUDE.md: agent-frameworks

Tutorial plus comparison of LangGraph, CrewAI and Strands Agents. Read README.md for the layout and
DECISIONS.md before changing anything structural.

## Rules

- **contracts/ is the single source of truth.** Specs, data, tool functions, schemas and metric
  definitions live there once. Builds import `agent_contracts`; never copy its files into a tutorial.
- **Changes to contracts/ need Jared's yes.** They affect all three builds. Who owns edits is still
  open (DECISIONS.md).
- **Tool logic lives in contracts/, not in a framework.** Each build only wraps the plain Python
  functions in its framework's tool type.
- **One environment per tutorial folder.** Run `uv` inside `langgraph-tutorial/`, `crewai-tutorial/`
  or `strands-tutorial/`, never at the repo root.
- **Agents live in importable modules** (`lg_agents/`, `crew_agents/`, `strands_agents_impl/`);
  notebooks and `run_eval.py` import them, so the code taught is the code tested.
- **Results go to `results/<agent>/<framework>.json`** in the shared format from the spec.
- **Notebooks must run top to bottom on a clean kernel.**
- Prose: no em-dashes or en-dashes as punctuation.

## Tracking

Work is tracked as 9 open Otoma tasks (created 2026-11-03): repo setup, shared contracts, component
notebooks per framework (x3), both agents per framework (x3), comparison.
