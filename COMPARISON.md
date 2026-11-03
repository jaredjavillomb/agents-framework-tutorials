# Comparison: LangGraph, CrewAI and Strands Agents

Written 2026-11-03. The evidence is `comparison/compare.ipynb` (reads `results/`, counts code, re-runs the
experiment below) and the 24 tutorial notebooks.

## What was compared

Each framework built the same two agents against one contract (`contracts/`): the same data, tools,
instructions, scoring and evaluation runner, and the same local model with the same settings.

| | |
|---|---|
| Agents | ShipFast support agent (8 scenarios: triage, tool use, a $100 approval gate, two-turn memory); research agent (10 questions over 30 fixed articles, with a critique loop and a 5-call tool budget) |
| Model | `qwen3-agents`: Qwen 3 8B on Ollama, temperature 0, seed 42, 16K context, thinking off |
| Versions | LangGraph 1.2.19 (langchain 1.4.8, langchain-ollama 1.1.0); CrewAI 1.15.34; Strands Agents 1.63.0 |
| Machine | one Apple M4 Mac, 16 GB |
| Runs | one run of every scenario and question per framework |

**Read every number with that last row in mind.** One run of 18 cases on a laptop with an 8B model is a
first measurement, not a ranking. Small models are sensitive to small differences in how a request is
packaged, and the most important finding below is exactly that.

## Results

| | LangGraph | CrewAI | Strands |
|---|---|---|---|
| **ShipFast** | **7 of 8** | **7 of 8** | **4 of 8** |
| tokens / seconds | 29,962 / 387 | 34,867 / 401 | 26,981 / 331 |
| **Research** | **10 of 10** | **9 of 10** | **9 of 10** |
| tokens / seconds | 44,361 / 1,464 | 60,243 / 1,194 | 48,803 / 1,147 |

Failures:

| Framework | Case | What happened |
|---|---|---|
| LangGraph | S5 | refunded the stand delivered 46 days ago; policy P4 forbids refunds after 30 days |
| CrewAI | S4 | refunded the broken vase on the first turn, before the photo policy P3 requires (the shared tool then refused a second refund, so only one $60 left) |
| CrewAI | Q10 | gave the correct total (PHP 1,635,000) and subtotals, but not the two unit prices the check expects |
| Strands | S3a, S3b | announced the $250 refund and ended the turn without calling `issue_refund`, so the approval gate never paused |
| Strands | S5 | the same mistake as LangGraph's S5 |
| Strands | S6 | the reply correctly asked the customer to check the order ID; the model's outcome choice labelled it `declined`, not `needs_info` |
| Strands | Q07 | mentioned the stale "9 provinces" beside the current 14 |

## What the results mean

**1. Strands' ShipFast gap comes from its tool schemas, not its agent loop.** A recording relay captured
the request each framework sent for the deciding model call in S3a. LangGraph's and Strands' requests were
identical in system prompt, options, messages, tool calls and tool results. The only difference: Strands
adds a placeholder description to every tool parameter that has none (`"description": "Parameter
order_id"`). Replaying the captured requests (`comparison/evidence/replay_tool_schema.py`):

| The deciding request | The model's next move |
|---|---|
| Strands, as sent | text, no refund attempt |
| Strands, placeholder descriptions removed | `issue_refund` |
| LangGraph, as sent | `issue_refund` |
| LangGraph, placeholder descriptions added | text, no refund attempt |

A content-free sentence per parameter was enough to move an 8B model from acting to talking. That is a
lesson about small models and tool schemas more than about Strands; with real parameter descriptions in
the shared tools' docstrings, all three frameworks would send meaningful text instead (an open question
below).

**2. The policy mistakes are the model's, and they move around.** LangGraph and Strands refunded outside the
30-day window (S5); CrewAI refunded before the photo (S4). The same model with the same instructions made
different mistakes under each framework, and different again across the component notebooks. Rules that
must hold (the $100 approval, no refund above what is left on the order) were enforced in code and held
in every framework; rules left to the prompt (30 days, photo first) were not reliable.

**3. A model checking a model misses what a plain check catches.** In research, the critic passed answers
that mentioned a stale figure to dismiss it (Strands' Q07 here; LangGraph's Q01 in notebook 05), and the
deterministic scorer failed them. Scoring in code, never by another model call, kept the comparison honest.

**4. Cost and time.** CrewAI used the most tokens on both agents: each agent call carries a longer prompt
(role, goal, backstory) and its research agent searched more often. LangGraph's research questions took the
longest on similar token counts; that was not investigated and may be environmental in a single run.

**5. Code built the Resolution, and that mattered.** Asked to fill in the scored Resolution itself, the
model claimed refunds that never happened and invented ticket numbers in the LangGraph and CrewAI
component notebooks (03), where the schema was imposed on the run. In Strands' notebook 03 the one
Resolution, asked for after the tools had run, was correct, which is too little to rely on. Building `action`, `refund_amount` and `ticket_id` from the ledger and tickets
(DECISIONS.md 16) is what made the ShipFast scores measure the frameworks.

## How each framework does it

| | LangGraph | CrewAI | Strands Agents |
|---|---|---|---|
| Mental model | a graph of steps over typed state | agents with roles, tasks, crews; Flows for structure | an agent: model, prompt, tools, loop |
| Control flow | `StateGraph`: nodes are plain functions or agents, conditional edges read typed state | `Flow`: `@start`, `@listen`, `@router` methods over typed state; agents inside steps | plain Python around agents; `GraphBuilder` and `Swarm` connect agents, passing text |
| Agent loop cap | `recursion_limit`, raises an error | `max_iter`, asks for a final answer | none by default; added with a hook |
| Structured output with tools on Ollama | provider strategy cannot call tools; tool strategy works but is not always called | `output_pydantic` skipped the tools | agent-level output cannot be forced on Ollama; the reliable native path reports no tokens |
| Approval gate | middleware calls `interrupt()` inside the tool call; `Command(resume=...)` continues | `@human_feedback` pauses **between Flow steps**; resumes only from saved state (`@persist`); the refund becomes its own step | a `BeforeToolCallEvent` hook interrupts inside the tool call; calling the agent with an interrupt response continues |
| What a pause needs | a checkpointer (in memory is enough in one process) | SQLite (or other) persistence, always | the same agent object, or a session manager |
| Conversation memory | checkpointer plus thread id | you pass the history yourself; `Memory` stores facts by meaning, not conversations | built in (`agent.messages`); session managers persist it |
| Measuring model calls | callbacks in the run's config | event bus listener; events are asynchronous, so flush before reading | hooks; token totals update after the model-call event, so settle per invocation |
| Lines of agent code (ShipFast / research / plumbing) | 109 / 78 / 34 | 206 / 77 / 79 | 82 / 72 / 43 |

Running locally, each framework needed something switched off or set explicitly:

- **CrewAI** defaulted to the cloud in four places: anonymous telemetry to telemetry.crewai.com, OpenAI
  embeddings for memory, an OpenAI model inside `Memory`, and an OpenAI model to read a supervisor's reply
  in `@human_feedback`. It also assumed an 8K context for an unknown model, reaches Ollama through the
  OpenAI-style API (which cannot carry a per-request context size), and refuses synchronous runs inside a
  notebook's event loop.
- **Strands** defaults to Amazon Bedrock when no model is given. Telemetry is opt-in.
- **LangGraph** needed nothing switched off. Its approval middleware is synchronous only, so the async
  capstone runs the graph in a worker thread.

Concepts a newcomer meets, from the notebooks:

- **LangGraph:** state, nodes, edges and conditional edges; `create_agent`; middleware; checkpointer and
  thread id; `interrupt` and `Command`; callbacks; two structured-output strategies.
- **CrewAI:** `LLM`; `Agent` (role, goal, backstory), `Task`, `Crew` and processes; Flows and their
  decorators and state; `@human_feedback` with providers; `@persist` and `from_pending`; `Memory` with an
  embedder; the event bus; async kickoff in notebooks; the environment switches above.
- **Strands:** `Agent` and `OllamaModel`; `@tool`; `agent.messages`; `structured_output_model`; hooks,
  interrupts and `cancel_tool`; session managers; `GraphBuilder` and `Swarm`; result metrics.

## When to pick which

- **LangGraph** when control flow must be exact and durable: approvals that wait for hours, workflows that
  mix fixed steps with agent steps, and state you can inspect, replay and resume. It asks you to draw more
  yourself, and gives the most predictable behaviour for it. In this project it had the best results and
  needed no defaults turned off.
- **CrewAI** when the work reads naturally as roles and tasks, and a team can live with its defaults or
  configure them once. Flows give real structure, but pauses only sit between steps and need saved state,
  so money-moving actions become flow steps. It cost the most tokens here, and running it fully locally
  took the most configuration.
- **Strands** when one capable agent with tools is the core of the system and you want the least code: it
  had the smallest ShipFast agent, built-in conversation memory, and hooks that pause inside a tool call.
  With small local models, check the tool schemas it generates; its placeholder parameter descriptions
  caused most of its failures here.

## Open questions

1. **Real parameter descriptions in the shared tools.** Documenting each argument in the tools' docstrings
   would give every framework meaningful parameter text and remove Strands' placeholders. That changes the
   contract and needs a re-run of every evaluation.
2. **The Q10 check.** The question asks only for the total, but the expected facts also require the two unit
   prices. Dropping them would change CrewAI's research score to 10 of 10.
3. **Repeats and a larger model.** One run per case was decided to keep cost and time down (DECISIONS.md 9).
   Three runs per case, or the same evaluation on a larger model, would show which differences are stable.
4. **LangGraph's research latency.** Higher per question on similar token counts; not investigated.
5. **Who edits `contracts/`** is still open (DECISIONS.md, Open).
