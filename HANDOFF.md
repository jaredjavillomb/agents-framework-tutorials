# Handoff: agent-frameworks

Written 2026-11-03 by the planning session (Claude Code, started in `~/work/tarkie-v5`). Read this,
then README.md, DECISIONS.md and CLAUDE.md, before doing anything.

## Goal

A LangGraph tutorial with the main components as notebooks, extended into a three-way comparison
with CrewAI and Strands Agents. Each framework builds the same two agents against one shared
contract, and the results are compared side by side.

## Where things stand

**Done**

- Repo scaffolded at `~/work/agent-frameworks` (git initialized, nothing committed yet).
- `contracts/agent_contracts/shipfast/` holds the ShipFast spec and data: `SPEC.md`,
  `data/orders.json`, `data/policy.md`, `scenarios.json`.
- Three tutorial folders, each with a `pyproject.toml` that installs `contracts/` as an editable
  path dependency. LangGraph keeps the version floors from the earlier project; CrewAI and Strands
  are unpinned.
- 9 open tasks in Otoma prod, all assigned to Jared, due 2026-11-03 (see below).

**Not done**

- No environments installed, no lockfiles. Run `uv sync` inside each tutorial folder.
- `contracts/` has no code yet: no `tools.py`, `schema.py` or harness.
- The research agent has no spec, fixtures or questions (only a README describing what is owed).
- No notebooks, no agents, no results.

## Decisions (Jared's, recorded in DECISIONS.md)

1. Both agents, ShipFast and the research agent, are built in every framework.
2. One repo: `contracts/` plus one `*-tutorial/` folder per framework, separate environments, not a
   uv workspace.
3. Strands Agents is the third framework, at the same depth as the other two.

**Open:** who edits `contracts/`. The planning session recommended one neutral owner, with
framework sessions only reading and requesting changes. Jared deferred it. Do not decide it for him;
until he does, treat every change to `contracts/` as needing his yes.

## History worth knowing

- ShipFast was first written by a separate session in `~/work/langgraph/shared/`. Jared reverted
  that session and it deleted the folder. The files here were recovered from that session's
  transcript: the three data files match the originals byte for byte, and SPEC.md is its final
  version, edited only to name all three frameworks, point at `contracts/` and set the results path
  to `results/shipfast/<framework>.json`.
- `~/work/langgraph` and `~/work/crew-ai` are now empty. Jared will remove them himself. Do not
  build there.
- A CrewAI session (`crew-ai-32`) was asked to coordinate earlier, then reverted at Jared's request.
  It has built nothing and is not working from this plan. Do not message it unless Jared asks.

## Next step: Otoma task 2, the contracts package

Start here; every build depends on it. Get Jared's yes on the research spec before writing code
against it.

1. **ShipFast `tools.py` and `schema.py`** from SPEC.md: `lookup_order`, `track_shipment`,
   `search_policy`, `issue_refund`, `escalate_to_human` as plain Python functions returning JSON
   strings, plus a resettable refund ledger. `Triage` and `Resolution` as pydantic models.
2. **Research agent spec**, in the style of the ShipFast one: `SPEC.md`,
   `data/search_fixtures.json`, `questions.json` (about 10, with expected key facts and aliases),
   `tools.py` (`web_search`, `calculator`), `schema.py` (the report).
3. **Harness**: one definition of tokens, latency and tool calls, and the results writer.

### Proposals for the research spec (not decided; put them to Jared)

- Plan, research (at most 5 tool calls), draft, critique, revise at most 2 times, final.
- `web_search` matches fixtures by token overlap on a normalized query and returns the top 3, so
  differently worded queries from different frameworks still get deterministic results.
- Enforce the tool-call budget inside the shared tool layer with a per-run counter (CrewAI's
  `max_iter` counts agent iterations, not tool calls). The same counter fills `tool_calls`.
- Score correctness deterministically: fraction of expected key facts present, case-insensitive,
  with aliases. Only add an LLM judge if Jared wants one.
- Tokens are summed over every LLM call in a run; latency is wall clock per question, median of 3
  runs, since temperature 0 is still not fully deterministic.

## Things that will bite

- **Approval gate parity.** ShipFast requires the framework, not the LLM, to pause before a refund
  over $100. LangGraph has `interrupt` plus a checkpointer. CrewAI has no direct equivalent
  (Flow state persistence is the nearest). Check what the installed Strands version supports.
  Document the gap honestly rather than faking parity; it is part of the comparison.
- **CrewAI memory defaults to an OpenAI embedder.** Configure it explicitly or it needs an OpenAI key.
- **Model ID formats differ.** Default `claude-sonnet-5-5`, temperature 0. LangChain style is
  `anthropic:claude-sonnet-5-5`, LiteLLM (CrewAI) style is `anthropic/claude-sonnet-5-5`.
- **CrewAI pins Python below 3.14**; its pyproject says so.
- **Import shadowing.** Do not rename `langgraph-tutorial/` to `langgraph/`, `crewai-tutorial/` to
  `crewai/`, or `strands_agents_impl/` to `strands_agents/`; each would shadow the real library.
- **Verify APIs against the installed versions.** Framework APIs move fast; read the installed
  package before trusting memory.

## How Jared works

- Propose before building anything structural, and wait for his explicit yes. Ask one question at
  a time. When there are two or more open decisions, keep a numbered ledger at the end of each
  message until they are closed, and record each one in DECISIONS.md with the rejected alternative.
- Lead with the long-term option, then the cheaper alternatives and what they cost.
- No em-dashes or en-dashes as punctuation in anything written for him.
- Subagents run on Sonnet. Never Fable without his explicit yes.
- Commit only when he asks.

## Otoma tasks (prod, all pending, due 2026-11-03)

| # | Task | Est | Id |
|---|---|---|---|
| 1 | Set up the agent-frameworks repo and move the existing work into it | 1 h | c3022654-a657-430e-ad03-5bbc675c88ba |
| 2 | Build the shared contracts package | 3 h | 65764891-4c39-4a39-8926-29a17686b4b6 |
| 3 | Write the LangGraph component notebooks (01 to 07) | 4 h | 29d10e79-8e5f-4659-86bb-aeb0a7a1338c |
| 4 | Write the CrewAI component notebooks | 4 h | 675829bd-c908-4eb4-afe3-27d244ccccd8 |
| 5 | Write the Strands Agents component notebooks | 4 h | 9d9da109-9500-445e-be17-07032754238c |
| 6 | Build ShipFast and the research agent in LangGraph | 3 h | 0498e879-9179-46a4-a9e4-6dd8a5af827a |
| 7 | Build ShipFast and the research agent in CrewAI | 3 h | 66b86c45-d431-45e7-b0bb-3bb6d6630c3c |
| 8 | Build ShipFast and the research agent in Strands | 3 h | 21ab7904-39f4-43fa-b2ae-25385e54b681 |
| 9 | Compare LangGraph, CrewAI and Strands | 2 h | ef172392-9973-4fbf-a4f9-ffb86e8d5fd1 |

Task 1 is mostly done (scaffold in place); what remains is the open ownership decision, installing
the environments and the first commit. The old Oct 27 "Explore the LangGraph / CrewAI / Strands
Agents framework" tasks are deliberately not linked to these; Jared asked to leave them alone.

The 27 h of estimates are the planning session's, agreed by Jared, and stored as agent-sourced.
Close tasks through the Otoma dashboard or the otoma-task-author / otoma-session-tasks skills.
