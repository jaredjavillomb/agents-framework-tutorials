# Decisions

Decided by Jared on 2026-11-03.

## 1. Common agents: both ShipFast and the research agent, in every framework

Rejected: ShipFast only (fewer builds, but no read on research-style agents); the research agent only
(drops the human approval gate and memory, the parts that separate the frameworks most).

## 2. One repo, contracts/ plus one tutorial folder per framework, separate environments

Rejected: a neutral `agent-shared/` folder beside separate `~/work/langgraph` and `~/work/crew-ai`
projects (two histories, the comparison has no home); leaving `shared/` inside the LangGraph project
and copying it (the copies drift). Not a uv workspace: one lockfile would force CrewAI and LangGraph
into a single dependency resolution. Tutorial folders are named `*-tutorial/` and agent packages
`lg_agents/`, `crew_agents/`, `strands_agents_impl/` so nothing shadows the real libraries on import.

## 3. Third framework: Strands Agents, same depth as LangGraph and CrewAI

## 4. Model: a local small model only, no API key (revised the same day)

Every build, notebook and results run uses one small model served locally on Jared's Mac (M4, 16 GB),
for both agents. No API key and no per-token cost. The specs treat the model as a setting (the
`MODEL` env var), not as Anthropic-specific; the Claude API rules in ShipFast SPEC.md section 2 come
out of the contract. Which local model and runtime is a separate decision. Accepted costs: small
models misformat tool calls and structured output more often, so pass rates reflect the model as well
as the framework (equally for all three builds), and latency measures the laptop.

First decided as `claude-haiku-5-5` (the cheapest Claude model), then replaced by this.

Rejected: a local model for building and notebooks plus one `claude-haiku-5-5` run for the published
results (recommended: framework differences show through a capable model, at under a dollar per run,
but it needs a paid key); `claude-haiku-5-5` for everything (needs a key for every run);
`claude-opus-5-5` and `claude-sonnet-5-5` (40x and 20x Haiku's input price).

## 5. Research agent flow: the spec sets observable rules, each framework builds it its own way

The spec requires only what can be checked from outside a build: the plan, every critique and the
final report are recorded; at most 5 tool calls (enforced by the shared tool layer); at most 2
revisions, enforced in code, not in the prompt. How the loop is built (a LangGraph graph, a CrewAI
Flow with agents inside, a Strands agent) is each tutorial's choice, and how each framework expresses
a bounded loop becomes a comparison row. Same approach as ShipFast's approval gate, which names the
behaviour and leaves the mechanism to the framework.

Rejected: fixing six named stages (plan, research, draft, critique, revise, final) in every build.
Easier to line up step by step, but it is LangGraph's natural shape, so the CrewAI and Strands
tutorials would teach a translation of LangGraph and not their own framework.

## 6. Research search fixtures: word-overlap matching, a made-up company, one stale article

`web_search` reads a fixed file of about 30 short articles, never the live web. Matching: lowercase
the query, strip punctuation, drop stopwords, score each article by shared words with its title and
snippet, return the top 3 (ties broken by file order), and return no results when nothing overlaps.
The articles describe a fictional company, so the model cannot answer from memory. One article is
deliberately out of date and contradicts a newer one, so the critique step has something real to catch.

Rejected: word-overlap matching alone over real-world topics. Quicker to write, but scores would
partly reward what the model already knows, and the self-check would have nothing to find.

## 7. Research tool-call limit: one shared counter, every tool counts, call 6 is refused

At most 5 tool calls per question, `web_search` and `calculator` together. A per-run counter in the
shared tool code enforces it, so the limit means the same thing in every framework, and the same
counter fills the `tool_calls` metric. Call 6 onward returns `{"error": "budget_exhausted"}` and does
not raise, so the agent must finish with what it has and graceful recovery is visible in the results.

Rejected: each framework's own limit setting. No shared code needed, but the settings count different
things (CrewAI's `max_iter` counts agent iterations, not tool calls), so the tool-call numbers would
not be comparable.

## 8. Research scoring: deterministic text matching, pass at 80%, no LLM judge

Each question lists expected facts, each with aliases. Checks, all case-insensitive: `fact_recall`
(share of expected facts present), `contradiction_free` (no `must_not_contain` hit, which is the stale
article's claim), `citations_valid` (every cited id exists in the fixtures). A question passes when
recall is at least 0.8, there is no contradiction and every citation is valid. Free, repeatable and
checkable by hand; an LLM judge can be added later without changing these checks.

Rejected: adding an LLM judge for answer quality. It would catch correct facts worded in ways no
alias anticipated, but it adds API calls to every run and its scores vary between runs, which blurs
the differences between frameworks.

## 9. Runs per test: once, for both agents

Every ShipFast scenario and research question runs once per framework. Latency, tokens and tool
calls are from that single run. ShipFast SPEC.md section 6 and its results format stay as written.
A single run can pass or fail by chance; the comparison should say its results are single-run. With
the local model (decision 4), temperature 0 and a fixed seed can be set, which makes a single run far
more repeatable than it would have been on Claude 5 models.

Rejected: 3 runs for both agents, reporting pass^3 (passes all 3 times), the pass rate and median
latency and tokens (recommended: reliability, not one lucky run, but three times the API calls);
3 runs for the research agent and 1 for ShipFast (two rules in one harness).

## 10. Local model: `qwen3:8b` served by Ollama

Ollama 0.40.13 (Homebrew) on the M4, 16 GB Mac. Every build sets the same options explicitly:
temperature 0, a fixed seed, and a context of 16384 tokens (Ollama's own default on this machine is
4096, which silently breaks long tool-calling runs). Thinking mode on or off is still to decide.

Evidence, from a framework-free smoke test with thinking off: structured triage 9 of 9 correct,
including both memory follow-ups; tool-calling loop 4 of 5 once one over-strict check was fixed. The
one real failure: on the lost package (A1002) it refunded $120 instead of $45 without looking up the
order total. Roughly 3 to 4 s per triage call and 8 to 40 s per tool-calling case.

Rejected: `llama3.1:8b` (older; the test was stopped before it downloaded, so it was never measured);
`gpt-oss:20b` (about 14 GB, too large to run beside notebooks on 16 GB).

## 11. Qwen thinking mode: off

Every build sends `think: false` (or its framework's equivalent) to `qwen3:8b`. Measured on the smoke
test: about 3.5 s per triage call and up to about 15 s per tool-calling step with thinking off, against
about 29 s and 55 s with it on, so a full ShipFast run in one framework is roughly 5 to 10 minutes,
not 30 to 60.

Rejected: thinking on. Slower by 8 to 10 times; whether it fixes the made-up refund amount was not
measured, because the run was stopped once speed decided it.

## Open

- Who edits `contracts/`. Recommended: one neutral owner, framework sessions only read and request
  changes, so no build shapes the contract in its own favour. Deferred by Jared.
