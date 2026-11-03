# Shared Agent Spec: Research Agent

This is the contract that all three builds implement (`langgraph-tutorial/`, `crewai-tutorial/`,
`strands-tutorial/` in `~/work/agent-frameworks`), so they can be compared like for like. Changes go
through Jared and land once, here; who edits `contracts/` is still undecided (see DECISIONS.md, which
also records why each rule below was chosen: decisions 4 to 11).

Source of truth: this package (`contracts/agent_contracts/research/`). Builds import it; never copy it.

- `data/search_fixtures.json`: 30 short articles (F01 to F30), the only "web" the agent can search.
  They describe **Veltara Aquatech, a fictional company**, so the model cannot answer from memory.
  Article F03 is deliberately out of date and contradicts newer articles. Every URL is on a reserved
  `.example` domain.
- `questions.json`: 10 questions (Q01 to Q10) with the facts a correct answer must contain.
- "Today" is fixed at **2026-11-01**, as in ShipFast.

## 1. What the agent does

The agent gets one question about Veltara Aquatech and returns a short, cited answer. It must:

1. **Plan** what it needs to find out, and record the plan.
2. **Research** with the two tools (section 3): at most **5 tool calls per question** in total.
3. **Draft** an answer that cites the articles it used.
4. **Critique** its own draft: is every fact supported by a cited article, is anything missing, does
   any cited article contradict a newer one? The critique ends with a verdict, `pass` or `revise`.
5. **Revise** when the verdict is `revise`, at most **2 times**.
6. Return the final `Report` (section 5).

**What the spec fixes, and what it leaves to each framework.** The spec fixes only what can be checked
from outside a build: the recorded plan, every critique and the final report; the 5-call limit
(enforced by the shared tools); the 2-revision limit, which must be enforced **in code, not in the
prompt**. How the loop is built (a LangGraph graph, a CrewAI Flow with agents inside, a Strands agent,
or anything else) is each tutorial's choice, and is one of the things the comparison reports on.

**Loop rule.** A critique follows every draft until a critique passes or the second revision is done.
The second revision is final without another critique. So the only valid shapes are:

| `critiques` verdicts | `revisions` |
|---|---|
| `[pass]` | 0 |
| `[revise, pass]` | 1 |
| `[revise, revise]` | 2 |

Tool calls may happen in any step (a revision may search again), as long as the total stays within 5.

## 2. Model

Identical to ShipFast SPEC.md section 2 (DECISIONS.md 4, 10, 11 and 12), which is the single
statement of it. In short: model **`qwen3-agents`** on Ollama (`qwen3:8b` with `num_ctx` 16384,
`temperature` 0 and `seed` 42 written in by `contracts/ollama/Modelfile`), thinking off on every call,
and `agent_contracts.harness.verify_model()` at the top of every runner. No API key.

## 3. Tools (same names, arguments and return shapes in all builds)

Both tools are plain Python functions in `research/tools.py` and return a JSON string. Each build only
wraps them in its framework's tool type.

| Tool | Args | Returns |
|---|---|---|
| `web_search` | `query: str` | `{"query": ..., "results": [{"id", "title", "url", "published", "snippet"}, ...]}`, at most 3 results, possibly none |
| `calculator` | `expression: str` | `{"expression": ..., "result": <number>}`, or `{"error": "invalid_expression", "expression": ...}` |

**`web_search` matching.** Deterministic, so differently worded queries from different frameworks get
predictable results:

1. Normalize the query and each article's `title + " " + snippet`: lowercase; remove commas between
   digits (`1,200` becomes `1200`); split on every character that is not a letter, digit or `.`;
   strip leading and trailing `.` from each token; drop stopwords (the fixed list in `tools.py`);
   strip a trailing `s` from tokens of 4 or more letters that do not end in `ss`.
2. Score each article by the number of distinct query tokens it contains.
3. Return the top 3 articles with a score above 0, highest first, ties broken by file order. If no
   article scores above 0, return `"results": []`. (Unlike ShipFast's `search_policy`, nothing is
   returned when nothing matches.)

**`calculator`.** Arithmetic only: numbers, `+ - * / **`, parentheses and unary minus, evaluated by
walking the Python AST (never `eval`). Division by zero, names, function calls or anything else return
`invalid_expression`.

**Tool-call budget.** A per-run counter inside `tools.py` counts every call to either tool. Calls 1 to
5 run normally. Call 6 onward does not run and returns
`{"error": "budget_exhausted", "limit": 5}`; it does not raise, so the agent must finish with what it
has. The runner resets the counter before each question (`reset()`), and reads the list of attempted
calls (`tool_calls()`) for the results, refused calls included.

## 4. Output: `Report` (structured, `research/schema.py`)

```json
{
  "question_id": "Q03_revenue_growth",
  "plan": ["Find FY2025 revenue", "Find FY2024 revenue", "Compute the growth"],
  "answer": "Revenue grew 30%, from PHP 1.20 billion in FY2024 to PHP 1.56 billion in FY2025 [F05][F04].",
  "citations": ["F04", "F05"],
  "critiques": [{"verdict": "pass", "issues": []}],
  "revisions": 0
}
```

- `answer`: customer-readable prose, a few sentences at most.
- `citations`: the fixture ids the answer relies on, at least one.
- `critiques`: every critique in order, each `{"verdict": "pass" | "revise", "issues": [str]}`.
- `revisions`: how many revisions ran; must match `critiques` per the loop rule in section 1.

## 5. Scoring

Deterministic text matching, no LLM judge (DECISIONS.md 8). For each question, on the final `answer`:

- **Matching rule.** Normalize the answer: lowercase, remove commas between digits, collapse
  whitespace. An alias matches when it appears in the normalized answer and the characters just before
  and after it are not letters or digits (so `30%` does not match inside `130%`, and `14` does not
  match inside `2014`). Aliases in `questions.json` are already written in normalized form.
- `fact_recall`: the share of `expected_facts` with at least one matching alias.
- `contradiction_free`: no entry of `must_not_contain` matches. An answer that repeats a stale figure
  fails this even if it calls the figure outdated, so the agent should state only current facts.
- `citations_valid`: `citations` is not empty and every id exists in `search_fixtures.json`.
- `passed`: `fact_recall >= 0.8` and `contradiction_free` and `citations_valid`.

With 4 or fewer expected facts, 0.8 means every fact is needed; Q01 to Q10 have 1 to 4 facts each.

The runner also checks the `Report` shape and the loop rule. A report that breaks either is recorded
with `"passed": false` and the reason, not dropped.

## 6. Test runner and results

Each build's `run_eval.py` calls the shared `agent_contracts.harness.runners.run_research_eval` with one
function that answers a question. For every question in `questions.json`, once (DECISIONS.md 9), it:

1. calls `tools.reset()`,
2. runs the agent on the question,
3. scores the final `Report` (section 5),
4. records the metrics through the shared harness: every LLM call's input and output tokens summed,
   wall-clock latency for the whole question, and the tool calls from `tools.tool_calls()`.

Write results to `results/research/<framework>.json` at the repo root:

```json
{
  "framework": "langgraph | crewai | strands",
  "framework_version": "x.y.z",
  "model": "qwen3-agents",
  "run_at": "ISO timestamp",
  "questions": [
    {
      "id": "Q03_revenue_growth",
      "passed": true,
      "scores": {
        "fact_recall": 1.0,
        "facts_missing": [],
        "contradiction_free": true,
        "citations_valid": true
      },
      "report": { "...": "final Report" },
      "tool_calls": ["web_search", "web_search", "calculator"],
      "budget_exhausted": false,
      "llm_calls": 5,
      "input_tokens": 6200,
      "output_tokens": 540,
      "latency_s": 48.3,
      "error": null
    }
  ],
  "summary": {"passed": 8, "total": 10, "total_tokens": 0, "total_latency_s": 0}
}
```

Results come from a single run on a laptop: latency measures that machine, and a small model's mistakes
show up alongside each framework's. The comparison says so next to the numbers.

## 7. Tutorial layout (each build)

As for ShipFast: the agent lives in the build's importable package (`lg_agents/`, `crew_agents/`,
`strands_agents_impl/`), and both the capstone notebook and `run_eval.py` import it, so the code
taught is the code tested.

## 8. Comparison (written jointly at the end)

Added to `COMPARISON.md` for this agent: how each framework expresses the bounded critique loop and
where the 2-revision limit is enforced; how it surfaces the `budget_exhausted` refusal to the model;
which of the section 2 model options it could pass; lines of code for the agent; pass rate, tokens and
latency per question.
