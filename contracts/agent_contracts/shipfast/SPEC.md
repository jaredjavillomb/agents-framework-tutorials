# Shared Agent Spec: ShipFast Customer Support Agent

This is the contract that all three builds implement (`langgraph-tutorial/`, `crewai-tutorial/`,
`strands-tutorial/` in `~/work/agent-frameworks`), so they can be compared like for like. Changes go
through Jared and land once, here; who edits `contracts/` is still undecided (see DECISIONS.md).

Source of truth: this package (`contracts/agent_contracts/shipfast/`). Builds import it; never copy it.
Data is in `data/` (orders, policy); test scenarios are in `scenarios.json` next to this file.

- `orders.json`: 5 mock orders (A1001 to A1005)
- `policy.md`: the refund and complaint policy (P1 to P7). "Today" is fixed at **2026-11-01** so runs are deterministic.
- `scenarios.json`: 8 test scenarios with expected outcomes

## 1. What the agent does

A customer writes in about a delivery or refund. The agent:

1. **Triages** the message into an intent (structured output).
2. **Resolves** it with a tool-using specialist that follows `policy.md`.
3. **Pauses for human approval** before any refund over $100 (policy P5), using the framework's
   native human-in-the-loop mechanism. The test runner supplies the decision (`approve` / `reject`).
4. **Remembers** the conversation per thread, so follow-up turns work without repeating the order ID.
5. Returns a customer-facing reply plus a structured `Resolution`.

## 2. Model

Shared with the research agent (DECISIONS.md 4, 10, 11 and 12). Everything runs locally; there is no
API key.

- Model **`qwen3-agents`**, served by **Ollama** at `http://localhost:11434` (override with
  `OLLAMA_HOST`). It is `qwen3:8b` with `num_ctx` 16384, `temperature` 0 and `seed` 42 written into
  the model by `contracts/ollama/Modelfile`; create it once with
  `ollama create qwen3-agents -f contracts/ollama/Modelfile`. Ollama's own default context on the
  reference Mac is 4096, which silently breaks long tool-calling runs, and the OpenAI-style API that
  CrewAI uses ignores a per-request `num_ctx`, so the settings must live in the model.
- `MODEL` env var overrides the model, written the plain Ollama way. Each build adds its framework's
  prefix itself (CrewAI `ollama/`; LangChain and Strands take the bare name).
- Thinking off on every call, which a Modelfile cannot set: `reasoning=False` (LangChain),
  `additional_args={"think": False}` (Strands), `reasoning_effort="none"` (CrewAI). Builds whose
  framework can also send `temperature`, `seed` and `num_ctx` per request do so
  (`agent_contracts.harness.load_settings().options()`).
- Every runner calls `agent_contracts.harness.verify_model()` first; it fails if Ollama is down or
  the model lacks the shared settings.
- If a framework cannot pass a setting, the build says so in its notebook and in the comparison. It
  does not work around it silently.
- Python 3.12 or later, managed with `uv` (the reference environments use 3.13).

## 3. Tools (same names, arguments and return shapes in all builds)

All tools return a JSON string. Tools are pure functions over the shared data plus an in-memory ledger
that the runner resets between scenarios.

| Tool | Args | Returns |
|---|---|---|
| `lookup_order` | `order_id: str` | the order object from `orders.json` without `tracking`, or `{"error": "order_not_found", "order_id": ...}` |
| `track_shipment` | `order_id: str` | `{"order_id", "status", "carrier", "eta", "delivered_on", "tracking": [...]}` or the not-found error |
| `search_policy` | `query: str` | `{"today": "2026-11-01", "sections": [{"id": "P1", "title": ..., "text": ...}, ...]}`. Keyword match over `policy.md` sections (split on `## `); return every section whose title or body shares a lowercased word of 4+ letters with the query; if none match, return all sections. `today` is the policy's fixed date, which sits above the first section and would otherwise be lost |
| `issue_refund` | `order_id: str, amount: float, reason: str` | `{"refund_id": "RF-<order_id>", "order_id", "amount", "status": "issued"}`. Appends to the ledger. Errors: the not-found error; `{"error": "amount_exceeds_order_total", "order_id", "amount", "order_total"}`; `{"error": "amount_exceeds_remaining", "order_id", "amount", "already_refunded", "remaining"}` when earlier refunds on the order leave less than the amount; `{"error": "invalid_amount", ...}` for zero, negative or non-numeric amounts |
| `escalate_to_human` | `order_id: str, summary: str` | `{"ticket_id": "TCK-<order_id>", "status": "open"}` (use `TCK-GENERAL` when order_id is empty) |

**Approval gate.** The $100 check is enforced by the framework, not by the LLM and not inside
`issue_refund`. When a refund over $100.00 is about to be issued, execution must pause and wait for a
human decision. Approve: the refund is issued. Reject: the refund is not issued and the agent calls
`escalate_to_human`.

## 4. Triage output (structured)

```json
{"intent": "order_status | refund_request | complaint | other", "order_id": "A1001 | null"}
```

- `order_status`: where is my order, when will it arrive
- `refund_request`: explicit ask for refund, return or cancellation
- `complaint`: something is wrong (damaged, wrong item) without an explicit refund ask yet
- `other`: out of scope (policy P7). Skip the specialist, reply with a polite decline.

On a follow-up turn, triage may use the conversation history to fill `order_id` and should keep the
thread's intent unless the customer clearly changes topic.

## 5. Final output (structured `Resolution`)

```json
{
  "intent": "order_status | refund_request | complaint | other",
  "order_id": "A1001 | null",
  "action": "answered | refunded | pending_approval | escalated | needs_info | declined",
  "refund_amount": 45.0,
  "ticket_id": "TCK-A1003 | null",
  "reply": "customer-facing text"
}
```

- **Code builds the Resolution from what happened** (DECISIONS.md 16), with the shared
  `shipfast.resolution.derive_resolution`: out-of-scope triage is `declined`; a pause at the approval gate
  is `pending_approval`; a refund in the ledger this turn is `refunded` with its amount; a ticket opened
  this turn is `escalated` with its id; otherwise the model chooses `answered`, `needs_info` or `declined`
  (the shared `Outcome` model and `prompts.OUTCOME`). `intent` and `order_id` come from triage, which uses
  `prompts.TRIAGE`. The model writes only the `reply`.
- `refunded` only if `issue_refund` actually succeeded (check the ledger, not the LLM's claim).
- `pending_approval` is what a turn returns while it is paused waiting for the supervisor.
- `needs_info`: we asked the customer for something (photo, correct order ID).

## 6. Test runner and results

Each build's `run_eval.py` calls the shared `agent_contracts.harness.runners.run_shipfast_eval` with a
small adapter (start a conversation, send a turn, resume a paused turn). For every scenario in
`scenarios.json` the runner:

1. resets the ledger and starts a fresh thread/memory,
2. feeds each turn in order on the same thread,
3. if a turn pauses for approval, resumes it with the scenario's `approval_decision`,
4. checks the final `Resolution` against `expected` (and `expected_per_turn` when present) on the
   fields `intent`, `order_id`, `action`, `refund_amount` (exact match; amounts compared to 2 dp),
5. records metrics.

Write results to `results/shipfast/<framework>.json` at the repo root:

```json
{
  "framework": "langgraph | crewai | strands",
  "framework_version": "x.y.z",
  "model": "qwen3-agents",
  "run_at": "ISO timestamp",
  "scenarios": [
    {
      "id": "S2_lost_refund_auto",
      "passed": true,
      "resolution": { "...": "final Resolution" },
      "turn_resolutions": [ { "...": "Resolution per turn" } ],
      "tool_calls": ["lookup_order", "search_policy", "issue_refund"],
      "llm_calls": 4,
      "input_tokens": 3120,
      "output_tokens": 410,
      "latency_s": 6.8
    }
  ],
  "summary": {"passed": 7, "total": 8, "total_tokens": 0, "total_latency_s": 0}
}
```

## 7. Tutorial layout (each build)

Notebooks teach the framework's building blocks one at a time, then a capstone notebook builds this
agent. The agent code itself also lives in a small importable package so the runner and notebooks share it.

## 8. Comparison (written jointly at the end)

`COMPARISON.md`, covering: lines of code for the agent, concepts a newcomer must learn, how control
flow is expressed, how the approval gate is done, memory, streaming and observability, scenario pass
rate, tokens and latency per scenario, and when to pick which.
