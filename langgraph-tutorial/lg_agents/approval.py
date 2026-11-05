"""ShipFast's approval gate (SPEC.md section 3, policy P5) as LangChain agent middleware.

Before any `issue_refund` call over $100.00 runs, the graph pauses with LangGraph's `interrupt()` and
waits for a supervisor's decision, supplied later with `Command(resume="approve" | "reject")`. The pause
is in the framework, not in the prompt and not in the shared tool. Needs a checkpointer, because the
paused state must be saved to resume it.

Approve: the refund runs. Reject: the refund does not run; the model gets a rejection it must act on by
escalating to a human.
"""

from __future__ import annotations

import json

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

from agent_contracts.shipfast.tools import APPROVAL_THRESHOLD  # policy P5, shared by all builds


# The refund amount the model asked for, as a number, or None if it is not one.
def _amount(args: dict) -> float | None:
    try:
        return float(args.get("amount"))
    except (TypeError, ValueError):
        return None  # the shared tool rejects a non-numeric amount itself


# @wrap_tool_call turns this function into middleware that runs around every tool call the agent makes.
# `request.tool_call` is the call the model asked for; `handler(request)` runs the real tool.
@wrap_tool_call
def refund_approval(request, handler):
    call = request.tool_call  # {"name": ..., "args": {...}, "id": ...}
    amount = _amount(call["args"]) if call["name"] == "issue_refund" else None
    # Any other tool, or a refund of $100.00 or less: run it straight away.
    if amount is None or amount <= APPROVAL_THRESHOLD:
        return handler(request)

    # Over $100: pause here. interrupt() saves the run through the checkpointer and hands this payload to
    # the caller. When the caller resumes with Command(resume=...), this function runs again from the
    # top and interrupt() returns the supervisor's decision instead of pausing.
    decision = interrupt({
        "type": "refund_approval",
        "order_id": call["args"].get("order_id"),
        "amount": amount,
        "reason": call["args"].get("reason"),
        "question": f"Approve a refund of ${amount:.2f}? Resume with 'approve' or 'reject'.",
    })
    if decision == "approve":
        return handler(request)  # approved: run the real refund, with the amount the supervisor saw
    # Rejected: the refund never runs. The model gets this tool result instead, telling it what to do next.
    return ToolMessage(
        name=call["name"],
        tool_call_id=call["id"],
        content=json.dumps({
            "status": "rejected_by_supervisor",
            "order_id": call["args"].get("order_id"),
            "amount": amount,
            "next_step": "Do not refund. Call escalate_to_human, then tell the customer a specialist will follow up.",
        }),
    )
