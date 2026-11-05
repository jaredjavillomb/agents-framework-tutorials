"""ShipFast's approval gate (SPEC.md section 3, policy P5) as a Strands hook.

A hook on `BeforeToolCallEvent` runs before every tool call. For an `issue_refund` over $100.00 it calls
`event.interrupt(...)`, which stops the agent and returns the interrupt in the result, before the tool
runs. The caller resumes by calling the agent with an `interruptResponse`; this time `interrupt()` returns
the supervisor's decision. Approve: the refund runs. Reject: `cancel_tool` skips it and hands the model a
rejection, telling it to escalate.
"""

from __future__ import annotations

import json

from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry

from agent_contracts.shipfast.tools import APPROVAL_THRESHOLD  # policy P5, shared by all builds


class RefundApproval(HookProvider):
    # Strands calls register_hooks once, when the hook is given to an Agent: run check() before every tool.
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeToolCallEvent, self.check)

    def check(self, event: BeforeToolCallEvent) -> None:
        call = event.tool_use  # the call the model asked for: {"name": ..., "input": {...}, "toolUseId": ...}
        if call["name"] != "issue_refund":
            return  # any other tool runs as normal
        args = call.get("input") or {}
        try:
            amount = float(args.get("amount"))
        except (TypeError, ValueError):
            return  # the shared tool rejects a non-numeric amount itself
        if amount <= APPROVAL_THRESHOLD:
            return  # $100.00 or less: no approval needed
        # Over $100: the first time through, interrupt() stops the agent before the tool runs, and the call
        # returns with stop_reason "interrupt". When the agent is called again with an interruptResponse,
        # this hook runs again and interrupt() returns the supervisor's decision instead.
        decision = event.interrupt("refund_approval", reason={
            "order_id": args.get("order_id"), "amount": amount, "reason": args.get("reason"),
            "question": f"Approve a refund of ${amount:.2f}? Resume with 'approve' or 'reject'."})
        if decision != "approve":
            # Rejected: cancel_tool skips the refund and hands the model this text as the tool's result.
            event.cancel_tool = json.dumps({
                "status": "rejected_by_supervisor", "order_id": args.get("order_id"), "amount": amount,
                "next_step": "Do not refund. Call escalate_to_human, then tell the customer a specialist will follow up."})


def resume_with(result, decision: str) -> list[dict]:
    """The prompt that resumes an interrupted agent with the same decision for each pending interrupt."""
    return [{"interruptResponse": {"interruptId": i.id, "response": decision}} for i in result.interrupts]
