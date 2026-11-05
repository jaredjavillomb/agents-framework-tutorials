"""ShipFast's approval gate (SPEC.md section 3, policy P5) as a CrewAI Flow with `@human_feedback`.

CrewAI pauses between flow steps, not inside a tool call, so the gate has two parts:

1. The specialist agent gets a gated `issue_refund`. Refunds up to $100.00 run at once; a larger one is
   not run, only recorded as pending in the flow's state, and the agent is told it awaits a supervisor.
2. A router sends a flow with a pending refund to a `@human_feedback` step, which pauses the flow
   (`kickoff` returns `HumanFeedbackPending`) and saves its state with `@persist`.
3. `resume(flow_id, "approve")` restores the flow from that saved state and issues exactly the pending
   refund, in code; `resume(flow_id, "reject")` tells the agent, which must escalate to a human. CrewAI
   only resumes from saved state, so persistence is required, here a SQLite file inside the tutorial.

The feedback is mapped to "approve" or "reject" by the local model: CrewAI's default for that mapping is
an OpenAI model.
"""

from __future__ import annotations

import json
from pathlib import Path

from crewai import Agent
from crewai.flow import human_feedback
from crewai.flow.async_feedback import HumanFeedbackPending
from crewai.flow.flow import Flow, listen, router, start
from crewai.flow.persistence import SQLiteFlowPersistence, persist
from crewai.tools import tool
from pydantic import BaseModel

from agent_contracts.shipfast import tools as shipfast
from agent_contracts.shipfast.tools import APPROVAL_THRESHOLD  # policy P5, shared by all builds
from crew_agents.llm import make_llm
from crew_agents.prompts import BACKSTORY, GOAL, ROLE

FEEDBACK_LLM = make_llm()  # maps the supervisor's free-text reply onto "approve" or "reject"
FLOW_DB = Path(__file__).resolve().parents[1] / ".crewai_flows" / "flows.db"  # inside the tutorial folder
FLOW_DB.parent.mkdir(exist_ok=True)
PERSISTENCE = SQLiteFlowPersistence(str(FLOW_DB))  # where paused flows are saved, so they can be resumed


class SupervisorQueue:
    """A feedback provider that never blocks: it pauses the flow and lets the caller resume it later."""

    # CrewAI's default provider waits for an answer typed in the console; a notebook or a server has none.
    # Raising HumanFeedbackPending tells CrewAI to save the flow and return instead of waiting.
    def request_feedback(self, context, flow):
        raise HumanFeedbackPending(context=context, callback_info={"queue": "supervisors"})


# The flow's state: one customer turn, saved by @persist after each step.
class RefundCase(BaseModel):
    message: str = ""  # the customer's message for this turn
    history: list[dict] = []  # the conversation so far, as chat messages
    pending_refund: dict | None = None  # a refund over $100 waiting for a supervisor, or None
    reply: str = ""  # what the customer is told


@persist(PERSISTENCE)  # save the state after every step, so a paused flow can be restored by its id
class ShipFastApprovalFlow(Flow[RefundCase]):
    def __init__(self, llm=None, **kwargs):
        kwargs.setdefault("suppress_flow_events", True)  # no console panels for every step
        super().__init__(**kwargs)
        self.llm = llm or make_llm()

    # The specialist's tools: the shared ones, except that issue_refund is swapped for a gated version.
    def _tools(self):
        def issue_refund(order_id: str, amount: float, reason: str) -> str:
            """Issue a refund for an order. The amount cannot exceed the order total minus anything already refunded."""
            try:
                value = float(amount)
            except (TypeError, ValueError):
                value = None  # not a number: let the shared tool reject it below
            if value is not None and value > APPROVAL_THRESHOLD:
                # Over $100: do not refund. Record it in the flow's state and tell the model it is waiting.
                self.state.pending_refund = {"order_id": order_id, "amount": value, "reason": reason}
                return json.dumps({"status": "pending_approval", "order_id": order_id, "amount": value,
                                   "next_step": "Do not call issue_refund again. Tell the customer a supervisor "
                                                "must approve this refund and they will hear back shortly."})
            return shipfast.issue_refund(order_id, amount, reason)  # $100 or less: the real shared tool

        others = [f for f in shipfast.TOOLS if f.__name__ != "issue_refund"]
        return [tool(f.__name__)(f) for f in others] + [tool("issue_refund")(issue_refund)]

    # One agent turn: a new agent is given the whole history, and its reply is added to it.
    async def _agent_turn(self, text: str) -> str:
        agent = Agent(role=ROLE, goal=GOAL, backstory=BACKSTORY, tools=self._tools(), llm=self.llm, verbose=False)
        self.state.history.append({"role": "user", "content": text})
        reply = (await agent.kickoff_async(self.state.history)).raw
        self.state.history.append({"role": "assistant", "content": reply})
        return reply

    # Step 1, the entry point: kickoff runs this first. The agent answers the customer.
    @start()
    async def handle_message(self):
        self.state.reply = await self._agent_turn(self.state.message)

    # Step 2, a router: runs after handle_message, and the label it returns picks the next step.
    @router(handle_message)
    def check_refund(self):
        return "needs_approval" if self.state.pending_refund else "done"  # nothing listens for "done"

    # Step 3, on "needs_approval": @human_feedback pauses the flow for a person once this method returns.
    # The person's reply is mapped by FEEDBACK_LLM onto one of the emit labels, "approve" or "reject".
    @listen("needs_approval")
    @human_feedback(message="Approve this refund? Reply approve or reject.", emit=["approve", "reject"],
                    llm=FEEDBACK_LLM, provider=SupervisorQueue())
    def ask_supervisor(self):
        return self.state.pending_refund  # what the supervisor is shown

    # Step 4a, on "approve": code, not the model, issues exactly the refund the supervisor saw.
    @listen("approve")
    def issue_approved_refund(self):
        p = self.state.pending_refund
        result = json.loads(shipfast.issue_refund(p["order_id"], p["amount"], p["reason"]))
        self.state.pending_refund = None
        if result.get("status") == "issued":
            self.state.reply = (f"Good news: your refund of ${result['amount']:.2f} for order {result['order_id']} "
                                "has been approved and issued.")
        else:
            self.state.reply = "The approved refund could not be issued; a specialist will follow up."

    # Step 4b, on "reject": no refund. The agent is told, and must escalate to a human (policy P5).
    @listen("reject")
    async def escalate_rejected_refund(self):
        p = self.state.pending_refund
        self.state.pending_refund = None
        self.state.reply = await self._agent_turn(
            f"(Supervisor) The refund of ${p['amount']:.2f} for order {p['order_id']} was rejected. Do not "
            "refund. Call escalate_to_human, then tell the customer a specialist will follow up.")


async def resume(flow_id: str, decision: str) -> ShipFastApprovalFlow:
    """Restore a paused flow from its saved state and continue it with the supervisor's decision."""
    flow = ShipFastApprovalFlow.from_pending(flow_id, persistence=PERSISTENCE)  # load the saved state
    await flow.resume_async(decision)  # the decision is the supervisor's reply; the flow continues from step 3
    return flow
