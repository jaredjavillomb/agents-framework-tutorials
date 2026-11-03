"""The ShipFast support agent in CrewAI (ShipFast SPEC.md), the capstone of notebooks 01 to 07.

One Flow run per customer turn; the conversation history travels in the flow's state from turn to turn.

    triage (router) --"out_of_scope"--> send_decline
                    \\--"in_scope"----> run_specialist (router) --"finished"--> done
                                                               \\--"needs_approval"--> ask_supervisor
                                                                    (@human_feedback: pauses, state saved)
                                                                    --"approve"--> issue_approved_refund (code)
                                                                    --"reject"---> escalate_rejected_refund (agent)

The approval gate works as in notebook 06 (crew_agents/approval.py): the specialist's `issue_refund` is
gated above the shared $100 limit, the flow pauses between steps, and it resumes only from saved state.
The Resolution is built by the shared `derive_resolution` from the ledger and tickets (DECISIONS.md 16);
the model only writes the reply and, when a turn left no record, picks the `Outcome`.
"""

from __future__ import annotations

import json
import uuid

from crewai import Agent
from crewai.flow import human_feedback
from crewai.flow.async_feedback import HumanFeedbackPending
from crewai.flow.flow import Flow, listen, router, start
from crewai.flow.persistence import persist
from crewai.tools import tool
from pydantic import BaseModel

from agent_contracts.harness import RunMetrics
from agent_contracts.harness.runners import Turn
from agent_contracts.shipfast import tools as shipfast
from agent_contracts.shipfast.prompts import OUTCOME, TRIAGE
from agent_contracts.shipfast.resolution import TurnMarker, derive_resolution
from agent_contracts.shipfast.schema import Outcome, Triage
from agent_contracts.shipfast.tools import APPROVAL_THRESHOLD
from crew_agents.approval import FEEDBACK_LLM, PERSISTENCE, SupervisorQueue
from crew_agents.llm import make_llm
from crew_agents.metrics import harness_listener
from crew_agents.prompts import BACKSTORY, GOAL, ROLE

DECLINE = "Sorry, I can only help with orders, deliveries, refunds and complaints."
WAITING = "Thanks for your patience: a refund of this size needs a supervisor's approval, and you will hear back shortly."
LLM = make_llm()


class Conversation(BaseModel):
    message: str = ""
    history: list[dict] = []
    triage: dict = {}
    pending_refund: dict | None = None
    reply: str = ""


@persist(PERSISTENCE)
class ShipFastFlow(Flow[Conversation]):
    def __init__(self, **kwargs):
        kwargs.setdefault("suppress_flow_events", True)
        super().__init__(**kwargs)

    def _tools(self):
        def issue_refund(order_id: str, amount: float, reason: str) -> str:
            """Issue a refund for an order. The amount cannot exceed the order total minus anything already refunded."""
            try:
                value = float(amount)
            except (TypeError, ValueError):
                value = None
            if value is not None and value > APPROVAL_THRESHOLD:
                self.state.pending_refund = {"order_id": order_id, "amount": value, "reason": reason}
                return json.dumps({"status": "pending_approval", "order_id": order_id, "amount": value,
                                   "next_step": "Do not call issue_refund again. Tell the customer a supervisor "
                                                "must approve this refund and they will hear back shortly."})
            return shipfast.issue_refund(order_id, amount, reason)

        others = [f for f in shipfast.TOOLS if f.__name__ != "issue_refund"]
        return [tool(f.__name__)(f) for f in others] + [tool("issue_refund")(issue_refund)]

    async def _agent_reply(self, text: str) -> str:
        agent = Agent(role=ROLE, goal=GOAL, backstory=BACKSTORY, tools=self._tools(), llm=LLM, verbose=False)
        self.state.history.append({"role": "user", "content": text})
        reply = (await agent.kickoff_async(self.state.history)).raw
        self.state.history.append({"role": "assistant", "content": reply})
        return reply

    @start()
    def triage(self):
        messages = [{"role": "system", "content": TRIAGE}] + self.state.history + [
            {"role": "user", "content": self.state.message}]
        self.state.triage = LLM.call(messages, response_model=Triage).model_dump()

    @router(triage)
    def route(self):
        return "out_of_scope" if self.state.triage["intent"] == "other" else "in_scope"

    @listen("out_of_scope")
    def send_decline(self):
        self.state.history += [{"role": "user", "content": self.state.message}, {"role": "assistant", "content": DECLINE}]
        self.state.reply = DECLINE

    @listen("in_scope")
    async def run_specialist(self):
        self.state.reply = await self._agent_reply(self.state.message)

    @router(run_specialist)
    def check_refund(self):
        return "needs_approval" if self.state.pending_refund else "finished"

    @listen("needs_approval")
    @human_feedback(message="Approve this refund? Reply approve or reject.", emit=["approve", "reject"],
                    llm=FEEDBACK_LLM, provider=SupervisorQueue())
    def ask_supervisor(self):
        return self.state.pending_refund

    @listen("approve")
    def issue_approved_refund(self):
        p = self.state.pending_refund
        result = json.loads(shipfast.issue_refund(p["order_id"], p["amount"], p["reason"]))
        self.state.pending_refund = None
        self.state.reply = (f"Good news: your refund of ${result['amount']:.2f} for order {result['order_id']} has "
                            "been approved and issued." if result.get("status") == "issued"
                            else "The approved refund could not be issued; a specialist will follow up.")
        self.state.history.append({"role": "assistant", "content": self.state.reply})

    @listen("reject")
    async def escalate_rejected_refund(self):
        p = self.state.pending_refund
        self.state.pending_refund = None
        self.state.reply = await self._agent_reply(
            f"(Supervisor) The refund of ${p['amount']:.2f} for order {p['order_id']} was rejected. Do not "
            "refund. Call escalate_to_human, then tell the customer a specialist will follow up.")


class CrewAIShipFast:
    """The adapter the shared runner drives (agent_contracts.harness.runners)."""

    async def start(self) -> dict:
        return {"id": str(uuid.uuid4()), "history": [], "marker": None, "flow_id": None}

    async def _finish(self, conv: dict, flow: ShipFastFlow, result, metrics: RunMetrics) -> Turn:
        t = Triage(**flow.state.triage)
        conv["history"] = list(flow.state.history)
        if isinstance(result, HumanFeedbackPending):
            conv["flow_id"] = result.context.flow_id
            return Turn(derive_resolution(t, conv["marker"], WAITING, paused=True), paused=True)
        reply = flow.state.reply
        recorded = shipfast.refunds()[conv["marker"].refunds:] or shipfast.tickets()[conv["marker"].tickets:]
        outcome = None
        if t.intent != "other" and not recorded:
            with harness_listener.measuring(metrics):
                outcome = LLM.call([{"role": "system", "content": OUTCOME},
                                    {"role": "user", "content": f"The agent's last reply:\n{reply}"}], response_model=Outcome)
        return Turn(derive_resolution(t, conv["marker"], reply, outcome=outcome))

    async def send(self, conv: dict, text: str, metrics: RunMetrics) -> Turn:
        conv["marker"] = TurnMarker.now()
        flow = ShipFastFlow()
        with harness_listener.measuring(metrics):
            result = await flow.kickoff_async(inputs={"message": text, "history": conv["history"]})
        return await self._finish(conv, flow, result, metrics)

    async def resume(self, conv: dict, decision: str, metrics: RunMetrics) -> Turn:
        flow = ShipFastFlow.from_pending(conv["flow_id"], persistence=PERSISTENCE)
        with harness_listener.measuring(metrics):
            result = await flow.resume_async(decision)
        return await self._finish(conv, flow, result, metrics)
