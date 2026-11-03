"""The ShipFast support agent in Strands Agents (ShipFast SPEC.md), the capstone of notebooks 01 to 07.

- One Strands `Agent` per customer conversation: it keeps the conversation itself (notebook 04).
- Triage first, as a native structured call over the conversation so far (shared `prompts.TRIAGE`);
  out-of-scope messages get a fixed decline and never reach the agent (policy P7).
- The agent has the shared tools and instructions, and the `RefundApproval` hook, which interrupts
  before an `issue_refund` over the shared $100 limit; `resume` answers the interrupt.

The Resolution is built by the shared `derive_resolution` from the ledger and tickets (DECISIONS.md 16);
the model only writes the reply and, when a turn left no record, picks the `Outcome`.
"""

from __future__ import annotations

from strands import Agent

from agent_contracts.harness import RunMetrics
from agent_contracts.harness.runners import Turn
from agent_contracts.shipfast import tools as shipfast
from agent_contracts.shipfast.prompts import OUTCOME, SPECIALIST, TRIAGE
from agent_contracts.shipfast.resolution import TurnMarker, derive_resolution
from agent_contracts.shipfast.schema import Outcome, Triage
from strands_agents_impl.approval import RefundApproval, resume_with
from strands_agents_impl.llm import make_model
from strands_agents_impl.metrics import HarnessHook
from strands_agents_impl.structured import structured_call
from strands_agents_impl.tools import shipfast_tools

DECLINE = "Sorry, I can only help with orders, deliveries, refunds and complaints."
WAITING = "Thanks for your patience: a refund of this size needs a supervisor's approval, and you will hear back shortly."


def conversation_messages(agent: Agent) -> list[dict]:
    """The customer's messages and the agent's text replies so far, as chat messages, without tool calls
    and tool results: the same view of the conversation the LangGraph and CrewAI triage steps get."""
    out = []
    for message in agent.messages:
        texts = [b["text"] for b in message["content"] if "text" in b and b["text"].strip()]
        if texts:
            out.append({"role": message["role"], "content": [{"text": "\n".join(texts)}]})
    return out


class StrandsShipFast:
    """The adapter the shared runner drives (agent_contracts.harness.runners)."""

    def __init__(self, model=None):
        self.model = model or make_model()

    async def start(self) -> dict:
        hook = HarnessHook(RunMetrics())
        agent = Agent(model=self.model, tools=shipfast_tools(), system_prompt=SPECIALIST,
                      hooks=[RefundApproval(), hook], callback_handler=None)
        return {"agent": agent, "hook": hook, "marker": None, "triage": None, "paused_result": None}

    async def _finish(self, conv: dict, result, metrics: RunMetrics) -> Turn:
        t = conv["triage"]
        if result.stop_reason == "interrupt":
            conv["paused_result"] = result
            return Turn(derive_resolution(t, conv["marker"], WAITING, paused=True), paused=True)
        reply = str(result).strip()
        recorded = shipfast.refunds()[conv["marker"].refunds:] or shipfast.tickets()[conv["marker"].tickets:]
        outcome = None if recorded else structured_call(
            self.model, Outcome, f"The agent's last reply:\n{reply}", system=OUTCOME, metrics=metrics)
        return Turn(derive_resolution(t, conv["marker"], reply, outcome=outcome))

    async def send(self, conv: dict, text: str, metrics: RunMetrics) -> Turn:
        conv["marker"], conv["hook"].metrics = TurnMarker.now(), metrics
        agent: Agent = conv["agent"]
        messages = conversation_messages(agent) + [{"role": "user", "content": [{"text": text}]}]
        conv["triage"] = structured_call(self.model, Triage, system=TRIAGE, metrics=metrics, messages=messages)
        if conv["triage"].intent == "other":
            agent.messages.extend([{"role": "user", "content": [{"text": text}]},
                                   {"role": "assistant", "content": [{"text": DECLINE}]}])
            return Turn(derive_resolution(conv["triage"], conv["marker"], DECLINE))
        result = await agent.invoke_async(text)
        return await self._finish(conv, result, metrics)

    async def resume(self, conv: dict, decision: str, metrics: RunMetrics) -> Turn:
        conv["hook"].metrics = metrics
        result = await conv["agent"].invoke_async(resume_with(conv["paused_result"], decision))
        return await self._finish(conv, result, metrics)
