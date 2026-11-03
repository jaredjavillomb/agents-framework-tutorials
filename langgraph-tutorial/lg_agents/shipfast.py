"""The ShipFast support agent in LangGraph (ShipFast SPEC.md), the capstone of notebooks 01 to 07.

    START -> triage --(intent == other)--> decline -> END
                    \\-----(otherwise)----> specialist -> END

- triage: structured `Triage` from the conversation so far (shared `prompts.TRIAGE`).
- decline: a fixed reply, no model call (policy P7).
- specialist: the `create_agent` loop with the shared tools and instructions, added to the graph as a
  subgraph, with the $100 approval middleware (`interrupt()` before a large refund).
- A checkpointer keeps each conversation (thread) and holds a paused approval until it is resumed.

The Resolution is built by the shared `derive_resolution` from the ledger and tickets (DECISIONS.md 16);
the model only writes the reply and, when a turn left no record, picks the `Outcome`.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from agent_contracts.harness import RunMetrics
from agent_contracts.harness.runners import Turn
from agent_contracts.shipfast import tools as shipfast
from agent_contracts.shipfast.prompts import OUTCOME, SPECIALIST, TRIAGE
from agent_contracts.shipfast.resolution import TurnMarker, derive_resolution
from agent_contracts.shipfast.schema import Outcome, Triage
from lg_agents.approval import refund_approval
from lg_agents.llm import make_llm
from lg_agents.metrics import HarnessCallback
from lg_agents.tools import shipfast_tools

DECLINE = "Sorry, I can only help with orders, deliveries, refunds and complaints."
WAITING = "Thanks for your patience: a refund of this size needs a supervisor's approval, and you will hear back shortly."


class ShipFastState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    triage: dict  # a Triage, stored as a plain dict so the checkpoint holds only plain data


def conversation_text(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The customer's messages and the agent's replies, without tool calls and tool results."""
    return [m for m in messages if m.type == "human" or (m.type == "ai" and not m.tool_calls and m.content)]


def build_shipfast(llm=None, checkpointer=None):
    llm = llm or make_llm()
    specialist = create_agent(llm, shipfast_tools(), system_prompt=SPECIALIST, middleware=[refund_approval])

    def triage(state: ShipFastState) -> ShipFastState:
        t = llm.with_structured_output(Triage).invoke([SystemMessage(TRIAGE)] + conversation_text(state["messages"]))
        return {"triage": t.model_dump()}

    def decline(state: ShipFastState) -> ShipFastState:
        return {"messages": [AIMessage(DECLINE)]}

    def route(state: ShipFastState) -> str:
        return "decline" if state["triage"]["intent"] == "other" else "specialist"

    g = StateGraph(ShipFastState)
    g.add_node("triage", triage)
    g.add_node("decline", decline)
    g.add_node("specialist", specialist)
    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", route, ["decline", "specialist"])
    g.add_edge("decline", END)
    g.add_edge("specialist", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


class LangGraphShipFast:
    """The adapter the shared runner drives (agent_contracts.harness.runners)."""

    def __init__(self, llm=None):
        self.llm = llm or make_llm()
        self.graph = build_shipfast(self.llm)

    async def start(self) -> dict:
        return {"thread_id": str(uuid.uuid4()), "marker": None}

    def _config(self, conv: dict, metrics: RunMetrics) -> dict:
        return {"configurable": {"thread_id": conv["thread_id"]}, "callbacks": [HarnessCallback(metrics)]}

    async def _finish(self, conv: dict, state: dict, metrics: RunMetrics) -> Turn:
        t = Triage(**state["triage"])
        if state.get("__interrupt__"):
            return Turn(derive_resolution(t, conv["marker"], WAITING, paused=True), paused=True)
        reply = next((m.content for m in reversed(state["messages"]) if m.type == "ai" and m.content and not m.tool_calls), "")
        recorded = (shipfast.refunds()[conv["marker"].refunds:] or shipfast.tickets()[conv["marker"].tickets:])
        outcome = None
        if t.intent != "other" and not recorded:
            outcome = await self.llm.with_structured_output(Outcome).ainvoke(
                [SystemMessage(OUTCOME), HumanMessage(f"The agent's last reply:\n{reply}")],
                config={"callbacks": [HarnessCallback(metrics)]})
        return Turn(derive_resolution(t, conv["marker"], reply, outcome=outcome))

    async def send(self, conv: dict, text: str, metrics: RunMetrics) -> Turn:
        conv["marker"] = TurnMarker.now()
        # The approval middleware is synchronous, so the graph runs with invoke() in a worker thread.
        state = await asyncio.to_thread(self.graph.invoke, {"messages": [HumanMessage(text)]}, self._config(conv, metrics))
        return await self._finish(conv, state, metrics)

    async def resume(self, conv: dict, decision: str, metrics: RunMetrics) -> Turn:
        state = await asyncio.to_thread(self.graph.invoke, Command(resume=decision), self._config(conv, metrics))
        return await self._finish(conv, state, metrics)
