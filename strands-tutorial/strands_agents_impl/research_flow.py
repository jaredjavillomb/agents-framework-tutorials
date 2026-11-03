"""The research agent in Strands (research SPEC.md section 1), taught in notebook 05.

Strands has no workflow engine with typed state; its multi-agent tools (`GraphBuilder`, `Swarm`) connect
agents and pass text between them. So the fixed steps here are plain Python around Strands agents, with
the same instructions as the LangGraph and CrewAI builds: plan, research (a tool-using agent), draft, then
critique and revise. The structured steps use `structured_call` (see structured.py for why). The revision limit is enforced by the loop, in code, never by the prompt; the tool
budget (5) by the shared tools.

    plan -> research -> draft -> critique --pass--> done
                                   ^   \\--revise--> revise --(revisions < 2)--> critique
                                                       \\--(revisions == 2)--> done
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from strands import Agent

from agent_contracts.harness import RunMetrics
from agent_contracts.research.prompts import CRITIC, PLANNER, RESEARCHER, WRITER  # shared by all builds
from agent_contracts.research.schema import Critique, Report
from strands_agents_impl.llm import make_model
from strands_agents_impl.metrics import HarnessHook
from strands_agents_impl.structured import structured_call
from strands_agents_impl.tools import research_tools

MAX_REVISIONS = 2


class Plan(BaseModel):
    steps: list[str] = Field(description="2 to 4 short research steps")


class Draft(BaseModel):
    answer: str = Field(description="A few sentences that cite article ids in square brackets, such as [F04]")
    citations: list[str] = Field(description="The article ids the answer relies on, such as F04")


def critique_draft(model, question: str, findings: str, answer: str, citations: list[str],
                   metrics: RunMetrics | None = None) -> Critique:
    prompt = f"Question: {question}\n\nFindings:\n{findings}\n\nDraft answer:\n{answer}\nCitations: {citations}"
    return structured_call(model, Critique, prompt, system=CRITIC, metrics=metrics)


def run_research(question_id: str, question: str, model=None, metrics: RunMetrics | None = None) -> dict:
    """Run the whole research workflow for one question and return its state. With `metrics`, every model
    call is counted: the structured steps directly, the research agent through `HarnessHook`."""
    model = model or make_model()
    hooks = [HarnessHook(metrics)] if metrics is not None else []
    state = {"question_id": question_id, "question": question, "critiques": [], "revisions": 0}

    state["plan"] = structured_call(model, Plan, question, system=PLANNER,
                                    metrics=metrics).steps

    researcher = Agent(model=model, tools=research_tools(), system_prompt=RESEARCHER, hooks=hooks, callback_handler=None)
    summary = str(researcher(f"Question: {question}\nPlan:\n" + "\n".join(f"- {s}" for s in state["plan"])))
    tool_results = [block["toolResult"]["content"][0]["text"] for message in researcher.messages
                    for block in message["content"] if "toolResult" in block]
    state["findings"] = "Tool results:\n" + "\n".join(tool_results) + "\n\nResearcher's summary:\n" + summary

    def write(extra: str = "") -> None:
        d = structured_call(model, Draft, f"Question: {question}\n\nFindings:\n{state['findings']}{extra}",
                            system=WRITER, metrics=metrics)
        state["answer"], state["citations"] = d.answer, d.citations

    write()
    while True:
        c = critique_draft(model, question, state["findings"], state["answer"], state["citations"], metrics)
        state["critiques"].append(c.model_dump())
        if c.verdict == "pass":
            break
        issues = "\n".join(f"- {i}" for i in c.issues) or "- (no details given)"
        write(f"\n\nYour previous draft:\n{state['answer']}\n\nFix these problems:\n{issues}")
        state["revisions"] += 1
        if state["revisions"] >= MAX_REVISIONS:  # the limit, enforced in code: no third critique
            break
    return state


def to_report(state: dict) -> Report:
    return Report(question_id=state["question_id"], plan=state.get("plan", []), answer=state.get("answer", ""),
                  citations=state.get("citations", []),
                  critiques=[Critique(**c) for c in state.get("critiques", [])], revisions=state.get("revisions", 0))


async def answer(question_id: str, question: str, metrics: RunMetrics | None = None, model=None) -> Report:
    """Answer one question: the function the shared research runner calls (Strands agents are synchronous
    here, so the workflow runs in a worker thread)."""
    import asyncio

    return to_report(await asyncio.to_thread(run_research, question_id, question, model, metrics))
