"""The research agent as a LangGraph workflow (research SPEC.md section 1), taught in notebook 05.

Workflow outside, agent inside: fixed steps (plan, draft, critique, revise) with a small tool-using agent
for the research step. The revision limit is enforced by the graph's edges, in code, never by the prompt.
The tool-call budget (5) is enforced by the shared tools.

    plan -> research -> draft -> critique --pass--> END
                                   ^   \\--revise--> revise --(revisions < 2)--> critique
                                                        \\--(revisions == 2)--> END
"""

from __future__ import annotations

import json
from typing import TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent_contracts.research.schema import Critique, Report
from lg_agents.llm import make_llm
from lg_agents.tools import research_tools

MAX_REVISIONS = 2


class Plan(BaseModel):
    steps: list[str] = Field(description="2 to 4 short research steps")


class Draft(BaseModel):
    answer: str = Field(description="A few sentences that cite article ids in square brackets, such as [F04]")
    citations: list[str] = Field(description="The article ids the answer relies on, such as F04")


class ResearchState(TypedDict, total=False):
    question_id: str
    question: str
    plan: list[str]
    findings: str
    answer: str
    citations: list[str]
    critiques: list[dict]
    revisions: int


RESEARCHER = """You research questions about the company Veltara Aquatech using web_search, which returns
short articles, each with an id (such as F04) and a published date. Use calculator for any arithmetic.
You may make at most 5 tool calls in total. Search with a few specific keywords. When articles disagree,
the most recently published one is current. Finish with a short list of the facts you found, each with
its article id and published date, and any calculation results."""

WRITER = """Answer the question in two or three sentences using only the findings given. State only current
facts: when sources disagree, use the most recently published one and do not mention older figures. Cite
the article ids you used in square brackets, such as [F04]."""

CRITIC = """You check a draft answer against research findings. Verdict "revise" if any fact in the draft is
not supported by the findings, if a fact the question asks for is missing, if it uses a figure from an
older article that a newer article contradicts, or if it mentions an outdated figure at all. Otherwise
verdict "pass". List each problem in issues."""


def critique_draft(llm, question: str, findings: str, answer: str, citations: list[str]) -> Critique:
    prompt = f"Question: {question}\n\nFindings:\n{findings}\n\nDraft answer:\n{answer}\nCitations: {citations}"
    return llm.with_structured_output(Critique).invoke([SystemMessage(CRITIC), HumanMessage(prompt)])


def route_after_critique(state: ResearchState) -> str:
    return "revise" if state["critiques"][-1]["verdict"] == "revise" else END


def route_after_revise(state: ResearchState) -> str:
    return "critique" if state["revisions"] < MAX_REVISIONS else END  # the limit, enforced in code


def build_research_flow(llm=None):
    llm = llm or make_llm()
    researcher = create_agent(llm, research_tools(), system_prompt=RESEARCHER)

    def plan(state: ResearchState) -> ResearchState:
        p = llm.with_structured_output(Plan).invoke(
            [SystemMessage("Plan how to research this question in 2 to 4 short steps."), HumanMessage(state["question"])])
        return {"plan": p.steps, "revisions": 0, "critiques": []}

    def research(state: ResearchState) -> ResearchState:
        task = f"Question: {state['question']}\nPlan:\n" + "\n".join(f"- {s}" for s in state["plan"])
        result = researcher.invoke({"messages": [HumanMessage(task)]})
        tool_results = [m.content for m in result["messages"] if m.type == "tool"]
        findings = "Tool results:\n" + "\n".join(tool_results) + "\n\nResearcher's summary:\n" + result["messages"][-1].content
        return {"findings": findings}

    def _write(state: ResearchState, extra: str = "") -> Draft:
        prompt = f"Question: {state['question']}\n\nFindings:\n{state['findings']}{extra}"
        return llm.with_structured_output(Draft).invoke([SystemMessage(WRITER), HumanMessage(prompt)])

    def draft(state: ResearchState) -> ResearchState:
        d = _write(state)
        return {"answer": d.answer, "citations": d.citations}

    def critique(state: ResearchState) -> ResearchState:
        c = critique_draft(llm, state["question"], state["findings"], state["answer"], state["citations"])
        return {"critiques": state["critiques"] + [c.model_dump()]}

    def revise(state: ResearchState) -> ResearchState:
        issues = "\n".join(f"- {i}" for i in state["critiques"][-1]["issues"]) or "- (no details given)"
        d = _write(state, f"\n\nYour previous draft:\n{state['answer']}\n\nFix these problems:\n{issues}")
        return {"answer": d.answer, "citations": d.citations, "revisions": state["revisions"] + 1}

    g = StateGraph(ResearchState)
    for name, fn in [("plan", plan), ("research", research), ("draft", draft), ("critique", critique), ("revise", revise)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "research")
    g.add_edge("research", "draft")
    g.add_edge("draft", "critique")
    g.add_conditional_edges("critique", route_after_critique, ["revise", END])
    g.add_conditional_edges("revise", route_after_revise, ["critique", END])
    return g.compile()


def to_report(state: ResearchState) -> Report:
    return Report(question_id=state["question_id"], plan=state.get("plan", []), answer=state.get("answer", ""),
                  citations=state.get("citations", []),
                  critiques=[Critique(**c) for c in state.get("critiques", [])], revisions=state.get("revisions", 0))
