"""The research agent as a CrewAI Flow (research SPEC.md section 1), taught in notebook 05.

Workflow outside, agent inside, as in lg_agents/research_flow.py and with the same instructions: fixed
steps (plan, draft, critique, revise) and a tool-using CrewAI agent for the research step. The revision
limit is enforced in code by the router, never by the prompt; the tool budget (5) by the shared tools.

    plan -> research -> draft -> critique (router) --"done"--> END
                                    ^        \\--"needs_revision"--> revise --+
                                    +---------------------------------------+
    The router skips the critique and returns "done" once 2 revisions have run.
"""

from __future__ import annotations

from crewai import Agent
from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel, Field

from agent_contracts.research.prompts import CRITIC, PLANNER, RESEARCHER, WRITER  # shared by all builds
from agent_contracts.research.schema import Critique, Report
from crew_agents.llm import make_llm
from crew_agents.tools import research_tools

MAX_REVISIONS = 2


class Plan(BaseModel):
    steps: list[str] = Field(description="2 to 4 short research steps")


class Draft(BaseModel):
    answer: str = Field(description="A few sentences that cite article ids in square brackets, such as [F04]")
    citations: list[str] = Field(description="The article ids the answer relies on, such as F04")


class ResearchState(BaseModel):
    question_id: str = ""
    question: str = ""
    plan: list[str] = []
    findings: str = ""
    answer: str = ""
    citations: list[str] = []
    critiques: list[dict] = []
    revisions: int = 0


def critique_draft(llm, question: str, findings: str, answer: str, citations: list[str]) -> Critique:
    prompt = f"Question: {question}\n\nFindings:\n{findings}\n\nDraft answer:\n{answer}\nCitations: {citations}"
    return llm.call([{"role": "system", "content": CRITIC}, {"role": "user", "content": prompt}],
                    response_model=Critique)


class ResearchFlow(Flow[ResearchState]):
    def __init__(self, llm=None, **kwargs):
        kwargs.setdefault("suppress_flow_events", True)  # no console panels for every step
        super().__init__(**kwargs)
        self.llm = llm or make_llm()
        self.researcher = Agent(role="Researcher", goal="Find the facts that answer the question, with article ids",
                                backstory=RESEARCHER, tools=research_tools(), llm=self.llm, verbose=False)

    def _write(self, extra: str = "") -> Draft:
        prompt = f"Question: {self.state.question}\n\nFindings:\n{self.state.findings}{extra}"
        return self.llm.call([{"role": "system", "content": WRITER}, {"role": "user", "content": prompt}],
                             response_model=Draft)

    @start()
    def plan(self):
        p = self.llm.call([{"role": "system", "content": PLANNER},
                           {"role": "user", "content": self.state.question}], response_model=Plan)
        self.state.plan = p.steps

    @listen(plan)
    async def research(self):
        task = f"Question: {self.state.question}\nPlan:\n" + "\n".join(f"- {s}" for s in self.state.plan)
        result = await self.researcher.kickoff_async(task)
        tool_results = [m["content"] for m in (result.messages or []) if m.get("role") == "tool"]
        self.state.findings = ("Tool results:\n" + "\n".join(map(str, tool_results))
                               + "\n\nResearcher's summary:\n" + result.raw)

    @listen(research)
    def draft(self):
        d = self._write()
        self.state.answer, self.state.citations = d.answer, d.citations

    @router(or_(draft, "revise"))  # "revise" here is the method below finishing
    def critique(self):
        # A router's return value is a label: a method decorated @listen("<label>") runs next.
        if self.state.revisions >= MAX_REVISIONS:  # the limit, enforced in code: no third critique
            return "done"  # nothing listens for "done", so the flow ends here
        # One structured model call with the critic's prompt (not a separate agent): a verdict and issues.
        c = critique_draft(self.llm, self.state.question, self.state.findings, self.state.answer, self.state.citations)
        self.state.critiques.append(c.model_dump())  # kept in the flow's state, for the report
        return "needs_revision" if c.verdict == "revise" else "done"  # "needs_revision" triggers revise()

    @listen("needs_revision")
    def revise(self):
        issues = "\n".join(f"- {i}" for i in self.state.critiques[-1]["issues"]) or "- (no details given)"
        d = self._write(f"\n\nYour previous draft:\n{self.state.answer}\n\nFix these problems:\n{issues}")
        self.state.answer, self.state.citations = d.answer, d.citations
        self.state.revisions += 1


def to_report(state: ResearchState) -> Report:
    return Report(question_id=state.question_id, plan=state.plan, answer=state.answer, citations=state.citations,
                  critiques=[Critique(**c) for c in state.critiques], revisions=state.revisions)


async def answer(question_id: str, question: str, metrics=None) -> Report:
    """Answer one question with a fresh flow: the function the shared research runner calls."""
    from contextlib import nullcontext

    from crew_agents.metrics import harness_listener

    flow = ResearchFlow()
    with harness_listener.measuring(metrics) if metrics is not None else nullcontext():
        await flow.kickoff_async(inputs={"question_id": question_id, "question": question})
    return to_report(flow.state)
