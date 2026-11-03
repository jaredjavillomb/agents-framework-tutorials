"""Run the shared evaluations for the LangGraph build and write results/<agent>/langgraph.json.

    uv run python run_eval.py                 # both agents
    uv run python run_eval.py shipfast        # one agent
    uv run python run_eval.py research Q01_headcount Q03_revenue_growth   # a subset (not written)

A subset is printed but not written, so results files always cover every scenario or question.
"""

from __future__ import annotations

import asyncio
import sys
from importlib.metadata import version

from agent_contracts.harness import RunMetrics, load_settings, verify_model
from agent_contracts.harness.runners import run_research_eval, run_shipfast_eval
from lg_agents.llm import make_llm
from lg_agents.research_flow import answer, build_research_flow
from lg_agents.shipfast import LangGraphShipFast

FRAMEWORK = "langgraph"


async def main(args: list[str]) -> None:
    verify_model()
    agents = [a for a in args if a in ("shipfast", "research")] or ["shipfast", "research"]
    ids = [a for a in args if a not in ("shipfast", "research")] or None
    model, fw_version = load_settings().model, version("langgraph")
    llm = make_llm()

    if "shipfast" in agents:
        print("== ShipFast")
        out = await run_shipfast_eval(LangGraphShipFast(llm), FRAMEWORK, fw_version, model,
                                      scenario_ids=ids, write=ids is None)
        print("written:", out["path"])

    if "research" in agents:
        print("== Research")
        flow = build_research_flow(llm)

        async def answer_question(question_id: str, question: str, metrics: RunMetrics):
            return await answer(question_id, question, metrics, flow=flow)

        out = await run_research_eval(answer_question, FRAMEWORK, fw_version, model, question_ids=ids, write=ids is None)
        print("written:", out["path"])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
