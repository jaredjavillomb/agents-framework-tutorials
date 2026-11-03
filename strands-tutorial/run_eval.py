"""Run the shared evaluations for the Strands Agents build and write results/<agent>/strands.json.

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
from strands_agents_impl.llm import make_model
from strands_agents_impl.research_flow import answer
from strands_agents_impl.shipfast import StrandsShipFast

FRAMEWORK = "strands"


async def main(args: list[str]) -> None:
    verify_model()
    agents = [a for a in args if a in ("shipfast", "research")] or ["shipfast", "research"]
    ids = [a for a in args if a not in ("shipfast", "research")] or None
    model_name, fw_version = load_settings().model, version("strands-agents")
    model = make_model()

    if "shipfast" in agents:
        print("== ShipFast")
        out = await run_shipfast_eval(StrandsShipFast(model), FRAMEWORK, fw_version, model_name,
                                      scenario_ids=ids, write=ids is None)
        print("written:", out["path"])

    if "research" in agents:
        print("== Research")

        async def answer_question(question_id: str, question: str, metrics: RunMetrics):
            return await answer(question_id, question, metrics, model=model)

        out = await run_research_eval(answer_question, FRAMEWORK, fw_version, model_name, question_ids=ids, write=ids is None)
        print("written:", out["path"])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
