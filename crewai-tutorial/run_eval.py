"""Run the shared evaluations for the CrewAI build and write results/<agent>/crewai.json.

    uv run python run_eval.py                 # both agents
    uv run python run_eval.py shipfast        # one agent
    uv run python run_eval.py research Q01_headcount Q03_revenue_growth   # a subset (not written)

A subset is printed but not written, so results files always cover every scenario or question.
"""

from __future__ import annotations

import asyncio
import sys
from importlib.metadata import version

import crew_agents  # noqa: F401  (first: switches CrewAI telemetry and console panels off)
from crew_agents.research_flow import answer
from crew_agents.shipfast import CrewAIShipFast

from agent_contracts.harness import load_settings, verify_model
from agent_contracts.harness.runners import run_research_eval, run_shipfast_eval

FRAMEWORK = "crewai"


async def main(args: list[str]) -> None:
    verify_model()
    agents = [a for a in args if a in ("shipfast", "research")] or ["shipfast", "research"]
    ids = [a for a in args if a not in ("shipfast", "research")] or None
    model, fw_version = load_settings().model, version("crewai")

    if "shipfast" in agents:
        print("== ShipFast")
        out = await run_shipfast_eval(CrewAIShipFast(), FRAMEWORK, fw_version, model, scenario_ids=ids, write=ids is None)
        print("written:", out["path"])

    if "research" in agents:
        print("== Research")

        out = await run_research_eval(answer, FRAMEWORK, fw_version, model, question_ids=ids, write=ids is None)
        print("written:", out["path"])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
