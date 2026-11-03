"""The evaluation runners every build's run_eval.py uses (ShipFast SPEC.md section 6, research section 6).

A build supplies a small adapter; the runner does everything that must be identical across frameworks:
resetting the shared tools, feeding turns, resuming paused approvals with the scenario's decision,
scoring, measuring and writing `results/<agent>/<framework>.json`.

ShipFast adapter (async):
    start() -> conversation            a fresh conversation (thread, history, agent ...)
    send(conversation, text, metrics) -> Turn
    resume(conversation, decision, metrics) -> Turn     continue a turn that paused for approval

Research adapter (async):
    answer(question_id, question, metrics) -> Report

Model calls are counted by the adapter through `metrics` (each framework's callbacks, events or hooks);
tool calls come from the shared tools' own logs.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from agent_contracts.research import tools as research_tools
from agent_contracts.research.schema import Report
from agent_contracts.research.scoring import load_questions, score_report
from agent_contracts.shipfast import tools as shipfast_tools
from agent_contracts.shipfast.schema import Resolution
from agent_contracts.shipfast.scoring import check_ledger, check_resolution, load_scenarios

from .metrics import RunMetrics
from .results import write_results


@dataclass
class Turn:
    resolution: Resolution
    paused: bool = False


class ShipFastAdapter(Protocol):
    async def start(self) -> Any: ...
    async def send(self, conversation: Any, text: str, metrics: RunMetrics) -> Turn: ...
    async def resume(self, conversation: Any, decision: str, metrics: RunMetrics) -> Turn: ...


async def run_scenario(adapter: ShipFastAdapter, scenario: dict) -> dict:
    """Run one ShipFast scenario and score it (SPEC.md section 6, steps 1 to 5)."""
    shipfast_tools.reset()
    metrics = RunMetrics()
    turns: list[dict] = []
    problems: list[str] = []
    error = None
    with metrics.measure():
        try:
            conversation = await adapter.start()
            for text in scenario["turns"]:
                turn = await adapter.send(conversation, text, metrics)
                if turn.paused:
                    if not scenario.get("approval_decision"):
                        problems.append(f"paused for approval, but the scenario has no decision: {text!r}")
                        turns.append(turn.resolution.model_dump())
                        break
                    turn = await adapter.resume(conversation, scenario["approval_decision"], metrics)
                turns.append(turn.resolution.model_dump())
        except Exception as exc:  # recorded, not raised, so one failure does not stop the evaluation
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
    final = turns[-1] if turns else None
    if final is None:
        problems.append("no resolution")
    else:
        problems += check_resolution(final, scenario["expected"])
        problems += check_ledger(final, shipfast_tools.refunds())
        for i, want in enumerate(scenario.get("expected_per_turn") or []):
            if i < len(turns):
                problems += [f"turn {i + 1}: {p}" for p in check_resolution(turns[i], want)]
            else:
                problems.append(f"turn {i + 1}: missing")
    return {"id": scenario["id"], "passed": not problems and error is None, "problems": problems,
            "resolution": final, "turn_resolutions": turns, "tool_calls": shipfast_tools.tool_calls(),
            **metrics.as_dict(), "error": error}


async def run_shipfast_eval(adapter: ShipFastAdapter, framework: str, framework_version: str, model: str,
                            scenario_ids: list[str] | None = None, results_dir: Path | None = None,
                            write: bool = True, log: Callable[[str], None] = print) -> dict:
    items = []
    for scenario in load_scenarios():
        if scenario_ids and scenario["id"] not in scenario_ids:
            continue
        item = await run_scenario(adapter, scenario)
        log(f"{'PASS' if item['passed'] else 'FAIL'} {item['id']:36} {item['latency_s']:6.1f}s  {item['problems'] or ''}")
        items.append(item)
    path = write_results("shipfast", framework, framework_version, model, items, results_dir) if write else None
    return {"items": items, "path": path}


async def run_question(answer: Callable[[str, str, RunMetrics], Awaitable[Report]], question: dict) -> dict:
    """Run and score one research question (research SPEC.md sections 5 and 6)."""
    research_tools.reset()
    metrics = RunMetrics()
    report, error = None, None
    with metrics.measure():
        try:
            report = await answer(question["id"], question["question"], metrics)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
    scored = score_report(report, question) if report is not None else {
        "passed": False, "scores": None, "problems": ["no report"]}
    return {"id": question["id"], "passed": scored["passed"] and error is None, "scores": scored["scores"],
            "problems": scored["problems"], "report": report.model_dump() if report else None,
            "tool_calls": research_tools.tool_calls(), "budget_exhausted": research_tools.budget_exhausted(),
            **metrics.as_dict(), "error": error}


async def run_research_eval(answer: Callable[[str, str, RunMetrics], Awaitable[Report]], framework: str,
                            framework_version: str, model: str, question_ids: list[str] | None = None,
                            results_dir: Path | None = None, write: bool = True,
                            log: Callable[[str], None] = print) -> dict:
    items = []
    for question in load_questions():
        if question_ids and question["id"] not in question_ids:
            continue
        item = await run_question(answer, question)
        log(f"{'PASS' if item['passed'] else 'FAIL'} {item['id']:22} {item['latency_s']:6.1f}s  {item['problems'] or ''}")
        items.append(item)
    path = write_results("research", framework, framework_version, model, items, results_dir) if write else None
    return {"items": items, "path": path}
