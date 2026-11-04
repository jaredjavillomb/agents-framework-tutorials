"""Write results/<agent>/<framework>.json in the shared format from each agent's SPEC.md.

Each item is one scenario (ShipFast) or one question (research) and must carry `passed`,
`input_tokens`, `output_tokens` and `latency_s`; the summary is computed here so every build sums the
same way. `RESULTS_DIR` overrides the output folder (the default is `results/` at the repo root).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Agent = Literal["shipfast", "research"]
Framework = Literal["langgraph", "crewai", "strands"]

REPO_ROOT = Path(__file__).resolve().parents[3]
_ITEMS_KEY = {"shipfast": "scenarios", "research": "questions"}


def summarize(items: list[dict]) -> dict:
    return {
        "passed": sum(1 for i in items if i.get("passed")),
        "total": len(items),
        "total_tokens": sum(int(i.get("input_tokens", 0)) + int(i.get("output_tokens", 0)) for i in items),
        "total_latency_s": round(sum(float(i.get("latency_s", 0.0)) for i in items), 3),
    }


def build_results(agent: Agent, framework: Framework, framework_version: str, model: str,
                  items: list[dict], run_at: datetime | None = None) -> dict:
    if agent not in _ITEMS_KEY:
        raise ValueError(f"unknown agent {agent!r}")
    return {
        "framework": framework,
        "framework_version": framework_version,
        "model": model,
        "run_at": (run_at or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        _ITEMS_KEY[agent]: items,
        "summary": summarize(items),
    }


def write_results(agent: Agent, framework: Framework, framework_version: str, model: str,
                  items: list[dict], results_dir: Path | None = None) -> Path:
    """Write the results file and return its path."""
    base = results_dir or Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "results"))
    path = Path(base) / agent / f"{framework}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = build_results(agent, framework, framework_version, model, items)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
