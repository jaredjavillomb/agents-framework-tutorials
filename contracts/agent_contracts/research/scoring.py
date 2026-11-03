"""Deterministic scoring for research reports (SPEC.md section 5). No LLM judge."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import Report
from .tools import FIXTURE_IDS

QUESTIONS: list[dict] = json.loads((Path(__file__).parent / "questions.json").read_text())
PASS_RECALL = 0.8
VALID_LOOPS = {(("pass",), 0), (("revise", "pass"), 1), (("revise", "revise"), 2)}


def load_questions() -> list[dict]:
    return [dict(q) for q in QUESTIONS]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"(?<=\d),(?=\d)", "", text.lower())).strip()


def matches(alias: str, normalized_text: str) -> bool:
    """Whole-token match: the characters just before and after must not be letters or digits."""
    return re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", normalized_text) is not None


def check_loop(report: Report) -> str | None:
    """Return why the critique loop is invalid, or None if it follows the loop rule."""
    shape = (tuple(c.verdict for c in report.critiques), report.revisions)
    if shape in VALID_LOOPS:
        return None
    return f"invalid critique loop: verdicts {list(shape[0])} with {report.revisions} revisions"


def score_report(report: Report | dict, question: dict) -> dict:
    """Score one report. `passed` also requires a valid loop; `problems` says why something failed."""
    report = report if isinstance(report, Report) else Report.model_validate(report)
    text = normalize(report.answer)
    found = [f["id"] for f in question["expected_facts"] if any(matches(a, text) for a in f["aliases"])]
    missing = [f["id"] for f in question["expected_facts"] if f["id"] not in found]
    stale = [m for m in question.get("must_not_contain", []) if matches(m, text)]
    unknown = [c for c in report.citations if c not in FIXTURE_IDS]
    recall = len(found) / len(question["expected_facts"])

    problems = []
    if stale:
        problems.append(f"repeats stale or wrong figures: {stale}")
    if not report.citations:
        problems.append("no citations")
    if unknown:
        problems.append(f"cites unknown articles: {unknown}")
    if recall < PASS_RECALL:
        problems.append(f"missing facts: {missing}")
    loop_problem = check_loop(report)
    if loop_problem:
        problems.append(loop_problem)
    if report.question_id != question["id"]:
        problems.append(f"question_id {report.question_id!r} does not match {question['id']!r}")

    return {
        "passed": not problems,
        "scores": {
            "fact_recall": round(recall, 3),
            "facts_missing": missing,
            "contradiction_free": not stale,
            "citations_valid": bool(report.citations) and not unknown,
        },
        "problems": problems,
    }
