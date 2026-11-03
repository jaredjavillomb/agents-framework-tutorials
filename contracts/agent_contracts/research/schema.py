"""Research agent structured output (SPEC.md section 4).

Kept to shape only. The loop rule (which `critiques` and `revisions` combinations are valid) is checked
by `scoring.check_loop`, so a report that breaks it is recorded with a reason, not rejected on parse.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Critique(BaseModel):
    verdict: Literal["pass", "revise"] = Field(description="pass if the draft is supported and complete")
    issues: list[str] = Field(default_factory=list, description="What is unsupported, missing or outdated")


class Report(BaseModel):
    question_id: str
    plan: list[str] = Field(description="The steps planned before researching")
    answer: str = Field(description="A few sentences that cite article ids such as [F04]")
    citations: list[str] = Field(description="The article ids the answer relies on")
    critiques: list[Critique]
    revisions: int = Field(ge=0)
