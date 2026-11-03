"""Check ShipFast resolutions against scenarios.json (SPEC.md section 6)."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Resolution

SCENARIOS: list[dict] = json.loads((Path(__file__).parent / "scenarios.json").read_text())
CHECKED_FIELDS = ("intent", "order_id", "action", "refund_amount")


def load_scenarios() -> list[dict]:
    return [dict(s) for s in SCENARIOS]


def _same(field: str, got, want) -> bool:
    if field == "refund_amount" and got is not None and want is not None:
        return round(float(got), 2) == round(float(want), 2)
    return got == want


def check_resolution(resolution: Resolution | dict, expected: dict) -> list[str]:
    """Return the mismatched fields as readable strings; an empty list means a match."""
    got = resolution.model_dump() if isinstance(resolution, Resolution) else dict(resolution)
    return [f"{f}: expected {expected.get(f)!r}, got {got.get(f)!r}"
            for f in CHECKED_FIELDS if not _same(f, got.get(f), expected.get(f))]


def check_ledger(resolution: Resolution | dict, refunds: list[dict]) -> list[str]:
    """`refunded` must be backed by a real refund in the ledger, for the same order and amount."""
    got = resolution.model_dump() if isinstance(resolution, Resolution) else dict(resolution)
    if got.get("action") != "refunded":
        return []
    matching = [r for r in refunds if r["order_id"] == got.get("order_id")
                and got.get("refund_amount") is not None
                and round(r["amount"], 2) == round(float(got["refund_amount"]), 2)]
    return [] if matching else ["action is 'refunded' but the ledger has no matching refund"]
