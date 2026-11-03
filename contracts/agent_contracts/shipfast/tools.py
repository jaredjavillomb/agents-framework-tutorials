"""ShipFast tools (SPEC.md section 3): plain functions over the shared data, each returning a JSON string.

Builds wrap these in their framework's tool type and never reimplement them. The refund ledger, the
ticket list and the tool-call log are module state; the runner calls `reset()` before every scenario.
The $100 approval gate is NOT here: the spec puts it in each framework's control flow.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from agent_contracts.harness import ToolCallLog

DATA = Path(__file__).parent / "data"
ORDERS: dict[str, dict] = json.loads((DATA / "orders.json").read_text())
POLICY_TEXT = (DATA / "policy.md").read_text()

log = ToolCallLog()
_lock = threading.Lock()
_refunds: list[dict] = []
_tickets: list[dict] = []


def _policy_sections() -> list[dict]:
    sections = []
    for chunk in POLICY_TEXT.split("\n## ")[1:]:
        heading, _, body = chunk.partition("\n")
        sid, _, title = heading.partition(". ")
        sections.append({"id": sid.strip(), "title": title.strip(), "text": body.strip()})
    return sections


POLICY_SECTIONS = _policy_sections()
# The policy's "today" line sits above the first section, so splitting on "## " alone would drop it.
TODAY = re.search(r"\*\*(\d{4}-\d{2}-\d{2})\*\*", POLICY_TEXT).group(1)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 4}


def _order(order_id: str) -> dict | None:
    return ORDERS.get((order_id or "").strip().upper())


def _not_found(order_id: str) -> str:
    return json.dumps({"error": "order_not_found", "order_id": order_id})


def lookup_order(order_id: str) -> str:
    """Look up an order by the order ID from the conversation; never guess one. Returns the order without tracking events."""
    log.attempt("lookup_order")
    order = _order(order_id)
    if order is None:
        return _not_found(order_id)
    return json.dumps({k: v for k, v in order.items() if k != "tracking"})


def track_shipment(order_id: str) -> str:
    """Get an order's shipment status, carrier, ETA, delivery date and tracking events."""
    log.attempt("track_shipment")
    order = _order(order_id)
    if order is None:
        return _not_found(order_id)
    return json.dumps({k: order[k] for k in ("order_id", "status", "carrier", "eta", "delivered_on", "tracking")})


def search_policy(query: str) -> str:
    """Search the ShipFast refund and complaint policy. Returns the matching sections (all if none match)."""
    log.attempt("search_policy")
    words = _words(query)
    hits = [s for s in POLICY_SECTIONS if words & _words(s["title"] + " " + s["text"])]
    return json.dumps({"today": TODAY, "sections": hits or POLICY_SECTIONS})


def issue_refund(order_id: str, amount: float, reason: str) -> str:
    """Issue a refund for an order. The amount cannot exceed the order total."""
    log.attempt("issue_refund")
    order = _order(order_id)
    if order is None:
        return _not_found(order_id)
    try:
        amount = round(float(amount), 2)
    except (TypeError, ValueError):
        return json.dumps({"error": "invalid_amount", "order_id": order["order_id"], "amount": amount})
    if amount <= 0:
        return json.dumps({"error": "invalid_amount", "order_id": order["order_id"], "amount": amount})
    if amount > order["total"]:
        return json.dumps({"error": "amount_exceeds_order_total", "order_id": order["order_id"],
                           "amount": amount, "order_total": order["total"]})
    refund = {"refund_id": f"RF-{order['order_id']}", "order_id": order["order_id"], "amount": amount,
              "status": "issued"}
    with _lock:
        _refunds.append({**refund, "reason": reason})
    return json.dumps(refund)


def escalate_to_human(order_id: str, summary: str) -> str:
    """Open a ticket for a human agent. Use an empty order_id if no order is involved."""
    log.attempt("escalate_to_human")
    oid = (order_id or "").strip().upper()
    ticket = {"ticket_id": f"TCK-{oid or 'GENERAL'}", "status": "open"}
    with _lock:
        _tickets.append({**ticket, "order_id": oid or None, "summary": summary})
    return json.dumps(ticket)


TOOLS = [lookup_order, track_shipment, search_policy, issue_refund, escalate_to_human]


def refunds() -> list[dict]:
    """Refunds actually issued since the last reset: the truth for `action: refunded`."""
    with _lock:
        return [dict(r) for r in _refunds]


def tickets() -> list[dict]:
    with _lock:
        return [dict(t) for t in _tickets]


def tool_calls() -> list[str]:
    return log.names()


def reset() -> None:
    """Clear the ledger, the tickets and the tool-call log. Call before every scenario."""
    with _lock:
        _refunds.clear()
        _tickets.clear()
    log.reset()
