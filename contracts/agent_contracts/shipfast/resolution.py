"""Build a turn's Resolution from what actually happened (DECISIONS.md 16).

Code records the facts; the model only writes the reply and, when a turn left no record, chooses between
`answered`, `needs_info` and `declined`. The rules, in order:

1. triage said `other` (out of scope)       -> declined
2. the turn paused at the approval gate    -> pending_approval
3. a refund reached the ledger this turn   -> refunded, with its amount
4. a ticket was opened this turn           -> escalated, with its id
5. otherwise                               -> the model's Outcome choice

`intent` and `order_id` come from the turn's triage. "This turn" means since `TurnMarker` was taken, so
a refund from an earlier turn of the same conversation does not count again.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import tools as shipfast
from .schema import Outcome, Resolution, Triage


@dataclass(frozen=True)
class TurnMarker:
    """How many refunds and tickets existed when a turn started."""
    refunds: int
    tickets: int

    @classmethod
    def now(cls) -> "TurnMarker":
        return cls(len(shipfast.refunds()), len(shipfast.tickets()))


def derive_resolution(triage: Triage, since: TurnMarker, reply: str, paused: bool = False,
                      outcome: Outcome | None = None) -> Resolution:
    new_refunds = shipfast.refunds()[since.refunds:]
    new_tickets = shipfast.tickets()[since.tickets:]
    base = {"intent": triage.intent, "order_id": triage.order_id, "reply": reply}
    if triage.intent == "other":
        return Resolution(**base, action="declined")
    if paused:
        return Resolution(**base, action="pending_approval")
    if new_refunds:
        return Resolution(**base, action="refunded", refund_amount=round(sum(r["amount"] for r in new_refunds), 2))
    if new_tickets:
        return Resolution(**base, action="escalated", ticket_id=new_tickets[-1]["ticket_id"])
    return Resolution(**base, action=(outcome.action if outcome else "answered"))
