"""ShipFast structured outputs (SPEC.md sections 4 and 5)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["order_status", "refund_request", "complaint", "other"]
Action = Literal["answered", "refunded", "pending_approval", "escalated", "needs_info", "declined"]


class Triage(BaseModel):
    intent: Intent = Field(description="order_status, refund_request, complaint, or other (out of scope)")
    order_id: str | None = Field(description="The order ID, such as A1001, or null if there is none")


class Resolution(BaseModel):
    intent: Intent
    order_id: str | None
    action: Action
    refund_amount: float | None = Field(default=None, description="Only when a refund was actually issued")
    ticket_id: str | None = None
    reply: str = Field(description="The customer-facing reply")


class Outcome(BaseModel):
    """The model's one judgement call in a Resolution, for turns that left no record (DECISIONS.md 16)."""
    action: Literal["answered", "needs_info", "declined"] = Field(
        description="answered: gave the customer information; needs_info: asked the customer for something "
                    "(a photo, a correct order ID); declined: refused the request under the policy")
