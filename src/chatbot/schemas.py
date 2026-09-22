"""Contracts shared across modules (see docs/system-plan.md §4)."""

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """A retrieved help-center passage."""

    source: str  # "returns_and_refunds.md"
    section: str  # "Restocking fee"
    content: str
    score: float  # cosine relevance, 0-1
    updated: str = ""

    @property
    def ref(self) -> str:
        return f"{self.source}#{self.section}"


class OrderItemView(BaseModel):
    sku: str
    name: str
    brand: str
    quantity: int
    unit_price: Decimal  # what the customer paid
    final_sale: bool


class OrderView(BaseModel):
    """What lookup_order shows the LLM. Deliberately no email, last name, phone, or address."""

    order_number: str
    status: str
    customer_first_name: str
    country: str
    placed_at: datetime
    shipped_at: datetime | None
    delivered_at: datetime | None
    days_since_delivery: int | None  # computed here so the LLM doesn't do date math
    cancelled_at: datetime | None
    refunded_at: datetime | None
    estimated_delivery: date | None
    shipping_method: str
    carrier: str | None
    tracking_number: str | None
    rma_number: str | None
    items: list[OrderItemView]
    shipping_cost: Decimal
    total: Decimal


class ChatAnswer(BaseModel):
    """What the assistant returns to the user for one turn."""

    answer: str
    sources: list[str] = Field(default_factory=list)  # "file.md#Section"
    escalated: bool = False
    ticket_id: str | None = None
    error: str | None = None  # set when the turn failed and nothing was saved
