"""The agent's three tools.

Each tool gets per-turn state through `ToolRuntime.context` (hidden from the model's schema).
Tools never raise into the agent loop: expected failures come back as text the model can act on.

LangGraph runs tool calls in worker threads, in parallel when the model makes several at once.
A SQLAlchemy Session isn't thread-safe, so anything touching the session or mutable context holds `ctx.lock`.
"""

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from langchain.tools import ToolRuntime, tool
from sqlalchemy.orm import Session

from chatbot.chains.rag import format_context
from chatbot.db.models import Order, TicketReason
from chatbot.db.repositories import create_ticket, find_order_for_customer
from chatbot.rag.retriever import retrieve_chunks
from chatbot.schemas import OrderItemView, OrderView, SourceChunk

ORDER_NUMBER = re.compile(r"NG-\d{5}")
MAX_LOOKUPS_PER_TURN = 3
MAX_SUMMARY_CHARS = 1000

ORDER_NOT_FOUND = (
    "No order matches that order number and email. Ask the customer to double-check both. "
    "Don't try other order numbers or emails."
)
LOOKUP_LIMIT_REACHED = (
    "Lookup limit reached for this message. Don't try more combinations. "
    "Ask the customer to check their confirmation email, or offer to connect them with a person."
)
BAD_ORDER_NUMBER = "Order numbers look like NG-12345. Ask the customer for the number in their confirmation email."


@dataclass
class AgentContext:
    """Per-turn state shared by the tools. A fresh one is created for every user message."""

    session: Session  # the caller owns the transaction (one commit per turn)
    conversation_id: str | None
    now: datetime
    search: Callable[[str], list[SourceChunk]] = retrieve_chunks

    # Written by tools, read by run_agent after the turn
    retrieved: set[str] = field(default_factory=set)  # refs of every passage shown to the model
    verified_orders: set[str] = field(default_factory=set)
    lookup_attempts: int = 0
    ticket_id: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def to_order_view(order: Order, now: datetime) -> OrderView:
    return OrderView(
        order_number=order.order_number,
        status=order.status.value,
        customer_first_name=order.customer.first_name,
        country=order.customer.country,
        placed_at=order.placed_at,
        shipped_at=order.shipped_at,
        delivered_at=order.delivered_at,
        days_since_delivery=(now - order.delivered_at).days if order.delivered_at else None,
        cancelled_at=order.cancelled_at,
        refunded_at=order.refunded_at,
        estimated_delivery=order.estimated_delivery,
        shipping_method=order.shipping_method.value,
        carrier=order.carrier,
        tracking_number=order.tracking_number,
        rma_number=order.rma_number,
        items=[
            OrderItemView(
                sku=item.sku,
                name=item.product.name,
                brand=item.product.brand,
                quantity=item.quantity,
                unit_price=item.unit_price,
                final_sale=item.product.final_sale,
            )
            for item in order.items
        ],
        shipping_cost=order.shipping_cost,
        total=order.total,
    )


@tool
def search_help_center(query: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Search the Nimbus Gear help center: returns, refunds, shipping, warranty, payments, account and privacy,
    product FAQs, and contact info. Returns the most relevant passages with their ids. Use a short, specific query.
    """
    chunks = runtime.context.search(query)
    with runtime.context.lock:
        runtime.context.retrieved.update(c.ref for c in chunks)
    return format_context(chunks) if chunks else "No relevant help-center passages found."


@tool
def lookup_order(order_number: str, email: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Look up one order's status, dates, tracking, and items. Both values must come from the customer:
    the order number (format NG-12345) and the email address used on that order.
    """
    ctx = runtime.context
    with ctx.lock:
        ctx.lookup_attempts += 1
        if ctx.lookup_attempts > MAX_LOOKUPS_PER_TURN:
            return LOOKUP_LIMIT_REACHED

        number = order_number.strip().upper()
        if not ORDER_NUMBER.fullmatch(number):
            return BAD_ORDER_NUMBER

        order = find_order_for_customer(ctx.session, number, email)
        if order is None:
            return ORDER_NOT_FOUND

        ctx.verified_orders.add(order.order_number)
        return to_order_view(order, ctx.now).model_dump_json()


@tool
def escalate_to_human(
    reason: TicketReason,
    summary: str,
    runtime: ToolRuntime[AgentContext],
    order_number: str | None = None,
) -> str:
    """Create a support ticket so a person on the team takes over. Use it when the customer asks for a person,
    an item arrived damaged or wrong, a package is lost, they're making a warranty claim, they need a refund or
    a policy exception, they suspect fraud, or you can't answer confidently.

    summary: one or two sentences a teammate can act on. Don't include the customer's email.
    order_number: only an order already verified with lookup_order in this conversation.
    """
    ctx = runtime.context
    with ctx.lock:
        if ctx.ticket_id:  # one ticket per turn, even if the model calls this twice (or in parallel)
            return f"Ticket {ctx.ticket_id} already created for this request."

        # Only link orders the customer proved they own; anything else could be a guess (or not exist, breaking the FK).
        number = order_number.strip().upper() if order_number else None
        linked = number if number in ctx.verified_orders else None

        ticket = create_ticket(
            ctx.session,
            reason=reason,
            summary=summary.strip()[:MAX_SUMMARY_CHARS],
            conversation_id=ctx.conversation_id,
            order_number=linked,
        )
        ctx.ticket_id = ticket.id
        return f"Ticket {ticket.id} created. A person on the support team will follow up."


TOOLS = [search_help_center, lookup_order, escalate_to_human]
