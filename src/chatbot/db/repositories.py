from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from chatbot.db.models import Customer, Order, OrderItem, Ticket, TicketReason


def find_order_for_customer(session: Session, order_number: str, email: str) -> Order | None:
    """The order, only if `email` belongs to its customer.

    One query that requires both: an unknown order and a wrong email take the same path
    and return the same None, so callers can't tell (or leak) which one it was.
    """
    stmt = (
        select(Order)
        .join(Order.customer)
        .where(Order.order_number == order_number, Customer.email == email.strip().lower())
        .options(
            selectinload(Order.customer),
            selectinload(Order.items).selectinload(OrderItem.product),
        )
    )
    return session.scalars(stmt).one_or_none()


def create_ticket(
    session: Session,
    *,
    reason: TicketReason,
    summary: str,
    conversation_id: str | None,
    order_number: str | None,
) -> Ticket:
    ticket = Ticket(reason=reason, summary=summary, conversation_id=conversation_id, order_number=order_number)
    session.add(ticket)
    session.flush()  # assign the id now; the caller owns the transaction
    return ticket
