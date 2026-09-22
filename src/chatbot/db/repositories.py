from datetime import datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from chatbot.db.models import Conversation, Customer, Message, MessageRole, Order, OrderItem, Ticket, TicketReason


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


def load_recent_messages(session: Session, conversation_id: str, turns: int) -> list[Message]:
    """The last `turns` user/assistant pairs, oldest first, always starting with a user message."""
    newest_first = session.scalars(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id.desc()).limit(2 * turns)
    ).all()
    messages = list(reversed(newest_first))
    while messages and messages[0].role is not MessageRole.USER:  # the API requires the first message be the user's
        messages.pop(0)
    return messages


def delete_expired_conversations(session: Session, now: datetime, retention_days: int) -> int:
    """Delete conversations idle longer than the retention window.

    Messages go with them (ON DELETE CASCADE); tickets stay, with conversation_id set to NULL.
    """
    cutoff = now - timedelta(days=retention_days)
    result = session.execute(delete(Conversation).where(Conversation.last_active_at < cutoff))
    return result.rowcount
