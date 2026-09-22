import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from chatbot.config import get_settings
from chatbot.db.models import (
    Conversation,
    Customer,
    Message,
    MessageRole,
    Order,
    OrderStatus,
    Product,
    Ticket,
    TicketReason,
)
from chatbot.db.seed import OrderSeed, SeedData, load_seed_data, seed
from chatbot.db.session import make_engine

SEED_DIR = get_settings().seed_dir


@pytest.fixture
def engine() -> Engine:
    engine = make_engine("sqlite://")
    seed(engine, load_seed_data(SEED_DIR))
    return engine


def _raw_order(order_number: str) -> dict:
    orders = json.loads((SEED_DIR / "orders.json").read_text())
    return next(o for o in orders if o["order_number"] == order_number)


# --- seed validation --------------------------------------------------------


def test_real_seed_files_are_valid():
    data = load_seed_data(SEED_DIR)
    assert (len(data.customers), len(data.products), len(data.orders)) == (6, 12, 11)
    assert {o.status for o in data.orders} == set(OrderStatus)


def test_shipped_order_without_shipped_at_is_rejected():
    raw = _raw_order("NG-10415") | {"shipped_at": None}
    with pytest.raises(ValidationError, match="missing \\['shipped_at'\\]"):
        OrderSeed.model_validate(raw)


def test_cancelled_order_with_tracking_timestamp_is_rejected():
    raw = _raw_order("NG-10397") | {"shipped_at": "2026-08-25T18:00:00Z"}
    with pytest.raises(ValidationError, match="should be null \\['shipped_at'\\]"):
        OrderSeed.model_validate(raw)


def test_out_of_order_timestamps_are_rejected():
    raw = _raw_order("NG-10402") | {"delivered_at": "2026-08-01T00:00:00Z"}
    with pytest.raises(ValidationError, match="violated"):
        OrderSeed.model_validate(raw)


def test_naive_datetime_is_rejected():
    raw = _raw_order("NG-10421") | {"placed_at": "2026-09-21T13:05:00"}
    with pytest.raises(ValidationError):
        OrderSeed.model_validate(raw)


def test_unknown_sku_is_rejected():
    data = json.loads((SEED_DIR / "orders.json").read_text())
    data[0]["items"][0]["sku"] = "NG-DOES-NOT-EXIST"
    raw = {n: json.loads((SEED_DIR / f"{n}.json").read_text()) for n in ("customers", "products")}
    with pytest.raises(ValidationError, match="unknown sku"):
        SeedData.model_validate(raw | {"orders": data})


# --- persisted data ---------------------------------------------------------


def test_row_counts(engine):
    with Session(engine) as s:
        counts = [s.scalar(select(func.count()).select_from(m)) for m in (Customer, Product, Order)]
    assert counts == [6, 12, 11]


def test_money_round_trips_exactly(engine):
    with Session(engine) as s:
        order = s.get(Order, "NG-10415")
        assert order.shipping_cost == Decimal("12.99")
        assert order.total == Decimal("91.99")
        # stored as integer cents
        assert s.execute(text("SELECT shipping_cost FROM orders WHERE order_number='NG-10415'")).scalar() == 1299


def test_unit_price_is_a_snapshot_not_catalog_price(engine):
    with Session(engine) as s:
        order = s.get(Order, "NG-10233")
        assert order.items[0].unit_price == Decimal("129.00")
        assert order.items[0].product.price == Decimal("149.00")


def test_datetimes_come_back_utc_aware(engine):
    with Session(engine) as s:
        order = s.get(Order, "NG-10388")
        assert order.delivered_at.tzinfo is UTC
        assert (datetime(2026, 9, 21, tzinfo=UTC) - order.delivered_at).days == 41


def test_emails_are_stored_lowercase(engine):
    with Session(engine) as s:
        emails = s.scalars(select(Customer.email)).all()
    assert all(e == e.lower() for e in emails)


def test_db_rejects_invalid_status(engine):
    with Session(engine) as s, pytest.raises(IntegrityError):
        s.execute(text("UPDATE orders SET status='teleported' WHERE order_number='NG-10421'"))


def test_foreign_keys_are_enforced(engine):
    with Session(engine) as s, pytest.raises(IntegrityError):
        s.execute(
            text(
                "INSERT INTO order_items (order_number, sku, quantity, unit_price) VALUES ('NG-99999', 'NG-HUB-7', 1, 100)"
            )
        )


# --- conversation lifecycle -------------------------------------------------


def test_deleting_conversation_cascades_messages_but_keeps_ticket(engine):
    with Session(engine) as s, s.begin():
        convo = Conversation()
        convo.messages = [
            Message(role=MessageRole.USER, content="talk to a person"),
            Message(role=MessageRole.ASSISTANT, content="Done, ticket created."),
        ]
        s.add(convo)
        s.flush()
        ticket = Ticket(conversation_id=convo.id, reason=TicketReason.REQUESTED_HUMAN, summary="wants a human")
        s.add(ticket)
        s.flush()
        convo_id, ticket_id = convo.id, ticket.id

    with Session(engine) as s, s.begin():
        s.execute(text("DELETE FROM conversations WHERE id = :id"), {"id": convo_id})

    with Session(engine) as s:
        assert s.scalar(select(func.count()).select_from(Message)) == 0
        surviving = s.get(Ticket, ticket_id)
        assert surviving is not None and surviving.conversation_id is None
        assert ticket_id.startswith("TCK-")
