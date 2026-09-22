from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

CENT = Decimal("0.01")


# --- Column types -----------------------------------------------------------


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetimes in Python, stored as naive UTC.

    SQLite has no timezone support, so without this, values come back naive and
    comparing them with `datetime.now(UTC)` raises TypeError.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"naive datetime not allowed: {value!r}")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


class Money(TypeDecorator[Decimal]):
    """Decimal dollars in Python, integer cents in the DB.

    SQLite has no exact decimal type; SQLAlchemy's Numeric would round-trip through float.
    """

    impl = Integer
    cache_ok = True

    def process_bind_param(self, value: Decimal | int | None, dialect) -> int | None:
        if value is None:
            return None
        cents = Decimal(value) * 100
        if cents != cents.to_integral_value():
            raise ValueError(f"{value} has more than 2 decimal places")
        return int(cents)

    def process_result_value(self, value: int | None, dialect) -> Decimal | None:
        return None if value is None else (Decimal(value) / 100).quantize(CENT)


def _enum(enum_cls: type[StrEnum]) -> Enum:
    # Store the enum's value ("processing"), not its name ("PROCESSING"), and add a CHECK constraint.
    return Enum(
        enum_cls,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda cls: [m.value for m in cls],
    )


def _now() -> datetime:
    return datetime.now(UTC)


# --- Enums ------------------------------------------------------------------


class OrderStatus(StrEnum):
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURN_IN_PROGRESS = "return_in_progress"
    REFUNDED = "refunded"


class ShippingMethod(StrEnum):
    STANDARD = "standard"
    EXPEDITED = "expedited"
    OVERNIGHT = "overnight"
    STANDARD_CA = "standard_ca"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class TicketReason(StrEnum):
    REQUESTED_HUMAN = "requested_human"
    DAMAGED_OR_WRONG_ITEM = "damaged_or_wrong_item"
    LOST_PACKAGE = "lost_package"
    WARRANTY_CLAIM = "warranty_claim"
    REFUND_EXCEPTION = "refund_exception"
    FRAUD = "fraud"
    LOW_CONFIDENCE = "low_confidence"
    OTHER = "other"


class TicketStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


# --- Tables -----------------------------------------------------------------


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict] = {datetime: UTCDateTime, Decimal: Money}


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)  # always stored lowercase
    phone: Mapped[str | None] = mapped_column(String(32))
    city: Mapped[str] = mapped_column(String(100))
    region: Mapped[str] = mapped_column(String(8))
    country: Mapped[str] = mapped_column(String(2))
    has_account: Mapped[bool]
    created_at: Mapped[datetime]

    orders: Mapped[list["Order"]] = relationship(back_populates="customer")


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (CheckConstraint("price >= 0", name="ck_products_price_nonneg"),)

    sku: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    brand: Mapped[str] = mapped_column(String(100))
    category: Mapped[str] = mapped_column(String(32))
    price: Mapped[Decimal]  # current catalog price, not what past customers paid
    final_sale: Mapped[bool]


class Order(Base):
    __tablename__ = "orders"

    order_number: Mapped[str] = mapped_column(String(16), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    status: Mapped[OrderStatus] = mapped_column(_enum(OrderStatus))
    shipping_method: Mapped[ShippingMethod] = mapped_column(_enum(ShippingMethod))
    shipping_cost: Mapped[Decimal]
    carrier: Mapped[str | None] = mapped_column(String(16))
    tracking_number: Mapped[str | None] = mapped_column(String(40))
    placed_at: Mapped[datetime]
    shipped_at: Mapped[datetime | None]
    delivered_at: Mapped[datetime | None]
    cancelled_at: Mapped[datetime | None]
    refunded_at: Mapped[datetime | None]
    estimated_delivery: Mapped[date | None]
    rma_number: Mapped[str | None] = mapped_column(String(16), unique=True)

    customer: Mapped[Customer] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(
        back_populates="order", cascade="all, delete-orphan", order_by="OrderItem.id"
    )

    @property
    def subtotal(self) -> Decimal:
        return sum((item.line_total for item in self.items), Decimal("0.00"))

    @property
    def total(self) -> Decimal:
        # Derived, never stored, so it can't drift from the line items.
        return self.subtotal + self.shipping_cost


class OrderItem(Base):
    __tablename__ = "order_items"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_order_items_quantity_pos"),
        CheckConstraint("unit_price >= 0", name="ck_order_items_unit_price_nonneg"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    order_number: Mapped[str] = mapped_column(ForeignKey("orders.order_number", ondelete="CASCADE"), index=True)
    sku: Mapped[str] = mapped_column(ForeignKey("products.sku"))
    quantity: Mapped[int]
    unit_price: Mapped[Decimal]  # snapshot at purchase time

    order: Mapped[Order] = relationship(back_populates="items")
    product: Mapped[Product] = relationship()

    @property
    def line_total(self) -> Decimal:
        return self.unit_price * self.quantity


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    started_at: Mapped[datetime] = mapped_column(default=_now)
    last_active_at: Mapped[datetime] = mapped_column(default=_now)
    failed_lookups: Mapped[int] = mapped_column(default=0)  # enumeration guard across turns

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.id",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[MessageRole] = mapped_column(_enum(MessageRole))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=_now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default=lambda: f"TCK-{uuid4().hex[:8].upper()}")
    # A ticket outlives its conversation (transcripts are deleted after 90 days), so SET NULL, not CASCADE.
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id", ondelete="SET NULL"), index=True)
    order_number: Mapped[str | None] = mapped_column(ForeignKey("orders.order_number"))
    reason: Mapped[TicketReason] = mapped_column(_enum(TicketReason))
    summary: Mapped[str] = mapped_column(Text)
    status: Mapped[TicketStatus] = mapped_column(_enum(TicketStatus), default=TicketStatus.OPEN)
    created_at: Mapped[datetime] = mapped_column(default=_now)
