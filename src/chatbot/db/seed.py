"""Validate data/seed/*.json with Pydantic, then reset the database and load it.

Usage: uv run python -m chatbot.db.seed

This drops and recreates every table, including conversations and tickets.
"""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from chatbot.config import get_settings
from chatbot.db.models import Base, Customer, Order, OrderItem, OrderStatus, Product, ShippingMethod
from chatbot.db.session import get_engine

TIMESTAMP_FIELDS = ("shipped_at", "delivered_at", "cancelled_at", "refunded_at")

# For each status: which timestamps must be set. All others must be null.
REQUIRED_TIMESTAMPS: dict[OrderStatus, set[str]] = {
    OrderStatus.PROCESSING: set(),
    OrderStatus.SHIPPED: {"shipped_at"},
    OrderStatus.DELIVERED: {"shipped_at", "delivered_at"},
    OrderStatus.CANCELLED: {"cancelled_at"},
    OrderStatus.RETURN_IN_PROGRESS: {"shipped_at", "delivered_at"},
    OrderStatus.REFUNDED: {"shipped_at", "delivered_at", "refunded_at"},
}
REQUIRES_RMA = {OrderStatus.RETURN_IN_PROGRESS, OrderStatus.REFUNDED}
FORBIDS_RMA = {OrderStatus.PROCESSING, OrderStatus.SHIPPED, OrderStatus.CANCELLED}


class _Seed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CustomerSeed(_Seed):
    id: str = Field(pattern=r"^CUST-\d{4}$")
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)
    email: EmailStr
    phone: str | None = None
    city: str
    region: str
    country: str = Field(pattern=r"^(US|CA)$")
    has_account: bool
    created_at: AwareDatetime

    @field_validator("email")
    @classmethod
    def lowercase_email(cls, v: str) -> str:
        return v.lower()


class ProductSeed(_Seed):
    sku: str = Field(pattern=r"^[A-Z]{2}-[A-Z0-9-]+$")
    name: str
    brand: str
    category: str
    price: Decimal = Field(ge=0, decimal_places=2)
    final_sale: bool


class OrderItemSeed(_Seed):
    sku: str
    quantity: int = Field(gt=0)
    unit_price: Decimal = Field(ge=0, decimal_places=2)


class OrderSeed(_Seed):
    order_number: str = Field(pattern=r"^NG-\d{5}$")
    customer_id: str
    status: OrderStatus
    shipping_method: ShippingMethod
    shipping_cost: Decimal = Field(ge=0, decimal_places=2)
    carrier: str | None
    tracking_number: str | None
    placed_at: AwareDatetime
    shipped_at: AwareDatetime | None
    delivered_at: AwareDatetime | None
    cancelled_at: AwareDatetime | None
    refunded_at: AwareDatetime | None
    estimated_delivery: date | None
    rma_number: Annotated[str, Field(pattern=r"^RMA-\d{5}$")] | None
    items: list[OrderItemSeed] = Field(min_length=1)
    scenario: str | None = Field(default=None, alias="_scenario")  # documentation only, not stored

    @model_validator(mode="after")
    def check_status_consistency(self) -> Self:
        required = REQUIRED_TIMESTAMPS[self.status]
        missing = [f for f in required if getattr(self, f) is None]
        unexpected = [f for f in TIMESTAMP_FIELDS if f not in required and getattr(self, f) is not None]
        if missing or unexpected:
            raise ValueError(f"status={self.status}: missing {missing}, should be null {unexpected}")

        timeline = [self.placed_at, self.shipped_at, self.delivered_at, self.refunded_at]
        timeline = [t for t in timeline if t is not None]
        if timeline != sorted(timeline):
            raise ValueError("placed_at <= shipped_at <= delivered_at <= refunded_at violated")
        if self.cancelled_at and self.cancelled_at < self.placed_at:
            raise ValueError("cancelled_at before placed_at")

        if self.shipped_at and not (self.carrier and self.tracking_number):
            raise ValueError("shipped orders need carrier and tracking_number")
        if self.status in REQUIRES_RMA and not self.rma_number:
            raise ValueError(f"status={self.status} requires rma_number")
        if self.status in FORBIDS_RMA and self.rma_number:
            raise ValueError(f"status={self.status} cannot have rma_number")
        return self


class SeedData(_Seed):
    customers: list[CustomerSeed]
    products: list[ProductSeed]
    orders: list[OrderSeed]

    @model_validator(mode="after")
    def check_references(self) -> Self:
        customers = {c.id: c for c in self.customers}
        skus = {p.sku for p in self.products}

        for label, keys in (
            ("customer id", [c.id for c in self.customers]),
            ("customer email", [c.email for c in self.customers]),
            ("sku", [p.sku for p in self.products]),
            ("order_number", [o.order_number for o in self.orders]),
            ("rma_number", [o.rma_number for o in self.orders if o.rma_number]),
        ):
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate {label}")

        for order in self.orders:
            customer = customers.get(order.customer_id)
            if customer is None:
                raise ValueError(f"{order.order_number}: unknown customer {order.customer_id}")
            if customer.created_at > order.placed_at:
                raise ValueError(f"{order.order_number}: placed before customer {customer.id} existed")
            if (customer.country == "CA") != (order.shipping_method is ShippingMethod.STANDARD_CA):
                raise ValueError(f"{order.order_number}: shipping method doesn't match country")
            for item in order.items:
                if item.sku not in skus:
                    raise ValueError(f"{order.order_number}: unknown sku {item.sku}")
        return self


def load_seed_data(seed_dir: Path) -> SeedData:
    raw = {name: json.loads((seed_dir / f"{name}.json").read_text()) for name in ("customers", "products", "orders")}
    return SeedData.model_validate(raw)


def seed(engine: Engine, data: SeedData) -> None:
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with Session(engine) as session, session.begin():
        session.add_all(Customer(**c.model_dump()) for c in data.customers)
        session.add_all(Product(**p.model_dump()) for p in data.products)
        for o in data.orders:
            order = Order(**o.model_dump(exclude={"items", "scenario"}))
            order.items = [OrderItem(**i.model_dump()) for i in o.items]
            session.add(order)


def main() -> None:
    settings = get_settings()
    data = load_seed_data(settings.seed_dir)
    engine = get_engine()
    seed(engine, data)
    print(
        f"Seeded {len(data.customers)} customers, {len(data.products)} products, "
        f"{len(data.orders)} orders into {engine.url}"
    )


if __name__ == "__main__":
    main()
