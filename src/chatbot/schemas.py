"""Contracts shared across modules (see docs/system-plan.md §4)."""

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


class ChatAnswer(BaseModel):
    """What the assistant returns to the user for one turn."""

    answer: str
    sources: list[str] = Field(default_factory=list)  # "file.md#Section"
    escalated: bool = False
    ticket_id: str | None = None
