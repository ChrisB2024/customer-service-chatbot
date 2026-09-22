"""Load the help-center markdown, split it into chunks, and sync them into Chroma.

Usage: uv run python -m chatbot.rag.ingest

The sync is idempotent: chunk IDs are deterministic, so re-running upserts the same
IDs, and chunks whose section no longer exists are deleted.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from chatbot.config import get_settings
from chatbot.rag.vector_store import get_vector_store

UPDATED_LINE = re.compile(r"^_Last updated: (\d{4}-\d{2}-\d{2})_\s*$", re.MULTILINE)
HEADERS = [("#", "title"), ("##", "section")]


@dataclass(frozen=True)
class SyncReport:
    files: int
    chunks: int
    new: int
    deleted: int


def chunk_id(source: str, section: str, index: int) -> str:
    return hashlib.sha256(f"{source}|{section}|{index}".encode()).hexdigest()[:32]


def split_markdown(source: str, text: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split one markdown file: first by H1/H2 headers, then by size if a section is too long.

    Each chunk's text starts with "Title > Section" so the embedding knows its context,
    even for a chunk like "Items $150 or less have no restocking fee." on its own.
    """
    match = UPDATED_LINE.search(text)
    updated = match.group(1) if match else ""
    text = UPDATED_LINE.sub("", text)

    sections = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True).split_text(text)
    by_size = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks: list[Document] = []
    for section_doc in sections:
        title = section_doc.metadata.get("title", source)
        section = section_doc.metadata.get("section", "")
        heading = f"{title} > {section}" if section else title
        for index, piece in enumerate(by_size.split_text(section_doc.page_content)):
            chunks.append(
                Document(
                    id=chunk_id(source, section, index),
                    page_content=f"{heading}\n\n{piece}",
                    # Chroma metadata values must be str/int/float/bool, never None.
                    metadata={"source": source, "title": title, "section": section, "chunk": index, "updated": updated},
                )
            )
    return chunks


def load_chunks(kb_dir: Path, chunk_size: int, chunk_overlap: int) -> tuple[int, list[Document]]:
    files = sorted(kb_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown files in {kb_dir}")
    chunks = [c for f in files for c in split_markdown(f.name, f.read_text(), chunk_size, chunk_overlap)]
    return len(files), chunks


def sync_collection(store: Chroma, chunks: list[Document]) -> tuple[int, int]:
    """Make the collection contain exactly `chunks`. Returns (new, deleted)."""
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs: two sections share a heading in the same file")

    existing = set(store.get(include=[])["ids"])
    stale = existing - set(ids)
    if stale:
        store.delete(ids=list(stale))
    if chunks:
        store.add_documents(chunks, ids=ids)  # upsert
    return len(set(ids) - existing), len(stale)


def ingest(store: Chroma, kb_dir: Path, chunk_size: int, chunk_overlap: int) -> SyncReport:
    files, chunks = load_chunks(kb_dir, chunk_size, chunk_overlap)
    new, deleted = sync_collection(store, chunks)
    return SyncReport(files=files, chunks=len(chunks), new=new, deleted=deleted)


def main() -> None:
    settings = get_settings()
    report = ingest(get_vector_store(), settings.knowledge_base_dir, settings.chunk_size, settings.chunk_overlap)
    print(
        f"Ingested {report.files} files -> {report.chunks} chunks "
        f"({report.new} new, {report.deleted} deleted) into {settings.chroma_dir}"
    )


if __name__ == "__main__":
    main()
