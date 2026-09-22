import shutil

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from chatbot.config import get_settings
from chatbot.rag.embeddings import get_embeddings
from chatbot.rag.ingest import chunk_id, ingest, split_markdown
from chatbot.rag.vector_store import make_vector_store

KB_DIR = get_settings().knowledge_base_dir

DOC = """# Returns

_Last updated: 2026-08-01_

## Return window

You have 30 days.

## Restocking fee

10% over $150.
"""


@pytest.fixture
def fake_store(tmp_path):
    return make_vector_store(tmp_path / "chroma", "test", DeterministicFakeEmbedding(size=16))


@pytest.fixture
def kb_copy(tmp_path):
    kb = tmp_path / "kb"
    shutil.copytree(KB_DIR, kb)
    return kb


# --- splitting --------------------------------------------------------------


def test_split_by_headers_with_context_prefix_and_metadata():
    chunks = split_markdown("returns.md", DOC, chunk_size=800, chunk_overlap=100)

    assert [c.metadata["section"] for c in chunks] == ["Return window", "Restocking fee"]
    assert chunks[0].page_content == "Returns > Return window\n\nYou have 30 days."
    assert chunks[0].metadata == {
        "source": "returns.md",
        "title": "Returns",
        "section": "Return window",
        "chunk": 0,
        "updated": "2026-08-01",
    }
    assert all("Last updated" not in c.page_content for c in chunks)


def test_chunk_ids_are_deterministic():
    first = [c.id for c in split_markdown("returns.md", DOC, 800, 100)]
    second = [c.id for c in split_markdown("returns.md", DOC, 800, 100)]
    assert first == second == [chunk_id("returns.md", "Return window", 0), chunk_id("returns.md", "Restocking fee", 0)]


def test_long_section_is_split_by_size():
    long_doc = "# T\n\n## S\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(10))
    chunks = split_markdown("t.md", long_doc, chunk_size=300, chunk_overlap=0)

    assert len(chunks) > 1
    assert [c.metadata["chunk"] for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(len(c.page_content) <= 300 + len("T > S\n\n") for c in chunks)


# --- syncing ----------------------------------------------------------------


def test_reingest_is_idempotent(fake_store, kb_copy):
    first = ingest(fake_store, kb_copy, 800, 100)
    second = ingest(fake_store, kb_copy, 800, 100)

    assert first.new == first.chunks > 0
    assert (second.new, second.deleted) == (0, 0)
    assert len(fake_store.get(include=[])["ids"]) == first.chunks


def test_removed_doc_chunks_are_deleted(fake_store, kb_copy):
    ingest(fake_store, kb_copy, 800, 100)
    warranty = kb_copy / "warranty.md"
    warranty_chunks = len(split_markdown(warranty.name, warranty.read_text(), 800, 100))
    warranty.unlink()

    report = ingest(fake_store, kb_copy, 800, 100)

    assert report.deleted == warranty_chunks
    assert not fake_store.get(where={"source": "warranty.md"}, include=[])["ids"]


def test_duplicate_section_heading_is_rejected(fake_store, tmp_path):
    kb = tmp_path / "dup"
    kb.mkdir()
    # Adjacent duplicates get merged by the header splitter; separated ones collide.
    (kb / "a.md").write_text("# A\n\n## Same\n\none\n\n## Other\n\nmid\n\n## Same\n\ntwo\n")
    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        ingest(fake_store, kb, 800, 100)


# --- real embeddings --------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    ("question", "source", "section"),
    [
        ("Is there a restocking fee?", "returns_and_refunds.md", "Restocking fee"),
        ("Do you ship to Canada?", "shipping.md", "Canada shipping"),
        ("Can I take the power bank on a plane?", "products_faq.md", "PowerCell 20K Power Bank"),
    ],
)
def test_real_embeddings_retrieve_the_right_section(tmp_path, question, source, section):
    store = make_vector_store(tmp_path / "chroma", "real", get_embeddings())
    ingest(store, KB_DIR, 800, 100)

    top, score = store.similarity_search_with_relevance_scores(question, k=1)[0]

    assert (top.metadata["source"], top.metadata["section"]) == (source, section)
    assert 0.0 <= score <= 1.0
