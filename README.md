# Nimbus Gear Support Assistant

RAG customer-support chatbot for a fictional electronics store. Uses LangChain, Claude, Chroma (local), fastembed (local embeddings), SQLAlchemy/SQLite, and Pydantic.

See [`docs/system-plan.md`](docs/system-plan.md) for the architecture, invariants, and build steps.

## Setup

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY
```

## Layout

```
data/knowledge_base/   help-center docs (RAG source)
data/seed/             customers, products, orders (SQLite source)
data/eval/             golden questions for evaluation
docs/                  system plan
src/chatbot/           application code
storage/               generated: app.db, chroma/, models/ (gitignored)
```
