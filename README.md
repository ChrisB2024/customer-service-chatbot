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

## Run

```bash
uv run python -m chatbot.db.seed   # validate seed JSON, reset + load storage/app.db
uv run python -m chatbot.rag.ingest  # chunk + embed help-center docs into storage/chroma (idempotent)
uv run python -m chatbot.chains.rag "Is there a restocking fee?"  # one grounded answer (needs ANTHROPIC_API_KEY)
uv run python -m chatbot.agent "Where is NG-10415? daniel.reyes@example.com"  # agent with tools (one turn)
uv run pytest           # offline tests (fake LLM)
uv run pytest -m live   # calls the Anthropic API
```
