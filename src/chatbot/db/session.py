from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, StaticPool, create_engine, event, make_url
from sqlalchemy.orm import Session, sessionmaker

from chatbot.config import get_settings


def make_engine(url: str) -> Engine:
    parsed = make_url(url)
    is_sqlite = parsed.get_backend_name() == "sqlite"
    in_memory = is_sqlite and parsed.database in (None, "", ":memory:")
    kwargs: dict = {}

    if is_sqlite:
        # Agent tools run in worker threads. Sessions are still used by one thread at a time (see
        # AgentContext.lock), but sqlite3 refuses cross-thread use of a connection by default.
        kwargs["connect_args"] = {"check_same_thread": False}
    if in_memory:
        kwargs["poolclass"] = StaticPool  # one shared connection, or each thread would get its own empty DB
    elif is_sqlite:
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url, **kwargs)

    if is_sqlite:
        # SQLite ignores foreign keys (and ON DELETE CASCADE) unless enabled per connection.
        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


@lru_cache
def get_engine() -> Engine:
    return make_engine(get_settings().database_url)


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """One unit of work: commits on success, rolls back on any exception."""
    with get_sessionmaker().begin() as session:
        yield session
