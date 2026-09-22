"""Chat with the Nimbus Gear support assistant in the terminal.

Usage:
  uv run chatbot                 start a new conversation
  uv run chatbot --resume <id>   continue a saved one
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from langchain_chroma import Chroma
from sqlalchemy import Engine, inspect

from chatbot.chat_service import ChatService
from chatbot.config import get_settings
from chatbot.db.models import Base, MessageRole
from chatbot.db.session import get_engine
from chatbot.rag.retriever import KnowledgeBaseEmpty, ensure_knowledge_base
from chatbot.schemas import ChatAnswer

RESET_DB = "Run: uv run python -m chatbot.db.seed (this resets the database)"
HELP = "Commands: /new starts a new conversation · /help shows this · /quit exits (or Ctrl-D)"


class SetupError(Exception):
    """Something the user must fix before chatting (not a bug)."""


def preflight(engine: Engine | None = None, store: Chroma | None = None) -> None:
    try:
        ensure_knowledge_base(store)
    except KnowledgeBaseEmpty as e:
        raise SetupError(str(e)) from e

    # No migrations in this project, so catch a stale schema here instead of as an error on the first message.
    db = inspect(engine or get_engine())
    for table in Base.metadata.sorted_tables:
        if not db.has_table(table.name):
            raise SetupError(f"Database isn't set up (missing table '{table.name}'). {RESET_DB}")
        missing = {c.name for c in table.columns} - {c["name"] for c in db.get_columns(table.name)}
        if missing:
            raise SetupError(f"Database schema is out of date ({table.name} lacks {sorted(missing)}). {RESET_DB}")


def configure_logging(log_file: Path) -> None:
    """Logs go to a file so they don't interleave with the chat."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger("chatbot").setLevel(logging.INFO)
    logging.captureWarnings(True)


class ChatLoop:
    def __init__(
        self,
        service: ChatService,
        *,
        conversation_id: str | None = None,
        read: Callable[[str], str] = input,
        out: TextIO = sys.stdout,
        color: bool | None = None,
    ) -> None:
        self.service = service
        self.conversation_id = conversation_id
        self._read = read
        self._out = out
        tty = hasattr(out, "isatty") and out.isatty()
        self._color = (tty and "NO_COLOR" not in os.environ) if color is None else color
        self._tty = tty

    # --- output helpers -----------------------------------------------------

    def _style(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self._color else text

    def _print(self, text: str = "") -> None:
        print(text, file=self._out, flush=True)

    def _thinking(self, on: bool) -> None:
        if self._tty:  # a transient status line; skipped when output is piped
            self._out.write(self._style("  thinking…", "2") if on else "\r\033[K")
            self._out.flush()

    def _show(self, answer: ChatAnswer) -> None:
        label = self._style("bot ›", "1;36")
        body = self._style(answer.answer, "33") if answer.error else answer.answer
        self._print(f"{label} {body}")
        if answer.sources:
            self._print(self._style(f"      sources: {' · '.join(answer.sources)}", "2"))
        if answer.escalated:
            self._print(self._style(f"      ticket {answer.ticket_id}: a person on the team will follow up", "35"))
        self._print()

    # --- loop ---------------------------------------------------------------

    def run(self) -> int:
        self._print(self._style("Nimbus Gear support", "1") + self._style("  (type /help for commands)", "2"))
        if self.conversation_id:
            self._show_resumed()
        self._print()

        while True:
            try:
                text = self._read(self._style("you › ", "1")).strip()
            except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C at the prompt
                self._print()
                break
            if not text:
                continue
            if text.startswith("/"):
                if not self._command(text):
                    break
                continue
            self._turn(text)

        if self.conversation_id:
            self._print(
                self._style(f"Conversation saved. Resume with: uv run chatbot --resume {self.conversation_id}", "2")
            )
        return 0

    def _turn(self, text: str) -> None:
        self._thinking(True)
        try:
            reply = self.service.send(text, conversation_id=self.conversation_id)
        except KeyboardInterrupt:  # the transaction rolls back, so nothing from this turn is saved
            self._thinking(False)
            self._print(self._style("(cancelled, nothing was saved)", "2") + "\n")
            return
        self._thinking(False)
        if reply.conversation_id:
            self.conversation_id = reply.conversation_id
        self._show(reply.answer)

    def _command(self, text: str) -> bool:
        """Handle a slash command. Returns False to exit."""
        command = text.split()[0].lower()
        if command in ("/quit", "/exit"):
            return False
        if command == "/new":
            self.conversation_id = None
            self._print(self._style("Started a new conversation.", "2") + "\n")
        elif command == "/help":
            self._print(self._style(HELP, "2") + "\n")
        else:
            self._print(self._style(f"Unknown command {command}. {HELP}", "2") + "\n")
        return True

    def _show_resumed(self) -> None:
        recent = self.service.recent_messages(self.conversation_id)
        if not recent:
            self._print(self._style("No saved conversation with that id (it may have expired). Starting fresh.", "33"))
            self.conversation_id = None
            return
        self._print(self._style("Resuming. Last messages:", "2"))
        for role, content in recent:
            who = "you" if role is MessageRole.USER else "bot"
            self._print(self._style(f"  {who} › {content}", "2"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chatbot", description="Chat with the Nimbus Gear support assistant.")
    parser.add_argument("--resume", metavar="ID", help="continue a saved conversation")
    args = parser.parse_args(argv)

    configure_logging(get_settings().log_file)
    try:
        preflight()
    except SetupError as e:
        print(f"Setup needed: {e}", file=sys.stderr)
        return 1

    service = ChatService()
    purged = service.purge_expired()
    if purged:
        logging.getLogger(__name__).info("Purged %d expired conversations", purged)

    return ChatLoop(service, conversation_id=args.resume).run()


if __name__ == "__main__":
    sys.exit(main())
