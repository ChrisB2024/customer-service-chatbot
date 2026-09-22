"""Test doubles shared across test modules."""

from itertools import count

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_ids = count()


class ScriptedModel(GenericFakeChatModel):
    """Replays scripted AIMessages (or raises scripted exceptions) and records what the agent sent on each call."""

    seen: list[list] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        item = next(self.messages)
        if isinstance(item, BaseException):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


def call(name: str, **args) -> dict:
    return {"name": name, "args": args, "id": f"call_{next(_ids)}"}


def scripted(*turns: AIMessage | BaseException) -> ScriptedModel:
    return ScriptedModel(messages=iter(turns))


def tool_results(model: ScriptedModel) -> list[ToolMessage]:
    """Tool results the model saw on its last call."""
    return [m for m in model.seen[-1] if isinstance(m, ToolMessage)]
