"""Language model interface."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pydantic import BaseModel

    from .message import AssistantMessage, Message
    from .tool import Tool


class LanguageModel(Protocol):
    """Interface for a language model."""

    async def generate_response(
        self,
        instruction: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> AssistantMessage:
        """Generate response from the given instruction and conversation history.

        Args:
            instruction: The system instruction.
            messages: Conversation history.
            tools: List of available tools for the model to call.
            response_format: Target output structure in text responses.
        """
        ...
