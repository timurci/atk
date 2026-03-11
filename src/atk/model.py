"""Language model interface."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pydantic import BaseModel

    from atk.message import AssistantMessage, Message


class LanguageModel(Protocol):
    """Interface for a language model."""

    async def generate_response(
        self,
        instruction: str,
        messages: list[Message],
        # tools: list[Tool],
        response_format: BaseModel | None = None,
    ) -> AssistantMessage:
        """Generate response from the given instruction and conversation history.

        Args:
            instruction: The system instruction.
            messages: Conversation history.
            response_format: Target output structure in text responses.
        """
        ...
