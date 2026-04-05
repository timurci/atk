"""Language model interface."""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import BaseModel

    from .message import (
        AssistantMessage,
        AssistantStream,
        Message,
    )
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


class StreamingLanguageModel(Protocol):
    """Interface for a language model that supports streaming responses."""

    def stream_response(
        self,
        instruction: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> AsyncIterator[AssistantStream | AssistantMessage]:
        """Stream a response as incremental chunks.

        Yields ``AssistantStream`` chunks during generation, ending with
        a single ``AssistantMessage`` containing the fully accumulated
        content.

        Args:
            instruction: The system instruction.
            messages: Conversation history.
            tools: List of available tools for the model to call.
            response_format: Target output structure in text responses.
        """
        ...
