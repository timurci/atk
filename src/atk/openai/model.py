"""OpenAI language model implementation."""

from typing import TYPE_CHECKING

from openai import AsyncOpenAI, Omit

from atk.core.model import LanguageModel

from .message import OpenAIMessageMapper
from .tool import OpenAIToolMapper

if TYPE_CHECKING:
    from pydantic import BaseModel

    from atk.core.message import AssistantMessage, Message
    from atk.core.tool import Tool


class OpenAILanguageModel(LanguageModel):
    """OpenAI language model implementation."""

    def __init__(
        self,
        base_url: str | None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the OpenAI language model.

        Args:
            base_url: Optional base URL for OpenAI-compatible API.
            api_key: Optional API key.
            model: Optional model name.
        """
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model: str = model if model is not None else "gpt-4o-mini"
        self.message_mapper = OpenAIMessageMapper()
        self.tool_mapper = OpenAIToolMapper()

    async def generate_response(
        self,
        instruction: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> AssistantMessage:
        """Generate a response from the language model.

        Args:
            instruction: System instruction.
            messages: Conversation history.
            tools: List of available tools for the model to call.
            response_format: Optional Pydantic model for structured output.

        Returns:
            AssistantMessage response.
        """
        response = await self.client.chat.completions.parse(
            model=self.model,
            messages=self.message_mapper.to_openai(instruction, messages),
            tools=self.tool_mapper.to_openai(tools) if tools else Omit(),
            response_format=response_format or Omit(),  # type: ignore[invalid-argument-type]
        )
        return self.message_mapper.from_openai(response.choices[0].message)
