"""OpenAI language model implementation."""

from typing import TYPE_CHECKING, cast

from openai import AsyncOpenAI, Omit
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)

from atk.core.message import AssistantMessage, Message, TextPart, UserMessage
from atk.core.model import LanguageModel

if TYPE_CHECKING:
    from pydantic import BaseModel


class OpenAIMessageMapper:
    """Maps between internal message types and OpenAI message formats."""

    @staticmethod
    def to_openai(
        instruction: str, messages: list[Message]
    ) -> list[ChatCompletionMessageParam]:
        """Convert internal messages to OpenAI message format.

        Args:
            instruction: System instruction to prepend.
            messages: Internal message list.

        Returns:
            List of OpenAI-compatible message parameters.
        """
        result: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": instruction}
        ]
        for msg in messages:
            if isinstance(msg, (UserMessage, AssistantMessage)):
                role = msg.role
            else:
                error_msg = f"Unsupported message role: {type(msg)}"
                raise NotImplementedError(error_msg)

            content: list[ChatCompletionContentPartParam] = []
            for part in msg.content:
                if isinstance(part, TextPart):
                    content.append(
                        ChatCompletionContentPartTextParam(
                            {"type": "text", "text": part.text}
                        )
                    )
                else:
                    error_msg = f"Unsupported content part type: {type(part)}"
                    raise NotImplementedError(error_msg)

            result.append(
                cast("ChatCompletionMessageParam", {"role": role, "content": content})
            )

        return result

    @staticmethod
    def from_openai(message: ChatCompletionMessage) -> AssistantMessage:
        """Convert OpenAI message to internal AssistantMessage format.

        Args:
            message: OpenAI chat completion message.

        Returns:
            Internal AssistantMessage.
        """
        content_raw = message.content

        if content_raw is None:
            content = []
        elif isinstance(content_raw, str):
            content = [TextPart(text=content_raw)]
        else:
            error_msg = f"Unsupported content type: {type(content_raw)}"
            raise NotImplementedError(error_msg)

        return AssistantMessage(content=content)


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
        self.mapper = OpenAIMessageMapper()

    async def generate_response(
        self,
        instruction: str,
        messages: list[Message],
        response_format: type[BaseModel] | None = None,
    ) -> AssistantMessage:
        """Generate a response from the language model.

        Args:
            instruction: System instruction.
            messages: Conversation history.
            response_format: Optional Pydantic model for structured output.

        Returns:
            AssistantMessage response.
        """
        response = await self.client.chat.completions.parse(
            model=self.model,
            messages=self.mapper.to_openai(instruction, messages),
            response_format=response_format or Omit(),  # type: ignore[invalid-argument-type]
        )
        return self.mapper.from_openai(response.choices[0].message)
