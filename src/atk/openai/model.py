"""OpenAI language model implementation."""

import json
from typing import TYPE_CHECKING, cast

from openai import AsyncOpenAI, Omit
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessage,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnionParam,
)

from atk.core.message import (
    AssistantMessage,
    Message,
    TextPart,
    ToolCallPart,
    ToolMessage,
    ToolResultPart,
    UserMessage,
)
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
            mapped_msg = {}
            if isinstance(msg, (UserMessage, AssistantMessage, ToolMessage)):
                role = msg.role
            else:
                error_msg = f"Unsupported message role: {type(msg)}"
                raise NotImplementedError(error_msg)

            mapped_msg["role"] = role

            content: list[ChatCompletionContentPartParam | str] = []
            tool_calls: list[ChatCompletionMessageToolCallUnionParam] = []
            for part in msg.content:
                match part:
                    case TextPart():
                        mapped_part = ChatCompletionContentPartTextParam(
                            {"type": "text", "text": part.text}
                        )
                        content.append(mapped_part)
                    case ToolResultPart():
                        mapped_part = part.content
                        mapped_msg["tool_call_id"] = part.tool_call_id
                    case ToolCallPart():
                        mapped_call = ChatCompletionMessageFunctionToolCallParam(
                            {
                                "type": "function",
                                "id": part.id,
                                "function": {
                                    "name": part.name,
                                    "arguments": json.dumps(part.arguments),
                                },
                            }
                        )
                        tool_calls.append(mapped_call)
                    case _:
                        error_msg = f"Unsupported content part type: {type(part)}"
                        raise NotImplementedError(error_msg)
            mapped_msg["content"] = content
            if len(tool_calls) > 0:
                mapped_msg["tool_calls"] = tool_calls
            result.append(cast("ChatCompletionMessageParam", mapped_msg))

        return result

    @staticmethod
    def from_openai(message: ChatCompletionMessage) -> AssistantMessage:
        """Convert OpenAI message to internal AssistantMessage format.

        Args:
            message: OpenAI chat completion message.

        Returns:
            Internal AssistantMessage.
        """
        assistant_content = []

        if message.content is not None:
            assistant_content.append(TextPart(text=message.content))

        if message.tool_calls is not None:
            for tool_call in message.tool_calls:
                match tool_call:
                    case ChatCompletionMessageFunctionToolCall():
                        assistant_content.append(
                            ToolCallPart(
                                id=tool_call.id,
                                name=tool_call.function.name,
                                arguments=json.loads(tool_call.function.arguments),
                            )
                        )
                    case ChatCompletionMessageCustomToolCall():
                        error_msg = "Custom tool calls are not supported yet."
                        raise NotImplementedError(error_msg)

        if message.audio is not None:
            error_msg = "Audio output is not supported yet."
            raise NotImplementedError(error_msg)

        return AssistantMessage(content=assistant_content)


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
