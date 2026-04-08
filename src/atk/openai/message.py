"""OpenAI message mapper."""

import json
from typing import cast

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
    ThinkingPart,
    ToolCallPart,
    ToolMessage,
    ToolResultPart,
    UserMessage,
)


class OpenAIMessageMapper:
    """Maps between internal message types and OpenAI message formats."""

    @staticmethod
    def _map_assistant_message(
        msg: AssistantMessage,
    ) -> ChatCompletionMessageParam:
        """Map an AssistantMessage to an OpenAI message parameter."""
        mapped_msg: dict[str, object] = {"role": msg.role}
        content: list[ChatCompletionContentPartParam | str] = []
        tool_calls: list[ChatCompletionMessageToolCallUnionParam] = []
        thinking_parts: list[str] = []
        for part in msg.content:
            match part:
                case TextPart():
                    content.append(
                        ChatCompletionContentPartTextParam(
                            {"type": "text", "text": part.text}
                        )
                    )
                case ToolCallPart():
                    tool_calls.append(
                        ChatCompletionMessageFunctionToolCallParam(
                            {
                                "type": "function",
                                "id": part.id,
                                "function": {
                                    "name": part.name,
                                    "arguments": json.dumps(part.arguments),
                                },
                            }
                        )
                    )
                case ThinkingPart():
                    thinking_parts.append(part.thinking)
                case _:
                    error_msg = f"Unsupported content part type: {type(part)}"
                    raise NotImplementedError(error_msg)
        mapped_msg["content"] = content
        if tool_calls:
            mapped_msg["tool_calls"] = tool_calls
        if thinking_parts:
            mapped_msg["reasoning_content"] = "".join(thinking_parts)
        return cast("ChatCompletionMessageParam", mapped_msg)

    @staticmethod
    def _map_user_message(msg: UserMessage) -> ChatCompletionMessageParam:
        """Map a UserMessage to an OpenAI message parameter."""
        content: list[ChatCompletionContentPartParam | str] = []
        for part in msg.content:
            match part:
                case TextPart():
                    content.append(
                        ChatCompletionContentPartTextParam(
                            {"type": "text", "text": part.text}
                        )
                    )
                case _:
                    error_msg = f"Unsupported content part type: {type(part)}"
                    raise NotImplementedError(error_msg)
        return cast(
            "ChatCompletionMessageParam", {"role": msg.role, "content": content}
        )

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
            if isinstance(msg, ToolMessage):
                for part in msg.content:
                    if isinstance(part, ToolResultPart):
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": part.tool_call_id,
                                "content": part.content,
                            }
                        )
                    else:
                        error_msg = (
                            f"Unsupported ToolMessage content part type: {type(part)}"
                        )
                        raise NotImplementedError(error_msg)
                continue

            if isinstance(msg, AssistantMessage):
                result.append(OpenAIMessageMapper._map_assistant_message(msg))
            elif isinstance(msg, UserMessage):
                result.append(OpenAIMessageMapper._map_user_message(msg))
            else:
                error_msg = f"Unsupported message role: {type(msg)}"
                raise NotImplementedError(error_msg)

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

        reasoning_content: str | None = getattr(message, "reasoning_content", None)
        if reasoning_content:
            assistant_content.append(ThinkingPart(thinking=reasoning_content))

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
