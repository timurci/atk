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
    ToolCallPart,
    ToolMessage,
    ToolResultPart,
    UserMessage,
)


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

            mapped_msg: dict[str, object] = {}
            if isinstance(msg, (UserMessage, AssistantMessage)):
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
