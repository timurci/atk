"""Message mapper for any-llm completions API."""

import json
from typing import Any

from any_llm.types.completion import (
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Function,
    Reasoning,
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


class MessageMapper:
    """Maps between internal message types and any-llm message formats."""

    @staticmethod
    def _map_assistant_message(
        msg: AssistantMessage,
    ) -> ChatCompletionMessage | dict[str, Any]:
        """Map an AssistantMessage to an any-llm message parameter.

        Returns a ChatCompletionMessage when thinking is present
        (since it supports the reasoning field), otherwise a dict
        for simplicity.
        """
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for part in msg.content:
            match part:
                case TextPart():
                    text_parts.append(part.text)
                case ToolCallPart():
                    tool_calls.append(
                        {
                            "type": "function",
                            "id": part.id,
                            "function": {
                                "name": part.name,
                                "arguments": json.dumps(part.arguments),
                            },
                        }
                    )
                case ThinkingPart():
                    thinking_parts.append(part.thinking)
                case _:
                    error_msg = f"Unsupported content part type: {type(part)}"
                    raise NotImplementedError(error_msg)

        content = "".join(text_parts) if text_parts else None
        reasoning = (
            Reasoning(content="".join(thinking_parts)) if thinking_parts else None
        )

        if tool_calls:
            mapped_tool_calls = [
                ChatCompletionMessageFunctionToolCall(
                    id=tc["id"],
                    function=Function(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                    type="function",
                )
                for tc in tool_calls
            ]
            return ChatCompletionMessage(
                content=content,
                reasoning=reasoning,
                tool_calls=mapped_tool_calls,
                role="assistant",
            )

        if reasoning is not None:
            return ChatCompletionMessage(
                content=content,
                reasoning=reasoning,
                role="assistant",
            )

        return {"role": "assistant", "content": content}

    @staticmethod
    def _map_user_message(msg: UserMessage) -> dict[str, Any]:
        """Map a UserMessage to an any-llm message parameter."""
        content: list[dict[str, Any]] | str = []
        for part in msg.content:
            match part:
                case TextPart():
                    content.append({"type": "text", "text": part.text})
                case _:
                    error_msg = f"Unsupported content part type: {type(part)}"
                    raise NotImplementedError(error_msg)
        if len(content) == 1:
            return {"role": "user", "content": content[0]["text"]}
        return {"role": "user", "content": content}

    @staticmethod
    def to_messages(
        instruction: str, messages: list[Message]
    ) -> list[dict[str, Any] | ChatCompletionMessage]:
        """Convert internal messages to any-llm message format.

        Args:
            instruction: System instruction to prepend.
            messages: Internal message list.

        Returns:
            List of message parameters for any-llm's completion API.
        """
        result: list[dict[str, Any] | ChatCompletionMessage] = [
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
                result.append(MessageMapper._map_assistant_message(msg))
            elif isinstance(msg, UserMessage):
                result.append(MessageMapper._map_user_message(msg))
            else:
                error_msg = f"Unsupported message role: {type(msg)}"
                raise NotImplementedError(error_msg)

        return result

    @staticmethod
    def from_completion(message: ChatCompletionMessage) -> AssistantMessage:
        """Convert an any-llm ChatCompletionMessage to internal AssistantMessage.

        Args:
            message: Completion message from any-llm.

        Returns:
            Internal AssistantMessage.
        """
        assistant_content = []

        if message.reasoning is not None:
            assistant_content.append(ThinkingPart(thinking=message.reasoning.content))

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
                    case _:
                        error_msg = "Custom tool calls are not supported yet."
                        raise NotImplementedError(error_msg)

        if message.audio is not None:
            error_msg = "Audio output is not supported yet."
            raise NotImplementedError(error_msg)

        return AssistantMessage(content=assistant_content)
