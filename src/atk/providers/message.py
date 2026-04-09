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

# Heterogeneous payload type for any-llm message parameters.
# Any is required because different LLM providers produce varied,
# untyped payload shapes with arbitrary keys/values (e.g. "role",
# "content", "tool_call_id"). This alias documents that intentional
# dynamic typing at the provider boundary.
AnyLLMPayload = dict[str, Any]


class MessageMapper:
    """Maps between internal message types and any-llm message formats."""

    @staticmethod
    def _map_assistant_message(msg: AssistantMessage) -> ChatCompletionMessage:
        """Map an AssistantMessage to an any-llm ChatCompletionMessage."""
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        tool_call_parts: list[ChatCompletionMessageFunctionToolCall] = []
        for part in msg.content:
            match part:
                case TextPart():
                    text_parts.append(part.text)
                case ToolCallPart():
                    tool_call_parts.append(
                        ChatCompletionMessageFunctionToolCall(
                            id=part.id,
                            function=Function(
                                name=part.name,
                                arguments=json.dumps(part.arguments),
                            ),
                            type="function",
                        )
                    )
                case ThinkingPart():
                    thinking_parts.append(part.thinking)
                case _:
                    error_msg = f"Unsupported content part type: {type(part)}"
                    raise NotImplementedError(error_msg)

        content_str = "".join(text_parts) if text_parts else None
        reasoning_obj = (
            Reasoning(content="".join(thinking_parts)) if thinking_parts else None
        )

        if tool_call_parts:
            return ChatCompletionMessage(
                content=content_str,
                reasoning=reasoning_obj,
                tool_calls=list(tool_call_parts),
                role="assistant",
            )
        return ChatCompletionMessage(
            content=content_str,
            reasoning=reasoning_obj,
            role="assistant",
        )

    @staticmethod
    def _map_user_message(msg: UserMessage) -> AnyLLMPayload:
        """Map a UserMessage to an any-llm message parameter."""
        content: list[AnyLLMPayload] | str = []
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
    ) -> list[AnyLLMPayload | ChatCompletionMessage]:
        """Convert internal messages to any-llm message format.

        Args:
            instruction: System instruction to prepend.
            messages: Internal message list.

        Returns:
            List of message parameters for any-llm's completion API.
        """
        result: list[AnyLLMPayload | ChatCompletionMessage] = [
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
