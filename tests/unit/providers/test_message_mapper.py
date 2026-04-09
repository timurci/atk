"""Tests for MessageMapper — bidirectional mapping between core and any-llm types."""

import json

import pytest
from any_llm.types.completion import (
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Function,
    Reasoning,
)
from pydantic import ValidationError

from atk.core.message import (
    AssistantMessage,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolMessage,
    ToolResultPart,
    UserMessage,
)
from atk.providers.message import MessageMapper


def _make_tool_call(
    call_id: str,
    name: str,
    arguments: dict,
) -> ChatCompletionMessageFunctionToolCall:
    """Construct a ChatCompletionMessageFunctionToolCall for tests."""
    return ChatCompletionMessageFunctionToolCall(
        id=call_id,
        function=Function(name=name, arguments=json.dumps(arguments)),
        type="function",
    )


# ------------------------------------------------------------------ #
# from_completion — basic message conversion                          #
# ------------------------------------------------------------------ #


class TestFromCompletionBasic:
    """Test ChatCompletionMessage -> AssistantMessage conversion."""

    @staticmethod
    def test_text_only() -> None:
        msg = ChatCompletionMessage(content="Hello, world!", role="assistant")
        result = MessageMapper.from_completion(msg)
        assert isinstance(result, AssistantMessage)
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextPart)
        assert result.content[0].text == "Hello, world!"

    @staticmethod
    def test_null_content() -> None:
        msg = ChatCompletionMessage(content=None, role="assistant")
        result = MessageMapper.from_completion(msg)
        assert isinstance(result, AssistantMessage)
        assert len(result.content) == 0

    @staticmethod
    def test_tool_calls() -> None:
        msg = ChatCompletionMessage(
            content=None,
            role="assistant",
            tool_calls=[_make_tool_call("call_1", "get_weather", {"city": "Paris"})],
        )
        result = MessageMapper.from_completion(msg)
        assert len(result.content) == 1
        part = result.content[0]
        assert isinstance(part, ToolCallPart)
        assert part.id == "call_1"
        assert part.name == "get_weather"
        assert part.arguments == {"city": "Paris"}

    @staticmethod
    def test_multiple_tool_calls() -> None:
        msg = ChatCompletionMessage(
            content=None,
            role="assistant",
            tool_calls=[
                _make_tool_call("call_a", "fn_a", {"x": 1}),
                _make_tool_call("call_b", "fn_b", {"y": 2}),
            ],
        )
        result = MessageMapper.from_completion(msg)
        tool_parts = [p for p in result.content if isinstance(p, ToolCallPart)]
        assert len(tool_parts) == 2
        assert tool_parts[0].name == "fn_a"
        assert tool_parts[1].name == "fn_b"


# ------------------------------------------------------------------ #
# from_completion — thinking / reasoning                              #
# ------------------------------------------------------------------ #


class TestFromCompletionThinking:
    """Test reasoning content is mapped to ThinkingPart."""

    @staticmethod
    def test_reasoning_before_text() -> None:
        msg = ChatCompletionMessage(
            content="The answer is 42.",
            role="assistant",
            reasoning=Reasoning(content="Let me think about this..."),
        )
        result = MessageMapper.from_completion(msg)
        thinking_parts = [p for p in result.content if isinstance(p, ThinkingPart)]
        text_parts = [p for p in result.content if isinstance(p, TextPart)]
        assert len(thinking_parts) == 1
        assert thinking_parts[0].thinking == "Let me think about this..."
        assert len(text_parts) == 1

    @staticmethod
    def test_reasoning_without_text() -> None:
        msg = ChatCompletionMessage(
            content=None,
            role="assistant",
            reasoning=Reasoning(content="Deep thoughts"),
        )
        result = MessageMapper.from_completion(msg)
        thinking_parts = [p for p in result.content if isinstance(p, ThinkingPart)]
        assert len(thinking_parts) == 1
        assert thinking_parts[0].thinking == "Deep thoughts"

    @staticmethod
    def test_reasoning_with_tool_calls() -> None:
        msg = ChatCompletionMessage(
            content=None,
            role="assistant",
            reasoning=Reasoning(content="I need to call a tool"),
            tool_calls=[_make_tool_call("tc_1", "lookup", {})],
        )
        result = MessageMapper.from_completion(msg)
        thinking_parts = [p for p in result.content if isinstance(p, ThinkingPart)]
        tool_parts = [p for p in result.content if isinstance(p, ToolCallPart)]
        assert len(thinking_parts) == 1
        assert len(tool_parts) == 1

    @staticmethod
    def test_no_reasoning() -> None:
        msg = ChatCompletionMessage(content="Hello", role="assistant", reasoning=None)
        result = MessageMapper.from_completion(msg)
        thinking_parts = [p for p in result.content if isinstance(p, ThinkingPart)]
        assert len(thinking_parts) == 0


# ------------------------------------------------------------------ #
# from_completion — unsupported features                             #
# ------------------------------------------------------------------ #


class TestFromCompletionUnsupported:
    """Test that unsupported features raise NotImplementedError."""

    @staticmethod
    def test_audio_raises() -> None:
        msg = ChatCompletionMessage(content="Hi", role="assistant")
        msg.audio = object()  # type: ignore[attr-defined]
        with pytest.raises(NotImplementedError, match="Audio"):
            MessageMapper.from_completion(msg)


# ------------------------------------------------------------------ #
# to_messages — basic message format conversion                       #
# ------------------------------------------------------------------ #


class TestToMessagesBasic:
    """Test core messages -> any-llm message list conversion."""

    @staticmethod
    def test_system_instruction_prepended() -> None:
        messages = [
            UserMessage(content=[TextPart(text="Hello")]),
        ]
        result = MessageMapper.to_messages("Be helpful", messages)
        assert result[0] == {"role": "system", "content": "Be helpful"}

    @staticmethod
    def test_user_message() -> None:
        messages = [
            UserMessage(content=[TextPart(text="What is 2+2?")]),
        ]
        result = MessageMapper.to_messages("System", messages)
        user_msg = result[1]
        assert isinstance(user_msg, dict)
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "What is 2+2?"

    @staticmethod
    def test_tool_message() -> None:
        messages = [
            ToolMessage(content=[ToolResultPart(tool_call_id="c1", content="result")]),
        ]
        result = MessageMapper.to_messages("System", messages)
        tool_msg = result[1]
        assert isinstance(tool_msg, dict)
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "c1"
        assert tool_msg["content"] == "result"

    @staticmethod
    def test_assistant_message_text() -> None:
        messages = [
            AssistantMessage(content=[TextPart(text="Hi there")]),
        ]
        result = MessageMapper.to_messages("System", messages)
        asst_msg = result[1]
        assert isinstance(asst_msg, ChatCompletionMessage)
        assert asst_msg.role == "assistant"
        assert asst_msg.content == "Hi there"


# ------------------------------------------------------------------ #
# to_messages — thinking / reasoning mapping for input                 #
# ------------------------------------------------------------------ #


class TestToMessagesThinking:
    """Test that ThinkingPart is mapped to reasoning in input messages."""

    @staticmethod
    def test_thinking_in_assistant_message() -> None:
        messages = [
            AssistantMessage(
                content=[
                    ThinkingPart(thinking="I need to think..."),
                    TextPart(text="The answer"),
                ]
            ),
        ]
        result = MessageMapper.to_messages("System", messages)
        asst_msg = result[1]
        assert isinstance(asst_msg, ChatCompletionMessage)
        assert asst_msg.reasoning is not None
        assert asst_msg.reasoning.content == "I need to think..."
        assert asst_msg.content == "The answer"

    @staticmethod
    def test_thinking_only_in_assistant_message() -> None:
        messages = [
            AssistantMessage(
                content=[
                    ThinkingPart(thinking="Pure reasoning"),
                ]
            ),
        ]
        result = MessageMapper.to_messages("System", messages)
        asst_msg = result[1]
        assert isinstance(asst_msg, ChatCompletionMessage)
        assert asst_msg.reasoning is not None
        assert asst_msg.reasoning.content == "Pure reasoning"
        assert asst_msg.content is None

    @staticmethod
    def test_thinking_with_tool_calls() -> None:
        messages = [
            AssistantMessage(
                content=[
                    ThinkingPart(thinking="Need a tool"),
                    ToolCallPart(
                        id="tc1",
                        name="search",
                        arguments={"query": "test"},
                    ),
                ]
            ),
        ]
        result = MessageMapper.to_messages("System", messages)
        asst_msg = result[1]
        assert isinstance(asst_msg, ChatCompletionMessage)
        assert asst_msg.reasoning is not None
        assert asst_msg.reasoning.content == "Need a tool"
        assert asst_msg.tool_calls is not None
        assert len(asst_msg.tool_calls) == 1


# ------------------------------------------------------------------ #
# to_messages — multiple content parts                                 #
# ------------------------------------------------------------------ #


class TestToMessagesMultiPart:
    """Test messages with multiple content parts."""

    @staticmethod
    def test_user_message_multiple_parts() -> None:
        messages = [
            UserMessage(
                content=[
                    TextPart(text="Hello"),
                    TextPart(text="World"),
                ]
            ),
        ]
        result = MessageMapper.to_messages("System", messages)
        user_msg = result[1]
        assert isinstance(user_msg, dict)
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
        num_content_parts = len(user_msg["content"])
        assert num_content_parts == 2

    @staticmethod
    def test_assistant_text_and_tool_call() -> None:
        messages = [
            AssistantMessage(
                content=[
                    TextPart(text="Let me look that up."),
                    ToolCallPart(
                        id="c1",
                        name="search",
                        arguments={"q": "test"},
                    ),
                ]
            ),
        ]
        result = MessageMapper.to_messages("System", messages)
        asst_msg = result[1]
        assert isinstance(asst_msg, ChatCompletionMessage)
        assert asst_msg.content == "Let me look that up."
        assert asst_msg.tool_calls is not None


# ------------------------------------------------------------------ #
# to_messages — error cases                                           #
# ------------------------------------------------------------------ #


class TestToMessagesErrors:
    """Test error handling for unsupported types."""

    @staticmethod
    def test_invalid_content_part_in_assistant() -> None:
        """Pydantic rejects ToolResultPart in AssistantMessage content."""
        with pytest.raises(ValidationError):
            AssistantMessage(
                content=[ToolResultPart(tool_call_id="x", content="y")],  # type: ignore[list-item]
            )
