"""Tests for AssistantStreamAccumulator."""

import json

import pytest

from atk.core import (
    AssistantMessage,
    AssistantStream,
    AssistantStreamAccumulator,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
)
from atk.core.message import ToolArgumentsParsingError


def _text_parts(message: AssistantMessage) -> list[TextPart]:
    return [part for part in message.content if isinstance(part, TextPart)]


def _thinking_parts(message: AssistantMessage) -> list[ThinkingPart]:
    return [part for part in message.content if isinstance(part, ThinkingPart)]


def _tool_call_parts(message: AssistantMessage) -> list[ToolCallPart]:
    return [part for part in message.content if isinstance(part, ToolCallPart)]


class TestAssistantStreamAccumulatorText:
    """Test text delta accumulation."""

    @staticmethod
    def test_single_text_delta() -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_delta(TextDelta(text="Hello"))

        text_parts = _text_parts(accumulator.build_message())

        assert len(text_parts) == 1
        assert text_parts[0].text == "Hello"

    @staticmethod
    def test_multiple_text_deltas() -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_delta(TextDelta(text="Hello"))
        accumulator.add_delta(TextDelta(text=" world"))

        text_parts = _text_parts(accumulator.build_message())

        assert len(text_parts) == 1
        assert text_parts[0].text == "Hello world"

    @staticmethod
    def test_no_text_deltas() -> None:
        accumulator = AssistantStreamAccumulator()

        assert _text_parts(accumulator.build_message()) == []


class TestAssistantStreamAccumulatorThinking:
    """Test thinking delta accumulation."""

    @staticmethod
    def test_single_thinking_delta() -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_delta(ThinkingDelta(thinking="I need to think..."))

        thinking_parts = _thinking_parts(accumulator.build_message())

        assert len(thinking_parts) == 1
        assert thinking_parts[0].thinking == "I need to think..."

    @staticmethod
    def test_multiple_thinking_deltas() -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_delta(ThinkingDelta(thinking="Step 1: "))
        accumulator.add_delta(ThinkingDelta(thinking="Step 2."))

        thinking_parts = _thinking_parts(accumulator.build_message())

        assert len(thinking_parts) == 1
        assert thinking_parts[0].thinking == "Step 1: Step 2."

    @staticmethod
    def test_thinking_before_text() -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_delta(ThinkingDelta(thinking="Thinking..."))
        accumulator.add_delta(TextDelta(text="The answer."))

        message = accumulator.build_message()

        assert [part.type for part in message.content] == ["thinking", "text"]


class TestAssistantStreamAccumulatorToolCalls:
    """Test tool call delta accumulation."""

    @staticmethod
    def test_single_tool_call_delta() -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_delta(
            ToolCallDelta(
                id="call_1",
                name="search",
                arguments_delta='{"q": "test"}',
            )
        )

        tool_parts = _tool_call_parts(accumulator.build_message())

        assert len(tool_parts) == 1
        assert tool_parts[0].id == "call_1"
        assert tool_parts[0].name == "search"
        assert tool_parts[0].arguments == {"q": "test"}

    @staticmethod
    def test_fragmented_tool_call_arguments() -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_delta(
            ToolCallDelta(id="call_1", name="search", arguments_delta='{"q')
        )
        accumulator.add_delta(
            ToolCallDelta(id="call_1", name="search", arguments_delta='": "test"}')
        )

        tool_parts = _tool_call_parts(accumulator.build_message())

        assert len(tool_parts) == 1
        assert tool_parts[0].arguments == {"q": "test"}

    @staticmethod
    def test_multiple_tool_calls_preserve_first_seen_order() -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_delta(
            ToolCallDelta(id="call_2", name="fetch", arguments_delta="")
        )
        accumulator.add_delta(
            ToolCallDelta(id="call_1", name="search", arguments_delta='{"q": "test"}')
        )
        accumulator.add_delta(
            ToolCallDelta(id="call_2", name="fetch", arguments_delta='{"id": 1}')
        )

        tool_parts = _tool_call_parts(accumulator.build_message())

        assert [part.id for part in tool_parts] == ["call_2", "call_1"]
        assert [part.arguments for part in tool_parts] == [
            {"id": 1},
            {"q": "test"},
        ]

    @staticmethod
    @pytest.mark.parametrize("arguments_delta", ["", "   "])
    def test_empty_tool_call_json_becomes_empty_dict(
        arguments_delta: str,
    ) -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_delta(
            ToolCallDelta(
                id="call_1",
                name="search",
                arguments_delta=arguments_delta,
            )
        )

        tool_parts = _tool_call_parts(accumulator.build_message())

        assert len(tool_parts) == 1
        assert tool_parts[0].arguments == {}

    @staticmethod
    def test_malformed_tool_call_json_raises_json_decode_error() -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_delta(
            ToolCallDelta(
                id="call_1",
                name="search",
                arguments_delta="{invalid",
            )
        )

        with pytest.raises(json.JSONDecodeError):
            accumulator.build_message()

    @staticmethod
    @pytest.mark.parametrize("arguments_delta", ['["not", "object"]', '"text"', "1"])
    def test_non_object_tool_call_json_raises_tool_arguments_parsing_error(
        arguments_delta: str,
    ) -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_delta(
            ToolCallDelta(
                id="call_1",
                name="search",
                arguments_delta=arguments_delta,
            )
        )

        with pytest.raises(
            ToolArgumentsParsingError,
            match="JSON parsing did not match to a dictionary format",
        ):
            accumulator.build_message()


class TestAssistantStreamAccumulatorCombined:
    """Test stream-level and combined accumulation."""

    @staticmethod
    def test_add_stream_adds_all_content_deltas() -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_stream(
            AssistantStream(
                content=[
                    ThinkingDelta(thinking="Think. "),
                    TextDelta(text="Answer."),
                    ToolCallDelta(id="call_1", name="lookup", arguments_delta="{}"),
                ]
            )
        )

        message = accumulator.build_message()

        assert [part.type for part in message.content] == [
            "thinking",
            "text",
            "tool_call",
        ]

    @staticmethod
    def test_empty_stream_produces_empty_message() -> None:
        accumulator = AssistantStreamAccumulator()
        accumulator.add_stream(AssistantStream(content=[]))

        message = accumulator.build_message()

        assert isinstance(message, AssistantMessage)
        assert message.content == []
