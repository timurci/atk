"""Tests for _StreamAccumulator.

Accumulating streaming chunks into AssistantMessage.
"""

from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    Reasoning,
)

from atk.core.message import (
    AssistantMessage,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
)
from atk.providers.model import _StreamAccumulator


def _make_chunk(
    *,
    content: str | None = None,
    reasoning: Reasoning | None = None,
    tool_calls: list[ChoiceDeltaToolCall] | None = None,
) -> ChatCompletionChunk:
    """Build a minimal ChatCompletionChunk for testing."""
    delta = ChoiceDelta(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
    )
    choice = ChunkChoice(delta=delta, finish_reason=None, index=0)
    return ChatCompletionChunk(
        id="test_chunk",
        choices=[choice],
        created=0,
        model="test",
        object="chat.completion.chunk",
    )


# ------------------------------------------------------------------ #
# Text accumulation                                                   #
# ------------------------------------------------------------------ #


class TestStreamAccumulatorText:
    """Test text chunk accumulation."""

    @staticmethod
    def test_single_text_chunk() -> None:
        acc = _StreamAccumulator()
        chunk = _make_chunk(content="Hello")
        events = acc.process_chunk(chunk)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, TextDelta)
        assert event.text == "Hello"
        msg = acc.build_message()
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextPart)
        assert msg.content[0].text == "Hello"

    @staticmethod
    def test_multiple_text_chunks() -> None:
        acc = _StreamAccumulator()
        acc.process_chunk(_make_chunk(content="Hello"))
        acc.process_chunk(_make_chunk(content=" world"))
        msg = acc.build_message()
        text_parts = [p for p in msg.content if isinstance(p, TextPart)]
        assert len(text_parts) == 1
        assert text_parts[0].text == "Hello world"

    @staticmethod
    def test_empty_chunks_produce_no_text() -> None:
        acc = _StreamAccumulator()
        acc.process_chunk(_make_chunk(content=None))
        msg = acc.build_message()
        text_parts = [p for p in msg.content if isinstance(p, TextPart)]
        assert len(text_parts) == 0


# ------------------------------------------------------------------ #
# Thinking / reasoning accumulation                                    #
# ------------------------------------------------------------------ #


class TestStreamAccumulatorThinking:
    """Test reasoning chunk accumulation into ThinkingPart."""

    @staticmethod
    def test_single_reasoning_chunk() -> None:
        acc = _StreamAccumulator()
        chunk = _make_chunk(reasoning=Reasoning(content="I need to think..."))
        events = acc.process_chunk(chunk)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, ThinkingDelta)
        assert event.thinking == "I need to think..."
        msg = acc.build_message()
        thinking_parts = [p for p in msg.content if isinstance(p, ThinkingPart)]
        assert len(thinking_parts) == 1
        assert thinking_parts[0].thinking == "I need to think..."

    @staticmethod
    def test_multiple_reasoning_chunks() -> None:
        acc = _StreamAccumulator()
        acc.process_chunk(_make_chunk(reasoning=Reasoning(content="Step 1: ")))
        acc.process_chunk(_make_chunk(reasoning=Reasoning(content="Step 2.")))
        msg = acc.build_message()
        thinking_parts = [p for p in msg.content if isinstance(p, ThinkingPart)]
        assert len(thinking_parts) == 1
        assert thinking_parts[0].thinking == "Step 1: Step 2."

    @staticmethod
    def test_reasoning_before_text() -> None:
        acc = _StreamAccumulator()
        acc.process_chunk(_make_chunk(reasoning=Reasoning(content="Thinking...")))
        acc.process_chunk(_make_chunk(content="The answer."))
        msg = acc.build_message()
        thinking_parts = [p for p in msg.content if isinstance(p, ThinkingPart)]
        text_parts = [p for p in msg.content if isinstance(p, TextPart)]
        assert len(thinking_parts) == 1
        assert len(text_parts) == 1

    @staticmethod
    def test_no_reasoning() -> None:
        acc = _StreamAccumulator()
        acc.process_chunk(_make_chunk(content="Hello"))
        msg = acc.build_message()
        thinking_parts = [p for p in msg.content if isinstance(p, ThinkingPart)]
        assert len(thinking_parts) == 0


# ------------------------------------------------------------------ #
# Tool call accumulation                                               #
# ------------------------------------------------------------------ #


class TestStreamAccumulatorToolCalls:
    """Test tool call chunk accumulation."""

    @staticmethod
    def test_single_tool_call() -> None:
        acc = _StreamAccumulator()
        chunk = _make_chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="call_1",
                    function=ChoiceDeltaToolCallFunction(
                        name="search", arguments='{"q": "test"}'
                    ),
                    type="function",
                ),
            ],
        )
        events = acc.process_chunk(chunk)
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, ToolCallDelta)
        assert event.id == "call_1"
        assert event.name == "search"
        assert event.arguments_delta == '{"q": "test"}'

        msg = acc.build_message()
        tool_parts = [p for p in msg.content if isinstance(p, ToolCallPart)]
        assert len(tool_parts) == 1
        assert tool_parts[0].id == "call_1"
        assert tool_parts[0].name == "search"
        assert tool_parts[0].arguments == {"q": "test"}

    @staticmethod
    def test_tool_call_streamed_in_fragments() -> None:
        acc = _StreamAccumulator()
        acc.process_chunk(
            _make_chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        function=ChoiceDeltaToolCallFunction(
                            name="search", arguments=None
                        ),
                        type="function",
                    ),
                ],
            )
        )
        acc.process_chunk(
            _make_chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id=None,
                        function=ChoiceDeltaToolCallFunction(
                            name=None, arguments='{"q'
                        ),
                        type="function",
                    ),
                ],
            )
        )
        acc.process_chunk(
            _make_chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id=None,
                        function=ChoiceDeltaToolCallFunction(
                            name=None, arguments='": "test"}'
                        ),
                        type="function",
                    ),
                ],
            )
        )
        msg = acc.build_message()
        tool_parts = [p for p in msg.content if isinstance(p, ToolCallPart)]
        assert len(tool_parts) == 1
        assert tool_parts[0].arguments == {"q": "test"}


# ------------------------------------------------------------------ #
# Combined accumulation                                                #
# ------------------------------------------------------------------ #


class TestStreamAccumulatorCombined:
    """Test combinations of text, thinking, and tool calls."""

    @staticmethod
    def test_thinking_text_and_tool_call() -> None:
        acc = _StreamAccumulator()
        acc.process_chunk(_make_chunk(reasoning=Reasoning(content="Let me think...")))
        acc.process_chunk(_make_chunk(content="Here's what I found:"))
        acc.process_chunk(
            _make_chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="tc1",
                        function=ChoiceDeltaToolCallFunction(
                            name="lookup", arguments="{}"
                        ),
                        type="function",
                    ),
                ],
            )
        )
        msg = acc.build_message()
        thinking_parts = [p for p in msg.content if isinstance(p, ThinkingPart)]
        text_parts = [p for p in msg.content if isinstance(p, TextPart)]
        tool_parts = [p for p in msg.content if isinstance(p, ToolCallPart)]
        assert len(thinking_parts) == 1
        assert len(text_parts) == 1
        assert len(tool_parts) == 1

    @staticmethod
    def test_empty_stream_produces_empty_message() -> None:
        acc = _StreamAccumulator()
        acc.process_chunk(_make_chunk())
        msg = acc.build_message()
        assert isinstance(msg, AssistantMessage)
        assert len(msg.content) == 0
