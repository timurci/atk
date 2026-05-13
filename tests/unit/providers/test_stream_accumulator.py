"""Tests for provider stream chunk mapping."""

from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    Reasoning,
)

from atk.core import TextDelta, ThinkingDelta, ToolCallDelta
from atk.providers.model import _ChunkDeltaMapper


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


class TestChunkDeltaMapper:
    """Test provider-specific chunk-to-delta mapping."""

    @staticmethod
    def test_content_chunk_maps_to_text_delta() -> None:
        mapper = _ChunkDeltaMapper()

        events = mapper.process_chunk(_make_chunk(content="Hello"))

        assert events == [TextDelta(text="Hello")]

    @staticmethod
    def test_reasoning_chunk_maps_to_thinking_delta() -> None:
        mapper = _ChunkDeltaMapper()

        events = mapper.process_chunk(
            _make_chunk(reasoning=Reasoning(content="I need to think..."))
        )

        assert events == [ThinkingDelta(thinking="I need to think...")]

    @staticmethod
    def test_tool_call_chunk_maps_to_tool_call_delta() -> None:
        mapper = _ChunkDeltaMapper()

        events = mapper.process_chunk(
            _make_chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        function=ChoiceDeltaToolCallFunction(
                            name="search",
                            arguments='{"q": "test"}',
                        ),
                        type="function",
                    ),
                ],
            )
        )

        assert events == [
            ToolCallDelta(
                id="call_1",
                name="search",
                arguments_delta='{"q": "test"}',
            )
        ]

    @staticmethod
    def test_tool_call_fragments_reuse_cached_id_and_name() -> None:
        mapper = _ChunkDeltaMapper()

        first_events = mapper.process_chunk(
            _make_chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        function=ChoiceDeltaToolCallFunction(
                            name="search",
                            arguments=None,
                        ),
                        type="function",
                    ),
                ],
            )
        )
        second_events = mapper.process_chunk(
            _make_chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id=None,
                        function=ChoiceDeltaToolCallFunction(
                            name=None,
                            arguments='{"q',
                        ),
                        type="function",
                    ),
                ],
            )
        )

        assert first_events == [
            ToolCallDelta(id="call_1", name="search", arguments_delta="")
        ]
        assert second_events == [
            ToolCallDelta(id="call_1", name="search", arguments_delta='{"q')
        ]

    @staticmethod
    def test_tool_call_fragment_without_cached_id_or_name_is_not_emitted() -> None:
        mapper = _ChunkDeltaMapper()

        events = mapper.process_chunk(
            _make_chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id=None,
                        function=ChoiceDeltaToolCallFunction(
                            name=None,
                            arguments='{"q',
                        ),
                        type="function",
                    ),
                ],
            )
        )

        assert events == []
