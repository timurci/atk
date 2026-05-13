"""Accumulate streaming assistant deltas into a final message."""

import json
from typing import Any

from .message import (
    AssistantMessage,
    AssistantStream,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
)


class AssistantStreamAccumulator:
    """Accumulates normalized assistant stream deltas."""

    def __init__(self) -> None:
        """Initialize an empty stream accumulator."""
        self._text_fragments: list[str] = []
        self._thinking_fragments: list[str] = []
        self._tool_call_order: list[str] = []
        self._tool_call_names: dict[str, str] = {}
        self._tool_call_arguments: dict[str, list[str]] = {}

    def add_delta(self, delta: TextDelta | ThinkingDelta | ToolCallDelta) -> None:
        """Add one normalized stream delta."""
        if isinstance(delta, TextDelta):
            self._text_fragments.append(delta.text)
            return

        if isinstance(delta, ThinkingDelta):
            self._thinking_fragments.append(delta.thinking)
            return

        if delta.id not in self._tool_call_arguments:
            self._tool_call_order.append(delta.id)
            self._tool_call_arguments[delta.id] = []
        self._tool_call_names[delta.id] = delta.name
        self._tool_call_arguments[delta.id].append(delta.arguments_delta)

    def add_stream(self, stream: AssistantStream) -> None:
        """Add all deltas from one assistant stream chunk."""
        for delta in stream.content:
            self.add_delta(delta)

    def build_message(self) -> AssistantMessage:
        """Build the final accumulated assistant message."""
        content: list[TextPart | ToolCallPart | ThinkingPart] = []

        if self._thinking_fragments:
            content.append(ThinkingPart(thinking="".join(self._thinking_fragments)))

        if self._text_fragments:
            content.append(TextPart(text="".join(self._text_fragments)))

        content.extend(
            (
                ToolCallPart(
                    id=tool_call_id,
                    name=self._tool_call_names[tool_call_id],
                    arguments=self._parse_arguments(
                        "".join(self._tool_call_arguments[tool_call_id]),
                    ),
                )
            )
            for tool_call_id in self._tool_call_order
        )

        return AssistantMessage(content=content)

    @staticmethod
    def _parse_arguments(raw_arguments: str) -> dict[str, Any]:
        stripped_arguments = raw_arguments.strip()
        if not stripped_arguments:
            return {}

        try:
            parsed_arguments = json.loads(stripped_arguments)
        except json.JSONDecodeError:
            return {}

        if not isinstance(parsed_arguments, dict):
            return {}

        return parsed_arguments
