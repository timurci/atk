"""Core specification for language models."""

from .message import (
    AssistantMessage,
    AssistantStream,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
    ToolMessage,
    ToolResultPart,
    UserMessage,
)
from .model import LanguageModel, StreamingLanguageModel
from .stream_accumulator import AssistantStreamAccumulator
from .tool import Tool
from .toolset import (
    CallableToolset,
    Toolset,
    invoke_tool_calls,
)

__all__ = [
    "AssistantMessage",
    "AssistantStream",
    "AssistantStreamAccumulator",
    "CallableToolset",
    "LanguageModel",
    "StreamingLanguageModel",
    "TextDelta",
    "TextPart",
    "ThinkingDelta",
    "ThinkingPart",
    "Tool",
    "ToolCallDelta",
    "ToolCallPart",
    "ToolMessage",
    "ToolResultPart",
    "Toolset",
    "UserMessage",
    "invoke_tool_calls",
]
