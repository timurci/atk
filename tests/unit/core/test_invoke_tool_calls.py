"""Unit tests for invoking assistant tool calls."""

from __future__ import annotations

import anyio
import pytest

from atk.core import (
    AssistantMessage,
    CallableToolset,
    TextPart,
    ToolCallPart,
    ToolMessage,
    invoke_tool_calls,
)


def _tool_call(call_id: str, tool_name: str, **arguments: object) -> ToolCallPart:
    return ToolCallPart(id=call_id, name=tool_name, arguments=arguments)


def test_no_tool_calls_returns_empty_tool_message() -> None:
    """Return an empty ToolMessage when no tool calls are present."""
    message = AssistantMessage(content=[TextPart(text="No tool needed.")])
    toolset = CallableToolset([])

    async def _run() -> ToolMessage:
        return await invoke_tool_calls(message, toolset)

    assert anyio.run(_run) == ToolMessage(content=[])


def test_single_tool_call_invokes_matching_tool() -> None:
    """Invoke a single matching tool call."""

    def greet(name: str) -> str:
        """Greet a person.

        Args:
            name: The person's name.
        """
        return f"Hello, {name}!"

    message = AssistantMessage(content=[_tool_call("call-1", "greet", name="Alice")])
    toolset = CallableToolset([greet])

    async def _run() -> ToolMessage:
        return await invoke_tool_calls(message, toolset)

    result = anyio.run(_run)
    assert result.content[0].content == "Hello, Alice!"


def test_multiple_tool_calls_preserve_assistant_content_order() -> None:
    """Invoke multiple tool calls in assistant content order."""

    class RecordingToolset:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        @property
        def tools(self) -> list:
            return []

        async def invoke_tool(self, tool_name: str, kwargs: dict[str, object]) -> str:
            self.calls.append((tool_name, kwargs))
            return tool_name

    toolset = RecordingToolset()
    message = AssistantMessage(
        content=[
            _tool_call("call-1", "first", value=1),
            TextPart(text="Between calls."),
            _tool_call("call-2", "second", value=2),
        ]
    )

    async def _run() -> ToolMessage:
        return await invoke_tool_calls(message, toolset)

    result = anyio.run(_run)
    assert toolset.calls == [
        ("first", {"value": 1}),
        ("second", {"value": 2}),
    ]
    assert [part.content for part in result.content] == ["first", "second"]


def test_tool_result_id_matches_assistant_tool_call_id() -> None:
    """Copy the assistant tool call id into the tool result."""

    def lookup(query: str) -> str:
        """Look up a query.

        Args:
            query: The query to look up.
        """
        return query

    message = AssistantMessage(
        content=[_tool_call("assistant-call-id", "lookup", query="status")]
    )
    toolset = CallableToolset([lookup])

    async def _run() -> ToolMessage:
        return await invoke_tool_calls(message, toolset)

    result = anyio.run(_run)
    assert result.content[0].tool_call_id == "assistant-call-id"


def test_unknown_tool_behavior_matches_toolset_invoke_tool() -> None:
    """Delegate unknown tool behavior to the toolset."""
    message = AssistantMessage(content=[_tool_call("call-1", "missing")])
    toolset = CallableToolset([])

    async def _run() -> ToolMessage:
        return await invoke_tool_calls(message, toolset)

    with pytest.raises(KeyError, match="missing"):
        anyio.run(_run)


def test_callable_toolset_execution_error_string_is_preserved() -> None:
    """Preserve CallableToolset execution error strings."""

    def failing() -> str:
        """Fail while running."""
        msg = "boom"
        raise RuntimeError(msg)

    message = AssistantMessage(content=[_tool_call("call-1", "failing")])
    toolset = CallableToolset([failing])

    async def _run() -> ToolMessage:
        return await invoke_tool_calls(message, toolset)

    result = anyio.run(_run)
    assert result.content[0].content == "Error executing 'failing': boom"
