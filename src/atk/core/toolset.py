"""Toolset protocol and callable-based implementation."""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeIs, cast

from .message import AssistantMessage, ToolCallPart, ToolMessage, ToolResultPart
from .tool import Tool

CallableToolEntry = Callable[..., object] | tuple[str, Callable[..., object]]


class Toolset(Protocol):
    """Protocol for a collection of tools that can be invoked."""

    @property
    def tools(self) -> list[Tool]:
        """Return the list of tool definitions in this toolset."""
        ...

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, object]) -> str:
        """Invoke a tool by name with the given keyword arguments.

        Args:
            tool_name: The name of the tool to invoke.
            kwargs: Keyword arguments to pass to the tool.

        Raises:
            KeyError: If no tool with the given name exists in the toolset.
        """
        ...


async def invoke_tool_calls(
    message: AssistantMessage,
    toolset: Toolset,
) -> ToolMessage:
    """Invoke tool calls from an ``AssistantMessage`` with a ``Toolset``.

    Processes each ``ToolCallPart`` in ``message.content`` and ignores non-tool
    parts. For every tool call, this calls ``Toolset.invoke_tool`` with the
    tool call name and arguments, then wraps the returned string in a
    ``ToolResultPart`` using the original tool call ID.

    Args:
        message: The ``AssistantMessage`` whose ``ToolCallPart`` items should be
            invoked.
        toolset: The ``Toolset`` used to invoke each requested tool.

    Returns:
        A ``ToolMessage`` containing one ``ToolResultPart`` for each processed
        ``ToolCallPart``, in message order. If the assistant message contains no
        tool calls, the returned ``ToolMessage`` has empty content.

    Raises:
        KeyError: Propagated from ``Toolset.invoke_tool`` when a tool name is not
            available in the toolset.
    """
    tool_results: list[ToolResultPart] = []
    for part in message.content:
        if not isinstance(part, ToolCallPart):
            continue
        result = await toolset.invoke_tool(part.name, part.arguments)
        tool_results.append(ToolResultPart(tool_call_id=part.id, content=result))
    return ToolMessage(content=tool_results)


def _is_async_callable(
    fn: Callable[..., object],
) -> TypeIs[Callable[..., Awaitable[object]]]:
    """Check whether a callable is async, handling edge cases."""
    if inspect.iscoroutinefunction(fn):
        return True
    if isinstance(fn, functools.partial):
        return inspect.iscoroutinefunction(fn.func)
    return callable(fn) and inspect.iscoroutinefunction(fn.__call__)


async def _call_maybe_async(
    fn: Callable[..., object], kwargs: dict[str, object]
) -> object:
    """Call a callable, awaiting the result if it is async."""
    if _is_async_callable(fn):
        return await fn(**kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(**kwargs))


class CallableToolset:
    """A Toolset built from a list of callable functions."""

    def __init__(self, callables: list[CallableToolEntry]) -> None:
        """Initialize a CallableToolset from a list of callables.

        Args:
            callables: Functions or ``(tool_name, function)`` entries to register.
        """
        self._callables: dict[str, Callable[..., object]] = {}
        self._tools: list[Tool] = []
        for entry in callables:
            name, fn = self._normalize_entry(entry)
            tool = Tool.from_callable(fn, name=name)
            if tool.name in self._callables:
                msg = f"Duplicate tool name: {tool.name!r}"
                raise KeyError(msg)
            self._tools.append(tool)
            self._callables[tool.name] = fn

    @staticmethod
    def _normalize_entry(
        entry: CallableToolEntry,
    ) -> tuple[str | None, Callable[..., object]]:
        """Return the optional name override and callable for a constructor entry."""
        if isinstance(entry, tuple):
            return cast("tuple[str, Callable[..., object]]", entry)
        return None, entry

    @property
    def tools(self) -> list[Tool]:
        """Return the list of tool definitions."""
        return self._tools

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, object]) -> str:
        """Invoke a tool by name with the given keyword arguments.

        Args:
            tool_name: The name of the tool to invoke.
            kwargs: Keyword arguments to pass to the tool.

        Returns:
            The tool's return value converted to a string.

        Raises:
            KeyError: If no tool with the given name exists in the toolset.
        """
        if tool_name not in self._callables:
            msg = f"Tool {tool_name!r} not found in toolset"
            raise KeyError(msg)
        fn = self._callables[tool_name]
        try:
            result = await _call_maybe_async(fn, kwargs)
        except Exception as e:  # noqa: BLE001
            return f"Error executing {tool_name!r}: {e}"
        if isinstance(result, str):
            return result
        return str(result)
