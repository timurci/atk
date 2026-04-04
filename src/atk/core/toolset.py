"""Toolset protocol and callable-based implementation."""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import TYPE_CHECKING, Protocol, TypeIs

from .tool import Tool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


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

    def __init__(self, callables: list[Callable[..., object]]) -> None:
        """Initialize a CallableToolset from a list of callables.

        Args:
            callables: Functions to register as tools in this toolset.
        """
        self._callables: dict[str, Callable[..., object]] = {}
        self._tools: list[Tool] = []
        for fn in callables:
            tool = Tool.from_callable(fn)
            self._tools.append(tool)
            self._callables[tool.name] = fn

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
