"""Unit tests for Toolset protocol and CallableToolset."""

from __future__ import annotations

import functools
from typing import Literal

import anyio
import pytest

from atk.core.tool import Tool
from atk.core.toolset import CallableToolset

# ------------------------------------------------------------------ #
# Test callables
# ------------------------------------------------------------------ #


def _sync_greet(name: str) -> str:
    """Greet a person by name.

    Args:
        name: The person's name.
    """
    return f"Hello, {name}!"


def _sync_add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer.
        b: Second integer.
    """
    return a + b


async def _async_fetch(url: str) -> str:
    """Fetch content from a URL.

    Args:
        url: The URL to fetch.
    """
    return f"fetched:{url}"


class _AsyncCallable:
    """Callable class with async __call__."""

    async def __call__(self, query: str) -> str:
        return f"searched:{query}"


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def sync_toolset():
    """CallableToolset with only sync callables."""
    return CallableToolset([_sync_greet, _sync_add])


@pytest.fixture
def mixed_toolset():
    """CallableToolset with sync and async callables."""
    return CallableToolset([_sync_greet, _async_fetch])


@pytest.fixture
def empty_toolset():
    """CallableToolset with no callables."""
    return CallableToolset([])


# ------------------------------------------------------------------ #
# TestCallableToolsetTools — tools property
# ------------------------------------------------------------------ #


class TestCallableToolsetTools:
    """Test the tools property returns correct Tool objects."""

    def test_tools_returns_list_of_tool(self, sync_toolset):
        tools = sync_toolset.tools
        assert isinstance(tools, list)
        assert len(tools) == 2
        for t in tools:
            assert isinstance(t, Tool)

    def test_tool_names_match_function_names(self, sync_toolset):
        names = [t.name for t in sync_toolset.tools]
        assert set(names) == {"_sync_greet", "_sync_add"}

    def test_alias_appears_in_tools(self):
        toolset = CallableToolset([("greet", _sync_greet)])
        assert [t.name for t in toolset.tools] == ["greet"]

    def test_alias_overrides_callable_name(self):
        toolset = CallableToolset([("greet", _sync_greet)])
        assert toolset.tools[0].name != "_sync_greet"

    def test_tool_order_follows_registration_order(self):
        toolset = CallableToolset(
            [
                ("greet", _sync_greet),
                _sync_add,
                ("fetch", _async_fetch),
            ]
        )
        assert [t.name for t in toolset.tools] == ["greet", "_sync_add", "fetch"]

    def test_tool_descriptions_from_docstrings(self, sync_toolset):
        greet_tool = next(t for t in sync_toolset.tools if t.name == "_sync_greet")
        assert "Greet a person" in greet_tool.description

    def test_tool_parameters_populated(self, sync_toolset):
        greet_tool = next(t for t in sync_toolset.tools if t.name == "_sync_greet")
        assert "name" in greet_tool.parameters
        assert greet_tool.required == ["name"]

    def test_empty_toolset_has_no_tools(self, empty_toolset):
        assert empty_toolset.tools == []


# ------------------------------------------------------------------ #
# TestInvokeToolSync — invoke sync callables
# ------------------------------------------------------------------ #


class TestInvokeToolSync:
    """Test invoking sync callables."""

    def test_invoke_sync_string_return(self, sync_toolset):
        async def _run() -> str:
            return await sync_toolset.invoke_tool("_sync_greet", {"name": "Alice"})

        result = anyio.run(_run)
        assert result == "Hello, Alice!"

    def test_invoke_sync_int_return_converted_to_str(self, sync_toolset):
        async def _run() -> str:
            return await sync_toolset.invoke_tool("_sync_add", {"a": 3, "b": 4})

        result = anyio.run(_run)
        assert result == "7"

    def test_invoke_aliased_tool(self):
        toolset = CallableToolset([("greet", _sync_greet)])

        async def _run() -> str:
            return await toolset.invoke_tool("greet", {"name": "Alice"})

        result = anyio.run(_run)
        assert result == "Hello, Alice!"

    def test_invoke_with_default_params(self):
        def _with_default(mode: Literal["fast", "slow"] = "fast") -> str:
            return f"mode={mode}"

        toolset = CallableToolset([_with_default])

        async def _run_empty() -> str:
            return await toolset.invoke_tool("_with_default", {})

        result = anyio.run(_run_empty)
        assert result == "mode=fast"

        async def _run_slow() -> str:
            return await toolset.invoke_tool("_with_default", {"mode": "slow"})

        result = anyio.run(_run_slow)
        assert result == "mode=slow"


# ------------------------------------------------------------------ #
# TestInvokeToolAsync — invoke async callables
# ------------------------------------------------------------------ #


class TestInvokeToolAsync:
    """Test invoking async callables."""

    def test_invoke_async_string_return(self, mixed_toolset):
        async def _run() -> str:
            return await mixed_toolset.invoke_tool("_async_fetch", {"url": "http://x"})

        result = anyio.run(_run)
        assert result == "fetched:http://x"

    def test_invoke_async_callable_object(self):
        toolset = CallableToolset([_AsyncCallable()])
        tool = toolset.tools[0]

        async def _run() -> str:
            return await toolset.invoke_tool(tool.name, {"query": "hello"})

        result = anyio.run(_run)
        assert result == "searched:hello"

    def test_invoke_async_partial(self):
        async def _greet(prefix: str, name: str) -> str:
            return f"{prefix}{name}"

        partial_fn = functools.partial(_greet, "Hi, ")
        toolset = CallableToolset([partial_fn])
        tool = toolset.tools[0]

        async def _run() -> str:
            return await toolset.invoke_tool(tool.name, {"name": "Bob"})

        result = anyio.run(_run)
        assert result == "Hi, Bob"

    def test_multiple_partials_have_distinct_names(self):
        async def _greet(prefix: str, name: str) -> str:
            return f"{prefix}{name}"

        async def _farewell(prefix: str, name: str) -> str:
            return f"{prefix}{name}"

        partial_greet = functools.partial(_greet, "Hello, ")
        partial_farewell = functools.partial(_farewell, "Goodbye, ")
        toolset = CallableToolset([partial_greet, partial_farewell])

        names = [t.name for t in toolset.tools]
        assert names == ["_greet", "_farewell"]
        assert len(names) == len(set(names))

        async def _run() -> tuple[str, str]:
            r1 = await toolset.invoke_tool("_greet", {"name": "Alice"})
            r2 = await toolset.invoke_tool("_farewell", {"name": "Bob"})
            return r1, r2

        result = anyio.run(_run)
        assert result == ("Hello, Alice", "Goodbye, Bob")


# ------------------------------------------------------------------ #
# TestInvokeToolErrors — error handling
# ------------------------------------------------------------------ #


class TestInvokeToolErrors:
    """Test error handling in invoke_tool."""

    @pytest.mark.parametrize(
        "callables",
        [
            [_sync_greet, _sync_greet],
            [("lookup", _sync_greet), ("lookup", _sync_add)],
            [_sync_greet, ("_sync_greet", _sync_add)],
        ],
    )
    def test_duplicate_tool_names_raise_keyerror(self, callables):
        with pytest.raises(KeyError, match="Duplicate tool name"):
            CallableToolset(callables)

    def test_unknown_tool_raises_keyerror(self, sync_toolset):
        async def _run() -> str:
            return await sync_toolset.invoke_tool("nonexistent", {})

        with pytest.raises(KeyError, match="nonexistent"):
            anyio.run(_run)

    def test_callable_exception_returns_error_str(self):
        def _failing_fn() -> str:
            msg = "boom"
            raise ValueError(msg)

        toolset = CallableToolset([_failing_fn])
        tool = toolset.tools[0]

        async def _run() -> str:
            return await toolset.invoke_tool(tool.name, {})

        result = anyio.run(_run)
        assert "Error executing" in result
        assert "boom" in result

    def test_async_callable_exception_returns_error_str(self):
        async def _async_failing() -> str:
            msg = "async boom"
            raise RuntimeError(msg)

        toolset = CallableToolset([_async_failing])
        tool = toolset.tools[0]

        async def _run() -> str:
            return await toolset.invoke_tool(tool.name, {})

        result = anyio.run(_run)
        assert "Error executing" in result
        assert "async boom" in result


# ------------------------------------------------------------------ #
# TestInvokeToolReturnTypes — non-string return value conversion
# ------------------------------------------------------------------ #


class TestInvokeToolReturnTypes:
    """Test that non-string return values are converted to str."""

    @pytest.mark.parametrize(
        ("return_val", "expected_str"),
        [(42, "42"), (None, "None")],
    )
    def test_non_str_return_converted(self, return_val, expected_str):
        def _fn() -> int | None:
            return return_val

        toolset = CallableToolset([_fn])
        tool = toolset.tools[0]

        async def _run() -> str:
            return await toolset.invoke_tool(tool.name, {})

        assert anyio.run(_run) == expected_str
