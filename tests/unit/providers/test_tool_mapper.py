"""Tests for ToolMapper — mapping internal Tool definitions to any-llm format."""

from __future__ import annotations

from atk.core.tool import (
    ArrayParameter,
    EnumParameter,
    ObjectParameter,
    PrimitiveParameter,
    Tool,
)
from atk.providers.tool import ToolMapper


def _simple_tool(name: str = "test_fn", description: str = "A test function") -> Tool:
    """Build a minimal Tool with one required and one optional parameter."""
    return Tool(
        name=name,
        description=description,
        parameters={
            "query": PrimitiveParameter(type="string", description="Search query"),
            "limit": PrimitiveParameter(type="integer", description="Max results"),
        },
        required=["query"],
    )


def _enum_tool() -> Tool:
    """Build a Tool with an EnumParameter."""
    return Tool(
        name="choose_align",
        description="Choose alignment",
        parameters={
            "align": EnumParameter(
                type="enum", description="Alignment", enum={"left", "center", "right"}
            ),
        },
        required=["align"],
    )


def _nested_tool() -> Tool:
    """Build a Tool with nested ArrayParameter inside ObjectParameter."""
    return Tool(
        name="search_index",
        description="Search an inverted index",
        parameters={
            "index": ObjectParameter(
                type="object",
                description="Token-to-offsets mapping",
                properties=ArrayParameter(
                    type="array",
                    description="Offset list",
                    items=PrimitiveParameter(type="integer", description="Byte offset"),
                ),
            ),
        },
        required=["index"],
    )


class TestToTools:
    """Test ToolMapper.to_tools produces correct any-llm tool schemas."""

    @staticmethod
    def test_top_level_shape() -> None:
        result = ToolMapper.to_tools([_simple_tool()])
        assert result == [
            {
                "type": "function",
                "function": {
                    "name": "test_fn",
                    "description": "A test function",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results",
                            },
                        },
                        "required": ["query"],
                    },
                    "strict": True,
                },
            }
        ]

    @staticmethod
    def test_required_omitted_when_empty() -> None:
        tool = Tool(
            name="no_req",
            description="No required params",
            parameters={
                "q": PrimitiveParameter(type="string", description="Query"),
            },
            required=[],
        )
        result = ToolMapper.to_tools([tool])
        params = result[0]["function"]["parameters"]
        assert "required" not in params

    @staticmethod
    def test_enum_tool_schema_is_deterministic() -> None:
        result = ToolMapper.to_tools([_enum_tool()])
        props = result[0]["function"]["parameters"]["properties"]
        assert props["align"]["enum"] == ["center", "left", "right"]

    @staticmethod
    def test_nested_tool_schema() -> None:
        result = ToolMapper.to_tools([_nested_tool()])
        props = result[0]["function"]["parameters"]["properties"]
        assert props["index"]["type"] == "object"
        assert props["index"]["additionalProperties"]["type"] == "array"
        assert props["index"]["additionalProperties"]["items"]["type"] == "integer"

    @staticmethod
    def test_multiple_tools() -> None:
        tools = [
            _simple_tool("fn_a", "First function"),
            _simple_tool("fn_b", "Second function"),
        ]
        result = ToolMapper.to_tools(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "fn_a"
        assert result[1]["function"]["name"] == "fn_b"

    @staticmethod
    def test_empty_tools_list() -> None:
        result = ToolMapper.to_tools([])
        assert result == []
