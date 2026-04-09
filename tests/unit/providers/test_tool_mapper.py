"""Tests for ToolMapper — mapping internal Tool definitions to any-llm format."""

from __future__ import annotations

import pytest

from atk.core.tool import (
    ArrayParameter,
    EnumParameter,
    ObjectParameter,
    PrimitiveParameter,
    Tool,
)
from atk.providers.tool import ToolMapper, _map_parameter

# ------------------------------------------------------------------ #
# _map_parameter — PrimitiveParameter                                  #
# ------------------------------------------------------------------ #


class TestMapPrimitiveParameter:
    """Test _map_parameter for each PrimitiveParameter variant."""

    @pytest.mark.parametrize(
        ("type_val", "expected_type"),
        [
            ("string", "string"),
            ("integer", "integer"),
            ("number", "number"),
            ("boolean", "boolean"),
        ],
    )
    def test_primitive_mapping(self, type_val, expected_type):
        param = PrimitiveParameter(type=type_val, description="test")
        result = _map_parameter(param)
        assert result == {"type": expected_type, "description": "test"}


# ------------------------------------------------------------------ #
# _map_parameter — EnumParameter                                       #
# ------------------------------------------------------------------ #


class TestMapEnumParameter:
    """Test _map_parameter for EnumParameter."""

    @staticmethod
    def test_enum_maps_to_string_with_enum_list() -> None:
        param = EnumParameter(
            type="enum", description="Alignment mode", enum={"left", "center", "right"}
        )
        result = _map_parameter(param)
        assert result["type"] == "string"
        assert sorted(result["enum"]) == ["center", "left", "right"]
        assert result["description"] == "Alignment mode"

    @staticmethod
    def test_enum_single_value() -> None:
        param = EnumParameter(type="enum", description="Single option", enum={"only"})
        result = _map_parameter(param)
        assert result["type"] == "string"
        assert result["enum"] == ["only"]


# ------------------------------------------------------------------ #
# _map_parameter — ArrayParameter                                      #
# ------------------------------------------------------------------ #


class TestMapArrayParameter:
    """Test _map_parameter for ArrayParameter."""

    @staticmethod
    def test_bare_array_no_items() -> None:
        param = ArrayParameter(type="array", description="A list of items", items=None)
        result = _map_parameter(param)
        assert result == {"type": "array", "description": "A list of items"}
        assert "items" not in result

    @staticmethod
    def test_typed_array_with_primitive_items() -> None:
        param = ArrayParameter(
            type="array",
            description="A list of strings",
            items=PrimitiveParameter(type="string", description="A string"),
        )
        result = _map_parameter(param)
        assert result["type"] == "array"
        assert result["description"] == "A list of strings"
        assert result["items"] == {"type": "string", "description": "A string"}

    @staticmethod
    def test_nested_array() -> None:
        param = ArrayParameter(
            type="array",
            description="A list of lists",
            items=ArrayParameter(
                type="array",
                description="Inner list",
                items=PrimitiveParameter(type="string", description=""),
            ),
        )
        result = _map_parameter(param)
        assert result["type"] == "array"
        assert result["items"]["type"] == "array"
        assert result["items"]["items"]["type"] == "string"


# ------------------------------------------------------------------ #
# _map_parameter — ObjectParameter                                     #
# ------------------------------------------------------------------ #


class TestMapObjectParameter:
    """Test _map_parameter for ObjectParameter."""

    @staticmethod
    def test_bare_object_no_properties() -> None:
        param = ObjectParameter(
            type="object", description="Free-form data", properties=None
        )
        result = _map_parameter(param)
        assert result == {"type": "object", "description": "Free-form data"}
        assert "properties" not in result
        assert "additionalProperties" not in result

    @staticmethod
    def test_object_with_typed_values() -> None:
        param = ObjectParameter(
            type="object",
            description="String-to-int mapping",
            properties=PrimitiveParameter(type="integer", description="Count"),
        )
        result = _map_parameter(param)
        assert result["type"] == "object"
        assert result["description"] == "String-to-int mapping"
        assert result["additionalProperties"] == {
            "type": "integer",
            "description": "Count",
        }
        assert "properties" not in result

    @staticmethod
    def test_object_with_structured_properties() -> None:
        param = ObjectParameter(
            type="object",
            description="A structured object",
            properties={
                "name": PrimitiveParameter(type="string", description="The name"),
                "age": PrimitiveParameter(type="integer", description="The age"),
            },
        )
        result = _map_parameter(param)
        assert result["type"] == "object"
        assert result["description"] == "A structured object"
        assert "name" in result["properties"]
        assert result["properties"]["name"]["type"] == "string"
        assert "age" in result["properties"]
        assert result["properties"]["age"]["type"] == "integer"
        assert "additionalProperties" not in result


# ------------------------------------------------------------------ #
# ToolMapper.to_tools — integration mapping                           #
# ------------------------------------------------------------------ #


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
        tools = [_simple_tool()]
        result = ToolMapper.to_tools(tools)
        assert len(result) == 1
        entry = result[0]
        assert entry["type"] == "function"
        assert "function" in entry
        fn = entry["function"]
        assert fn["name"] == "test_fn"
        assert fn["description"] == "A test function"
        assert fn["strict"] is True

    @staticmethod
    def test_parameters_shape() -> None:
        tools = [_simple_tool()]
        result = ToolMapper.to_tools(tools)
        params = result[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert "limit" in params["properties"]
        assert params["properties"]["limit"]["type"] == "integer"

    @staticmethod
    def test_required_included_when_present() -> None:
        tools = [_simple_tool()]
        result = ToolMapper.to_tools(tools)
        params = result[0]["function"]["parameters"]
        assert "required" in params
        assert params["required"] == ["query"]

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
    def test_enum_tool_schema() -> None:
        result = ToolMapper.to_tools([_enum_tool()])
        props = result[0]["function"]["parameters"]["properties"]
        assert props["align"]["type"] == "string"
        assert sorted(props["align"]["enum"]) == ["center", "left", "right"]

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
