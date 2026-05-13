"""Unit tests for core tool JSON Schema conversion."""

from __future__ import annotations

import pytest

from atk.core.tool import (
    ArrayParameter,
    EnumParameter,
    ObjectParameter,
    PrimitiveParameter,
    Tool,
)


class TestToolParameterToJsonSchema:
    """Test ToolParameter schema conversion."""

    @pytest.mark.parametrize(
        ("type_val", "expected_type"),
        [
            ("string", "string"),
            ("integer", "integer"),
            ("number", "number"),
            ("boolean", "boolean"),
        ],
    )
    def test_primitive_parameter_schema(self, type_val, expected_type):
        param = PrimitiveParameter(type=type_val, description="test")
        result = param.to_json_schema()
        assert result == {"type": expected_type, "description": "test"}

    @staticmethod
    def test_enum_parameter_schema_is_deterministic() -> None:
        param = EnumParameter(
            type="enum", description="Alignment mode", enum={"left", "center", "right"}
        )
        result = param.to_json_schema()
        assert result == {
            "type": "string",
            "description": "Alignment mode",
            "enum": ["center", "left", "right"],
        }

    @staticmethod
    def test_bare_array_parameter_schema() -> None:
        param = ArrayParameter(type="array", description="A list of items", items=None)
        result = param.to_json_schema()
        assert result == {"type": "array", "description": "A list of items"}

    @staticmethod
    def test_typed_array_parameter_schema() -> None:
        param = ArrayParameter(
            type="array",
            description="A list of strings",
            items=PrimitiveParameter(type="string", description="A string"),
        )
        result = param.to_json_schema()
        assert result == {
            "type": "array",
            "description": "A list of strings",
            "items": {"type": "string", "description": "A string"},
        }

    @staticmethod
    def test_bare_object_parameter_schema() -> None:
        param = ObjectParameter(
            type="object", description="Free-form data", properties=None
        )
        result = param.to_json_schema()
        assert result == {"type": "object", "description": "Free-form data"}

    @staticmethod
    def test_homogeneous_dictionary_parameter_schema() -> None:
        param = ObjectParameter(
            type="object",
            description="String-to-int mapping",
            properties=PrimitiveParameter(type="integer", description="Count"),
        )
        result = param.to_json_schema()
        assert result == {
            "type": "object",
            "description": "String-to-int mapping",
            "additionalProperties": {"type": "integer", "description": "Count"},
        }

    @staticmethod
    def test_structured_object_parameter_schema() -> None:
        param = ObjectParameter(
            type="object",
            description="A structured object",
            properties={
                "name": PrimitiveParameter(type="string", description="The name"),
                "age": PrimitiveParameter(type="integer", description="The age"),
            },
        )
        result = param.to_json_schema()
        assert result == {
            "type": "object",
            "description": "A structured object",
            "properties": {
                "name": {"type": "string", "description": "The name"},
                "age": {"type": "integer", "description": "The age"},
            },
        }


class TestToolToJsonSchema:
    """Test complete Tool schema conversion."""

    @staticmethod
    def test_top_level_required_parameter_schema() -> None:
        tool = Tool(
            name="search",
            description="Search indexed documents.",
            parameters={
                "query": PrimitiveParameter(type="string", description="Search query."),
                "limit": PrimitiveParameter(type="integer", description="Max results."),
            },
            required=["query"],
        )
        result = tool.to_json_schema()
        assert result == {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "limit": {"type": "integer", "description": "Max results."},
            },
            "required": ["query"],
        }

    @staticmethod
    def test_top_level_required_omitted_when_empty() -> None:
        tool = Tool(
            name="search",
            description="Search indexed documents.",
            parameters={
                "query": PrimitiveParameter(type="string", description="Search query."),
            },
            required=[],
        )
        result = tool.to_json_schema()
        assert result == {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
            },
        }

    @staticmethod
    def test_tool_json_schema_describes_llm_function_parameters() -> None:
        tool = Tool(
            name="search",
            description="Search indexed documents.",
            parameters={
                "query": PrimitiveParameter(type="string", description="Search query."),
            },
            required=["query"],
        )

        result = tool.to_json_schema()
        properties = result["properties"]

        assert isinstance(properties, dict)
        assert properties == {
            "query": {"type": "string", "description": "Search query."},
        }
        assert "name" not in properties

    @staticmethod
    def test_model_json_schema_still_describes_the_pydantic_model() -> None:
        model_schema = Tool.model_json_schema()

        assert "name" in model_schema["properties"]
        assert "parameters" in model_schema["properties"]
