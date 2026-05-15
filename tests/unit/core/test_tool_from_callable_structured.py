"""Structured parameter tests for Tool.from_callable()."""

from __future__ import annotations

import pytest

from atk.core.tool import (
    ArrayParameter,
    ObjectParameter,
    PrimitiveParameter,
    Tool,
    ToolParameter,
)


def _object_props(parameter: object) -> dict[str, ToolParameter]:
    assert isinstance(parameter, ObjectParameter)
    assert isinstance(parameter.properties, dict)
    return parameter.properties


def _primitive_type(parameter) -> str:
    assert isinstance(parameter, PrimitiveParameter)
    return parameter.type


class TestTypedDictParameters:
    """Test TypedDict arguments become nested object parameters."""

    @staticmethod
    def test_flat_typed_dict_fields_are_expanded(example_flat_typed_dict):
        tool = Tool.from_callable(example_flat_typed_dict)
        props = _object_props(tool.parameters["crop"])

        assert "crop" in tool.required
        assert set(props) == {"x", "y", "width", "height"}
        assert {_primitive_type(param) for param in props.values()} == {"integer"}

    @staticmethod
    def test_nested_typed_dict_fields_are_expanded(example_nested_typed_dict):
        tool = Tool.from_callable(example_nested_typed_dict)
        props = _object_props(tool.parameters["hint"])
        box_props = _object_props(props["box"])

        assert set(props) == {"label", "confidence", "box"}
        assert _primitive_type(props["label"]) == "string"
        assert _primitive_type(props["confidence"]) == "number"
        assert set(box_props) == {"x", "y", "width", "height"}
        assert {_primitive_type(param) for param in box_props.values()} == {"integer"}

    @staticmethod
    def test_nested_typed_dict_optional_filter_is_not_required(
        example_nested_typed_dict,
    ):
        tool = Tool.from_callable(example_nested_typed_dict)

        assert _primitive_type(tool.parameters["label_filter"]) == "string"
        assert "label_filter" not in tool.required


class TestNestedContainers:
    """Test nested list and dict container arguments."""

    @staticmethod
    @pytest.mark.parametrize(
        ("name", "expected_item_type", "required"),
        [
            ("index", "integer", True),
            ("label_scores", "number", False),
        ],
    )
    def test_dict_of_list_structure(
        example_nested_containers,
        name,
        expected_item_type,
        required,
    ):
        tool = Tool.from_callable(example_nested_containers)
        parameter = tool.parameters[name]

        assert isinstance(parameter, ObjectParameter)
        assert isinstance(parameter.properties, ArrayParameter)
        assert _primitive_type(parameter.properties.items) == expected_item_type
        assert (name in tool.required) is required

    @staticmethod
    def test_list_of_list_structure(example_nested_containers):
        parameter = Tool.from_callable(example_nested_containers).parameters[
            "tag_groups"
        ]

        assert isinstance(parameter, ArrayParameter)
        assert isinstance(parameter.items, ArrayParameter)
        assert _primitive_type(parameter.items.items) == "string"
