"""Unit tests for Tool.from_callable()."""

# Fixtures are provided by tests/unit/core/conftest.py and discovered
# automatically by pytest — no imports needed for fixture functions.
# The raw ``_example_*`` callables are imported directly only where
# @pytest.mark.parametrize requires module-level names.
#
# Test organisation
# -----------------
#   TestToolMetadata            name and description extraction
#   TestRequiredParameters      required list correctness
#   TestPrimitiveParameters     str / int / float / bool → PrimitiveParameter
#   TestEnumParameters          Literal and Enum → EnumParameter
#   TestArrayParameters         list variants → ArrayParameter
#   TestObjectParameters        dict variants → ObjectParameter
#   TestTypedDictParameters     TypedDict (flat + nested) → ObjectParameter
#   TestOptionalParameters      Optional[T] + None default handling
#   TestNestedContainers        list[list[T]], dict[str, list[T]]
#   TestDegradedDocstrings      missing / partial docstrings — no crash

from __future__ import annotations

import functools
from typing import TypeIs

import pytest

from atk.core.tool import (
    ArrayParameter,
    EnumParameter,
    ObjectParameter,
    PrimitiveParameter,
    Tool,
    ToolParameter,
)

# Raw callables imported for @pytest.mark.parametrize (fixtures cannot be
# used directly inside parametrize decorators).
from tests.unit.core.conftest import (
    _example_description_only,
    _example_no_docstring,
    _example_partial_args_doc,
)

# ======================================================================
# Helpers
# ======================================================================


def _param(tool: Tool, name: str) -> ToolParameter:
    """Return the ToolParameter for ``name``, with a clear error if absent."""
    assert name in tool.parameters, (
        f"Parameter {name!r} not found in tool.parameters. "
        f"Present keys: {list(tool.parameters)}"
    )
    return tool.parameters[name]


def _assert_enum(p: ToolParameter) -> TypeIs[EnumParameter]:
    return isinstance(p, EnumParameter)


def _assert_array(p: ToolParameter) -> TypeIs[ArrayParameter]:
    return isinstance(p, ArrayParameter)


def _assert_object(p: ToolParameter) -> TypeIs[ObjectParameter]:
    return isinstance(p, ObjectParameter)


def _assert_object_dict_props(
    p: ToolParameter,
) -> dict[str, ToolParameter]:
    assert isinstance(p, ObjectParameter)
    assert isinstance(p.properties, dict)
    return p.properties


# ======================================================================
# TestToolMetadata — name and description
# ======================================================================


class TestToolMetadata:
    """Test converter can extract information from docstring."""

    def test_name_is_function_name(self, example_primitives):
        tool = Tool.from_callable(example_primitives)
        assert tool.name == "_example_primitives"

    def test_name_override_replaces_function_name(self, example_primitives):
        tool = Tool.from_callable(example_primitives, name="send_log")
        assert tool.name == "send_log"

    def test_description_is_first_docstring_paragraph(self, example_primitives):
        tool = Tool.from_callable(example_primitives)
        # First paragraph of the docstring, stripped of leading/trailing whitespace.
        assert tool.description == (
            "Send a formatted log message to the configured output sink."
        )

    def test_description_strips_args_section(self, example_primitives):
        # The Args: block must not bleed into the description.
        tool = Tool.from_callable(example_primitives)
        assert "Args:" not in tool.description
        assert "message:" not in tool.description

    def test_no_args_function_has_empty_parameters(self, example_no_args):
        tool = Tool.from_callable(example_no_args)
        assert tool.parameters == {}
        assert tool.required == []

    def test_no_args_description_populated(self, example_no_args):
        tool = Tool.from_callable(example_no_args)
        assert tool.description == "Return the current health status of the service."


# ======================================================================
# TestRequiredParameters — required list
# ======================================================================


class TestRequiredParameters:
    """Test required parameters are correctly mapped to ``Tool.required``."""

    def test_required_and_optional_params(self, example_primitives):
        tool = Tool.from_callable(example_primitives)
        assert set(tool.required) == {"message", "count", "threshold", "verbose"}
        for name in ("prefix", "retries", "jitter", "dry_run"):
            assert name not in tool.required

    def test_required_order_matches_signature_order(self, example_primitives):
        tool = Tool.from_callable(example_primitives)
        # required must be a subset of parameter keys in declaration order
        sig_order = ["message", "count", "threshold", "verbose"]
        required_in_sig_order = [p for p in sig_order if p in tool.required]
        assert required_in_sig_order == sig_order

    def test_optional_args_required_and_optional(self, example_optional_args):
        tool = Tool.from_callable(example_optional_args)
        assert set(tool.required) == {"query", "limit"}
        for name in ("filter_tag", "min_score", "include_deleted", "page_size"):
            assert name not in tool.required


# ======================================================================
# TestPrimitiveParameters — str / int / float / bool
# ======================================================================


class TestPrimitiveParameters:
    """Test primitive type arguments in callables."""

    @pytest.mark.parametrize(
        ("name", "expected_type"),
        [
            ("message", "string"),
            ("count", "integer"),
            ("threshold", "number"),
            ("verbose", "boolean"),
            ("prefix", "string"),
            ("retries", "integer"),
            ("jitter", "number"),
            ("dry_run", "boolean"),
        ],
    )
    def test_primitive_param_types(self, example_primitives, name, expected_type):
        p = _param(Tool.from_callable(example_primitives), name)
        assert isinstance(p, PrimitiveParameter)
        assert p.type == expected_type

    def test_description_populated_from_docstring(self, example_primitives):
        p = _param(Tool.from_callable(example_primitives), "message")
        assert p.description != ""
        assert "log entry" in p.description.lower()

    def test_description_populated_for_optional_param(self, example_optional_args):
        p = _param(Tool.from_callable(example_optional_args), "filter_tag")
        assert p.description != ""


# ======================================================================
# TestEnumParameters — Literal and Enum subclass
# ======================================================================


class TestEnumParameters:
    """Test Literal of enum type arguments in callables."""

    # --- string Literal ---

    @pytest.mark.parametrize(
        ("fn", "param_name", "expected_values"),
        [
            (
                "example_string_literal",
                "align",
                {"left", "center", "right"},
            ),
            (
                "example_string_literal",
                "mode",
                {"truncate", "wrap", "overflow"},
            ),
        ],
    )
    def test_string_literal_values_correct(
        self, request, fn, param_name, expected_values
    ):
        fixture = request.getfixturevalue(fn)
        p = _param(Tool.from_callable(fixture), param_name)
        assert _assert_enum(p)
        assert p.enum == expected_values

    def test_string_literal_optional_not_in_required(self, example_string_literal):
        tool = Tool.from_callable(example_string_literal)
        assert "mode" not in tool.required

    # --- non-string Literal → NotImplementedError ---

    @pytest.mark.parametrize(
        "fn",
        [
            "example_int_literal",
            "example_mixed_literal",
        ],
    )
    def test_non_string_literal_raises_not_implemented_error(self, request, fn):
        fixture = request.getfixturevalue(fn)
        with pytest.raises(NotImplementedError):
            Tool.from_callable(fixture)

    # --- Enum subclass ---

    def test_enum_subclass_values_are_member_values(self, example_enum_param):
        # Severity enum values are strings; they should appear as-is.
        p = _param(Tool.from_callable(example_enum_param), "severity")
        assert _assert_enum(p)
        assert p.enum == {"low", "medium", "high", "critical"}

    def test_enum_subclass_description_from_docstring(self, example_enum_param):
        p = _param(Tool.from_callable(example_enum_param), "severity")
        assert p.description != ""


# ======================================================================
# TestArrayParameters — bare list + typed list[T]
# ======================================================================


class TestArrayParameters:
    """Test list type arguments in callables."""

    def test_bare_list_items_is_none(self, example_list_params):
        p = _param(Tool.from_callable(example_list_params), "tags")
        assert _assert_array(p)
        assert p.items is None

    @pytest.mark.parametrize(
        ("name", "expected_type"),
        [
            ("names", "string"),
            ("scores", "integer"),
            ("weights", "number"),
            ("enabled_flags", "boolean"),
        ],
    )
    def test_list_item_types(self, example_list_params, name, expected_type):
        p = _param(Tool.from_callable(example_list_params), name)
        assert _assert_array(p)
        assert isinstance(p.items, PrimitiveParameter)
        assert p.items.type == expected_type

    def test_list_required_and_optional_params(self, example_list_params):
        tool = Tool.from_callable(example_list_params)
        assert set(tool.required) == {"tags", "names", "scores"}
        assert "weights" not in tool.required
        assert "enabled_flags" not in tool.required


# ======================================================================
# TestObjectParameters — bare dict + typed dict[str, T]
# ======================================================================


class TestObjectParameters:
    """Test dict type arguments in callables."""

    def test_bare_dict_is_object_parameter(self, example_dict_params):
        p = _param(Tool.from_callable(example_dict_params), "metadata")
        assert isinstance(p, ObjectParameter)

    def test_bare_dict_properties_is_none(self, example_dict_params):
        # Unconstrained — no type information available.
        p = _param(Tool.from_callable(example_dict_params), "metadata")
        assert _assert_object(p)
        assert p.properties is None

    @pytest.mark.parametrize(
        ("name", "expected_type"),
        [
            ("word_counts", "integer"),
            ("feature_flags", "boolean"),
            ("score_overrides", "number"),
        ],
    )
    def test_dict_property_types(self, example_dict_params, name, expected_type):
        p = _param(Tool.from_callable(example_dict_params), name)
        assert _assert_object(p)
        assert isinstance(p.properties, PrimitiveParameter)
        assert p.properties.type == expected_type

    def test_dict_required_and_optional_params(self, example_dict_params):
        tool = Tool.from_callable(example_dict_params)
        assert set(tool.required) == {"metadata", "word_counts", "feature_flags"}
        assert "score_overrides" not in tool.required


# ======================================================================
# TestTypedDictParameters — TypedDict (flat + one-level nested)
# ======================================================================


class TestTypedDictParameters:
    """Test TypedDict type arguments in callables."""

    # --- flat TypedDict (BoundingBox: x, y, width, height — all int) ---

    def test_flat_typed_dict_properties_is_dict(self, example_flat_typed_dict):
        p = _param(Tool.from_callable(example_flat_typed_dict), "crop")
        assert _assert_object(p)
        props = _assert_object_dict_props(p)
        assert isinstance(props, dict)

    def test_flat_typed_dict_all_fields_present(self, example_flat_typed_dict):
        p = _param(Tool.from_callable(example_flat_typed_dict), "crop")
        assert _assert_object(p)
        props = _assert_object_dict_props(p)
        assert set(props.keys()) == {"x", "y", "width", "height"}

    def test_flat_typed_dict_field_types_are_integer(self, example_flat_typed_dict):
        p = _param(Tool.from_callable(example_flat_typed_dict), "crop")
        assert _assert_object(p)
        props = _assert_object_dict_props(p)
        for field_name, field_param in props.items():
            assert isinstance(field_param, PrimitiveParameter), (
                f"Field {field_name!r} should be PrimitiveParameter, "
                f"got {type(field_param).__name__}"
            )
            assert field_param.type == "integer", (
                f"Field {field_name!r} should have type 'integer', "
                f"got {field_param.type!r}"
            )

    def test_flat_typed_dict_outer_param_required(self, example_flat_typed_dict):
        tool = Tool.from_callable(example_flat_typed_dict)
        assert "crop" in tool.required

    # --- nested TypedDict (DetectionResult contains BoundingBox) ---

    def test_nested_typed_dict_top_level_fields(self, example_nested_typed_dict):
        p = _param(Tool.from_callable(example_nested_typed_dict), "hint")
        assert _assert_object(p)
        props = _assert_object_dict_props(p)
        assert set(props.keys()) == {"label", "confidence", "box"}

    def test_nested_typed_dict_primitive_fields(self, example_nested_typed_dict):
        p = _param(Tool.from_callable(example_nested_typed_dict), "hint")
        assert _assert_object(p)
        props = _assert_object_dict_props(p)
        assert isinstance(props["label"], PrimitiveParameter)
        assert props["label"].type == "string"
        assert isinstance(props["confidence"], PrimitiveParameter)
        assert props["confidence"].type == "number"

    def test_nested_typed_dict_box_fields_expanded(self, example_nested_typed_dict):
        p = _param(Tool.from_callable(example_nested_typed_dict), "hint")
        assert _assert_object(p)
        props = _assert_object_dict_props(p)
        box = props["box"]
        assert isinstance(box, ObjectParameter)
        box_props = _assert_object_dict_props(box)
        assert set(box_props.keys()) == {"x", "y", "width", "height"}

    def test_nested_typed_dict_box_field_types(self, example_nested_typed_dict):
        p = _param(Tool.from_callable(example_nested_typed_dict), "hint")
        assert _assert_object(p)
        props = _assert_object_dict_props(p)
        box = props["box"]
        assert isinstance(box, ObjectParameter)
        box_props = _assert_object_dict_props(box)
        for field_name, field_param in box_props.items():
            assert isinstance(field_param, PrimitiveParameter), (
                f"box.{field_name!r} should be PrimitiveParameter"
            )
            assert field_param.type == "integer"

    def test_nested_typed_dict_optional_label_filter(self, example_nested_typed_dict):
        # label_filter: Optional[str] = None on the same fixture — Gap 1 + Gap 8.
        tool = Tool.from_callable(example_nested_typed_dict)
        p = _param(tool, "label_filter")
        assert isinstance(p, PrimitiveParameter)
        assert p.type == "string"
        assert "label_filter" not in tool.required


# ======================================================================
# TestOptionalParameters — Optional[T] + None default
# ======================================================================


class TestOptionalParameters:
    """Test optional arguments in callables."""

    def test_all_parameters_present_in_schema(self, example_optional_args):
        tool = Tool.from_callable(example_optional_args)
        assert set(tool.parameters.keys()) == {
            "query",
            "limit",
            "filter_tag",
            "min_score",
            "include_deleted",
            "page_size",
        }

    def test_optional_params_are_plain_primitive_not_anyof(self, example_optional_args):
        # None of the Optional[T] params should produce anything other than
        # a plain PrimitiveParameter — anyOf / nullable schemas are out of scope.
        tool = Tool.from_callable(example_optional_args)
        for name in ("filter_tag", "min_score", "include_deleted"):
            p = tool.parameters[name]
            assert isinstance(p, PrimitiveParameter), (
                f"{name!r} should be PrimitiveParameter, got {type(p).__name__}"
            )
        # page_size: plain int with non-None default is also optional
        assert "page_size" not in tool.required
        p = _param(tool, "page_size")
        assert isinstance(p, PrimitiveParameter)
        assert p.type == "integer"

    def test_optional_param_description_populated(self, example_optional_args):
        tool = Tool.from_callable(example_optional_args)
        for name in ("filter_tag", "min_score", "include_deleted", "page_size"):
            assert tool.parameters[name].description != "", (
                f"Description for {name!r} should not be empty"
            )


# ======================================================================
# TestNestedContainers — list[list[T]] and dict[str, list[T]]
# ======================================================================


class TestNestedContainers:
    """Test nested container (list or dict) arguments in callables."""

    # --- list[list[str]] ---

    def test_list_of_list_structure(self, example_nested_containers):
        p = _param(Tool.from_callable(example_nested_containers), "tag_groups")
        assert _assert_array(p)
        assert isinstance(p.items, ArrayParameter)
        assert isinstance(p.items.items, PrimitiveParameter)
        assert p.items.items.type == "string"

    # --- dict[str, list[int]] ---

    def test_dict_of_list_structure(self, example_nested_containers):
        p = _param(Tool.from_callable(example_nested_containers), "index")
        assert _assert_object(p)
        assert isinstance(p.properties, ArrayParameter)
        assert isinstance(p.properties.items, PrimitiveParameter)
        assert p.properties.items.type == "integer"

    def test_dict_of_list_float_optional(self, example_nested_containers):
        # label_scores: dict[str, list[float]] = {} — optional, same structure
        tool = Tool.from_callable(example_nested_containers)
        assert "label_scores" not in tool.required
        p = _param(tool, "label_scores")
        assert _assert_object(p)
        assert isinstance(p.properties, ArrayParameter)
        assert isinstance(p.properties.items, PrimitiveParameter)
        assert p.properties.items.type == "number"


# ======================================================================
# TestDegradedDocstrings — converter must not raise; graceful fallback
# ======================================================================


class TestDegradedDocstrings:
    """Test conversion with incompatible or insufficient docstrings."""

    @pytest.mark.parametrize(
        "fn",
        [
            _example_no_docstring,
            _example_description_only,
            _example_partial_args_doc,
        ],
    )
    def test_converter_does_not_raise(self, fn):
        result = Tool.from_callable(fn)
        assert result is not None

    @pytest.mark.parametrize(
        ("fn", "expected_description"),
        [
            (_example_no_docstring, ""),
            (_example_description_only, "Store a named integer value in the registry."),
        ],
    )
    def test_degraded_docstring_tool_description(self, fn, expected_description):
        tool = Tool.from_callable(fn)
        assert tool.description == expected_description

    def test_partial_args_doc_documented_and_undocumented_params(
        self, example_partial_args_doc
    ):
        tool = Tool.from_callable(example_partial_args_doc)
        for name in ("name", "value", "active"):
            assert tool.parameters[name].description != "", (
                f"Documented parameter {name!r} should have a non-empty description"
            )
        assert "offset" in tool.parameters, (
            "Parameter 'offset' must appear in schema even when absent from Args"
        )
        assert tool.parameters["offset"].description == ""

    def test_partial_args_doc_required_list_correct(self, example_partial_args_doc):
        # name and value have no defaults; offset and active do.
        tool = Tool.from_callable(example_partial_args_doc)
        assert set(tool.required) == {"name", "value"}


# ======================================================================
# TestEdgeCases — *args, **kwargs, unannotated params, unsupported types
# ======================================================================


class TestEdgeCases:
    """Test edge cases in Tool.from_callable parameter handling."""

    def test_var_positional_args_skipped(self) -> None:
        def _fn(*args: str, name: str) -> None:  # noqa: D417 — *args intentionally undocumented
            """Accept var-args.

            Args:
                name: A name.
            """

        tool = Tool.from_callable(_fn)
        assert "name" in tool.parameters
        assert "args" not in tool.parameters

    def test_var_keyword_args_skipped(self) -> None:
        def _fn(name: str, **kwargs: int) -> None:  # noqa: D417 — **kwargs intentionally undocumented
            """Accept kwargs.

            Args:
                name: A name.
            """

        tool = Tool.from_callable(_fn)
        assert "name" in tool.parameters
        assert "kwargs" not in tool.parameters

    def test_unannotated_parameter_omitted(self) -> None:
        def _fn(name: str, unannotated=None) -> None:  # type: ignore[arg-type]  # reason: testing unannotated param handling  # noqa: D417 — unannotated param intentionally undocumented
            """Accept unannotated param.

            Args:
                name: A name.
            """

        tool = Tool.from_callable(_fn)
        assert "name" in tool.parameters
        assert "unannotated" not in tool.parameters

    def test_dict_non_string_key_raises(self) -> None:
        def _fn(mapping: dict[int, str]) -> None:  # type: ignore[type-arg]  # reason: intentionally testing non-str dict key
            """Use dict with non-str keys.

            Args:
                mapping: A mapping.
            """

        with pytest.raises(NotImplementedError, match="dict key type must be str"):
            Tool.from_callable(_fn)

    def test_functools_partial_name_extracts_func_name(self) -> None:
        def _greet(greeting: str, name: str) -> None:
            """Greet someone.

            Args:
                greeting: The greeting.
                name: The person's name.
            """

        partial_fn = functools.partial(_greet, "Hello")
        tool = Tool.from_callable(partial_fn)
        assert tool.name == "_greet"

    def test_unsupported_type_raises(self) -> None:
        def _fn(value: set) -> None:  # type: ignore[type-arg]  # reason: intentionally testing unsupported type
            """Use unsupported type.

            Args:
                value: A set.
            """

        with pytest.raises(NotImplementedError, match="Unsupported type"):
            Tool.from_callable(_fn)
