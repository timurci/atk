"""Tool definitions for the language model."""

from __future__ import annotations

import enum
import functools
import inspect
import types
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from griffe import (
    Docstring,
    DocstringSectionParameters,
    DocstringSectionText,
    parse_google,
)
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable


class PrimitiveParameter(BaseModel):
    """A primitive parameter represents a simple value type."""

    type: Literal["string", "integer", "number", "boolean"]
    description: str


class EnumParameter(BaseModel):
    """Represents an enumeration parameter with a fixed set of allowed values."""

    type: Literal["enum"]
    description: str
    enum: set[str]


class ArrayParameter(BaseModel):
    """Represents an array parameter with a variable number of elements.

    When ``items`` is ``None`` the element type is unconstrained (bare
    ``list`` annotation).  When set, all elements are expected to conform
    to the described parameter type.
    """

    type: Literal["array"]
    description: str
    items: ToolParameter | None = None


class ObjectParameter(BaseModel):
    """Represents a mapping parameter.

    ``properties`` has three distinct states:

    * ``None`` — unconstrained mapping; bare ``dict`` annotation with no
      type information (keys and values may be anything).
    * ``ToolParameter`` — homogeneous mapping; a typed ``dict[str, T]``
      annotation where all values share one type but keys are arbitrary.
    * ``dict[str, ToolParameter]`` — structured mapping; a ``TypedDict``
      annotation where each named field has its own type.
    """

    type: Literal["object"]
    description: str
    properties: dict[str, ToolParameter] | ToolParameter | None = None


ToolParameter = Annotated[
    PrimitiveParameter | EnumParameter | ArrayParameter | ObjectParameter,
    Field(discriminator="type"),
]


def _resolve_optional(annotation: object) -> tuple[object, bool]:
    """Return (inner_type, is_optional) for ``T | None`` annotations.

    This function intentionally handles only ``T | None`` (or
    ``Union[T, None]``) — the sole optional form that maps to a plain
    ``T`` schema with the parameter omitted from ``required``.  Other
    union types (``A | B`` with no ``None``) are intentionally not
    supported because a tool argument should declare a single, clear
    type to avoid ambiguity for the LLM.
    """
    origin = get_origin(annotation)
    if origin is types.UnionType or origin is Union:
        args = get_args(annotation)
        if type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return non_none[0], True
    return annotation, False


def _map_primitive(annotation: type, description: str) -> ToolParameter | None:
    """Map a primitive type annotation to a ToolParameter, or None if not primitive."""
    type_map: dict[type, Literal["string", "integer", "number", "boolean"]] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }
    prim_type = type_map.get(annotation)
    if prim_type is not None:
        return PrimitiveParameter(type=prim_type, description=description)
    return None


def _map_container_type(
    origin: object, args: tuple, description: str
) -> ToolParameter | None:
    """Map a container type (list/dict/Literal) to a ToolParameter, or None."""
    if origin is Literal:
        if not all(isinstance(v, str) for v in args):
            msg = f"Non-string Literal values are not supported: {args}"
            raise NotImplementedError(msg)
        return EnumParameter(type="enum", description=description, enum=set(args))
    if origin is list:
        items = None if not args else _map_type(args[0], "")
        return ArrayParameter(type="array", description=description, items=items)
    if origin is dict:
        if args and args[0] is not str:
            msg = f"dict key type must be str, got {args[0]!r}"
            raise NotImplementedError(msg)
        props = None if not args else _map_type(args[1], "")
        return ObjectParameter(type="object", description=description, properties=props)
    return None


def _map_bare_type(annotation: type, description: str) -> ToolParameter | None:
    """Map a bare (non-generic) type to a ToolParameter, or None."""
    if annotation is list:
        return ArrayParameter(type="array", description=description, items=None)
    if annotation is dict:
        return ObjectParameter(type="object", description=description, properties=None)
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return EnumParameter(
            type="enum",
            description=description,
            enum={m.value for m in annotation},
        )
    if hasattr(annotation, "__required_keys__") and hasattr(
        annotation, "__optional_keys__"
    ):
        try:
            resolved_hints = get_type_hints(annotation)
        except NameError:
            resolved_hints = annotation.__annotations__
        props = {
            field_name: _map_type(field_type, "")
            for field_name, field_type in resolved_hints.items()
        }
        return ObjectParameter(type="object", description=description, properties=props)
    return None


def _map_type(annotation: object, description: str) -> ToolParameter:
    """Map a Python type annotation to a ToolParameter."""
if isinstance(annotation, type):
        result = _map_primitive(annotation, description)
        if result is not None:
            return result

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is not None:
        result = _map_container_type(origin, args, description)
        if result is not None:
            return result

    if isinstance(annotation, type):
        result = _map_bare_type(annotation, description)
        if result is not None:
            return result

    msg = f"Unsupported type annotation: {annotation}"
    raise NotImplementedError(msg)


def _parse_docstring(fn: Callable) -> tuple[str, dict[str, str]]:
    """Parse a Google-style docstring and return (description, param_descriptions)."""
    doc = inspect.getdoc(fn)
    if not doc:
        return "", {}

    ds = Docstring(doc, lineno=1)
    sections = parse_google(ds, warnings=False)
    description = ""
    param_descs: dict[str, str] = {}
    for section in sections:
        if isinstance(section, DocstringSectionText):
            description = section.value.strip().split("\n\n")[0].strip()
            break
    for section in sections:
        if isinstance(section, DocstringSectionParameters):
            for param in section.value:
                param_descs[param.name] = param.description or ""
    return description, param_descs


class Tool(BaseModel):
    """Tool represents a callable function that can be invoked by the language model."""

    name: str
    description: str
    parameters: dict[str, ToolParameter]
    required: list[str] = Field(
        default_factory=list, description="List of required parameter names."
    )

    @staticmethod
    def from_callable(fn: Callable[..., object]) -> Tool:
        """Create a Tool from a callable function.

        Args:
            fn: The callable object to convert into a Tool.

        Returns:
            A Tool instance representing the callable function.
        """
        sig = inspect.signature(fn)
        annotations = inspect.get_annotations(fn, eval_str=True)
        description, param_descs = _parse_docstring(fn)
        parameters: dict[str, ToolParameter] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            annotation = annotations.get(param_name)
            if annotation is None:
                continue
            resolved, is_optional = _resolve_optional(annotation)
            tool_param = _map_type(resolved, param_descs.get(param_name, ""))
            has_default = param.default is not inspect.Parameter.empty
            if not has_default and not is_optional:
                required.append(param_name)
            parameters[param_name] = tool_param
        if isinstance(fn, functools.partial):
            tool_name = getattr(fn.func, "__name__", fn.func.__class__.__name__)
        else:
            tool_name = getattr(fn, "__name__", fn.__class__.__name__)
        return Tool(
            name=tool_name,
            description=description,
            parameters=parameters,
            required=required,
        )
