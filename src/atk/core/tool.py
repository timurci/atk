"""Tool definitions for the language model."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


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

    The types of the elements in the array might be heterogenous.
    """

    type: Literal["array"]
    description: str


class ObjectParameter(BaseModel):
    """Represents a mapping parameter with a fixed set of properties."""

    type: Literal["object"]
    description: str
    properties: dict[str, ToolParameter]


ToolParameter = Annotated[
    PrimitiveParameter | EnumParameter | ArrayParameter | ObjectParameter,
    Field(discriminator="type"),
]


class Tool(BaseModel):
    """Tool represents a callable function that can be invoked by the language model."""

    name: str
    description: str
    parameters: dict[str, ToolParameter]
    required: list[str] = Field(
        default_factory=list, description="List of required parameter names."
    )
