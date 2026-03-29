"""Tool definitions for the language model."""

from typing import TYPE_CHECKING, Annotated, Literal

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


class Tool(BaseModel):
    """Tool represents a callable function that can be invoked by the language model."""

    name: str
    description: str
    parameters: dict[str, ToolParameter]
    required: list[str] = Field(
        default_factory=list, description="List of required parameter names."
    )

    @staticmethod
    def from_callable(fn: Callable) -> Tool:
        """Create a Tool from a callable function.

        Args:
            fn: The callable object to convert into a Tool.

        Returns:
            A Tool instance representing the callable function.
        """
        raise NotImplementedError
