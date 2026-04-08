"""Maps between internal tools and any-llm tool schema."""

from typing import Any

from atk.core.tool import (
    ArrayParameter,
    EnumParameter,
    ObjectParameter,
    PrimitiveParameter,
    Tool,
    ToolParameter,
)


def _map_parameter(param: ToolParameter) -> dict[str, Any]:
    """Map a single ToolParameter to JSON schema property."""
    base: dict[str, Any] = {"description": param.description}

    match param:
        case PrimitiveParameter():
            base["type"] = param.type
        case EnumParameter():
            base["type"] = "string"
            base["enum"] = list(param.enum)
        case ArrayParameter():
            base["type"] = "array"
            if param.items is not None:
                base["items"] = _map_parameter(param.items)
        case ObjectParameter():
            base["type"] = "object"
            match param.properties:
                case dict():
                    base["properties"] = {
                        k: _map_parameter(v) for k, v in param.properties.items()
                    }
                case None:
                    pass
                case _:
                    base["additionalProperties"] = _map_parameter(param.properties)

    return base


class ToolMapper:
    """Maps between internal tools and any-llm tool format."""

    @staticmethod
    def to_tools(tools: list[Tool]) -> list[dict[str, Any]]:
        """Maps a list of tools to any-llm function tool format.

        Args:
            tools: List of internal Tool definitions.

        Returns:
            List of tool dictionaries compatible with any-llm's completion API.
        """
        result: list[dict[str, Any]] = []

        for tool in tools:
            function_schema: dict[str, Any] = {
                "type": "object",
                "properties": {
                    name: _map_parameter(param)
                    for name, param in tool.parameters.items()
                },
            }
            if tool.required:
                function_schema["required"] = tool.required

            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": function_schema,
                        "strict": True,
                    },
                }
            )

        return result
