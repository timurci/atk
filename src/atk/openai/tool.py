"""Maps between internal tools and OpenAI tool schema."""

from typing import TYPE_CHECKING

from atk.core.tool import (
    ArrayParameter,
    EnumParameter,
    ObjectParameter,
    PrimitiveParameter,
    Tool,
    ToolParameter,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionFunctionToolParam


def _map_parameter(param: ToolParameter) -> dict:
    """Map a single ToolParameter to JSON schema property."""
    base: dict = {"description": param.description}

    match param:
        case PrimitiveParameter():
            base["type"] = param.type
        case EnumParameter():
            base["type"] = "string"
            base["enum"] = list(param.enum)
        case ArrayParameter():
            base["type"] = "array"
        case ObjectParameter():
            base["type"] = "object"
            base["properties"] = {
                k: _map_parameter(v) for k, v in param.properties.items()
            }

    return base


class OpenAIToolMapper:
    """Maps between internal tools and OpenAI tool schema."""

    @staticmethod
    def to_openai(
        tools: list[Tool],
    ) -> list[ChatCompletionFunctionToolParam]:
        """Maps a list of tools to OpenAI function tool params.

        Args:
            tools: List of internal Tool definitions.

        Returns:
            List of OpenAI-compatible function tool parameters.
        """
        result: list[ChatCompletionFunctionToolParam] = []

        for tool in tools:
            function_schema: dict = {
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
