"""Maps between internal tools and any-llm tool schema."""

from typing import Any

from atk.core.tool import Tool, tool_to_json_schema

# any-llm accepts heterogeneous nested tool dictionaries without a precise
# exported schema type.
AnyLlmTool = dict[str, Any]


class ToolMapper:
    """Maps between internal tools and any-llm tool format."""

    @staticmethod
    def to_tools(tools: list[Tool]) -> list[AnyLlmTool]:
        """Maps a list of tools to any-llm function tool format.

        Args:
            tools: List of internal Tool definitions.

        Returns:
            List of tool dictionaries compatible with any-llm's completion API.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool_to_json_schema(tool),
                    "strict": True,
                },
            }
            for tool in tools
        ]
