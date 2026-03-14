"""Example tool calling from a pre-defined set of tools."""

from atk.core.tool import (
    EnumParameter,
    PrimitiveParameter,
    Tool,
)
from atk.openai.model import OpenAILanguageModel
from examples.openai.chat import chat_loop, parse_args

instruction = "You are a helpful assistant that can use tools to help users."
tools = [
    Tool(
        name="get_weather",
        description="Get the current weather for a specified location",
        parameters={
            "location": PrimitiveParameter(
                type="string",
                description="The city name, e.g. San Francisco",
            ),
            "unit": EnumParameter(
                type="enum",
                description="The unit of temperature. (Default: celsius)",
                enum={"celsius", "fahrenheit"},
            ),
        },
        required=["location"],
    ),
    Tool(
        name="calculate",
        description="Perform a mathematical calculation",
        parameters={
            "operation": EnumParameter(
                type="enum",
                description="The operation to perform",
                enum={"add", "subtract", "multiply", "divide"},
            ),
            "a": PrimitiveParameter(
                type="number",
                description="First number",
            ),
            "b": PrimitiveParameter(
                type="number",
                description="Second number",
            ),
        },
        required=["operation", "a", "b"],
    ),
    Tool(
        name="get_user_profile",
        description="Retrieve user profile information",
        parameters={
            "user_id": PrimitiveParameter(
                type="string",
                description="The unique identifier of the user",
            ),
            "include_history": PrimitiveParameter(
                type="boolean",
                description="Whether to include user's history",
            ),
        },
        required=["user_id"],
    ),
]


def main() -> None:
    """Run chat loop with tool definitions."""
    args = parse_args()
    llm = OpenAILanguageModel(
        base_url=args["base_url"],
        api_key=args["api_key"],
        model=args["model"],
    )
    print("List of available tools:")  # noqa: T201
    for tool in tools:
        print(f"  {tool.name}({list(tool.parameters)}): {tool.description}")  # noqa: T201
    print("")  # noqa: T201, FURB105
    chat_loop(llm, instruction, tools=tools)


if __name__ == "__main__":
    main()
