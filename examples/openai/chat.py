"""OpenAI-compatible simple chat loop example."""

import argparse
import asyncio
from typing import TYPE_CHECKING, TypedDict

from atk.core.message import (
    AssistantMessage,
    Message,
    TextPart,
    ToolCallPart,
    UserMessage,
)
from atk.openai.model import OpenAILanguageModel

if TYPE_CHECKING:
    from pydantic import BaseModel

    from atk.core.tool import Tool


class CLIArgs(TypedDict):
    """CLI arguments for OpenAI-compatible endpoints."""

    base_url: str
    api_key: str
    model: str | None


def parse_args() -> CLIArgs:
    """Parse CLI arguments for OpenAI-compatible endpoints."""
    parser = argparse.ArgumentParser(description="OpenAI-compatible chat")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="Base URL for OpenAI-compatible API (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key for the API (default: empty)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model name",
    )
    args = parser.parse_args()
    return CLIArgs(base_url=args.base_url, api_key=args.api_key, model=args.model)


def chat_loop(
    llm: OpenAILanguageModel,
    instruction: str,
    output_schema: type[BaseModel] | None = None,
    tools: list[Tool] | None = None,
) -> None:
    """Run interactive chat loop with the language model.

    Args:
        llm: Language model instance.
        instruction: System instruction.
        output_schema: Optional Pydantic model for structured output.
        tools: Optional list of tools available to the model.
    """
    messages: list[Message] = []

    while True:
        try:
            user_message = input("User:\n")
        except EOFError, KeyboardInterrupt:
            break
        if user_message in {"exit", "quit"}:
            break
        print("")  # noqa: T201, FURB105
        messages.append(UserMessage(content=[TextPart(text=user_message)]))

        response: AssistantMessage = asyncio.run(
            llm.generate_response(
                instruction=instruction,
                messages=messages,
                tools=tools,
                response_format=output_schema,
            )
        )

        messages.append(response)

        for part in response.content:
            match part:
                case TextPart():
                    print(f"Assistant:\n{part.text}\n")  # noqa: T201
                case ToolCallPart():
                    print(f"Assistant (tool call):\n{part.model_dump_json()}\n")  # noqa: T201


def main() -> None:
    """Run simple chat loop."""
    args = parse_args()
    llm = OpenAILanguageModel(
        base_url=args["base_url"],
        api_key=args["api_key"],
        model=args["model"],
    )
    instruction = "You are a helpful assistant."

    chat_loop(llm, instruction)


if __name__ == "__main__":
    main()
