"""OpenAI-compatible chat loop example with optional tool support."""

import argparse
import asyncio
from typing import TypedDict

from chat.chat_loop import chat_loop  # ty: ignore[unresolved-import]
from chat.display import display_tools  # ty: ignore[unresolved-import]
from chat.tools import (  # ty: ignore[unresolved-import]
    calculate,
    read_file,
    search_path,
)
from rich.console import Console

from atk.core.toolset import CallableToolset
from atk.openai.model import OpenAILanguageModel

console = Console()


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


def main() -> None:
    """Run streaming chat loop with tool definitions."""
    args = parse_args()
    llm = OpenAILanguageModel(
        base_url=args["base_url"],
        api_key=args["api_key"],
        model=args["model"],
    )
    toolset = CallableToolset([calculate, search_path, read_file])
    instruction = "You are a helpful assistant that can use tools to help users."
    display_tools(toolset)
    asyncio.run(chat_loop(llm, instruction, toolset=toolset))
    console.print("\n[blue]Exiting chat...[/blue]\n")


if __name__ == "__main__":
    main()
