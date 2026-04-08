"""Chat loop example using any-llm provider integration."""

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
from atk.providers.model import AnyLanguageModel

console = Console()


class CLIArgs(TypedDict):
    """CLI arguments for provider configuration."""

    provider: str
    model: str | None
    api_key: str | None
    api_base: str | None


def parse_args() -> CLIArgs:
    """Parse CLI arguments for provider configuration."""
    parser = argparse.ArgumentParser(description="Chat with an LLM provider")
    parser.add_argument(
        "--provider",
        default="openai",
        help="LLM provider identifier (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use (provider default if omitted)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (falls back to environment variable)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Base URL for the provider API",
    )
    args = parser.parse_args()
    return CLIArgs(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )


def main() -> None:
    """Run streaming chat loop with tool definitions."""
    args = parse_args()
    llm = AnyLanguageModel(
        provider=args["provider"],
        model=args["model"],
        api_key=args["api_key"],
        api_base=args["api_base"],
    )
    toolset = CallableToolset([calculate, search_path, read_file])
    instruction = "You are a helpful assistant that can use tools to help users."
    display_tools(toolset)
    asyncio.run(chat_loop(llm, instruction, toolset=toolset))
    console.print("\n[blue]Exiting chat...[/blue]\n")


if __name__ == "__main__":
    main()
