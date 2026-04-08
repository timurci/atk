"""Structured output example using any-llm provider integration.

Makes a single API call requesting structured JSON output and exits.
No interactive chat loop.
"""

import argparse
import asyncio
from typing import TypedDict

from pydantic import BaseModel
from rich.console import Console
from rich.pretty import pretty_repr
from rich.prompt import Prompt

from atk.core.message import TextPart, UserMessage
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
    parser = argparse.ArgumentParser(
        description="Structured output with an LLM provider"
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    args = parser.parse_args()
    return CLIArgs(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )


class Word(BaseModel):
    """A word with syllables."""

    syllables: list[str]


def main() -> None:
    """Run a single structured output call."""
    args = parse_args()

    llm = AnyLanguageModel(
        provider=args["provider"],
        model=args["model"],
        api_key=args["api_key"],
        api_base=args["api_base"],
    )
    instruction = (
        "You are a helpful assistant. You help users write syllables of a word."
    )
    prompt = Prompt.ask("[bold cyan]Enter a word[/bold cyan]")

    response = asyncio.run(
        llm.generate_response(
            instruction=instruction,
            messages=[UserMessage(content=[TextPart(text=prompt)])],
            response_format=Word,
        )
    )

    console.print(f"\n[bold green]Response:[/bold green] {pretty_repr(response)}\n")


if __name__ == "__main__":
    main()
