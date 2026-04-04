"""OpenAI-compatible structured output example.

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
from atk.openai.model import OpenAILanguageModel

console = Console()


class CLIArgs(TypedDict):
    """CLI arguments for OpenAI-compatible endpoints."""

    base_url: str
    api_key: str
    model: str | None


def parse_args() -> CLIArgs:
    """Parse CLI arguments for OpenAI-compatible endpoints."""
    parser = argparse.ArgumentParser(description="OpenAI-compatible structured output")
    parser.add_argument("--base-url", default="http://localhost:8080/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    return CLIArgs(base_url=args.base_url, api_key=args.api_key, model=args.model)


class Word(BaseModel):
    """A word with syllables."""

    syllables: list[str]


def main() -> None:
    """Run a single structured output call."""
    args = parse_args()

    llm = OpenAILanguageModel(
        base_url=args["base_url"],
        api_key=args["api_key"],
        model=args["model"],
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
