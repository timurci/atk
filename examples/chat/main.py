"""OpenAI-compatible chat loop example with optional tool support."""

import argparse
import asyncio
import json
from typing import TYPE_CHECKING, TypedDict

from chat.tools import (  # ty: ignore[unresolved-import]
    calculate,
    read_file,
    search_path,
)
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from atk.core.message import (
    AssistantMessage,
    Message,
    TextPart,
    ToolCallPart,
    ToolMessage,
    ToolResultPart,
    UserMessage,
)
from atk.core.toolset import CallableToolset
from atk.openai.model import OpenAILanguageModel

if TYPE_CHECKING:
    from pydantic import BaseModel

    from atk.core.toolset import Toolset

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


def _display_tools(toolset: CallableToolset) -> None:
    """Display available tools in a formatted table."""
    table = Table(title="Available Tools", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Parameters", style="yellow")
    for tool in toolset.tools:
        params = ", ".join(tool.parameters.keys())
        table.add_row(tool.name, tool.description, params)
    console.print(table)
    console.print()


def _truncate_id(call_id: str, length: int = 8) -> str:
    """Truncate a tool call ID for display."""
    return call_id[:length] if len(call_id) > length else call_id


def _handle_tool_calls(
    response: AssistantMessage,
    toolset: Toolset,
) -> ToolMessage | None:
    """Invoke tools from the assistant response and return a ToolMessage."""
    tool_calls = [p for p in response.content if isinstance(p, ToolCallPart)]
    if not tool_calls:
        return None
    tool_results: list[ToolResultPart] = []
    for part in tool_calls:
        result = asyncio.run(toolset.invoke_tool(part.name, part.arguments))
        tool_results.append(ToolResultPart(tool_call_id=part.id, content=result))
    return ToolMessage(content=tool_results)


def _display_assistant_response(response: AssistantMessage) -> None:
    """Print assistant response parts to the console."""
    for part in response.content:
        match part:
            case TextPart():
                if part.text.strip():
                    console.print(
                        Panel(
                            part.text,
                            title="[green]Assistant[/green]",
                            border_style="green",
                        )
                    )
            case ToolCallPart():
                console.print(
                    Panel(
                        JSON(json.dumps(part.arguments, indent=2)),
                        title=(
                            f"[blue]Tool call: {part.name}"
                            f"(#{_truncate_id(part.id)})[/blue]"
                        ),
                        border_style="blue",
                    )
                )


def _display_tool_response(tool_message: ToolMessage) -> None:
    """Print tool result parts to the console."""
    for result_part in tool_message.content:
        console.print(
            Panel(
                result_part.content,
                title=(
                    f"[yellow]Tool result: "
                    f"#{_truncate_id(result_part.tool_call_id)}[/yellow]"
                ),
                border_style="yellow",
            )
        )


def chat_loop(
    llm: OpenAILanguageModel,
    instruction: str,
    output_schema: type[BaseModel] | None = None,
    toolset: Toolset | None = None,
) -> None:
    """Run interactive chat loop with the language model.

    Args:
        llm: Language model instance.
        instruction: System instruction.
        output_schema: Optional Pydantic model for structured output.
        toolset: Optional toolset available to the model.
    """
    messages: list[Message] = []
    tools = toolset.tools if toolset else None

    try:
        while True:
            if not messages or isinstance(messages[-1], AssistantMessage):
                user_message = Prompt.ask("\n[bold cyan]User[/bold cyan]")
                if user_message in {"exit", "quit"}:
                    break
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

            _display_assistant_response(response)

            if toolset:
                tool_message = _handle_tool_calls(response, toolset)
                if tool_message:
                    _display_tool_response(tool_message)
                    messages.append(tool_message)
    except EOFError, KeyboardInterrupt:
        pass


def main() -> None:
    """Run chat loop with tool definitions."""
    args = parse_args()
    llm = OpenAILanguageModel(
        base_url=args["base_url"],
        api_key=args["api_key"],
        model=args["model"],
    )
    toolset = CallableToolset([calculate, search_path, read_file])
    instruction = "You are a helpful assistant that can use tools to help users."
    _display_tools(toolset)
    chat_loop(llm, instruction, toolset=toolset)
    console.print("\n[blue]Exiting chat...[/blue]\n")


if __name__ == "__main__":
    main()
