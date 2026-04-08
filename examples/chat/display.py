"""Rich-based display helpers for the chat demo."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import get_console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from atk.core.message import ToolMessage
    from atk.core.toolset import CallableToolset


def _truncate_id(call_id: str, length: int = 8) -> str:
    """Truncate a tool call ID for display."""
    return call_id[:length] if len(call_id) > length else call_id


def display_tools(toolset: CallableToolset) -> None:
    """Display available tools in a formatted table."""
    console = get_console()
    table = Table(title="Available Tools", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Parameters", style="yellow")
    for tool in toolset.tools:
        params = ", ".join(tool.parameters.keys())
        table.add_row(tool.name, tool.description, params)
    console.print(table)
    console.print()


def build_tool_call_panel(name: str, call_id: str, raw_args: str) -> Panel:
    """Build a panel for a tool call being streamed.

    ``raw_args`` may be an incomplete JSON string during streaming.
    """
    return Panel(
        raw_args or "…",
        title=f"[blue]Tool call: {name}(#{_truncate_id(call_id)})[/blue]",
        border_style="blue",
    )


def build_thinking_panel(thinking: str) -> Panel:
    """Build a panel for thinking/reasoning content."""
    return Panel(
        thinking or "…",
        title="[magenta]Thinking[/magenta]",
        border_style="magenta",
    )


def display_tool_response(tool_message: ToolMessage) -> None:
    """Print tool result parts to the console."""
    console = get_console()
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
