"""Streaming chat loop with Rich Live rendering."""

from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt

from atk.core.message import (
    AssistantMessage,
    AssistantStream,
    Message,
    TextDelta,
    TextPart,
    ToolCallDelta,
    ToolCallPart,
    ToolMessage,
    ToolResultPart,
    UserMessage,
)

from .display import (
    build_tool_call_panel,
    display_tool_response,
)

if TYPE_CHECKING:
    from atk.core.model import StreamingLanguageModel
    from atk.core.toolset import Toolset

console = Console()


class _StreamState:
    """Tracks accumulated state during a streaming turn."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.tool_call_ids: set[str] = set()
        self.tool_call_names: dict[str, str] = {}
        self.tool_call_args: dict[str, str] = {}

    def add_text(self, text: str) -> None:
        self.text.append(text)

    def add_tool_delta(self, delta: ToolCallDelta) -> None:
        if delta.id not in self.tool_call_ids:
            self.tool_call_ids.add(delta.id)
            self.tool_call_names[delta.id] = delta.name
            self.tool_call_args[delta.id] = ""
        self.tool_call_args[delta.id] += delta.arguments_delta

    def build_group(self) -> Group:
        renderables: list[Panel] = []
        renderables.append(
            Panel(
                "".join(self.text) or "…",
                title="[green]Assistant[/green]",
                border_style="green",
            )
        )
        renderables.extend(
            build_tool_call_panel(
                self.tool_call_names[cid],
                cid,
                self.tool_call_args[cid],
            )
            for cid in self.tool_call_ids
        )
        return Group(*renderables)


async def _handle_tool_calls(
    response: AssistantMessage,
    toolset: Toolset,
) -> ToolMessage:
    """Invoke tools from the assistant response and return a ToolMessage."""
    tool_calls = [p for p in response.content if isinstance(p, ToolCallPart)]
    tool_results: list[ToolResultPart] = []
    for part in tool_calls:
        result = await toolset.invoke_tool(part.name, part.arguments)
        tool_results.append(ToolResultPart(tool_call_id=part.id, content=result))
    return ToolMessage(content=tool_results)


async def _stream_turn(
    llm: StreamingLanguageModel,
    instruction: str,
    messages: list[Message],
    tools: list | None,
) -> AssistantMessage:
    """Stream a single turn and return the final AssistantMessage.

    Updates a Rich Live display with accumulating text and tool calls
    during streaming.
    """
    state = _StreamState()
    final_response: AssistantMessage | None = None

    with Live(state.build_group(), console=console, refresh_per_second=10) as live:
        async for chunk in llm.stream_response(
            instruction=instruction,
            messages=messages,
            tools=tools,
        ):
            match chunk:
                case AssistantStream():
                    for delta in chunk.content:
                        match delta:
                            case TextDelta():
                                state.add_text(delta.text)
                            case ToolCallDelta():
                                state.add_tool_delta(delta)
                    live.update(state.build_group())
                case AssistantMessage():
                    final_response = chunk

    if final_response is None:
        error_msg = "Stream ended without producing an AssistantMessage"
        raise RuntimeError(error_msg)
    return final_response


async def chat_loop(
    llm: StreamingLanguageModel,
    instruction: str,
    toolset: Toolset | None = None,
) -> None:
    """Run interactive chat loop with streaming responses.

    Args:
        llm: Language model instance with streaming support.
        instruction: System instruction.
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

            response = await _stream_turn(llm, instruction, messages, tools)
            messages.append(response)

            if toolset:
                tool_calls = [
                    p for p in response.content if isinstance(p, ToolCallPart)
                ]
                if tool_calls:
                    tool_message = await _handle_tool_calls(response, toolset)
                    display_tool_response(tool_message)
                    messages.append(tool_message)

    except EOFError, KeyboardInterrupt:
        pass
