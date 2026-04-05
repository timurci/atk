"""OpenAI language model implementation."""

import json
from typing import TYPE_CHECKING

from openai import AsyncOpenAI, Omit

from atk.core.message import (
    AssistantMessage,
    AssistantStream,
    Message,
    TextDelta,
    TextPart,
    ToolCallDelta,
    ToolCallPart,
)
from atk.core.model import LanguageModel, StreamingLanguageModel

from .message import OpenAIMessageMapper
from .tool import OpenAIToolMapper

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from openai.types.chat import ChatCompletionChunk
    from pydantic import BaseModel

    from atk.core.tool import Tool


class _StreamAccumulator:
    """Accumulates streaming chunks into a final AssistantMessage.

    OpenAI streams tool call data across multiple chunks, with ``id``
    and ``name`` only present in the first chunk for each tool call
    index.  This class tracks per-index state so that every emitted
    ``ToolCallDelta`` has complete ``id`` and ``name`` fields.
    """

    def __init__(self) -> None:
        self.accumulated_text: list[str] = []
        # Maps OpenAI tool call index (from ``delta.tool_calls[].index``)
        # to the unique call ID string.  Only set once, on the first chunk.
        self.tool_call_ids: dict[int, str] = {}
        # Maps tool call index to the function name.  Only set once,
        # on the first chunk for that index.
        self.tool_call_names: dict[int, str] = {}
        # Maps tool call index to a list of JSON argument fragments.
        # Fragments are appended in order and concatenated at the end.
        self.tool_call_args: dict[int, list[str]] = {}

    def process_chunk(
        self, chunk: ChatCompletionChunk
    ) -> list[TextDelta | ToolCallDelta]:
        """Process a single chunk and return mapped delta events."""
        delta = chunk.choices[0].delta
        events: list[TextDelta | ToolCallDelta] = []

        if delta.content is not None:
            self.accumulated_text.append(delta.content)
            events.append(TextDelta(text=delta.content))

        if delta.tool_calls is not None:
            for tc in delta.tool_calls:
                idx = tc.index
                if tc.id is not None:
                    self.tool_call_ids[idx] = tc.id
                if tc.function is not None:
                    if tc.function.name is not None:
                        self.tool_call_names[idx] = tc.function.name
                    args_fragment = tc.function.arguments or ""
                    self.tool_call_args.setdefault(idx, []).append(args_fragment)

                    events.append(
                        ToolCallDelta(
                            id=self.tool_call_ids[idx],
                            name=self.tool_call_names[idx],
                            arguments_delta=args_fragment,
                        )
                    )

        return events

    def build_message(self) -> AssistantMessage:
        """Build the final accumulated AssistantMessage."""
        content: list[TextPart | ToolCallPart] = []
        if self.accumulated_text:
            content.append(TextPart(text="".join(self.accumulated_text)))
        content.extend(
            ToolCallPart(
                id=self.tool_call_ids[idx],
                name=self.tool_call_names[idx],
                arguments=json.loads("".join(self.tool_call_args[idx])),
            )
            for idx in sorted(self.tool_call_ids)
        )
        return AssistantMessage(content=content)


class OpenAILanguageModel(LanguageModel, StreamingLanguageModel):
    """OpenAI language model implementation."""

    def __init__(
        self,
        base_url: str | None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the OpenAI language model.

        Args:
            base_url: Optional base URL for OpenAI-compatible API.
            api_key: Optional API key.
            model: Optional model name.
        """
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model: str = model if model is not None else "gpt-4o-mini"
        self.message_mapper = OpenAIMessageMapper()
        self.tool_mapper = OpenAIToolMapper()

    async def generate_response(
        self,
        instruction: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> AssistantMessage:
        """Generate a response from the language model.

        Args:
            instruction: System instruction.
            messages: Conversation history.
            tools: List of available tools for the model to call.
            response_format: Optional Pydantic model for structured output.

        Returns:
            AssistantMessage response.
        """
        response = await self.client.chat.completions.parse(
            model=self.model,
            messages=self.message_mapper.to_openai(instruction, messages),
            tools=self.tool_mapper.to_openai(tools) if tools else Omit(),
            response_format=response_format or Omit(),  # type: ignore[invalid-argument-type]
        )
        return self.message_mapper.from_openai(response.choices[0].message)

    async def stream_response(
        self,
        instruction: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> AsyncGenerator[AssistantStream | AssistantMessage]:
        """Stream a response from the language model.

        Args:
            instruction: System instruction.
            messages: Conversation history.
            tools: List of available tools for the model to call.
            response_format: Optional Pydantic model for structured output.

        Yields:
            AssistantStream chunks during generation, ending with
            AssistantMessage containing the fully accumulated content.
        """
        if response_format is not None:
            error_msg = "OpenAI does not support response_format with streaming"
            raise NotImplementedError(error_msg)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=self.message_mapper.to_openai(instruction, messages),
            tools=self.tool_mapper.to_openai(tools) if tools else Omit(),
            stream=True,
        )

        accumulator = _StreamAccumulator()

        async for chunk in stream:
            deltas = accumulator.process_chunk(chunk)
            yield AssistantStream(content=deltas)

        yield AssistantMessage(content=accumulator.build_message().content)
