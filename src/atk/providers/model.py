"""any-llm language model implementation."""

import json
from typing import TYPE_CHECKING, Any

from any_llm import AnyLLM
from any_llm.types.completion import ChatCompletion

from atk.core.message import (
    AssistantMessage,
    AssistantStream,
    Message,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
)
from atk.core.model import LanguageModel, StreamingLanguageModel

from .message import MessageMapper
from .tool import ToolMapper

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from any_llm.types.completion import ChatCompletionChunk
    from pydantic import BaseModel

    from atk.core.tool import Tool


class _StreamAccumulator:
    """Accumulates streaming chunks into a final AssistantMessage.

    Provider streams tool call data across multiple chunks, with ``id``
    and ``name`` only present in the first chunk for each tool call
    index.  This class tracks per-index state so that every emitted
    ``ToolCallDelta`` has complete ``id`` and ``name`` fields.
    """

    def __init__(self) -> None:
        self.accumulated_text: list[str] = []
        self.accumulated_reasoning: list[str] = []
        self.tool_call_ids: dict[int, str] = {}
        self.tool_call_names: dict[int, str] = {}
        self.tool_call_args: dict[int, list[str]] = {}

    def process_chunk(
        self, chunk: ChatCompletionChunk
    ) -> list[TextDelta | ToolCallDelta | ThinkingDelta]:
        """Process a single chunk and return mapped delta events."""
        delta = chunk.choices[0].delta
        events: list[TextDelta | ToolCallDelta | ThinkingDelta] = []

        if delta.content is not None:
            self.accumulated_text.append(delta.content)
            events.append(TextDelta(text=delta.content))

        if delta.reasoning is not None:
            self.accumulated_reasoning.append(delta.reasoning.content)
            events.append(ThinkingDelta(thinking=delta.reasoning.content))

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
        content: list[TextPart | ToolCallPart | ThinkingPart] = []
        if self.accumulated_reasoning:
            content.append(ThinkingPart(thinking="".join(self.accumulated_reasoning)))
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


class AnyLanguageModel(LanguageModel, StreamingLanguageModel):
    """Language model implementation backed by any-llm providers."""

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        reasoning_effort: str | None = "auto",
        **client_args: Any,  # noqa: ANN401 — Any is required for passthrough kwargs to AnyLLM.create
    ) -> None:
        """Initialize the any-llm language model.

        Args:
            provider: Provider identifier (e.g. 'openai', 'anthropic', 'mistral').
            model: Default model name. If None, provider-specific default is used.
            api_key: Optional API key. Falls back to environment variable.
            api_base: Optional base URL for the provider API.
            reasoning_effort: Reasoning effort level for models that support it.
                One of 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'auto'.
            **client_args: Additional arguments passed to AnyLLM client creation.
        """
        self._provider = provider
        self._default_model = model
        self._reasoning_effort = reasoning_effort
        self._client = AnyLLM.create(
            provider,
            api_key=api_key,
            api_base=api_base,
            **client_args,
        )
        self._message_mapper = MessageMapper()
        self._tool_mapper = ToolMapper()

    @property
    def model(self) -> str:
        """Return the default model name, falling back to provider default."""
        if self._default_model is not None:
            return self._default_model
        error_msg = (
            "A model name must be provided either in the constructor or per-call"
        )
        raise ValueError(error_msg)

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
        response = await self._client.acompletion(
            model=self.model,
            messages=self._message_mapper.to_messages(instruction, messages),
            tools=self._tool_mapper.to_tools(tools) if tools else None,
            response_format=response_format or None,
            reasoning_effort=self._reasoning_effort,
        )
        if not isinstance(response, ChatCompletion):
            error_msg = "Expected ChatCompletion response for non-streaming call"
            raise TypeError(error_msg)
        return self._message_mapper.from_completion(response.choices[0].message)

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
            error_msg = "Streaming with response_format is not supported"
            raise NotImplementedError(error_msg)

        stream = await self._client.acompletion(
            model=self.model,
            messages=self._message_mapper.to_messages(instruction, messages),
            tools=self._tool_mapper.to_tools(tools) if tools else None,
            stream=True,
            reasoning_effort=self._reasoning_effort,
        )

        accumulator = _StreamAccumulator()

        async for chunk in stream:
            deltas = accumulator.process_chunk(chunk)
            yield AssistantStream(content=deltas)

        yield accumulator.build_message()
