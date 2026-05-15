"""Tests for AnyLanguageModel."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import anyio
import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
)
from pydantic import BaseModel

from atk.core.message import (
    AssistantMessage,
    AssistantStream,
    TextDelta,
    TextPart,
    UserMessage,
)
from atk.core.tool import PrimitiveParameter, Tool
from atk.providers.model import AnyLanguageModel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _DummyResponse(BaseModel):
    """Minimal model to pass as response_format."""

    answer: str


class _FakeClient:
    def __init__(self, *, response=None, stream=None, models=None) -> None:
        self.response = response
        self.stream = stream
        self.models = models or []
        self.completion_calls = []
        self.list_models_calls = 0

    async def acompletion(
        self,
        **kwargs: Any,  # noqa: ANN401 — fake captures arbitrary provider kwargs.
    ) -> object:
        self.completion_calls.append(kwargs)
        if kwargs.get("stream"):
            return self.stream
        return self.response

    async def alist_models(self) -> list[SimpleNamespace]:
        self.list_models_calls += 1
        return self.models


class _FakeClientFactory:
    def __init__(self, *clients: _FakeClient) -> None:
        self.clients = list(clients)
        self.calls = []

    def __call__(
        self,
        provider: str,
        **kwargs: Any,  # noqa: ANN401 — fake captures arbitrary provider kwargs.
    ) -> _FakeClient:
        self.calls.append((provider, kwargs))
        return self.clients.pop(0)


class _AsyncChunkStream:
    def __init__(self, chunks) -> None:
        self.chunks = chunks

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ChatCompletionChunk]:
        for chunk in self.chunks:
            yield chunk


def _completion(content: str = "Hello") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl_test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(content=content, role="assistant"),
            )
        ],
        created=0,
        model="test-model",
        object="chat.completion",
    )


def _chunk(content: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chunk_test",
        choices=[
            ChunkChoice(
                delta=ChoiceDelta(content=content),
                finish_reason=None,
                index=0,
            )
        ],
        created=0,
        model="test-model",
        object="chat.completion.chunk",
    )


def _user_message(text: str = "Hi") -> UserMessage:
    return UserMessage(content=[TextPart(text=text)])


def _tool() -> Tool:
    return Tool(
        name="search",
        description="Search documents",
        parameters={
            "query": PrimitiveParameter(type="string", description="Search query"),
        },
        required=["query"],
    )


def _patch_client_factory(monkeypatch, factory: _FakeClientFactory) -> None:
    monkeypatch.setattr("atk.providers.model.AnyLLM.create", factory)


class TestCreate:
    """Test AnyLanguageModel.create factory behavior."""

    @staticmethod
    def test_create_uses_explicit_model_without_listing_models(monkeypatch) -> None:
        client = _FakeClient(response=_completion())
        factory = _FakeClientFactory(client)
        _patch_client_factory(monkeypatch, factory)

        async def run() -> None:
            model = await AnyLanguageModel.create(
                provider="openai",
                model="gpt-4o",
                api_key="key",
                api_base="https://example.test",
                reasoning_effort="low",
                timeout=30,
            )

            assert model.model == "gpt-4o"
            assert client.list_models_calls == 0
            assert factory.calls == [
                (
                    "openai",
                    {
                        "api_key": "key",
                        "api_base": "https://example.test",
                        "timeout": 30,
                    },
                )
            ]

        anyio.run(run)

    @staticmethod
    def test_create_auto_discovers_single_model(monkeypatch) -> None:
        discovery_client = _FakeClient(models=[SimpleNamespace(id="only-model")])
        completion_client = _FakeClient(response=_completion())
        factory = _FakeClientFactory(discovery_client, completion_client)
        _patch_client_factory(monkeypatch, factory)

        async def run() -> None:
            model = await AnyLanguageModel.create(provider="openai")

            assert model.model == "only-model"
            assert discovery_client.list_models_calls == 1
            assert len(factory.calls) == 2

        anyio.run(run)

    @staticmethod
    @pytest.mark.parametrize(
        "models",
        [[], [SimpleNamespace(id="a"), SimpleNamespace(id="b")]],
    )
    def test_create_without_model_requires_exactly_one_available_model(
        monkeypatch,
        models,
    ) -> None:
        factory = _FakeClientFactory(_FakeClient(models=models))
        _patch_client_factory(monkeypatch, factory)

        async def run() -> None:
            with pytest.raises(
                NotImplementedError,
                match="could not determine a default model",
            ):
                await AnyLanguageModel.create(provider="openai")

        anyio.run(run)


class TestGenerateResponse:
    """Test non-streaming response generation."""

    @staticmethod
    def test_generate_response_builds_request_and_maps_message(monkeypatch) -> None:
        client = _FakeClient(response=_completion("Found it."))
        _patch_client_factory(monkeypatch, _FakeClientFactory(client))
        model = AnyLanguageModel(
            provider="openai",
            model="gpt-4o",
            reasoning_effort="medium",
        )

        async def run() -> None:
            result = await model.generate_response(
                instruction="Be concise",
                messages=[_user_message("Search docs")],
                tools=[_tool()],
                response_format=_DummyResponse,
            )

            assert result == AssistantMessage(content=[TextPart(text="Found it.")])
            assert client.completion_calls == [
                {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "Be concise"},
                        {"role": "user", "content": "Search docs"},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "description": "Search documents",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "Search query",
                                        },
                                    },
                                    "required": ["query"],
                                },
                                "strict": True,
                            },
                        }
                    ],
                    "response_format": _DummyResponse,
                    "reasoning_effort": "medium",
                }
            ]

        anyio.run(run)

    @staticmethod
    def test_generate_response_omits_tools_when_none(monkeypatch) -> None:
        client = _FakeClient(response=_completion())
        _patch_client_factory(monkeypatch, _FakeClientFactory(client))
        model = AnyLanguageModel(provider="openai", model="gpt-4o")

        async def run() -> None:
            await model.generate_response(
                instruction="Be helpful",
                messages=[_user_message()],
                tools=None,
            )

            assert client.completion_calls[0]["tools"] is None
            assert client.completion_calls[0]["response_format"] is None
            assert client.completion_calls[0]["reasoning_effort"] == "auto"

        anyio.run(run)

    @staticmethod
    def test_generate_response_rejects_non_chat_completion(monkeypatch) -> None:
        client = _FakeClient(response={"unexpected": "shape"})
        _patch_client_factory(monkeypatch, _FakeClientFactory(client))
        model = AnyLanguageModel(provider="openai", model="gpt-4o")

        async def run() -> None:
            with pytest.raises(TypeError, match="Expected ChatCompletion"):
                await model.generate_response(
                    instruction="Be helpful",
                    messages=[_user_message()],
                )

        anyio.run(run)


class TestStreamResponse:
    """Test streaming response generation."""

    @staticmethod
    def test_stream_response_accumulates_chunks(monkeypatch) -> None:
        client = _FakeClient(stream=_AsyncChunkStream([_chunk("Hel"), _chunk("lo")]))
        _patch_client_factory(monkeypatch, _FakeClientFactory(client))
        model = AnyLanguageModel(
            provider="openai",
            model="gpt-4o",
            reasoning_effort="high",
        )

        async def run() -> None:
            results = [
                event
                async for event in model.stream_response(
                    instruction="Be helpful",
                    messages=[_user_message()],
                    tools=None,
                )
            ]

            assert results == [
                AssistantStream(content=[TextDelta(text="Hel")]),
                AssistantStream(content=[TextDelta(text="lo")]),
                AssistantMessage(content=[TextPart(text="Hello")]),
            ]
            assert client.completion_calls == [
                {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "Be helpful"},
                        {"role": "user", "content": "Hi"},
                    ],
                    "tools": None,
                    "stream": True,
                    "reasoning_effort": "high",
                }
            ]

        anyio.run(run)

    @staticmethod
    def test_stream_response_format_not_implemented(monkeypatch) -> None:
        client = _FakeClient()
        _patch_client_factory(monkeypatch, _FakeClientFactory(client))
        model = AnyLanguageModel(provider="openai", model="gpt-4o")

        async def run() -> None:
            with pytest.raises(
                NotImplementedError,
                match="Streaming with response_format",
            ):
                async for _ in model.stream_response(
                    instruction="Be helpful",
                    messages=[_user_message()],
                    response_format=_DummyResponse,
                ):
                    pass

        anyio.run(run)
