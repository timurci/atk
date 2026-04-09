"""Tests for AnyLanguageModel — stream_response guard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import anyio
import pytest
from pydantic import BaseModel

from atk.core.message import TextPart, UserMessage
from atk.providers.model import AnyLanguageModel


class _DummyResponse(BaseModel):
    """Minimal model to pass as response_format for the guard test."""

    answer: str


class TestStreamResponseGuard:
    """Test stream_response raises NotImplementedError w/ response_format."""

    @staticmethod
    def test_stream_response_format_not_implemented() -> None:
        mock_client = MagicMock()
        with patch("atk.providers.model.AnyLLM.create", return_value=mock_client):
            model = AnyLanguageModel(provider="openai", model="gpt-4o")

        async def _run() -> None:
            with pytest.raises(
                NotImplementedError, match="Streaming with response_format"
            ):
                async for _ in model.stream_response(
                    instruction="Be helpful",
                    messages=[UserMessage(content=[TextPart(text="Hi")])],
                    response_format=_DummyResponse,
                ):
                    pass

        anyio.run(_run)
