"""Tests for the connection mid-failure paths."""

from __future__ import annotations

from typing import Any

import pytest

from custom_components.openai_whisper_cloud.whisper_provider import custom_provider

from .conftest import TEST_TRANSCRIPTION

PCM_CHUNK = b"\x01\x02" * 16000

pytestmark = pytest.mark.usefixtures("socket_enabled")


async def fast_stream(chunks: int = 3) -> Any:
    """Produce audio immediately."""
    for _ in range(chunks):
        yield PCM_CHUNK


async def make_client(hass: Any, url: str, streaming: bool) -> Any:
    from custom_components.openai_whisper_cloud.client import (
        WhisperTranscriptionClient,
    )

    return WhisperTranscriptionClient(
        hass,
        custom_provider(url, "whisper-1"),
        api_key="k",
        model_name="whisper-1",
        temperature=0.0,
        streaming=streaming,
    )


async def test_response_read_failure_fails_transcription(
    hass: Any, transcription_server_factory: Any, caplog: Any
) -> None:
    """A connection that dies before the response body fails cleanly."""
    server = await transcription_server_factory([("abort", None)])
    client = await make_client(hass, server.url, streaming=False)
    assert (
        await client.transcribe(
            language=None,
            prompt="",
            channels=1,
            sample_rate=16000,
            stream=fast_stream(),
        )
        is None
    )
    assert "Error during transcription" in caplog.text


async def test_streamed_request_reset_mid_upload(
    hass: Any, transcription_server_factory: Any
) -> None:
    """A connection reset during the streamed upload is retried buffered."""
    server = await transcription_server_factory([("reset", None), (200, None)])
    client = await make_client(hass, server.url, streaming=True)
    assert (
        await client.transcribe(
            language=None,
            prompt="",
            channels=1,
            sample_rate=16000,
            stream=fast_stream(),
        )
        == TEST_TRANSCRIPTION
    )
    # the reset streamed attempt plus the buffered retry
    assert len(server.requests) == 2
    assert server.requests[0]["reset"] is True
    assert server.requests[1].get("file_field") == "file"


async def test_streamed_response_cut_mid_body(
    hass: Any, transcription_server_factory: Any
) -> None:
    """A response that dies mid-body is retried buffered."""
    server = await transcription_server_factory([("abort", None), (200, None)])
    client = await make_client(hass, server.url, streaming=True)
    assert (
        await client.transcribe(
            language=None,
            prompt="",
            channels=1,
            sample_rate=16000,
            stream=fast_stream(),
        )
        == TEST_TRANSCRIPTION
    )
    assert len(server.requests) == 2
