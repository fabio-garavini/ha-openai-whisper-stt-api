"""Tests for the transcription client."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

import custom_components.openai_whisper_cloud.client as client_mod
from custom_components.openai_whisper_cloud.client import (
    UNKNOWN_SIZE,
    WhisperTranscriptionClient,
    encode_wav,
    missing_multipart_field,
    wav_header,
)
from custom_components.openai_whisper_cloud.const import MAX_AUDIO_SIZE_BYTES
from custom_components.openai_whisper_cloud.whisper_provider import (
    ApiStyle,
    WhisperModel,
    WhisperProvider,
    custom_provider,
)

from .conftest import TEST_TRANSCRIPTION

# 32 KB == 1 second of 16 kHz mono audio
PCM_CHUNK = b"\x01\x02" * 16000


def openai_style(server_url: str) -> WhisperProvider:
    """A multipart provider pointing at the local test server."""
    return custom_provider(server_url, "whisper-1")


def make_client(
    hass: Any,
    provider: WhisperProvider,
    *,
    streaming: bool = True,
    temperature: float = 0.0,
) -> WhisperTranscriptionClient:
    """Create a client bound to the patched aiohttp session."""
    return WhisperTranscriptionClient(
        hass,
        provider,
        api_key="secret",
        model_name=provider.default_model,
        temperature=temperature,
        streaming=streaming,
    )


async def fast_stream(chunks: int = 3, chunk: bytes = PCM_CHUNK) -> Any:
    """Produce audio immediately."""
    for _ in range(chunks):
        yield chunk


async def slow_stream(
    chunks: int = 8,
    delay: float = 0.04,
    chunk: bytes = PCM_CHUNK,
    times: list[float] | None = None,
) -> Any:
    """Produce audio slowly, like a live microphone."""
    for _ in range(chunks):
        if times is not None:
            times.append(time.monotonic())
        yield chunk
        await asyncio.sleep(delay)


# These tests talk to scripted local HTTP servers; the plugin restricts
# real sockets to 127.0.0.1.
pytestmark = pytest.mark.usefixtures("socket_enabled")



# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_wav_header_exact_size() -> None:
    """Exact-size headers describe the payload precisely."""
    header = wav_header(1, 16000, 32000)
    assert len(header) == 44
    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    assert header[12:16] == b"fmt "
    assert int.from_bytes(header[16:20], "little") == 16
    assert int.from_bytes(header[20:22], "little") == 1  # PCM
    assert int.from_bytes(header[22:24], "little") == 1  # mono
    assert int.from_bytes(header[24:28], "little") == 16000
    assert int.from_bytes(header[28:32], "little") == 32000  # byte rate
    assert int.from_bytes(header[32:34], "little") == 2  # block align
    assert int.from_bytes(header[34:36], "little") == 16  # bits
    assert header[36:40] == b"data"
    assert int.from_bytes(header[40:44], "little") == 32000
    assert int.from_bytes(header[4:8], "little") == 36 + 32000


def test_wav_header_unknown_size() -> None:
    """Unknown-size headers mark both sizes with the ffmpeg sentinel."""
    header = wav_header(2, 8000, UNKNOWN_SIZE)
    assert header[4:8] == b"\xff\xff\xff\xff"
    assert header[40:44] == b"\xff\xff\xff\xff"
    assert int.from_bytes(header[22:24], "little") == 2  # stereo


def test_encode_wav() -> None:
    """encode_wav prepends an exact-size header to the samples."""
    data = PCM_CHUNK * 2
    wav = encode_wav(data, 1, 16000)
    assert wav[:44] == wav_header(1, 16000, len(data))
    assert wav[44:] == data
    assert len(wav) == 44 + len(data)


def test_missing_multipart_field() -> None:
    """The FastAPI 422 field name is extracted."""
    body = (
        '{"detail":[{"loc":["body","audio_file"],'
        '"type":"value_error.missing","msg":"Field required"}]}'
    )
    assert missing_multipart_field(body) == "audio_file"


def test_missing_multipart_field_by_message() -> None:
    """A 'required' message is also recognized."""
    body = (
        '{"detail":[{"loc":["body","data"],"type":"value_error","msg":"is required"}]}'
    )
    assert missing_multipart_field(body) == "data"


def test_missing_multipart_field_negative_cases() -> None:
    """Malformed or unrelated bodies produce no field name."""
    assert missing_multipart_field("not json") is None
    assert missing_multipart_field('{"detail": "boom"}') is None
    assert missing_multipart_field('{"detail": [1, 2]}') is None
    assert (
        missing_multipart_field(
            '{"detail":[{"loc":["query","x"],"type":"value_error.missing"}]}'
        )
        is None
    )
    assert (
        missing_multipart_field(
            '{"detail":[{"loc":["body"],"type":"value_error.missing"}]}'
        )
        is None
    )
    assert (
        missing_multipart_field(
            '{"detail":[{"loc":["body","body"],"type":"value_error.missing"}]}'
        )
        is None
    )
    assert (
        missing_multipart_field(
            '{"detail":[{"loc":["body",123],"type":"value_error.missing"}]}'
        )
        is None
    )


def test_retry_delay_parsing() -> None:
    """Retry-After is honored and clamped; garbage falls back to default."""
    from unittest.mock import MagicMock

    response = MagicMock()
    response.headers = {"Retry-After": "0.2"}
    assert WhisperTranscriptionClient._retry_delay(response) == pytest.approx(0.2)
    response.headers = {"Retry-After": "99"}
    assert (
        WhisperTranscriptionClient._retry_delay(response) == client_mod.RETRY_DELAY_MAX
    )
    response.headers = {"Retry-After": "-5"}
    assert WhisperTranscriptionClient._retry_delay(response) == 0.0
    response.headers = {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}
    assert WhisperTranscriptionClient._retry_delay(response) == client_mod.RETRY_DELAY
    response.headers = {}
    assert WhisperTranscriptionClient._retry_delay(response) == client_mod.RETRY_DELAY


# ---------------------------------------------------------------------------
# buffered path
# ---------------------------------------------------------------------------


async def test_transcribe_buffered_success(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """A buffered upload sends all form fields and parses the text."""
    server = await transcription_server_factory([(200, None)])
    client = make_client(hass, openai_style(server.url), streaming=False)
    text = await client.transcribe(
        language="en",
        prompt="kitchen, garage",
        channels=1,
        sample_rate=16000,
        stream=fast_stream(),
    )
    assert text == TEST_TRANSCRIPTION
    req = server.requests[0]
    assert req["chunked"] is False
    assert req["auth"] == "Bearer secret"
    assert req["file_field"] == "file"
    assert req["fields"]["model"] == "whisper-1"
    assert req["fields"]["language"] == "en"
    assert req["fields"]["temperature"] == "0.0"
    assert req["fields"]["prompt"] == "kitchen, garage"
    assert req["fields"]["response_format"] == "json"
    audio = req["audio"]
    assert audio[44:] == PCM_CHUNK * 3
    assert int.from_bytes(audio[40:44], "little") == len(audio) - 44


async def test_transcribe_empty_audio_fails_without_request(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """Empty audio returns None and never reaches the provider."""

    async def nothing() -> Any:
        return
        yield  # pragma: no cover - async generator

    server = await transcription_server_factory([(200, None)])
    client = make_client(hass, openai_style(server.url), streaming=False)
    assert (
        await client.transcribe(
            language="en",
            prompt="",
            channels=1,
            sample_rate=16000,
            stream=nothing(),
        )
        is None
    )
    assert server.requests == []


async def test_transcribe_plain_text_response(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """Plain text responses (whisper-asr-webservice) are parsed."""
    server = await transcription_server_factory([(200, TEST_TRANSCRIPTION)])
    client = make_client(hass, openai_style(server.url), streaming=False)
    text = await client.transcribe(
        language=None,
        prompt="",
        channels=1,
        sample_rate=16000,
        stream=fast_stream(),
    )
    assert text == TEST_TRANSCRIPTION


async def test_transcribe_json_string_response(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """A JSON-quoted string response is accepted."""
    server = await transcription_server_factory([(200, '"hello world"')])
    client = make_client(hass, openai_style(server.url), streaming=False)
    text = await client.transcribe(
        language=None,
        prompt="",
        channels=1,
        sample_rate=16000,
        stream=fast_stream(),
    )
    assert text == "hello world"


async def test_transcribe_dict_without_text_fails(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """A JSON dict without a text field fails."""
    server = await transcription_server_factory([(200, {"raw": True})])
    client = make_client(hass, openai_style(server.url), streaming=False)
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


async def test_transcribe_error_status(
    hass: Any,
    transcription_server_factory: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-200 responses fail the transcription."""
    server = await transcription_server_factory([(400, "nope")])
    client = make_client(hass, openai_style(server.url), streaming=False)
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
    assert "status 400" in caplog.text


async def test_transcribe_network_error(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """Connection failures return None."""
    # a port that is closed: the factory server is stopped again
    server = await transcription_server_factory([(200, None)])
    url = server.url
    await server.stop()
    client = make_client(hass, openai_style(url), streaming=False)
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


async def test_transcribe_too_large_buffered(
    hass: Any,
    transcription_server_factory: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Audio above the provider cap fails before any request."""
    import dataclasses

    server = await transcription_server_factory([(200, None)])
    provider = dataclasses.replace(
        openai_style(server.url), max_audio_bytes=len(PCM_CHUNK) * 2
    )
    client = make_client(hass, provider, streaming=False)
    assert (
        await client.transcribe(
            language=None,
            prompt="",
            channels=1,
            sample_rate=16000,
            stream=fast_stream(chunks=10),
        )
        is None
    )
    assert "exceeds the maximum allowed size" in caplog.text
    assert server.requests == []


async def test_transcriptio_retry_on_transient_error(
    hass: Any,
    transcription_server_factory: Any,
    fast_retries: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rate limits and server errors are retried once."""
    server = await transcription_server_factory([(500, None), (200, None)])
    client = make_client(hass, openai_style(server.url), streaming=False)
    text = await client.transcribe(
        language=None,
        prompt="",
        channels=1,
        sample_rate=16000,
        stream=fast_stream(),
    )
    assert text == TEST_TRANSCRIPTION
    assert len(server.requests) == 2
    assert "retrying" in caplog.text


async def test_retry_on_rate_limit_honored_only_once(
    hass: Any,
    transcription_server_factory: Any,
    fast_retries: Any,
) -> None:
    """A persistent 429 gives up after one retry."""
    server = await transcription_server_factory([(429, "slow down")])
    client = make_client(hass, openai_style(server.url), streaming=False)
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
    assert len(server.requests) == 2


async def test_422_field_learning_buffered(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """A 422 naming the expected field is learned and the retry uses it."""
    server = await transcription_server_factory(
        [
            lambda record: (
                (422, "audio_file") if record["file_field"] == "file" else (200, None)
            )
        ]
    )
    client = make_client(hass, openai_style(server.url), streaming=False)
    text = await client.transcribe(
        language=None,
        prompt="",
        channels=1,
        sample_rate=16000,
        stream=fast_stream(),
    )
    assert text == TEST_TRANSCRIPTION
    assert server.requests[0]["file_field"] == "file"
    assert server.requests[1]["file_field"] == "audio_file"


async def test_422_without_hint_not_retried(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """A 422 without a learnable field name fails immediately."""
    server = await transcription_server_factory([(422, {"error": "unparsable"})])
    client = make_client(hass, openai_style(server.url), streaming=False)
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
    assert len(server.requests) == 1


async def test_empty_transcription_fails(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """An empty transcription is an error."""
    server = await transcription_server_factory([(200, {"text": ""})])
    client = make_client(hass, openai_style(server.url), streaming=False)
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


async def test_capability_flags_shape_form(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """Providers without optional capabilities do not send those fields."""
    server = await transcription_server_factory([(200, None)])
    provider = WhisperProvider(
        key="restricted",
        name="Restricted",
        base_url=server.url,
        models=(WhisperModel("m"),),
        default_model="m",
        supports_temperature=False,
        supports_prompt=False,
        supports_response_format=False,
        supports_streaming=False,
    )
    client = make_client(hass, provider, streaming=True)
    assert (
        await client.transcribe(
            language="zh-hans",
            prompt="ignored",
            channels=1,
            sample_rate=16000,
            stream=fast_stream(),
        )
        == TEST_TRANSCRIPTION
    )
    req = server.requests[0]
    assert req["chunked"] is False
    assert req["fields"] == {"model": "m", "language": "zh-hans"}


# ---------------------------------------------------------------------------
# streaming path
# ---------------------------------------------------------------------------


async def test_streamed_upload_overlaps_capture(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """Audio reaches the server while the microphone is still recording."""
    server = await transcription_server_factory([(200, None)])
    client = make_client(hass, openai_style(server.url), streaming=True)
    yield_times: list[float] = []
    text = await client.transcribe(
        language="en",
        prompt="",
        channels=1,
        sample_rate=16000,
        stream=slow_stream(times=yield_times),
    )
    assert text == TEST_TRANSCRIPTION
    req = server.requests[0]
    assert req["chunked"] is True
    # unknown-size WAV header, intact payload
    assert req["audio"][4:8] == b"\xff\xff\xff\xff"
    assert req["audio"][40:44] == b"\xff\xff\xff\xff"
    assert req["audio"][44:] == PCM_CHUNK * 8
    # upload overlapped capture: early chunks arrived before the last one
    # was produced
    assert len(req["audio_arrivals"]) > 1
    assert req["audio_arrivals"][2] < yield_times[-1]


async def test_streamed_rejected_falls_back_and_latches(
    hass: Any,
    transcription_server_factory: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rejected streamed request falls back buffered and latches off."""
    server = await transcription_server_factory(
        [
            lambda record: (
                (400, "chunked not allowed") if record["chunked"] else (200, None)
            )
        ]
    )
    client = make_client(hass, openai_style(server.url), streaming=True)
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
    # streamed attempt (400) then buffered retry (200)
    assert len(server.requests) == 2
    assert server.requests[0]["chunked"] is True
    assert server.requests[1]["chunked"] is False
    assert client._streaming_supported is False
    assert "using buffered uploads from now on" in caplog.text
    # subsequent requests skip the doomed streamed attempt entirely
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
    assert len(server.requests) == 3
    assert server.requests[2]["chunked"] is False


async def test_streamed_failure_without_latch_when_both_fail(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """When buffered also fails, streaming is not latched off."""
    server = await transcription_server_factory([(400, "no")])
    client = make_client(hass, openai_style(server.url), streaming=True)
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
    assert client._streaming_supported is None
    assert len(server.requests) == 2


async def test_streamed_422_learns_field_and_retries_streamed(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """A 422 field hint is learned; the next request streams with it."""
    server = await transcription_server_factory(
        [
            lambda record: (
                (422, "audio_file") if record["file_field"] == "file" else (200, None)
            )
        ]
    )
    client = make_client(hass, openai_style(server.url), streaming=True)
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
    assert server.requests[0]["file_field"] == "file"
    assert server.requests[1]["file_field"] == "audio_file"
    assert client._file_field == "audio_file"
    # the field name was the problem, not streaming: no latch
    assert client._streaming_supported is None
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
    last = server.requests[-1]
    assert last["chunked"] is True
    assert last["file_field"] == "audio_file"


async def test_streamed_too_large_aborts(
    hass: Any,
    transcription_server_factory: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The size cap aborts a streamed upload mid-request."""
    import dataclasses

    server = await transcription_server_factory([(200, None)])
    provider = dataclasses.replace(
        openai_style(server.url), max_audio_bytes=len(PCM_CHUNK) * 2
    )
    client = make_client(hass, provider, streaming=True)
    assert (
        await client.transcribe(
            language=None,
            prompt="",
            channels=1,
            sample_rate=16000,
            stream=fast_stream(chunks=10),
        )
        is None
    )
    assert "exceeds the maximum allowed size" in caplog.text
    if server.requests:
        assert server.requests[0]["aborted"] or (
            len(server.requests[0]["audio"]) <= len(PCM_CHUNK) * 3
        )


async def test_streamed_network_failure_drains_and_retries_buffered(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """A streamed failure keeps the audio and retries it buffered."""
    server = await transcription_server_factory([(200, None)])
    # make every streamed request fail with 400, buffered succeed
    server.responses = [
        lambda record: (400, "no chunked") if record["chunked"] else (200, None)
    ]
    client = make_client(hass, openai_style(server.url), streaming=True)
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
    assert server.requests[1]["audio"][44:] == PCM_CHUNK * 3


async def test_streaming_disabled_by_provider_capability(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """Chat-style providers never stream, even if the user enabled it."""
    provider = WhisperProvider(
        key="chat",
        name="Chat",
        base_url="http://127.0.0.1:1",
        models=(WhisperModel("m"),),
        default_model="m",
        api_style=ApiStyle.CHAT_COMPLETIONS,
        supports_streaming=True,  # even if wrongly enabled
    )
    client = make_client(hass, provider, streaming=True)
    assert client._streaming_enabled() is False


# ---------------------------------------------------------------------------
# chat completions (Xiaomi MiMo)
# ---------------------------------------------------------------------------


def mimo_style(server_url: str) -> WhisperProvider:
    """A MiMo-like provider pointing at the local test server."""
    from custom_components.openai_whisper_cloud.const import MIMO_MAX_AUDIO_BYTES
    from custom_components.openai_whisper_cloud.whisper_provider import (
        MIMO_ASR_LANGUAGES,
    )

    return WhisperProvider(
        key="mimo",
        name="MiMo",
        base_url=server_url,
        models=(WhisperModel("mimo-v2.5-asr", languages=MIMO_ASR_LANGUAGES),),
        default_model="mimo-v2.5-asr",
        transcription_path="/v1/chat/completions",
        api_style=ApiStyle.CHAT_COMPLETIONS,
        supports_temperature=False,
        supports_prompt=False,
        supports_response_format=False,
        supports_streaming=False,
        max_audio_bytes=MIMO_MAX_AUDIO_BYTES,
    )


async def test_chat_completions_request_shape(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """MiMo requests are JSON chat completions with a base64 data URL."""
    server = await transcription_server_factory(
        [(200, {"choices": [{"message": {"content": TEST_TRANSCRIPTION}}]})]
    )
    client = make_client(hass, mimo_style(server.url), streaming=True)
    assert (
        await client.transcribe(
            language="zh-hans",
            prompt="ignored",
            channels=1,
            sample_rate=16000,
            stream=fast_stream(),
        )
        == TEST_TRANSCRIPTION
    )
    req = server.requests[0]
    assert req["kind"] == "chat_completions"
    assert req["chunked"] is False
    payload = req["json"]
    assert payload["model"] == "mimo-v2.5-asr"
    part = payload["messages"][0]["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"
    assert payload["asr_options"] == {"language": "zh"}
    assert "temperature" not in payload
    assert "response_format" not in payload
    # decoded data URL is an exact-size WAV carrying the samples
    audio = req["audio"]
    assert audio[:44] == wav_header(1, 16000, len(audio) - 44)
    assert audio[44:] == PCM_CHUNK * 3


async def test_chat_completions_language_mapping(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """Only zh/en are sent; anything else is left to auto detection."""
    server = await transcription_server_factory(
        [(200, {"choices": [{"message": {"content": TEST_TRANSCRIPTION}}]})] * 3
    )
    client = make_client(hass, mimo_style(server.url))
    for language, expected in [("en", {"language": "en"}), ("zh", {"language": "zh"})]:
        await client.transcribe(
            language=language,
            prompt="",
            channels=1,
            sample_rate=16000,
            stream=fast_stream(),
        )
        assert server.requests[-1]["json"]["asr_options"] == expected
    await client.transcribe(
        language=None,
        prompt="",
        channels=1,
        sample_rate=16000,
        stream=fast_stream(),
    )
    assert "asr_options" not in server.requests[-1]["json"]


async def test_chat_completions_size_cap(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """The 10 MB encoded limit is enforced on the raw audio."""
    import dataclasses

    server = await transcription_server_factory(
        [(200, {"choices": [{"message": {"content": TEST_TRANSCRIPTION}}]})]
    )
    provider = dataclasses.replace(
        mimo_style(server.url), max_audio_bytes=len(PCM_CHUNK) * 2
    )
    client = make_client(hass, provider)
    assert (
        await client.transcribe(
            language=None,
            prompt="",
            channels=1,
            sample_rate=16000,
            stream=fast_stream(chunks=10),
        )
        is None
    )


async def test_chat_completions_bad_response_shape(
    hass: Any,
    transcription_server_factory: Any,
) -> None:
    """Malformed chat responses fail instead of crashing."""
    server = await transcription_server_factory([(200, {"choices": []})])
    client = make_client(hass, mimo_style(server.url))
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
    server.responses = [(200, {"choices": [{"message": {}}]})]
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


async def test_chat_completions_transient_retry(
    hass: Any,
    transcription_server_factory: Any,
    fast_retries: Any,
) -> None:
    """Retries work the same for chat completions requests."""
    server = await transcription_server_factory(
        [
            (500, "boom"),
            (200, {"choices": [{"message": {"content": TEST_TRANSCRIPTION}}]}),
        ]
    )
    client = make_client(hass, mimo_style(server.url))
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


async def test_provider_max_audio_bytes_default() -> None:
    """The default cap is the documented 24.5 MB."""
    assert MAX_AUDIO_SIZE_BYTES == int(24.5 * 1024 * 1024)
