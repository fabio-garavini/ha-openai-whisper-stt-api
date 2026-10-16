"""HTTP client for OpenAI-compatible speech-to-text endpoints.

The client implements two upload strategies:

* **Streaming**: the audio is uploaded while it is still being recorded.
  The PCM samples are piped into a WAV container whose sizes are set to the
  "unknown" value (``0xFFFFFFFF``), which is what ffmpeg itself writes when
  streaming WAV to a non-seekable output and what every ffmpeg based server
  accepts. Uploading during capture removes the whole upload time from the
  end-to-end latency, which matters most on slow uplinks.

* **Buffered**: the audio is collected, wrapped into a WAV container with
  exact sizes and uploaded with a known content length. This is the most
  compatible path and is always used as a fallback: if a provider rejects a
  streamed request, the audio captured so far is completed from the stream
  and retried buffered, and streaming is latched off for the lifetime of
  the entity so the failed attempt is not repeated on every request.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
import json
import logging
import struct
import uuid
from typing import Any, Final

import aiohttp

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    MAX_AUDIO_SIZE_BYTES,
    REQUEST_TIMEOUT,
    TRANSCRIPTION_TIMEOUT,
)
from .whisper_provider import WhisperProvider

_LOGGER = logging.getLogger(__name__)

# Statuses worth one retry: rate limiting and transient server errors.
RETRYABLE_STATUSES: Final = frozenset({429, 500, 502, 503, 504})
RETRY_DELAY: Final = 1.0
RETRY_DELAY_MAX: Final = 5.0

# The buffered request starts after the recording is complete, so the whole
# timeout can be spent on upload plus processing.
BUFFERED_TIMEOUT: Final = aiohttp.ClientTimeout(total=TRANSCRIPTION_TIMEOUT)
# The streamed request spans the recording, which can take arbitrarily long;
# only a stalled connection (connect or a silent server) trips a timeout.
STREAMED_TIMEOUT: Final = aiohttp.ClientTimeout(
    total=None, connect=REQUEST_TIMEOUT, sock_read=TRANSCRIPTION_TIMEOUT
)

# RIFF/data sizes used for WAVs of unknown length, as written by ffmpeg
# when the output is not seekable.
UNKNOWN_SIZE: Final = 0xFFFFFFFF


class AudioTooLargeError(Exception):
    """Raised when the audio stream exceeds the maximum upload size."""


def wav_header(channels: int, sample_rate: int, data_size: int) -> bytes:
    """Build a canonical 44 byte PCM WAV header for 16 bit samples."""
    block_align = channels * 2
    byte_rate = sample_rate * block_align
    riff_size = UNKNOWN_SIZE if data_size == UNKNOWN_SIZE else 36 + data_size
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, 16
        )
        + b"data"
        + struct.pack("<I", data_size)
    )


def encode_wav(data: bytes, channels: int, sample_rate: int) -> bytes:
    """Wrap raw PCM samples into a WAV container with exact sizes."""
    return wav_header(channels, sample_rate, len(data)) + data


def missing_multipart_field(body: str) -> str | None:
    """Extract the expected multipart field name from a FastAPI 422 error body.

    Servers like whisper-asr-webservice answer with e.g.
    ``{"detail":[{"loc":["body","audio_file"],"type":"value_error.missing"}]}``
    when the audio is uploaded under the wrong multipart field name.
    """
    try:
        detail = json.loads(body).get("detail")
    except (ValueError, AttributeError):
        return None
    if not isinstance(detail, list):
        return None
    for item in detail:
        if not isinstance(item, dict):
            continue
        loc = item.get("loc")
        if not (isinstance(loc, list) and len(loc) >= 2 and loc[0] == "body"):
            continue
        error_type = str(item.get("type", ""))
        message = str(item.get("msg", "")).lower()
        if error_type.endswith("missing") or "required" in message:
            name = loc[-1]
            if isinstance(name, str) and name and name != "body":
                return name
    return None


class WhisperTranscriptionClient:
    """Send transcription requests to an OpenAI-compatible endpoint."""

    def __init__(
        self,
        hass: HomeAssistant,
        provider: WhisperProvider,
        *,
        api_key: str,
        model_name: str,
        temperature: float,
        streaming: bool,
    ) -> None:
        """Initialize the client.

        ``streaming`` is the user preference; whether it is actually used is
        also gated by the provider capability and by what previous requests
        taught us (``_streaming_supported``).
        """
        self._hass = hass
        self._provider = provider
        self._session = async_get_clientsession(hass)
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._model_name = model_name
        self._temperature = temperature
        self._streaming = streaming
        # Runtime knowledge, learned from responses:
        self._streaming_supported: bool | None = None
        self._file_field = provider.multipart_file_field

    async def transcribe(
        self,
        *,
        language: str | None,
        prompt: str,
        channels: int,
        sample_rate: int,
        stream: AsyncIterable[bytes],
    ) -> str | None:
        """Transcribe audio from a stream, returning the text or None."""
        try:
            if self._streaming_enabled():
                chunks: list[bytes] = []
                field_before = self._file_field
                text = await self._transcribe_streamed(
                    language=language,
                    prompt=prompt,
                    channels=channels,
                    sample_rate=sample_rate,
                    stream=stream,
                    chunks=chunks,
                )
                if text is not None:
                    return text
                _LOGGER.debug("Falling back to a buffered upload")
                await self._collect(stream, chunks)
                text = await self._transcribe_buffered(
                    chunks,
                    language=language,
                    prompt=prompt,
                    channels=channels,
                    sample_rate=sample_rate,
                )
                if text is not None and self._file_field == field_before:
                    # The buffered upload worked where the streamed one did
                    # not, so stop paying for a doomed attempt on every
                    # request. When the streamed attempt taught us the right
                    # multipart field name instead, give streaming another
                    # chance with the corrected name.
                    self._disable_streaming()
                return text

            chunks = await self._collect(stream, [])
            return await self._transcribe_buffered(
                chunks,
                language=language,
                prompt=prompt,
                channels=channels,
                sample_rate=sample_rate,
            )
        except AudioTooLargeError:
            _LOGGER.error(
                "Audio stream exceeds the maximum allowed size of %.1f MB",
                MAX_AUDIO_SIZE_BYTES / (1024 * 1024),
            )
            return None
        except (TimeoutError, aiohttp.ClientError) as err:
            # Unexpected network failure (e.g. the connection dropped while
            # reading the response); the specific paths log their own errors.
            _LOGGER.error("Error during transcription: %s", err)
            return None

    async def _transcribe_streamed(
        self,
        *,
        language: str | None,
        prompt: str,
        channels: int,
        sample_rate: int,
        stream: AsyncIterable[bytes],
        chunks: list[bytes],
    ) -> str | None:
        """Upload the audio while it is still being captured.

        The captured audio is teed into ``chunks`` so that a failed streamed
        attempt can be retried buffered without losing any samples.
        """
        boundary = uuid.uuid4().hex
        headers = {
            **self._headers,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        preamble = self._multipart_preamble(boundary, language, prompt)
        preamble += wav_header(channels, sample_rate, UNKNOWN_SIZE)
        # The CRLF is the delimiter of the audio part; it must not be
        # part of the audio data.
        tail = f"\r\n--{boundary}--\r\n".encode()
        size = 0
        too_large = False

        async def body() -> AsyncIterator[bytes]:
            nonlocal size, too_large
            yield preamble
            async for chunk in stream:
                size += len(chunk)
                chunks.append(chunk)
                if size > MAX_AUDIO_SIZE_BYTES:
                    too_large = True
                    raise AudioTooLargeError
                yield chunk
            yield tail

        _LOGGER.debug("Streaming audio to %s", self._provider.transcription_url)
        try:
            response = await self._session.post(
                self._provider.transcription_url,
                data=body(),
                headers=headers,
                timeout=STREAMED_TIMEOUT,
            )
        except (TimeoutError, aiohttp.ClientError) as err:
            if too_large:
                # The generator aborts the request when the cap is hit; the
                # error is raised by aiohttp wrapped in a ClientError.
                raise AudioTooLargeError from err
            _LOGGER.debug("Streamed upload failed: %s", err)
            return None

        if too_large:
            raise AudioTooLargeError
        if response.status != 200:
            error_body = await response.text(errors="replace")
            _LOGGER.debug(
                "Streamed upload rejected with status %d: %s",
                response.status,
                error_body[:300],
            )
            if response.status == 422 and (
                missing := missing_multipart_field(error_body)
            ):
                self._learn_file_field(missing)
            return None
        return await self._parse_response(response)

    async def _transcribe_buffered(
        self,
        chunks: list[bytes],
        *,
        language: str | None,
        prompt: str,
        channels: int,
        sample_rate: int,
    ) -> str | None:
        """Upload the complete audio with a known content length."""
        if not chunks:
            _LOGGER.error("No audio data received")
            return None

        wav_data = await self._hass.async_add_executor_job(
            encode_wav, b"".join(chunks), channels, sample_rate
        )
        _LOGGER.debug("Encoded %.2f MB of audio to WAV", len(wav_data) / (1024 * 1024))

        try:
            response = await self._post_with_retries(
                wav_data, language=language, prompt=prompt
            )
        except (TimeoutError, aiohttp.ClientError) as err:
            _LOGGER.error("Error during transcription: %s", err)
            return None

        if response.status != 200:
            error_body = await response.text(errors="replace")
            _LOGGER.error(
                "Transcription request failed with status %d: %s",
                response.status,
                error_body[:500],
            )
            return None
        return await self._parse_response(response)

    async def _post_with_retries(
        self, wav_data: bytes, *, language: str | None, prompt: str
    ) -> aiohttp.ClientResponse:
        """Post the buffered form, retrying once on recoverable responses.

        Two cases are worth a retry: a 422 that names the multipart field
        the server expects (learned and used from then on), and rate
        limiting or transient server errors (honoring Retry-After).
        """
        response: aiohttp.ClientResponse
        for attempt in (1, 2):
            response = await self._session.post(
                self._provider.transcription_url,
                data=self._build_form(wav_data, language, prompt),
                headers=self._headers,
                timeout=BUFFERED_TIMEOUT,
            )
            if attempt == 2:
                break
            if response.status == 422:
                body = await response.text(errors="replace")
                missing = missing_multipart_field(body)
                if missing and missing != self._file_field:
                    self._learn_file_field(missing)
                    continue
            if response.status in RETRYABLE_STATUSES:
                delay = self._retry_delay(response)
                _LOGGER.info(
                    "Transcription endpoint returned %d, retrying in %.1fs",
                    response.status,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            break
        return response

    def _build_form(
        self, wav_data: bytes, language: str | None, prompt: str
    ) -> aiohttp.FormData:
        """Build the multipart form for a buffered transcription request."""
        form = aiohttp.FormData()
        for name, value in self._form_fields(language, prompt):
            form.add_field(name, value)
        form.add_field(
            self._file_field,
            wav_data,
            filename="audio.wav",
            content_type="audio/wav",
        )
        return form

    def _form_fields(self, language: str | None, prompt: str) -> list[tuple[str, str]]:
        """Return the text fields shared by both upload strategies."""
        fields = [("model", self._model_name)]
        if language:
            fields.append(("language", language))
        if self._provider.supports_temperature:
            fields.append(("temperature", str(self._temperature)))
        if self._provider.supports_prompt and prompt:
            fields.append(("prompt", prompt))
        if self._provider.supports_response_format:
            fields.append(("response_format", "json"))
        return fields

    def _multipart_preamble(
        self, boundary: str, language: str | None, prompt: str
    ) -> bytes:
        """Build the multipart head: text fields plus the audio part header."""
        parts: list[str] = []
        for name, value in self._form_fields(language, prompt):
            parts.append(
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n'
                f"\r\n{value}\r\n"
            )
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{self._file_field}"; '
            f'filename="audio.wav"\r\n'
            f"Content-Type: audio/wav\r\n"
            f"\r\n"
        )
        return "".join(parts).encode()

    async def _collect(
        self, stream: AsyncIterable[bytes], chunks: list[bytes]
    ) -> list[bytes]:
        """Read the remaining audio into ``chunks``, enforcing the size cap."""
        size = sum(len(chunk) for chunk in chunks)
        async for chunk in stream:
            size += len(chunk)
            chunks.append(chunk)
            if size > MAX_AUDIO_SIZE_BYTES:
                raise AudioTooLargeError
        return chunks

    async def _parse_response(self, response: aiohttp.ClientResponse) -> str | None:
        """Extract the transcription text from a provider response."""
        body = await response.text(errors="replace")
        try:
            result: Any = json.loads(body)
        except ValueError:
            # Plain text response (e.g. whisper-asr-webservice /asr)
            result = body
        if isinstance(result, dict):
            transcription = result.get("text", "")
        else:
            transcription = str(result).strip()
        if not transcription:
            _LOGGER.error("Provider returned an empty transcription")
            return None
        return transcription

    def _learn_file_field(self, name: str) -> None:
        """Remember the multipart field the provider expects."""
        if name != self._file_field:
            _LOGGER.info(
                "Provider expects the audio in multipart field '%s' instead of '%s'",
                name,
                self._file_field,
            )
            self._file_field = name

    def _disable_streaming(self) -> None:
        """Latch that this provider does not accept streamed uploads."""
        if self._streaming_supported is not False:
            _LOGGER.info(
                "Provider %s does not accept streamed uploads; "
                "using buffered uploads from now on",
                self._provider.name,
            )
            self._streaming_supported = False

    def _streaming_enabled(self) -> bool:
        """Return True if audio should be streamed while it is captured."""
        return (
            self._streaming
            and self._provider.supports_streaming
            and self._streaming_supported is not False
        )

    @staticmethod
    def _retry_delay(response: aiohttp.ClientResponse) -> float:
        """Return the delay before a retry, honoring a numeric Retry-After."""
        try:
            delay = float(response.headers.get("Retry-After", RETRY_DELAY))
        except ValueError:
            delay = RETRY_DELAY
        return min(max(delay, 0.0), RETRY_DELAY_MAX)
