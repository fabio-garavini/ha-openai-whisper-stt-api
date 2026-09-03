"""OpenAI Whisper API speech-to-text entity."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from collections.abc import AsyncIterable
import wave

import aiohttp

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_BASE_URL,
    CONF_FILE_FIELD,
    CONF_PROMPT,
    CONF_PROVIDER_TYPE,
    CONF_TEMPERATURE,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    MAX_AUDIO_SIZE_BYTES,
    TRANSCRIPTION_TIMEOUT,
)
from .whisper_provider import WhisperModel, WhisperProvider, custom_provider, get_provider

_LOGGER = logging.getLogger(__name__)


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


def resolve_language(language: str | None, supported: list[str]) -> str | None:
    """Map a requested language onto one supported by the model.

    Regional variants (e.g. ``pt-br``) are folded onto their base language
    (``pt``) when the base language is supported. This is needed because
    Whisper-style APIs only accept base language codes, while Home Assistant
    may request regional variants for its intent system.
    """
    if not language:
        return None
    lang = language.lower()
    if lang in supported:
        return lang
    base = lang.split("-")[0]
    if base in supported:
        return base
    return lang


def encode_wav(data: bytes, channels: int, sample_rate: int) -> bytes:
    """Wrap raw PCM samples into a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data)
    return buffer.getvalue()


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Whisper speech-to-text platform via config entry."""
    _LOGGER.debug("Setting up STT entity for entry %s", config_entry.entry_id)

    provider_type = config_entry.data[CONF_PROVIDER_TYPE]
    if provider_type == "custom":
        provider = custom_provider(
            config_entry.data.get(CONF_BASE_URL, ""),
            config_entry.options.get(CONF_MODEL, "whisper-1"),
            file_field=config_entry.data.get(CONF_FILE_FIELD, "file"),
        )
    else:
        provider = get_provider(provider_type)

    async_add_entities(
        [
            OpenAIWhisperEntity(
                provider=provider,
                api_key=config_entry.data.get(CONF_API_KEY, ""),
                temperature=float(
                    config_entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
                ),
                prompt=config_entry.options.get(CONF_PROMPT, DEFAULT_PROMPT),
                model_name=config_entry.options.get(CONF_MODEL),
                name=config_entry.data[CONF_NAME],
                unique_id=config_entry.entry_id,
            )
        ]
    )


class OpenAIWhisperEntity(SpeechToTextEntity):
    """OpenAI Whisper API provider entity."""

    def __init__(
        self,
        provider: WhisperProvider,
        api_key: str,
        temperature: float,
        prompt: str,
        model_name: str | None,
        name: str,
        unique_id: str,
    ) -> None:
        """Init the STT entity."""
        self.provider = provider
        self.api_key = api_key
        self.temperature = temperature
        self.prompt = prompt
        self.model: WhisperModel = provider.get_model(model_name)
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, unique_id)},
            name=name,
            manufacturer=provider.name,
            model=self.model.name,
        )

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return self.model.languages

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bit rates."""
        return [
            AudioBitRates.BITRATE_8,
            AudioBitRates.BITRATE_16,
            AudioBitRates.BITRATE_24,
            AudioBitRates.BITRATE_32,
        ]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported sample rates."""
        return [
            AudioSampleRates.SAMPLERATE_8000,
            AudioSampleRates.SAMPLERATE_16000,
            AudioSampleRates.SAMPLERATE_44100,
            AudioSampleRates.SAMPLERATE_48000,
        ]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO, AudioChannels.CHANNEL_STEREO]

    def _build_form(
        self, wav_data: bytes, language: str | None, file_field: str
    ) -> aiohttp.FormData:
        """Build the multipart form for a transcription request."""
        form = aiohttp.FormData()
        form.add_field(
            file_field, wav_data, filename="audio.wav", content_type="audio/wav"
        )
        form.add_field("model", self.model.name)
        if language:
            form.add_field("language", language)
        if self.provider.supports_temperature:
            form.add_field("temperature", str(self.temperature))
        if self.provider.supports_prompt and self.prompt:
            form.add_field("prompt", self.prompt)
        if self.provider.supports_response_format:
            form.add_field("response_format", "json")
        return form

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process an audio stream to text."""
        _LOGGER.debug("Processing audio stream: %s", metadata)

        chunks: list[bytes] = []
        size = 0
        async for chunk in stream:
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_AUDIO_SIZE_BYTES:
                _LOGGER.error(
                    "Audio stream exceeds the maximum allowed size of %.1f MB",
                    MAX_AUDIO_SIZE_BYTES / (1024 * 1024),
                )
                return SpeechResult("", SpeechResultState.ERROR)

        if size == 0:
            _LOGGER.error("No audio data received")
            return SpeechResult("", SpeechResultState.ERROR)

        wav_data = await self.hass.async_add_executor_job(
            encode_wav, b"".join(chunks), int(metadata.channel), int(metadata.sample_rate)
        )
        _LOGGER.debug(
            "Encoded %.2f MB of audio to WAV", len(wav_data) / (1024 * 1024)
        )

        language = resolve_language(metadata.language, self.model.languages)

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        session = async_get_clientsession(self.hass)

        file_field = self.provider.multipart_file_field
        try:
            response = await session.post(
                self.provider.transcription_url,
                data=self._build_form(wav_data, language, file_field),
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=TRANSCRIPTION_TIMEOUT),
            )
            if response.status == 422:
                body = await response.text()
                missing = missing_multipart_field(body)
                if missing and missing != file_field:
                    _LOGGER.info(
                        "Provider expects the audio in multipart field '%s' instead of '%s', retrying",
                        missing,
                        file_field,
                    )
                    file_field = missing
                    response = await session.post(
                        self.provider.transcription_url,
                        data=self._build_form(wav_data, language, file_field),
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=TRANSCRIPTION_TIMEOUT),
                    )

            if response.status != 200:
                body = await response.text()
                _LOGGER.error(
                    "Transcription request failed with status %d: %s",
                    response.status,
                    body[:500],
                )
                return SpeechResult("", SpeechResultState.ERROR)

            body_text = await response.text()
            try:
                result = json.loads(body_text)
            except ValueError:
                # Plain text response (e.g. whisper-asr-webservice /asr)
                result = body_text
            if isinstance(result, dict):
                transcription = result.get("text", "")
            else:
                transcription = str(result).strip()
        except (TimeoutError, asyncio.TimeoutError, aiohttp.ClientError, ValueError) as err:
            _LOGGER.error("Error during transcription: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)

        if not transcription:
            _LOGGER.error("Provider returned an empty transcription")
            return SpeechResult("", SpeechResultState.ERROR)

        _LOGGER.debug("Transcription: %s", transcription)
        return SpeechResult(transcription, SpeechResultState.SUCCESS)
