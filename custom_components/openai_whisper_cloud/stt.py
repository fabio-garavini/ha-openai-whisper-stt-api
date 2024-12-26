"""OpenAI Whisper API speech-to-text entity."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
import io
import wave

import requests

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
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_NAME, CONF_SOURCE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import _LOGGER
from .const import CONF_PROMPT, CONF_TEMPERATURE
from .whisper_provider import WhisperModel, whisper_providers


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Demo speech platform via config entry."""
    _LOGGER.debug(f"STT setup Entry {config_entry.entry_id}")

    async_add_entities([
        OpenAIWhisperCloudEntity(
            api_url=whisper_providers[config_entry.data[CONF_SOURCE]].url,
            api_key=config_entry.data[CONF_API_KEY],
            model=whisper_providers[config_entry.data[CONF_SOURCE]].models[config_entry.options[CONF_MODEL]],
            temperature=config_entry.options[CONF_TEMPERATURE],
            prompt=config_entry.options[CONF_PROMPT],
            name=config_entry.data[CONF_NAME],
            unique_id=config_entry.entry_id
        )
    ])


class OpenAIWhisperCloudEntity(SpeechToTextEntity):
    """OpenAI Whisper API provider entity."""

    def __init__(self, api_url: str, api_key: str, model: WhisperModel, temperature, prompt, name, unique_id) -> None:
        """Init STT service."""
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.prompt = prompt
        self._attr_name = name
        self._attr_unique_id = unique_id

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

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process an audio stream to STT service."""

        _LOGGER.debug("Processing audio stream: %s", metadata)

        data = b""
        async for chunk in stream:
            data += chunk
            if len(data) / (1024 * 1024) > 24.5:
                _LOGGER.error("Audio stream size exceed the maximum allowed by OpenAI which is 25Mb")
                return SpeechResult("", SpeechResultState.ERROR)

        if not data:
            _LOGGER.error("No audio data received")
            return SpeechResult("", SpeechResultState.ERROR)

        try:
            temp_file = io.BytesIO()
            with wave.open(temp_file, "wb") as wav_file:
                wav_file.setnchannels(metadata.channel)
                wav_file.setframerate(metadata.sample_rate)
                wav_file.setsampwidth(2)
                wav_file.writeframes(data)

            # Ensure the buffer is at the start before passing it
            temp_file.seek(0)

            _LOGGER.debug("Temp wav audio file created of %.2f Mb", temp_file.getbuffer().nbytes / (1024 * 1024))

            # Prepare the files parameter with a proper filename
            files = {
                "file": ("audio.wav", temp_file, "audio/wav"),
            }

            # Prepare the data payload
            data = {
                "model": self.model.name,
                "language": metadata.language,
                "temperature": self.temperature,
                "prompt": self.prompt,
                "response_format": "json",
            }

            # Make the request in a separate thread
            response = await asyncio.to_thread(
                requests.post,
                f"{self.api_url}/v1/audio/transcriptions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
                files=files,
                data=data
            )

            _LOGGER.debug("Transcription request took %f s and returned %d - %s", response.elapsed.seconds, response.status_code, response.reason)

            # Parse the JSON response
            transcription = response.json().get("text", "")

            _LOGGER.debug("TRANSCRIPTION: %s", transcription)

            if not transcription:
                _LOGGER.error(response.text)
                return SpeechResult("", SpeechResultState.ERROR)

            return SpeechResult(transcription, SpeechResultState.SUCCESS)

        except requests.exceptions.RequestException as e:
            _LOGGER.error(e)
            return SpeechResult("", SpeechResultState.ERROR)
