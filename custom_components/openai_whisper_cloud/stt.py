"""Support for OpenAI Whisper API speech-to-text service."""
from __future__ import annotations

from collections.abc import AsyncIterable
import logging
import wave
import io
import asyncio
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
from homeassistant.const import CONF_API_KEY, CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    SUPPORT_LANGUAGES,
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up speech platform via config entry."""
    async_add_entities([OpenAIWhisperCloudEntity(config_entry)])


class OpenAIWhisperCloudEntity(SpeechToTextEntity):
    """OpenAI Whisper API provider entity."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Init STT service."""
        self.api_key = config_entry.data.get(CONF_API_KEY)
        self.model = config_entry.data.get(CONF_MODEL)
        self.temperature = config_entry.data.get(CONF_TEMPERATURE)
        self.prompt = config_entry.data.get(CONF_PROMPT)
        self._attr_name = "Whisper Cloud STT"
        self._attr_unique_id = config_entry.entry_id

        if self.temperature is None:
            self.temperature = DEFAULT_TEMPERATURE
        if self.prompt is None:
            self.prompt = DEFAULT_PROMPT

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORT_LANGUAGES

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

        data = b""
        async for chunk in stream:
            data += chunk

        if not data:
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

            # Prepare the files parameter with a proper filename
            files = {
                'file': ('audio.wav', temp_file, 'audio/wav'),
            }

            # Prepare the data payload
            data = {
                'model': self.model,
                'language': metadata.language,
                'temperature': self.temperature,
                'prompt': self.prompt,
            }

            # Make the request in a separate thread
            response = await asyncio.to_thread(
                requests.post,
                'https://api.openai.com/v1/audio/transcriptions',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                },
                files=files,
                data=data,
            )

            # Parse the JSON response
            transcription = response.json()

            logging.debug(transcription)

            # Retrieve the transcribed text
            transcribed_text = transcription.get('text', '')

            if transcribed_text is not None:
                return SpeechResult(transcribed_text, SpeechResultState.SUCCESS)
            else:
                return SpeechResult("", SpeechResultState.ERROR)

        except Exception as e:
            logging.error(e)
            return SpeechResult("", SpeechResultState.ERROR)