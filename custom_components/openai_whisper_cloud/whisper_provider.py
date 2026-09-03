"""Providers for the OpenAI Whisper Cloud integration."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Final
from urllib.parse import urlparse

from .const import SUPPORTED_LANGUAGES


@dataclass(frozen=True)
class WhisperModel:
    """A single transcription model exposed by a provider."""

    name: str
    languages: list[str] = field(default_factory=lambda: list(SUPPORTED_LANGUAGES))


@dataclass(frozen=True)
class WhisperProvider:
    """An OpenAI-compatible transcription API provider."""

    key: str
    name: str
    base_url: str = ""
    models: tuple[WhisperModel, ...] = ()
    default_model: str = ""
    transcription_path: str = "/v1/audio/transcriptions"
    supports_temperature: bool = True
    supports_prompt: bool = True
    supports_response_format: bool = True
    multipart_file_field: str = "file"
    custom: bool = False

    @property
    def transcription_url(self) -> str:
        """Return the full URL of the transcription endpoint."""
        if self.custom:
            return resolve_transcription_url(self.base_url)
        return f"{self.base_url.rstrip('/')}{self.transcription_path}"

    def get_model(self, name: str | None) -> WhisperModel:
        """Return the model matching name, falling back gracefully."""
        for model in self.models:
            if model.name == name:
                return model
        if self.models:
            return self.models[0]
        return WhisperModel(name or "whisper-1")


def resolve_transcription_url(base_url: str) -> str:
    """Resolve the transcription endpoint from a user supplied URL.

    Rules are intentionally predictable:
      - no path (``https://host``)            -> ``https://host/v1/audio/transcriptions``
      - versioned base (``https://host/v1``)  -> ``https://host/v1/audio/transcriptions``
      - anything else (any URL that already includes a path) -> used as-is,
        e.g. ``https://host/any/prefix/v1/audio/transcriptions`` or
        ``http://host:8000/asr`` (whisper-asr-webservice)
    """
    url = base_url.strip().rstrip("/")
    path = urlparse(url).path
    if not path:
        return f"{url}/v1/audio/transcriptions"
    if re.search(r"/v\d+$", path):
        return f"{url}/audio/transcriptions"
    return url


def custom_provider(
    base_url: str, model_name: str, file_field: str = "file"
) -> WhisperProvider:
    """Build a provider instance for a user defined OpenAI-compatible endpoint.

    ``file_field`` allows overriding the multipart field name used to upload
    the audio (e.g. ``audio_file`` for whisper-asr-webservice).
    """
    return WhisperProvider(
        key="custom",
        name="Custom",
        base_url=base_url,
        default_model=model_name,
        multipart_file_field=file_field or "file",
        custom=True,
    )


def get_provider(key: str) -> WhisperProvider:
    """Return a provider by key, falling back to OpenAI for unknown keys."""
    return PROVIDERS.get(key, PROVIDERS["openai"])


PROVIDERS: Final[dict[str, WhisperProvider]] = {
    provider.key: provider
    for provider in (
        WhisperProvider(
            key="openai",
            name="OpenAI",
            base_url="https://api.openai.com",
            models=(
                WhisperModel("whisper-1"),
                WhisperModel("gpt-4o-transcribe"),
                WhisperModel("gpt-4o-mini-transcribe"),
            ),
            default_model="gpt-4o-mini-transcribe",
        ),
        WhisperProvider(
            key="groqcloud",
            name="GroqCloud",
            base_url="https://api.groq.com/openai",
            models=(
                WhisperModel("whisper-large-v3"),
                WhisperModel("whisper-large-v3-turbo"),
            ),
            default_model="whisper-large-v3-turbo",
        ),
        WhisperProvider(
            key="mistral",
            name="Mistral AI",
            base_url="https://api.mistral.ai",
            models=(
                WhisperModel(
                    "voxtral-mini-latest",
                    languages=["en", "fr", "de", "es", "it", "pt", "nl", "hi", "ar"],
                ),
            ),
            default_model="voxtral-mini-latest",
        ),
        WhisperProvider(
            key="cohere",
            name="Cohere",
            base_url="https://api.cohere.com",
            models=(WhisperModel("cohere-transcribe-03-2026"),),
            default_model="cohere-transcribe-03-2026",
            transcription_path="/v2/audio/transcriptions",
            supports_temperature=False,
            supports_prompt=False,
            supports_response_format=False,
        ),
        WhisperProvider(key="custom", name="Custom", custom=True),
    )
}

# Mapping of the legacy integer CONF_SOURCE indexes (config entry version <= 1)
# to provider keys. Order must never change: migrations depend on it.
LEGACY_SOURCE_KEYS: Final = ("openai", "groqcloud", "mistral", "cohere")
