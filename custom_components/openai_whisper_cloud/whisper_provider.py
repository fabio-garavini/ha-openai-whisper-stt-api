"""OpenAI Whisper API Providers."""

from .const import SUPPORTED_LANGUAGES


class WhisperModel:
    """Whisper Model."""

    def __init__(self, name: str, languages: list) -> None:
        """Init."""
        self.name = name
        self.languages = languages


class WhisperProvider:
    """Whisper API Provider."""

    def __init__(self, name: str, url: str, models: list, default_model: int) -> None:
        """Init."""
        self.name = name
        self.url = url
        self.models = models
        self.default_model = default_model


whisper_providers = [
    WhisperProvider(
        "OpenAI",
        "https://api.openai.com",
        [
            WhisperModel("whisper-1", SUPPORTED_LANGUAGES)
        ],
        0
    ),
    WhisperProvider(
        "GroqCloud",
        "https://api.groq.com/openai",
        [
            WhisperModel("whisper-large-v3", SUPPORTED_LANGUAGES),
            WhisperModel("whisper-large-v3-turbo", SUPPORTED_LANGUAGES),
            WhisperModel("distil-whisper-large-v3-en", [ "en" ])
        ],
        0
    ),
    WhisperProvider("Custom", "", [], 0),
]
