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
            WhisperModel("whisper-1", SUPPORTED_LANGUAGES),
            WhisperModel("gpt-4o-transcribe", SUPPORTED_LANGUAGES),
            WhisperModel("gpt-4o-mini-transcribe", SUPPORTED_LANGUAGES),
        ],
        2
    ),
    WhisperProvider(
        "GroqCloud",
        "https://api.groq.com/openai",
        [
            WhisperModel("whisper-large-v3", SUPPORTED_LANGUAGES),
            WhisperModel("whisper-large-v3-turbo", SUPPORTED_LANGUAGES)
        ],
        1
    ),
    WhisperProvider(
        "Mistral AI",
        "https://api.mistral.ai",
        [
            WhisperModel("voxtral-mini-latest", languages = ["en", "fr", "de", "es", "it", "pt", "nl", "hi", "ar"])
        ],
        0
    ),
    WhisperProvider("Custom", "", [], 0),
]
