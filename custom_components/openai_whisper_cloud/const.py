"""Constants for the OpenAI Whisper Cloud integration."""

from typing import Final

DOMAIN: Final = "openai_whisper_cloud"

# Config entry data keys
CONF_PROVIDER_TYPE: Final = "provider_type"
CONF_BASE_URL: Final = "base_url"
CONF_FILE_FIELD: Final = "file_field"

# Options keys
CONF_PROMPT: Final = "prompt"
CONF_TEMPERATURE: Final = "temperature"

# Defaults
DEFAULT_NAME: Final = "Whisper Cloud"
DEFAULT_TEMPERATURE: Final = 0.0
DEFAULT_PROMPT: Final = ""
DEFAULT_CUSTOM_NAME: Final = "Custom Whisper"
DEFAULT_CUSTOM_MODEL: Final = "whisper-1"

# Networking
REQUEST_TIMEOUT: Final = 30
TRANSCRIPTION_TIMEOUT: Final = 60
MAX_AUDIO_SIZE_BYTES: Final = int(24.5 * 1024 * 1024)

# Config entry version this release migrates to
CONFIG_VERSION: Final = 2
CONFIG_MINOR_VERSION: Final = 1

SUPPORTED_LANGUAGES: Final = [
    "af",
    "ar",
    "hy",
    "az",
    "be",
    "bs",
    "bg",
    "ca",
    "zh",
    "zh-cn",
    "zh-tw",
    "zh-hk",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "gl",
    "de",
    "el",
    "he",
    "hi",
    "hu",
    "is",
    "id",
    "it",
    "ja",
    "kn",
    "kk",
    "ko",
    "lv",
    "lt",
    "mk",
    "ms",
    "mr",
    "mi",
    "ne",
    "no",
    "fa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sr",
    "sk",
    "sl",
    "es",
    "sw",
    "sv",
    "tl",
    "ta",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
    "cy",
]
